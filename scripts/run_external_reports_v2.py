"""Rerun existing paper's adapted external report protocols on v2 evidence.

These are task adaptations, not exact reproductions of the cited papers.
Original local adapters are retained under supplement_adapters for provenance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src_report'), str(ROOT/'supplement_adapters')]
from design_evidence import format_measurement, is_v2
from generate import _call_api, _image_content_block
import clip2safety_original_adapter as clip
import construction_original_adapter as construction


def call_with_retry(messages, model, base_url, key, attempts=5):
    for attempt in range(attempts):
        try:
            return _call_api(messages, model, base_url, key, timeout=120)
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            retryable = isinstance(exc, (requests.Timeout, requests.ConnectionError)) or status in (408, 429, 500, 502, 503, 504)
            if not retryable or attempt == attempts - 1:
                raise
            delay = min(10 * 2**attempt, 60)
            print(f'[RETRY] HTTP {status or "connection"}; attempt {attempt+2}/{attempts} in {delay}s', flush=True)
            time.sleep(delay)


def prompt(rec, evidence):
    text = ('Inspect the rebar installation and write one concise professional inspection paragraph. '
            'Describe observable construction-quality evidence. ')
    if evidence:
        text += ('The visual inputs show the RGB scene and its predicted segmentation mask. '
                 'Use the supplied measurements and region-specific design tolerances as authoritative. '
                 'Include the count, spacing statistics, compliance result and non-compliant interval indices '
                 'when available, then provide a corrective or readiness statement. '
                 'Do not estimate millimeter values from pixels or invent regulatory clauses.\n'
                 + format_measurement(rec['measurement'], include_rule=True))
    else:
        text += ('Use only the RGB image. Do not invent precise spacing values, design specifications, '
                 'tolerances or a compliance decision that cannot be established from the image.')
    return text


def inputs(a, site):
    source = a.input_root/f'{site}_full_seed42/within/report_inputs.jsonl'
    rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
    ids = (a.data_root/site/'gt_sets/val.txt').read_text().split()
    keys = [str(r['id']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(ids):
        raise ValueError('Input IDs differ from validation split: '+str(source))
    for r in rows:
        if r.get('subset') != site or r.get('split') != 'val' or not is_v2(r.get('measurement', {})):
            raise ValueError('Requires v2 validation evidence from the correct site')
        r['image_path'] = str(a.data_root/site/'images'/f"{r['id']}.png")
        r['seg_path'] = str(a.input_root/f'{site}_full_seed42/predictions/within/model'/site/'seg'/f"{r['id']}.png")
        for key in ('image_path', 'seg_path'):
            if not Path(r[key]).is_file():
                raise FileNotFoundError(r[key])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['construction', 'clip'], required=True)
    p.add_argument('--input-root', type=Path, default=ROOT/'runs/revision_full_seed42')
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--out-root', type=Path, default=ROOT/'runs/revision_external_reports_v2')
    p.add_argument('--model', default='qwen3.8-omni-flash')
    p.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    p.add_argument('--hf-model', default='llava-hf/llava-v1.6-mistral-7b-hf')
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    for key in ('input_root', 'data_root', 'out_root'):
        setattr(a, key, getattr(a, key).resolve())
    sites = {s:inputs(a,s) for s in ('site-1','site-2')}
    settings = ('rgb_only', 'rgb_seg_measure_rule')
    print('Requests:', sum(len(rows) for rows in sites.values())*2, '(full run)')
    print('Protocols:', settings, 'Method:', a.method)
    if a.plan_only:
        return
    model_id = a.model if a.method == 'construction' else a.hf_model
    manifest = dict(method=a.method, model=model_id, base_url=a.base_url,
                    max_new_tokens=512, temperature=.2 if a.method=='construction' else 0,
                    load_in_4bit=a.load_in_4bit if a.method=='clip' else False,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    sources={s:hashlib.sha256((a.input_root/f'{s}_full_seed42/within/report_inputs.jsonl').read_bytes()).hexdigest() for s in sites},
                    images={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for rows in sites.values() for r in rows for p in (Path(r['image_path']),Path(r['seg_path']))})
    a.out_root.mkdir(parents=True, exist_ok=True)
    config = a.out_root/(a.method+'_manifest.json')
    if config.exists() and json.loads(config.read_text()) != manifest:
        previous = json.loads(config.read_text())
        old_hash = previous.get('script_sha256')
        compatible = dict(previous, script_sha256=manifest['script_sha256'])
        if old_hash != '57b30b4705d383f92efa3c99b8113a66db83fa0e8f71fa280c3cba42fd5cb0df' or compatible != manifest:
            raise ValueError('Model, code or evidence changed; use another --out-root')
        backup = config.with_name(config.stem+'_before_retry.json')
        if not backup.exists():
            backup.write_text(json.dumps(previous, indent=2), encoding='utf-8')
        print('Applying transport-retry-only update; prompts and generation parameters unchanged.')
    config.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if a.method == 'construction':
        key = os.environ.get('DASHSCOPE_API_KEY')
        if not key:
            raise SystemExit('Set DASHSCOPE_API_KEY in this terminal')
    else:
        import torch
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if a.load_in_4bit else None
        processor = AutoProcessor.from_pretrained(a.hf_model)
        model = LlavaNextForConditionalGeneration.from_pretrained(a.hf_model,
            torch_dtype=torch.float16, device_map='auto', quantization_config=quant).eval()
    for site, rows in sites.items():
        outputs = []
        for setting in settings:
            path = a.out_root/site/(a.method+'_'+setting+'.jsonl')
            outputs.append(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            done = {}
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    rec = json.loads(line)
                    if rec.get('error') or not rec.get('prediction'):
                        raise ValueError('Unexpected failed record: '+str(path))
                    if str(rec['id']) in done:
                        raise ValueError('Duplicate record: '+str(path))
                    done[str(rec['id'])] = rec
            attempts = 0
            with path.open('a', encoding='utf-8') as f:
                for r in rows:
                    if str(r['id']) in done:
                        continue
                    if a.limit is not None and attempts >= a.limit:
                        break
                    evidence = setting != 'rgb_only'
                    text = prompt(r, evidence)
                    try:
                        if a.method == 'construction':
                            content = [_image_content_block(r['image_path'])]
                            if evidence:
                                content.append(_image_content_block(r['seg_path']))
                            messages = [{'role':'system','content':construction.SYSTEM_PROMPT},
                                        {'role':'user','content':content+[{'type':'text','text':text}]}]
                            prediction = call_with_retry(messages,a.model,a.base_url,key)
                        else:
                            image = clip._load_image(r, 'rgb_seg' if evidence else 'rgb')
                            conversation = [{'role':'user','content':[{'type':'image'}, {'type':'text','text':text}]}]
                            rendered = processor.apply_chat_template(conversation, add_generation_prompt=True)
                            batch = processor(text=rendered,images=image,return_tensors='pt').to(model.device)
                            with torch.inference_mode():
                                tokens = model.generate(**batch,max_new_tokens=512,do_sample=False)
                            prediction = processor.decode(tokens[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                        if not prediction:
                            raise ValueError('Empty report')
                    except Exception as exc:
                        # Keep failures separate so retries never count them as completed reports.
                        error = str(exc)
                        if a.method == 'construction':
                            error = error.replace(key,'<redacted>')
                        with (path.parent/'failures.jsonl').open('a',encoding='utf-8') as err:
                            err.write(json.dumps(dict(id=r['id'],setting=setting,error=error))+'\n')
                        raise SystemExit('Stopped; fix failure and rerun the same command: '+error)
                    record = {k:r.get(k) for k in ('subset','id','variant','reference_report','measurement')}
                    record.update(setting=a.method+'_'+setting,model=model_id,prediction=prediction,error=None,
                                  prompt_hash=hashlib.sha256(text.encode()).hexdigest())
                    f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()
                    attempts += 1
                    print('[OK]',site,r['id'],setting,flush=True)
                    if a.method == 'construction':
                        time.sleep(1.5)
        if a.limit is None:
            for path in outputs:
                generated = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
                if {str(r['id']) for r in generated} != {str(r['id']) for r in rows}:
                    raise ValueError('Incomplete output: '+str(path))
            subprocess.run([sys.executable,str(ROOT/'src_report/eval.py'),'--gen_files',*map(str,outputs),
                            '--out_dir',str(a.out_root/site/(a.method+'_metrics'))],check=True)


if __name__ == '__main__':
    main()
