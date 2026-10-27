"""Prepare independent-evidence report drafts, review them, and export groundtruth."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'src_report'))
from generate import _call_api, _image_content_block
from design_evidence import format_measurement

SYSTEM = (
    'Draft a reference rebar inspection report for subsequent human verification. '
    'Use only the attached RGB photograph and the independently supplied physical reference readings. '
    'Do not infer numerical dimensions from pixels or invent unavailable facts. '
    'Do not use model predictions, segmentation outputs, QAEP plans, or other generated reports. '
    'Write one English paragraph of six complete sentences, with no Markdown or headings. '
    'Include directly visible bar arrangement or distribution and installation context only where the image supports them; '
    'separate visual observations from numerical findings. Include the supplied count, mean spacing, '
    'standard deviation, design spacing and signed tolerance, failed zero-based interval indices, '
    'maximum design deviation, supplied risk level and final result where available. '
    'Do not invent structural safety, hidden conditions, causal explanations, or statutory acceptance. '
    'Do not state the installation is uniformly spaced merely because the mean matches the design. '
    'Finish with a brief recommendation consistent with the supplied evidence. '
    'These instructions produce a candidate reference, not an automatically verified groundtruth.'
)


def load(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def prepare(out, data_root, physical_root, sites):
    if (out/'inputs.jsonl').exists():
        raise FileExistsError('Prepared inputs already exist; use the existing folder for generation/review.')
    records = []
    for site in sites:
        source = physical_root/f'{site}_full_seed42/within/physical_references.jsonl'
        refs = load(source)
        if not refs:
            raise ValueError(f'Empty physical references for {site}')
        ids = data_root/site/'gt_sets/val.txt'
        if ids.exists():
            expected = {s.strip() for s in ids.read_text().splitlines() if s.strip()}
            if expected != {str(r['id']) for r in refs}:
                raise ValueError(f'Validation IDs differ for {site}')
        for r in refs:
            m = r['measurement']
            if m.get('schema_version') != 'design_physical_v2' or not m.get('compliance',{}).get('rule'):
                raise ValueError(f'Missing current physical/design reference: {site}/{r["id"]}')
            image = data_root/site/'images'/f'{r["id"]}.png'
            if not image.exists():
                raise FileNotFoundError(image)
            records.append({'subset':site,'id':str(r['id']),'measurement':m,
                            'image_path':str(image),'physical_source':str(source),
                            'physical_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    if len({(r['subset'],r['id']) for r in records}) != len(records):
        raise ValueError('Duplicate validation sample')
    out.mkdir(parents=True,exist_ok=True)
    (out/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(f'Prepared {len(records)} RGB + physical-reference inputs; no predicted measurements or model reports used.')


def generate(args):
    records = load(args.out/'inputs.jsonl')
    drafts = args.out/'drafts.jsonl'
    done = {(r['subset'],r['id']) for r in load(drafts)} if drafts.exists() else set()
    pending = [r for r in records if (r['subset'],r['id']) not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        print('No pending generations.'); return
    key = os.environ.get('DASHSCOPE_API_KEY')
    if not key:
        raise ValueError('Set DASHSCOPE_API_KEY; never put the key in report files.')
    review = args.out/'reports_for_review'
    review.mkdir(exist_ok=True)
    for r in pending:
        content = format_measurement(r['measurement'],include_rule=True)
        messages = [{'role':'system','content':SYSTEM},
                    {'role':'user','content':[_image_content_block(r['image_path']),
                      {'type':'text','text':'Physical reference and regional design evidence:\n'+content}]}]
        for attempt in range(4):
            try:
                report = _call_api(messages,args.model,args.base_url,key,timeout=120)
                if not report.strip(): raise ValueError('Empty generated draft')
                break
            except Exception as exc:
                response = getattr(exc,'response',None)
                status = getattr(response,'status_code',None)
                if status not in (None,408,429,500,502,503,504) or attempt==3:
                    raise
                print(f'Retrying {r["subset"]}/{r["id"]}; attempt {attempt+2}',flush=True)
                time.sleep(min(2**(attempt+1),10))
        target = review/f'{r["subset"]}_{r["id"]}.txt'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite a potentially edited report: {target}')
        target.write_text(report+'\n',encoding='utf-8')
        row = dict(r,reference_report=report,draft_model=args.model,base_url=args.base_url,
                   generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   text_prompt_sha256=hashlib.sha256((SYSTEM+content).encode()).hexdigest(),
                   image_sha256=hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest(),
                   reviewed=False,report_file=str(target))
        with drafts.open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(f'OK {r["subset"]}/{r["id"]}',flush=True)
    print('Drafts generated. Run make-review to create the review sheet.')


def make_review(out):
    target = out/'review.csv'
    if target.exists():
        raise FileExistsError('Review sheet exists; do not overwrite completed review.')
    rows = load(out/'drafts.jsonl')
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['subset','id','report_file','image_path','numbers_and_rules_checked','visual_description_checked','approved','notes'])
        writer.writeheader()
        for r in rows:
            writer.writerow({k:r.get(k,'') for k in writer.fieldnames})
    print(f'Created review sheet for {len(rows)} drafts. Edit report TXT files and approve each after checking.')


def export(out):
    source = {(r['subset'],r['id']):r for r in load(out/'drafts.jsonl')}
    inputs = {(r['subset'],r['id']) for r in load(out/'inputs.jsonl')}
    with (out/'review.csv').open(encoding='utf-8-sig',newline='') as f:
        review = list(csv.DictReader(f))
    keys = [(r['subset'],r['id']) for r in review]
    if len(keys)!=len(set(keys)) or set(keys)!=inputs or set(source)!=inputs:
        raise ValueError('Export requires one reviewed report for each prepared sample.')
    rows = []
    for r in review:
        if any(r[k].strip().lower()!='yes' for k in ('numbers_and_rules_checked','visual_description_checked','approved')):
            raise ValueError(f'Pending human review: {r["subset"]}/{r["id"]}')
        row = dict(source[(r['subset'],r['id'])])
        text = Path(row['report_file']).read_text(encoding='utf-8-sig').strip()
        if not text: raise ValueError(f'Empty reviewed report: {row["report_file"]}')
        row.update(reference_report=text,reviewed=True,review_notes=r['notes'],
                   reviewed_report_sha256=hashlib.sha256(text.encode()).hexdigest())
        rows.append(row)
    target = out/'groundtruth_reviewed.jsonl'
    if target.exists(): raise FileExistsError('Reviewed export exists; preserve it and use a new version for changes.')
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    print(f'Exported {len(rows)} reviewed groundtruth reports: {target}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','generate','make-review','export'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--physical-root',type=Path,
                   help='Full-run directory containing <site>_full_seed42/within/physical_references.jsonl')
    p.add_argument('--sites',nargs='+',default=['site-1','site-2'])
    p.add_argument('--model',default='qwen3.8-omni-flash')
    p.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.stage=='prepare':
        if args.physical_root is None:
            p.error('prepare requires --physical-root')
        prepare(args.out,args.data_root,args.physical_root,args.sites)
    elif args.stage=='generate': generate(args)
    elif args.stage=='make-review': make_review(args.out)
    else: export(args.out)


if __name__=='__main__':
    main()
