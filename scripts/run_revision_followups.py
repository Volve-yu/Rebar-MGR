"""Revision 3: auxiliary-loss ablation; revision 4: fresh QAEP/report evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ('rgb_only', 'rgb_rule', 'rgb_measure', 'rgb_measure_rule',
            'rgb_seg', 'rgb_seg_measure', 'rgb_seg_measure_rule', 'rgb_seg_measure_rule_qaep')


def stage(a, base, name, command, output=None, checkpoint=None):
    command = list(map(str, command))
    print(name, json.dumps(command), flush=True)
    if a.plan_only:
        return
    base.mkdir(parents=True, exist_ok=True)
    marker = base/(name+'.done.json')
    if marker.exists():
        if not a.resume or json.loads(marker.read_text()) != command:
            raise ValueError('Existing or changed stage: '+str(marker))
        print('Skip completed:', name)
        return
    actual = list(command)
    if a.resume and checkpoint and checkpoint.exists():
        actual += ['--resume', str(checkpoint)]
    if output and output.exists():
        if not a.resume:
            raise FileExistsError(output)
        output.rename(output.with_name(output.name+'.incomplete.'+str(time.time_ns())))
    with (base/(name+'.log')).open('a', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-u', *actual], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if proc.wait():
            raise RuntimeError(name+' failed; fix error and use --resume')
    marker.write_text(json.dumps(command), encoding='utf-8')


def ablation(a):
    for seed in a.seeds:
        for variant in a.variants:
            for site in ('site-1', 'site-2'):
                b = a.run_root/f'{site}_{variant}_seed{seed}'
                ck = b/'checkpoints'
                edge = .1 if variant in ('full', 'edge_aux') else 0
                proj = .15 if variant in ('full', 'measure_only') else 0
                stage(a, b, 'train', ['src_measure_guided/train.py', '--subsets', site,
                    '--seed', seed, '--epochs', 50, '--batch_size', 4, '--alpha_edge', edge,
                    '--beta_proj', proj, '--lambda_c', 0, '--max_edge_to_seg', .04,
                    '--max_proj_to_seg', .05, '--aux_schedule', 'warmup_cosine_decay',
                    '--aux_warmup_epochs', 5, '--aux_min_scale', .2, '--seg_refine_epochs', 5,
                    '--save_dir', ck, '--log_dir', b/'logs', '--device', 'cuda'], checkpoint=ck/'latest.pth')
                stage(a, b, 'seg_eval', ['src_measure_guided/eval.py', '--checkpoint', ck/'best.pth',
                    '--subsets', site, '--device', 'cuda', '--out_json', b/'segmentation_metrics.json'])
                for split in ('train', 'val'):
                    pred = b/'predictions'/split/'model'
                    dest = b/('calibration' if split == 'train' else 'measurement')
                    stage(a, b, 'predict_'+split, ['src_measure_guided/predict.py', '--checkpoint', ck/'best.pth',
                        '--subsets', site, '--split', split, '--pred_dir', pred, '--device', 'cuda'], output=pred)
                    cmd = ['scripts/run_physical_inspection.py', '--mode', 'calibrate' if split == 'train' else 'evaluate',
                        '--pred_root', pred.parent, '--model', 'model', '--subsets', site, '--split', split, '--out_dir', dest]
                    if split == 'val':
                        cmd += ['--profile', b/'calibration/peak_profile.json']
                    stage(a, b, 'measure_'+split, cmd, output=dest)


def reports(a):
    fingerprints = {}
    for site in ('site-1', 'site-2'):
        source = a.source_root/f'{site}_full_seed42'
        for split, expected in (('train', 'train'), ('within', 'val')):
            path = source/split/'report_inputs.jsonl'
            rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
            if not rows:
                raise ValueError('Empty input: '+str(path))
            for r in rows:
                if r.get('schema_version') != 'design_physical_v2' or r.get('subset') != site or r.get('split') != expected:
                    raise ValueError('Legacy/wrong site/split: '+str(path))
                for key in ('image_path', 'seg_path'):
                    if not Path(r[key]).is_file():
                        raise FileNotFoundError(r[key])
            fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        path = source/'within/template_reports.jsonl'
        fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(inputs=fingerprints, model=a.model, base_url=a.base_url, data_root=str(a.data_root))
    if not a.plan_only:
        a.run_root.mkdir(parents=True, exist_ok=True)
        saved = a.run_root/'input_manifest.json'
        if saved.exists() and json.loads(saved.read_text()) != manifest:
            raise ValueError('Inputs/config changed: use a new output root')
        saved.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if a.run_api and not a.plan_only and not os.environ.get('DASHSCOPE_API_KEY'):
        raise RuntimeError('Set DASHSCOPE_API_KEY first')
    for site in ('site-1', 'site-2'):
        src = a.source_root/f'{site}_full_seed42'
        b = a.run_root/site
        val = src/'within/report_inputs.jsonl'
        qaep = b/'qaep/validation.jsonl'
        commands = [
            ('features', ['src_report/build_qaep_dataset.py', '--input_jsonl', src/'train/report_inputs.jsonl', '--out_jsonl', b/'qaep/train.jsonl']),
            ('qaep_train', ['src_report/train_qaep.py', '--train_jsonl', b/'qaep/train.jsonl', '--model_out', b/'qaep/model.json']),
            ('qaep_apply', ['src_report/apply_qaep.py', '--input_jsonl', val, '--model_path', b/'qaep/model.json', '--out_jsonl', qaep]),
            ('template_eval', ['src_report/eval.py', '--gen_files', src/'within/template_reports.jsonl', '--out_dir', b/'template_metrics'])]
        for name, cmd in commands:
            if name in ('features', 'qaep_apply'):
                cmd += ['--data_root', a.data_root]
            stage(a, b, name, cmd)
        if not a.run_api:
            continue
        outputs = []
        for setting in SETTINGS:
            out = b/'generations'/(setting+'.jsonl')
            outputs.append(out)
            stage(a, b, setting, ['src_report/generate.py', '--input_jsonl', qaep if setting.endswith('_qaep') else val,
                '--setting', setting, '--out_jsonl', out, '--model', a.model, '--base_url', a.base_url, '--resume'])
            if not a.plan_only:
                rows = [json.loads(s) for s in out.read_text(encoding='utf-8').splitlines() if s.strip()]
                print(setting, 'API errors retained:', sum(bool(r.get('error')) for r in rows))
        stage(a, b, 'report_eval', ['src_report/eval.py', '--gen_files', *outputs, '--out_dir', b/'report_metrics'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['3', '4'])
    p.add_argument('--run-root', type=Path)
    p.add_argument('--source-root', type=Path, default=Path('runs/revision_full_seed42'))
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--variants', nargs='+', choices=['full', 'seg_only', 'edge_aux', 'measure_only'], default=['seg_only', 'edge_aux', 'measure_only'])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--run-api', action='store_true')
    p.add_argument('--model', default='qwen-vl-plus')
    p.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    a = p.parse_args()
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', TORCH_HOME=str(ROOT/'vendor/torch'))
    a.run_root = a.run_root or Path('runs/experiment3_ablation' if a.experiment == '3' else 'runs/experiment4_reports_v2')
    if a.run_root.exists() and not a.resume and not a.plan_only:
        raise FileExistsError('Use --resume or a new --run-root')
    (ablation if a.experiment == '3' else reports)(a)
    print('Finished', a.experiment, a.run_root)


if __name__ == '__main__':
    main()
