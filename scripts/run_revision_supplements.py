"""Evaluation-only supplements. Does not train models or modify experiment four."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def call(command, plan=False):
    command = [str(x) for x in command]
    print(json.dumps(command), flush=True)
    if not plan:
        subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True)


def full_seg(a):
    for source in ('site-1', 'site-2'):
        ck = a.runs_root/'revision_full_seed42'/f'{source}_full_seed42/checkpoints/best.pth'
        if not a.plan_only and not ck.is_file():
            raise FileNotFoundError(f'{ck}: use the AutoDL machine holding the trained Full checkpoints')
        for target in ('site-1', 'site-2'):
            out = a.out_root/'full_seg'/f'{source}_to_{target}.json'
            if out.exists():
                raise FileExistsError(f'{out}: use a fresh --out-root')
            call([ROOT/'src_measure_guided/eval.py', '--checkpoint', ck,
                  '--data_root', a.data_root, '--subsets', target,
                  '--device', a.device, '--num_workers', 0, '--out_json', out], a.plan_only)


def mask_paths(a, site):
    for method in ('deeplab', 'gscnn', 'sctransnet', 'afenet'):
        yield method, a.runs_root/f'experiment1_external/{method}/{site}/predictions/val/{method}/{site}/seg'
    yield 'mobile_seed', a.runs_root/f'experiment1_mobile_seed/{site}/predictions/val/mobile_seed/{site}/seg'
    yield 'full', a.runs_root/f'revision_full_seed42/{site}_full_seed42/predictions/within/model/{site}/seg'


def mask_eval(a):
    import numpy as np
    from PIL import Image
    eps = 1e-6
    jobs = []
    for site in ('site-1', 'site-2'):
        ids = (a.data_root/site/'gt_sets/val.txt').read_text().split()
        for method, directory in mask_paths(a, site):
            if method not in a.methods:
                continue
            if method == 'full' and not directory.exists():
                directory = a.runs_root/f'revision_full_seed42/{site}_full_seed42/predictions/val/model/{site}/seg'
            for sid in ids:
                for path in (directory/f'{sid}.png', a.data_root/site/f'segmentation/{sid}.png'):
                    if not path.is_file():
                        raise FileNotFoundError(path)
            jobs.append((site, method, ids, directory))
    if a.plan_only:
        print([(site, method, len(ids), str(directory)) for site, method, ids, directory in jobs])
        return
    for site, method, ids, directory in jobs:
        out = a.out_root/'mask_seg'/f'{site}_{method}.json'
        if out.exists():
            raise FileExistsError(out)
        rows, hashes = [], {}
        for sid in ids:
            paths = [directory/f'{sid}.png', a.data_root/site/f'segmentation/{sid}.png']
            masks = []
            for path in paths:
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as im:
                    pixels = np.asarray(im.convert('L'))
                    if not set(np.unique(pixels)).issubset({0, 1, 255}):
                        raise ValueError(f'Expected binary mask: {path}')
                    binary = Image.fromarray((pixels > 0).astype('uint8')*255)
                    masks.append(np.asarray(binary.resize((640, 480), Image.Resampling.NEAREST)) > 0)
            pred, gt = masks
            tp, fp, fn, tn = (int(x.sum()) for x in (pred & gt, pred & ~gt, ~pred & gt, ~pred & ~gt))
            iou = (tp+eps)/(tp+fp+fn+eps)
            rows.append(dict(id=sid, seg_iou=iou,
                seg_miou=(iou+(tn+eps)/(tn+fp+fn+eps))/2,
                seg_f1=(2*tp+eps)/(2*tp+fp+fn+eps),
                seg_precision=(tp+eps)/(tp+fp+eps), seg_recall=(tp+eps)/(tp+fn+eps),
                seg_acc=(tp+tn+eps)/(tp+tn+fp+fn+eps)))
        save(out, dict(subset=site, method=method, n_images=len(rows),
            protocol='exported_binary_masks_640x480_nearest_per_image_macro',
            metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'id'},
            per_image=rows, file_sha256=hashes))
        print('Saved', out)


def peaks(a):
    # Keep each site's independently trained model under a separate manifest.
    stage = a.out_root/'peak_inputs'
    if stage.exists():
        raise FileExistsError(f'{stage}: use a fresh --out-root after interrupted execution')
    jobs = []
    for split, source_split in (('train', 'train'), ('val', 'within')):
        for site in ('site-1', 'site-2'):
            source = a.runs_root/f'revision_full_seed42/{site}_full_seed42/predictions/{source_split}/model'
            if not source.exists() and source_split == 'within':
                source = a.runs_root/f'revision_full_seed42/{site}_full_seed42/predictions/val/model'
            if not (source/'prediction_manifest.json').is_file():
                raise FileNotFoundError(source/'prediction_manifest.json')
            jobs.append((source, stage/split/f'subset_{site}/model'))
    common = [ROOT/'scripts/run_physical_inspection.py', '--annotations', a.annotations,
              '--data_root', a.data_root, '--model', 'model', '--subsets', 'site-1', 'site-2']
    profile = a.out_root/'unified_peaks/calibration/peak_profile.json'
    if not a.plan_only:
        for source, dest in jobs:
            shutil.copytree(source, dest)
    call([*common, '--mode', 'calibrate', '--split', 'train', '--pred_root', stage/'train',
          '--out_dir', profile.parent, '--bootstrap', 0], a.plan_only)
    call([*common, '--mode', 'evaluate', '--split', 'val', '--pred_root', stage/'val',
          '--profile', profile, '--out_dir', a.out_root/'unified_peaks/validation'], a.plan_only)
    if a.plan_only:
        print('Also evaluate six predeclared one-factor peak perturbations; no validation-based selection.')
        return
    config = read_json(profile)
    for key, values in [('smooth_size', [3, 11]), ('min_prom_ratio', [.02, .06]), ('min_distance_px', [30, 50])]:
        for value in values:
            adjusted = dict(config, peak_config=dict(config['peak_config'], **{key:value}))
            adjusted['sensitivity_only'] = True
            dest = a.out_root/'peak_sensitivity'/f'{key}_{value}'
            parameter_file = dest/'profile.json'
            save(parameter_file, adjusted)
            call([*common, '--mode', 'evaluate', '--split', 'val', '--pred_root', stage/'val',
                  '--profile', parameter_file, '--out_dir', dest/'evaluation', '--bootstrap', 0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['full-seg', 'mask-seg', 'peaks'])
    p.add_argument('--runs-root', type=Path, default=ROOT/'runs')
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--annotations', type=Path, default=ROOT/'annotation_all_images/all_images_annotations.csv')
    p.add_argument('--out-root', type=Path, default=ROOT/'runs/revision_supplements_20261011')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--methods', nargs='+', default=['deeplab','gscnn','sctransnet','afenet','mobile_seed','full'])
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    for key in ('runs_root', 'data_root', 'annotations', 'out_root'):
        setattr(a, key, getattr(a, key).resolve())
    {'full-seg':full_seg, 'mask-seg':mask_eval, 'peaks':peaks}[a.task](a)


if __name__ == '__main__':
    main()
