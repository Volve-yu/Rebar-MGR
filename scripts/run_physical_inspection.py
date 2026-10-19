"""V2 calibration/evaluation/report-input entry using verified physical references."""
import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'src_report'), str(ROOT/'scripts')]
from inspection_data import AnnotationStore, SCHEMA, measurement_record, matched_intervals
from measure import RulerConfig, load_depth_labels
from audit_measurement_wholeset import PeakConfig, extract_centers, load_scaled_gt_xs, read_ids, resolve_seg_path
from design_evidence import template_report
from experiment_snapshot import snapshot


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_samples(args, store):
    samples = []
    for subset in args.subsets:
        ids = read_ids(args.data_root, subset, args.split)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Empty or duplicated split: '+subset+'/'+args.split)
        depths = load_depth_labels(str(args.data_root/subset/'depth_labels.txt'))
        for sid in ids:
            path = resolve_seg_path(args.pred_root, args.model, subset, sid)
            if not path.exists():
                raise FileNotFoundError('Missing prediction (not silently excluded): '+str(path))
            with Image.open(path) as im:
                mask = np.array(im.convert('L')) > 127
            xs = load_scaled_gt_xs(args.data_root, subset, sid, mask.shape[1])
            truth = store.spacings(subset, sid)
            if not xs or len(truth) != len(xs)-1:
                raise ValueError('Center/interval count mismatch: '+subset+'/'+sid)
            if sid not in depths or not np.isfinite(depths[sid]) or depths[sid] <= 0:
                raise ValueError('Missing/invalid distance: '+subset+'/'+sid)
            cfg = RulerConfig(hfov_deg=args.hfov, image_width_px=mask.shape[1], image_height_px=mask.shape[0])
            samples.append(dict(subset=subset, sid=sid, mask=mask, xs=xs, truth=truth,
                                rules=store.rules(subset, sid), distance=depths[sid], mpp=cfg.mm_per_pixel(depths[sid]),
                                seg_path=str(path), image_path=str(args.data_root/subset/'images'/(sid+'.png'))))
    return samples


def evaluate(samples, peak, tolerance):
    results = []
    for sample in samples:
        pred = extract_centers(sample['mask'], peak)
        xs, truth, mpp = sample['xs'], sample['truth'], sample['mpp']
        spacings = [(b-a)*mpp for a,b in zip(pred,pred[1:])]
        pairs, gaps = matched_intervals(xs, pred, mpp, tolerance)
        errors = [abs(spacings[j]-truth[i]) for i,j in gaps]
        count_error = len(pred)-len(xs)
        record = measurement_record(spacings, len(pred), sample['rules'], distance_m=sample['distance'], mpp=mpp,
                                    warning=None if spacings else 'Fewer than two predicted centers; assessment unavailable.')
        metrics = dict(subset=sample['subset'], id=sample['sid'], count_error=count_error,
                       count_abs_error=abs(count_error), count_within_1=int(abs(count_error)<=1),
                       count_within_2=int(abs(count_error)<=2), center_tp=len(pairs),
                       center_fp=len(pred)-len(pairs), center_fn=len(xs)-len(pairs),
                       reference_intervals=len(truth), matched_intervals=len(errors),
                       matched_spacing_abs_error_sum_mm=sum(errors),
                       matched_spacing_mae_mm=float(np.mean(errors)) if errors else None,
                       mean_spacing_abs_error_mm=abs(float(np.mean(spacings))-float(np.mean(truth))) if spacings else None,
                       conditional_spacing_mae_mm=float(np.mean(errors)) if len(pairs)==len(xs)==len(pred) and errors else None,
                       assessment_status=record['compliance']['status'])
        results.append((sample, metrics, record))
    return results


def summary(results):
    rows = [r[1] for r in results]
    tp,fp,fn = (sum(r[k] for r in rows) for k in ('center_tp','center_fp','center_fn'))
    matched = sum(r['matched_intervals'] for r in rows)
    means = [r['mean_spacing_abs_error_mm'] for r in rows if r['mean_spacing_abs_error_mm'] is not None]
    return dict(n=len(rows), count_mae=float(np.mean([r['count_abs_error'] for r in rows])),
                count_acc_pm1=float(np.mean([r['count_within_1'] for r in rows])),
                count_acc_pm2=float(np.mean([r['count_within_2'] for r in rows])),
                center_f1=2*tp/max(2*tp+fp+fn,1), mean_spacing_mae_mm=float(np.mean(means)) if means else None,
                mean_spacing_valid_n=len(means), matched_intervals=matched,
                interval_coverage=matched/max(sum(r['reference_intervals'] for r in rows),1),
                matched_spacing_mae_mm=sum(r['matched_spacing_abs_error_sum_mm'] for r in rows)/matched if matched else None,
                unavailable_assessments=sum(r['assessment_status']=='N/A' for r in rows))


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['calibrate','evaluate'], required=True)
    parser.add_argument('--annotations', type=Path, default=ROOT/'annotation_all_images/all_images_annotations.csv')
    parser.add_argument('--data_root', type=Path, default=Path('data'))
    parser.add_argument('--pred_root', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--subsets', nargs='+', required=True, choices=['site-1','site-2'])
    parser.add_argument('--split', default='val', choices=['train','val','test','trainval'])
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--hfov', type=float, default=87.0)
    parser.add_argument('--center_tol_mm', type=float, default=20.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--protocol', choices=['within_site','cross_site'], default='within_site')
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError('Use a new output directory; existing results are immutable.')
    if not 0 < args.hfov < 180 or args.center_tol_mm <= 0 or args.bootstrap < 0:
        parser.error('Invalid geometry, matching tolerance or bootstrap count')
    if args.mode=='calibrate' and args.split!='train':
        parser.error('Peak calibration may only use the training split')
    if args.mode=='evaluate' and not args.profile:
        parser.error('Evaluation requires a previously frozen --profile')
    store = AnnotationStore(args.annotations)
    provenance=[]
    for site in args.subsets:
        candidates=[args.pred_root/('subset_'+site)/args.model/'prediction_manifest.json',
                    args.pred_root/args.model/'prediction_manifest.json']
        path=next((p for p in candidates if p.exists()),None)
        if path is None:
            raise FileNotFoundError('Missing prediction_manifest.json; re-export predictions with the updated entry.')
        item=json.loads(path.read_text(encoding='utf-8'))
        current=snapshot(args.data_root,[site],[args.split])
        for name,digest in current['files'].items():
            if item['prediction_snapshot']['files'].get(name)!=digest:
                raise ValueError('Prediction snapshot mismatch: '+name)
        if not item.get('training_snapshot'):
            raise ValueError('Checkpoint has no training snapshot; use a freshly recorded training run.')
        training=item['training_snapshot']
        # Detect data replacement under the same IDs after training.
        from experiment_snapshot import file_sha
        for name,digest in training['files'].items():
            if file_sha(args.data_root/name)!=digest:
                raise ValueError('Training dataset changed since checkpoint creation: '+name)
        if args.mode=='calibrate' and site not in training['subsets']:
            raise ValueError('Do not calibrate peaks on an unseen target site')
        if args.mode=='evaluate' and args.protocol=='cross_site' and site in training['subsets']:
            raise ValueError('Target site participated in model training/selection')
        provenance.append(item)
    samples = load_samples(args, store)
    profile = None
    if args.mode=='evaluate':
        profile=json.loads(args.profile.read_text(encoding='utf-8'))
        if profile['schema_version']!=SCHEMA or profile['annotation_sha256']!=store.sha256:
            raise ValueError('Profile schema/data snapshot differs; recalibrate on the intended training data')
        if profile['hfov']!=args.hfov or profile['center_tol_mm']!=args.center_tol_mm:
            raise ValueError('Geometry/matching settings differ from the frozen calibration protocol')
        if not {p['checkpoint_sha256'] for p in provenance} <= set(profile['checkpoint_sha256']):
            raise ValueError('Peak profile and prediction checkpoints differ')
        if args.protocol=='cross_site' and set(profile['calibration_subsets']) & set(args.subsets):
            raise ValueError('Target site participated in peak calibration')
        if args.split in ('train','trainval') and set(profile['calibration_subsets']) & set(args.subsets):
            raise ValueError('Training images cannot be used for this held-out evaluation entry')
        peak=PeakConfig(**profile['peak_config'])
    else:
        from calibrate_peak_params_fast import config_grid
        candidates=[]
        for cfg in config_grid(quick=True):
            scores=summary(evaluate(samples,cfg,args.center_tol_mm))
            mae=scores['mean_spacing_mae_mm']
            score=scores['center_f1']-.03*scores['count_mae']-.005*(mae if mae is not None else 100)
            candidates.append((score,cfg,scores))
        _,peak,_=max(candidates,key=lambda x:x[0])
    results=evaluate(samples,peak,args.center_tol_mm)
    args.out_dir.mkdir(parents=True)
    manifest=dict(schema_version=SCHEMA,annotation_sha256=store.sha256, annotations=str(args.annotations),
                  protocol=args.protocol, checkpoint_sha256=sorted({p['checkpoint_sha256'] for p in provenance}),
                  model=args.model, subsets=args.subsets, split=args.split, mode=args.mode,
                  hfov=args.hfov, center_tol_mm=args.center_tol_mm, peak_config=asdict(peak),
                  reference_type='user_verified_physical_spacing', center_reference='image_centerlines',
                  measurement_protocol='column_projection_peaks', bootstrap_unit='image_not_individual_gap',
                  split_sha256={s:hashlib.sha256((args.data_root/s/'gt_sets'/(args.split+'.txt')).read_bytes()).hexdigest() for s in args.subsets})
    dump(args.out_dir/'manifest.json',manifest)
    if args.mode=='calibrate':
        dump(args.out_dir/'peak_profile.json',dict(manifest,calibration_subsets=args.subsets,
              candidates=[dict(score=s,config=asdict(c),metrics=m) for s,c,m in candidates]))
    metrics=summary(results)
    if args.bootstrap:
        rng=np.random.default_rng(args.seed)
        groups=[[r for r in results if r[0]['subset']==site] for site in args.subsets]
        draws=[]
        for _ in range(args.bootstrap):
            draw=[group[i] for group in groups for i in rng.integers(0,len(group),len(group))]
            draws.append(summary(draw))
        metrics['image_bootstrap_95ci']={key:[float(x) for x in np.percentile([d[key] for d in draws if d[key] is not None],[2.5,97.5])]
            for key in ('count_mae','count_acc_pm1','center_f1','mean_spacing_mae_mm','matched_spacing_mae_mm') if any(d[key] is not None for d in draws)}
    metrics['by_site']={site:summary([r for r in results if r[0]['subset']==site]) for site in args.subsets}
    dump(args.out_dir/'summary.json',metrics)
    with (args.out_dir/'details.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0][1]));writer.writeheader();writer.writerows(r[1] for r in results)
    inputs=[];refs=[];templates=[]
    for sample,_,prediction in results:
        reference=measurement_record(sample['truth'],len(sample['xs']),sample['rules'])
        common=dict(subset=sample['subset'],id=sample['sid'],variant=args.model,schema_version=SCHEMA,split=args.split,protocol=args.protocol,
                    image_path=sample['image_path'],seg_path=sample['seg_path'],
                    reference_report=template_report(prediction),measurement=prediction,
                    reference_scope='predicted_evidence_consistency')
        inputs.append(common)
        # Physical references live in a separate artifact, never in prompt measurements.
        refs.append(dict(subset=sample['subset'],id=sample['sid'],measurement=reference,reference_report=template_report(reference)))
        templates.append(dict(common,setting='deterministic_template',prediction=template_report(prediction),error=None))
    write_jsonl(args.out_dir/'report_inputs.jsonl',inputs)
    write_jsonl(args.out_dir/'physical_references.jsonl',refs)
    write_jsonl(args.out_dir/'template_reports.jsonl',templates)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
