"""Train-selected centerpoint Hough baseline, independent of EPM projection."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from run_traditional_measurement import (
    AnnotationStore, RulerConfig, dump, load_depth_labels, load_scaled_gt_xs,
    read_ids, score, summarize,
)


def candidates(mask):
    """Vote with row-run midpoints, not paired edges or projection peaks."""
    h, w = mask.shape
    canvas = np.zeros_like(mask, dtype=np.uint8)
    points = []
    for y in range(0, h, 3):
        row = mask[y] > 127
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        for left, right in zip(starts, ends):
            if max(2, w*.0015) <= right-left <= w*.07:
                x = (left+right-1)/2
                points.append((x, y))
                canvas[y, round(x)] = 255
    cv2.setRNGSeed(42)
    lines = cv2.HoughLinesP(canvas, 1, np.pi/720, threshold=max(12, h//45),
                            minLineLength=int(h*.08), maxLineGap=int(h*.2))
    axes = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y2-y1) < h*.08:
                continue
            slope = (x2-x1)/float(y2-y1)
            center = float(x1 + (h/2-y1)*slope)
            if abs(slope) <= np.tan(np.deg2rad(20)) and 0 <= center < w:
                axes.append((center, slope, abs(int(y2)-int(y1))))
    return axes, np.asarray(points, dtype=float).reshape(-1,2)


def fit_centers(data, shape, cfg):
    axes, points = data
    h, w = shape
    groups = []
    for axis in sorted(axes):
        if groups and axis[0]-np.average([a[0] for a in groups[-1]], weights=[a[2] for a in groups[-1]]) <= cfg['merge']:
            groups[-1].append(axis)
        else:
            groups.append([axis])
    centers = []
    for group in groups:
        center, slope = np.average(np.array(group)[:,:2], axis=0, weights=[a[2] for a in group])
        residual = abs(points[:,0]-(center+(points[:,1]-h/2)*slope))
        keep = residual <= cfg['band']
        if np.unique(points[keep,1]).size < (h/3)*cfg['support']:
            continue
        x, y = points[keep,0], points[keep,1]-h/2
        # Fit supported center observations; reference labels never enter this fit.
        slope, center = np.linalg.lstsq(np.column_stack([y, np.ones(len(y))]), x, rcond=None)[0]
        if abs(slope) <= np.tan(np.deg2rad(20)) and 0 <= center < w:
            centers.append(float(center))
    # Regression can move duplicate candidates together; merge them once again.
    groups = []
    for center in sorted(centers):
        if groups and center-np.mean(groups[-1]) <= cfg['merge']:
            groups[-1].append(center)
        else:
            groups.append([center])
    return [float(np.mean(group)) for group in groups]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--pred-root', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    cv2.setNumThreads(4)
    store = AnnotationStore(a.annotations)
    grid = [dict(merge=m, band=b, support=s) for m in (8,16,24)
            for b in (4,8,12) for s in (.12,.25,.4)]
    manifest, summary = {}, []
    for site in ('site-1','site-2'):
        out = a.out/site
        out.mkdir()
        distances = load_depth_labels(str(a.data_root/site/'depth_labels.txt'))
        training = [[] for _ in grid]
        selected = None
        for split in ('train','val'):
            rows = []
            ids = read_ids(a.data_root,site,split)
            if not ids:
                raise ValueError('Empty split')
            for i,sid in enumerate(ids):
                path = a.pred_root/(site+'_full_seed42')/'predictions'/('train' if split=='train' else 'within')/'model'/site/'seg'/(sid+'.png')
                mask = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(path)
                h,w = mask.shape
                for source in (path,a.data_root/site/'rebar_labels'/(sid+'.txt')):
                    manifest[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
                xs = load_scaled_gt_xs(a.data_root,site,sid,w)
                truth = store.spacings(site,sid)
                if not xs or len(xs)-1 != len(truth):
                    raise ValueError('Unaligned references')
                mpp = RulerConfig(hfov_deg=87,image_width_px=w,image_height_px=h).mm_per_pixel(distances[sid])
                if not np.isfinite(mpp) or mpp<=0:
                    raise ValueError('Invalid scale')
                start = time.perf_counter()
                data = candidates(mask)
                elapsed = time.perf_counter()-start
                for k,cfg in enumerate(grid if split=='train' else [selected]):
                    start = time.perf_counter()
                    pred = fit_centers(data,mask.shape,cfg)
                    record = score(site,sid,pred,xs,truth,mpp,elapsed+time.perf_counter()-start)
                    if split=='train':
                        training[k].append(record)
                    else:
                        rows.append(record)
                if (i+1)%100==0:
                    print(site,split,i+1,'/',len(ids),flush=True)
            if split=='train':
                stats = [summarize(r) for r in training]
                # Optimize interval recovery explicitly, penalized by unmatched gaps.
                # This F1 prevents a flood of spurious centers from gaming recall.
                for stat,records in zip(stats,training):
                    predicted_gaps = sum(max(0,r['predicted_count']-1) for r in records)
                    stat['interval_f1'] = 2*stat['matched_intervals']/max(1,stat['reference_intervals']+predicted_gaps)
                best = max(range(len(grid)),key=lambda k:(stats[k]['interval_f1'],stats[k]['center_f1'],-stats[k]['count_mae']))
                coverage_best = max(range(len(grid)),key=lambda k:stats[k]['interval_coverage'])
                selected = grid[best]
                dump(out/'calibration.json',dict(selected=selected,selection='train interval F1; center F1 and count MAE tie-breaks',
                    maximum_train_coverage_candidate=dict(config=grid[coverage_best],metrics=stats[coverage_best]),
                    candidates=[dict(config=c,metrics=s) for c,s in zip(grid,stats)]))
                print(site,'frozen config',selected,'train',stats[best],flush=True)
            else:
                report = summarize(rows)
                rng = np.random.default_rng(42)
                boots = [summarize([rows[j] for j in rng.integers(0,len(rows),len(rows))]) for _ in range(1000)]
                keys = ('spacing_mae_mm','spacing_rmse_mm','interval_coverage','count_mae','failure_rate')
                report['image_bootstrap_95ci'] = {key:np.percentile([b[key] for b in boots if b[key] is not None],[2.5,97.5]).tolist() if any(b[key] is not None for b in boots) else None for key in keys}
                report['undefined_bootstrap_replicates'] = {key:sum(b[key] is None for b in boots) for key in keys}
                dump(out/'centerpoint_hough_metrics.json',report)
                (out/'centerpoint_hough_per_image.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                summary.append(dict(site=site,**report))
                print(site,'VALIDATION',json.dumps(report),flush=True)
        dump(out/'protocol.json',dict(method='predicted_mask_row_run_midpoint_hough',selected=selected,
            hfov_deg=87,center_tol_mm=20,annotation_sha256=store.sha256,
            selection='27 configurations; train-only interval F1, then center F1, then count MAE',
            warning='Generic improved baseline, not exact published reproduction. Validation already inspected in earlier baseline; exploratory follow-up, not an untouched confirmatory test.',
            scale='Historical planar laser/FOV scale, not newly verified camera calibration',
            timing='CPU geometry diagnostics; excludes neural inference and I/O'))
    for site in ('site-1','site-2'):
        for relative in ('depth_labels.txt','gt_sets/train.txt','gt_sets/val.txt'):
            path=a.data_root/site/relative
            manifest[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest[str(a.annotations.resolve())]=store.sha256
    script=Path(__file__).resolve()
    manifest[str(script)]=hashlib.sha256(script.read_bytes()).hexdigest()
    dump(a.out/'input_hashes.json',manifest)
    dump(a.out/'summary.json',summary)


if __name__=='__main__':
    main()
