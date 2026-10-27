"""CPU-only 2D baselines; train-only tuning and held-out physical evaluation."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / 'src'), str(CODE / 'scripts')]
from inspection_data import AnnotationStore, matched_intervals
from audit_measurement_wholeset import PeakConfig, extract_centers, load_scaled_gt_xs, read_ids
from measure import RulerConfig, load_depth_labels


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def hough_centers(image, config, mask=False, cache=None):
    """Fit long near-vertical edges, merge duplicate lines, then pair boundaries.

    No reference coordinates, counts, design spacings or physical readings enter
    extraction. RGB pairing uses opposite edge contrast; mask pairing uses filled
    foreground support between boundaries. Ambiguous/unpaired edges are dropped.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if mask:
        gray = ((gray > 127) * 255).astype(np.uint8)
    key = (mask, config['low'])
    if cache is None or key not in cache:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), config['low'], config['low'] * 2)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=max(20, int(h * .08)),
                                minLineLength=int(h * .2), maxLineGap=int(h * .08))
        if cache is not None:
            cache[key] = lines
    else:
        lines = cache[key]
    if lines is None:
        return []
    candidates = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dy) < h * config['length'] or abs(dx / dy) > np.tan(np.deg2rad(10)):
            continue
        slope = dx / dy
        mid = x1 + (h / 2 - y1) * slope
        if 0 <= mid < w:
            candidates.append((mid, slope, abs(dy), min(y1, y2), max(y1, y2)))
    groups = []
    for line in sorted(candidates):
        if groups and line[0] - np.average([x[0] for x in groups[-1]], weights=[x[2] for x in groups[-1]]) <= max(2, w / 320):
            groups[-1].append(line)
        else:
            groups.append([line])
    boundaries = []
    for group in groups:
        weights = [x[2] for x in group]
        mid = float(np.average([x[0] for x in group], weights=weights))
        slope = float(np.average([x[1] for x in group], weights=weights))
        lo, hi = min(x[3] for x in group), max(x[4] for x in group)
        ys = np.arange(lo, hi + 1, max(1, h // 100))
        xs = np.rint(mid + (ys - h / 2) * slope).astype(int)
        offset = max(2, round(w / 640))
        left, right = np.clip(xs - offset, 0, w - 1), np.clip(xs + offset, 0, w - 1)
        contrast = float(np.mean(gray[ys, right].astype(float) - gray[ys, left].astype(float)))
        boundaries.append((mid, slope, lo, hi, contrast))
    centers, i = [], 0
    while i + 1 < len(boundaries):
        a, b = boundaries[i:i + 2]
        width = b[0] - a[0]
        lo, hi = max(a[2], b[2]), min(a[3], b[3])
        paired = (max(3, w * .002) <= width <= w * .04 and
                  hi - lo >= h * .15 and abs(a[1] - b[1]) <= .08)
        if paired and mask:
            ys = np.linspace(lo, hi, 50).astype(int)
            xs = np.rint((a[0] + b[0]) / 2 + (ys - h / 2) * (a[1] + b[1]) / 2).astype(int)
            paired = np.mean(gray[ys, np.clip(xs, 0, w - 1)] > 127) >= .7
        elif paired:
            paired = a[4] * b[4] < 0 and min(abs(a[4]), abs(b[4])) >= 3
        if paired:
            centers.append((a[0] + b[0]) / 2)
            i += 2
        else:
            i += 1
    return centers


def score(site, sid, pred, xs, truth, mpp, seconds):
    pairs, gaps = matched_intervals(xs, pred, mpp, 20)
    spacing = np.diff(pred) * mpp
    errors = [float(spacing[j] - truth[i]) for i, j in gaps]
    return dict(site=site, id=sid, predicted_centers=pred,
                predicted_spacings_mm=spacing.tolist(), predicted_count=len(pred),
                reference_count=len(xs), matched_centers=len(pairs),
                reference_intervals=len(truth), matched_intervals=len(errors),
                interval_errors_mm=errors, abs_error_sum=sum(abs(x) for x in errors),
                squared_error_sum=sum(x*x for x in errors), signed_error_sum=sum(errors),
                count_abs_error=abs(len(pred)-len(xs)), geometry_seconds=seconds)


def summarize(rows):
    n = sum(r['matched_intervals'] for r in rows)
    ref = sum(r['reference_intervals'] for r in rows)
    tp = sum(r['matched_centers'] for r in rows)
    total = sum(r['predicted_count'] + r['reference_count'] for r in rows)
    return dict(n_images=len(rows), matched_intervals=n, reference_intervals=ref,
                interval_coverage=n / ref if ref else None,
                spacing_mae_mm=sum(r['abs_error_sum'] for r in rows)/n if n else None,
                spacing_rmse_mm=(sum(r['squared_error_sum'] for r in rows)/n)**.5 if n else None,
                spacing_bias_mm=sum(r['signed_error_sum'] for r in rows)/n if n else None,
                center_f1=2*tp/total if total else 0,
                count_mae=float(np.mean([r['count_abs_error'] for r in rows])),
                failure_rate=float(np.mean([r['predicted_count'] < 2 for r in rows])),
                mean_geometry_seconds=float(np.mean([r['geometry_seconds'] for r in rows])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--pred-root', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    cv2.setNumThreads(4)
    store = AnnotationStore(a.annotations)
    methods = ('rgb_canny_hough', 'predmask_hough', 'epm_projection')
    candidates = [dict(length=l, low=c) for l in (.2, .35, .5) for c in (50, 100)]
    all_summaries, manifest = [], {}
    for site in ('site-1', 'site-2'):
        out = a.out / site
        out.mkdir()
        run = a.pred_root / (site + '_full_seed42')
        profile = json.loads((run / 'train/peak_profile.json').read_text(encoding='utf-8'))
        if profile['annotation_sha256'] != store.sha256 or profile['hfov'] != 87 or profile['center_tol_mm'] != 20:
            raise ValueError('EPM reference/scale mismatch')
        peak = PeakConfig(**profile['peak_config'])
        distances = load_depth_labels(str(a.data_root / site / 'depth_labels.txt'))
        training = {m: [[] for _ in candidates] for m in methods[:2]}
        selected, validation = {}, {m: [] for m in methods}
        for split in ('train', 'val'):
            ids = read_ids(a.data_root, site, split)
            if not ids:
                raise ValueError('Empty split')
            for index, sid in enumerate(ids):
                rgb_path = a.data_root / site / 'images' / (sid + '.png')
                seg_path = run / 'predictions' / ('within' if split == 'val' else 'train') / 'model' / site / 'seg' / (sid + '.png')
                image = cv2.imread(str(rgb_path))
                mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
                if image is None or mask is None:
                    raise FileNotFoundError(str(rgb_path) + ' or ' + str(seg_path))
                h, w = mask.shape
                image = cv2.resize(image, (w, h)) if image.shape[:2] != mask.shape else image
                xs = load_scaled_gt_xs(a.data_root, site, sid, w)
                truth = store.spacings(site, sid)
                if not xs or len(truth) != len(xs)-1:
                    raise ValueError('Reference alignment mismatch: ' + site + '/' + sid)
                mpp = RulerConfig(hfov_deg=87, image_width_px=w, image_height_px=h).mm_per_pixel(distances[sid])
                if not np.isfinite(mpp) or mpp <= 0:
                    raise ValueError('Invalid laser scale')
                for path in (rgb_path, seg_path, a.data_root/site/'rebar_labels'/(sid+'.txt')):
                    manifest[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
                line_cache = {}
                for method in methods:
                    if split == 'train' and method == 'epm_projection':
                        continue
                    configs = candidates if split == 'train' else [selected[method]]
                    for k, cfg in enumerate(configs):
                        start = time.perf_counter()
                        if method == 'epm_projection':
                            pred = extract_centers(mask > 127, peak)
                        else:
                            cv2.setRNGSeed(42)
                            pred = hough_centers(mask if method == 'predmask_hough' else image, cfg,
                                                 mask=method == 'predmask_hough', cache=line_cache)
                        seconds = time.perf_counter() - start
                        row = score(site, sid, pred, xs, truth, mpp, seconds)
                        if split == 'train':
                            training[method][k].append(row)
                        else:
                            validation[method].append(row)
                if (index+1) % 50 == 0:
                    print(site, split, index+1, '/', len(ids), flush=True)
            if split == 'train':
                calibration = {}
                for method in methods[:2]:
                    stats = [summarize(rows) for rows in training[method]]
                    best = max(range(len(stats)), key=lambda k: (stats[k]['center_f1'], -stats[k]['count_mae']))
                    selected[method] = candidates[best]
                    calibration[method] = dict(selected=selected[method], candidates=[dict(config=c, metrics=s) for c, s in zip(candidates, stats)])
                selected['epm_projection'] = profile['peak_config']
                dump(out/'calibration.json', calibration)
                print(site, 'train-only selected:', selected, flush=True)
        for method, records in validation.items():
            report = summarize(records)
            rng = np.random.default_rng(42)
            boots = [summarize([records[i] for i in rng.integers(0, len(records), len(records))]) for _ in range(a.bootstrap)]
            keys = ('spacing_mae_mm', 'spacing_rmse_mm', 'spacing_bias_mm', 'interval_coverage', 'failure_rate')
            report['image_bootstrap_95ci'] = {k: np.percentile([b[k] for b in boots if b[k] is not None], [2.5, 97.5]).tolist() if any(b[k] is not None for b in boots) else None for k in keys}
            report['undefined_bootstrap_replicates'] = {k: sum(b[k] is None for b in boots) for k in keys}
            dump(out/(method+'_metrics.json'), report)
            (out/(method+'_per_image.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
            all_summaries.append(dict(site=site, method=method, **{k:v for k,v in report.items() if not isinstance(v, dict)}))
            print(site, method, json.dumps(report), flush=True)
        dump(out/'protocol.json', dict(hfov_deg=87, center_tolerance_mm=20, selected=selected,
             annotation_sha256=store.sha256, epm_profile=profile,
             scale='Common laser scalar range and planar pinhole approximation; historical HFOV, not newly calibrated',
             scope='Near-vertical bars; horizontal center distances at image mid-row',
             tuning='Each site separately; train micro center F1, count MAE tie-break; no validation tuning',
             timing='CPU geometry only; excludes I/O and neural mask inference; not end-to-end runtime',
             status='Generic traditional baselines, not exact reproductions of published methods'))
    with (a.out/'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0]))
        writer.writeheader()
        writer.writerows(all_summaries)
    for site in ('site-1', 'site-2'):
        for rel in ('depth_labels.txt', 'gt_sets/train.txt', 'gt_sets/val.txt'):
            path = a.data_root/site/rel
            manifest[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest[str(a.annotations.resolve())] = store.sha256
    script = Path(__file__).resolve()
    manifest[str(script)] = hashlib.sha256(script.read_bytes()).hexdigest()
    dump(a.out/'input_hashes.json', manifest)
    print('Saved:', a.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
