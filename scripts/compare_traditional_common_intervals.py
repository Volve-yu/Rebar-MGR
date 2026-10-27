"""Paired comparison on shared matched reference gaps; no parameter tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run_traditional_measurement import (
    AnnotationStore, RulerConfig, dump, load_depth_labels,
    load_scaled_gt_xs, matched_intervals,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--pred-root', type=Path, required=True)
    args = parser.parse_args()
    store = AnnotationStore(args.annotations)
    reports = []
    for site in ('site-1', 'site-2'):
        root = args.results/site
        protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
        if protocol['annotation_sha256'] != store.sha256:
            raise ValueError('Reference hash mismatch')
        distances = load_depth_labels(str(args.data_root/site/'depth_labels.txt'))
        errors = {}
        for method in ('rgb_canny_hough', 'predmask_hough', 'epm_projection'):
            rows = [json.loads(line) for line in (root/(method+'_per_image.jsonl')).read_text(encoding='utf-8').splitlines()]
            errors[method] = {}
            for row in rows:
                mask = args.pred_root/(site+'_full_seed42')/'predictions/within/model'/site/'seg'/(row['id']+'.png')
                with Image.open(mask) as image:
                    width, height = image.size
                xs = load_scaled_gt_xs(args.data_root, site, row['id'], width)
                mpp = RulerConfig(hfov_deg=protocol['hfov_deg'], image_width_px=width,
                                  image_height_px=height).mm_per_pixel(distances[row['id']])
                pred = row['predicted_centers']
                _, gaps = matched_intervals(xs, pred, mpp, protocol['center_tolerance_mm'])
                if len(gaps) != row['matched_intervals']:
                    raise ValueError('Coordinate scale mismatch')
                truth = store.spacings(site, row['id'])
                errors[method][row['id']] = {i: abs((pred[j+1]-pred[j])*mpp-truth[i]) for i, j in gaps}
        for method in ('rgb_canny_hough', 'predmask_hough'):
            paired = []
            for sid, ours in errors['epm_projection'].items():
                other = errors[method][sid]
                common = sorted(set(ours) & set(other))
                paired.append(dict(id=sid, reference_interval_indices=common, n=len(common),
                    baseline_abs_error_sum=sum(other[i] for i in common),
                    epm_abs_error_sum=sum(ours[i] for i in common)))
            def stats(records):
                n = sum(row['n'] for row in records)
                if not n:
                    return None
                baseline = sum(row['baseline_abs_error_sum'] for row in records)/n
                epm = sum(row['epm_abs_error_sum'] for row in records)/n
                return baseline, epm, baseline-epm
            values = stats(paired)
            rng = np.random.default_rng(42)
            boot = [stats([paired[i] for i in rng.integers(0, len(paired), len(paired))]) for _ in range(1000)]
            valid = [b[2] for b in boot if b is not None]
            reports.append(dict(site=site, baseline=method,
                common_intervals=sum(row['n'] for row in paired),
                images_with_common_intervals=sum(row['n'] > 0 for row in paired),
                baseline_mae_mm=values[0] if values else None,
                epm_mae_mm=values[1] if values else None,
                baseline_minus_epm_mae_mm=values[2] if values else None,
                difference_image_bootstrap_95ci=np.percentile(valid, [2.5, 97.5]).tolist() if valid else None,
                undefined_bootstrap_replicates=1000-len(valid),
                note='Conditional on shared matched gaps, not all reference gaps'))
            dump(root/(method+'_common_with_epm.json'), paired)
    dump(args.results/'common_interval_comparison.json', reports)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
