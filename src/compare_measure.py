"""
Compare model measurement results (from run_measure.py CSV) against
manual GT labels (from label_rebars.py).

Usage:
    python src/compare_measure.py \
        --csv measure_output/results.csv \
        --data_root data \
        --subset site-1

Output:
    Per-image table + aggregate MAE stats (same format as eval_measure.py).
    With --visualize: saves comparison images to --out_dir.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from measure import RulerConfig, load_depth_labels
from eval_measure import load_manual_label, spacings_from_xs, save_viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       required=True,
                    help="results.csv from run_measure.py")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--subset",    default="site-1", choices=["site-1", "site-2"])
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--out_dir",   default="compare_out")
    ap.add_argument("--count_mismatch_only", action="store_true")
    args = ap.parse_args()

    import csv as _csv
    with open(args.csv, encoding="utf-8") as f:
        all_rows = list(_csv.DictReader(f))

    rows = [r for r in all_rows if r["subset"] == args.subset]
    if not rows:
        print(f"No rows for subset={args.subset} in {args.csv}")
        return

    data_root  = Path(args.data_root)
    sub_path   = data_root / args.subset
    label_dir  = sub_path / "rebar_labels"
    img_dir    = sub_path / "images"
    seg_dir    = sub_path / "segmentation"

    depth_file = sub_path / "depth_labels.txt"
    depth_labels = load_depth_labels(str(depth_file)) if depth_file.exists() else {}
    cfg = RulerConfig()

    if args.visualize:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    maes, mean_errors, skipped = [], [], 0
    count_match_n = 0

    for row in rows:
        sid = row["image_id"]
        label_path = label_dir / f"{sid}.txt"

        if not label_path.exists():
            skipped += 1
            continue

        gt_xs = load_manual_label(label_path)
        if gt_xs is None or len(gt_xs) < 2:
            skipped += 1
            continue

        mpp = float(row["mpp"]) if row["mpp"] not in ("", "nan") else None
        if mpp is None and sid in depth_labels:
            mpp = cfg.mm_per_pixel(depth_labels[sid])
        if mpp is None:
            skipped += 1
            continue

        gt_spacings  = spacings_from_xs(gt_xs, mpp)
        gt_count     = len(gt_xs)
        gt_mean      = round(float(np.mean(gt_spacings)), 1) if gt_spacings else None

        # Model results from CSV
        algo_count   = int(row["rebar_count"]) if row["rebar_count"] else 0
        algo_mean    = float(row["mean_spacing_mm"]) if row["mean_spacing_mm"] not in ("", "None") else None
        algo_spacings_str = row.get("spacings_mm", "")
        algo_spacings = [float(s) for s in algo_spacings_str.split(";") if s.strip()] \
                        if algo_spacings_str else []

        count_match = (gt_count == algo_count)
        if count_match:
            count_match_n += 1

        status = "OK " if count_match else "CNT"
        mae_str = "MAE=  N/A  "
        spacing_errors = []

        if count_match and gt_spacings and len(algo_spacings) == len(gt_spacings):
            errors = [g - a for g, a in zip(gt_spacings, algo_spacings)]
            spacing_errors = [round(e, 1) for e in errors]
            mae = float(np.mean(np.abs(errors)))
            maes.append(mae)
            mae_str = f"MAE={mae:5.1f}mm"
            if gt_mean is not None and algo_mean is not None:
                mean_errors.append(gt_mean - algo_mean)

        if args.count_mismatch_only and count_match:
            continue

        print(f"[{status}] {sid:>6}  GT={gt_count} Model={algo_count}  "
              f"{mae_str}  GT_mean={gt_mean}  Model_mean={algo_mean}  "
              f"warn={row.get('warning','')[:40]}")
        if spacing_errors:
            print(f"         errors(mm): {spacing_errors}")

        if args.visualize:
            algo_xs_approx = []
            if algo_spacings and mpp:
                # Reconstruct approximate x positions from spacings
                # (exact positions not in CSV, use image center as anchor)
                from PIL import Image as _Image
                try:
                    iw = _Image.open(img_dir / f"{sid}.png").width
                except Exception:
                    iw = 640
                total_span = sum(algo_spacings) / mpp
                start_x = (iw - total_span) / 2
                cx = start_x
                algo_xs_approx.append(cx)
                for sp in algo_spacings:
                    cx += sp / mpp
                    algo_xs_approx.append(cx)

            save_viz(
                sid,
                seg_dir / f"{sid}.png",
                img_dir / f"{sid}.png",
                gt_xs, algo_xs_approx,
                [round(s, 1) for s in gt_spacings],
                [round(s, 1) for s in algo_spacings],
                mpp,
                out_dir / f"{sid}.png",
            )

    labeled = len(rows) - skipped
    print(f"\n{'='*60}")
    print(f"CSV: {args.csv}  Subset: {args.subset}")
    print(f"Images compared      : {labeled}  (skipped no-label: {skipped})")
    if labeled:
        print(f"Count match          : {count_match_n}/{labeled} "
              f"({100*count_match_n/labeled:.1f}%)")
    if maes:
        print(f"Spacing MAE          : {np.mean(maes):.1f} mm  "
              f"(median {np.median(maes):.1f}, max {np.max(maes):.1f})")
    if mean_errors:
        print(f"Mean spacing bias    : {np.mean(mean_errors):+.1f} mm  "
              f"(std {np.std(mean_errors):.1f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
