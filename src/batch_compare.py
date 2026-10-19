"""
Batch compare all model CSVs against GT rebar labels and print summary table.

Usage:
    python src/batch_compare.py --result_dir result --data_root data
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from measure import RulerConfig, load_depth_labels
from eval_measure import load_manual_label, spacings_from_xs


def compare_csv(csv_path: Path, data_root: Path, subsets=("site-1", "site-2")):
    with open(csv_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    total_labeled = 0
    total_count_match = 0
    all_maes = []

    for subset in subsets:
        rows = [r for r in all_rows if r["subset"] == subset]
        if not rows:
            continue

        sub_path  = data_root / subset
        label_dir = sub_path / "rebar_labels"
        depth_file = sub_path / "depth_labels.txt"
        depth_labels = load_depth_labels(str(depth_file)) if depth_file.exists() else {}
        cfg = RulerConfig()

        for row in rows:
            sid = row["image_id"]
            label_path = label_dir / f"{sid}.txt"
            if not label_path.exists():
                continue

            gt_xs = load_manual_label(label_path)
            if gt_xs is None or len(gt_xs) < 2:
                continue

            mpp = float(row["mpp"]) if row.get("mpp") not in ("", "nan", None) else None
            if mpp is None:
                if sid in depth_labels:
                    mpp = cfg.mm_per_pixel(depth_labels[sid])
                else:
                    continue

            gt_spacings = spacings_from_xs(gt_xs, mpp)
            gt_count    = len(gt_xs)
            algo_count  = int(row["rebar_count"]) if row.get("rebar_count") else 0

            total_labeled += 1
            count_match = (gt_count == algo_count)
            if count_match:
                total_count_match += 1

            if not count_match:
                continue

            algo_spacings_str = row.get("spacings_mm", "")
            algo_spacings = [float(s) for s in algo_spacings_str.split(";") if s.strip()]
            if len(algo_spacings) != len(gt_spacings):
                continue

            errors = [abs(g - a) for g, a in zip(gt_spacings, algo_spacings)]
            all_maes.append(float(np.mean(errors)))

    return {
        "labeled":      total_labeled,
        "count_match":  total_count_match,
        "count_rate":   total_count_match / total_labeled if total_labeled else 0,
        "spacing_mae":  round(float(np.mean(all_maes)), 2) if all_maes else None,
        "mae_n":        len(all_maes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="result")
    ap.add_argument("--data_root",  default="data")
    ap.add_argument("--subsets",    nargs="+", default=["site-1", "site-2"])
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    data_root  = Path(args.data_root)

    # Ordered display list (skip ours_v2 - deprecated)
    order = [
        "seg_only", "edge_only", "no_coupling", "symmetric",
        "gscnn", "pidinet", "mobile_seed", "boundmatch",
        "no_consistency", "ours",
    ]

    csv_files = {p.stem: p for p in result_dir.glob("*.csv")
                 if p.stem not in ("ours_v2",)}

    print(f"\n{'Method':<18} {'Labeled':>7} {'Count%':>7} {'MAE(mm)':>9} {'N_mae':>6}")
    print("-" * 52)

    results = {}
    for name in order:
        if name not in csv_files:
            print(f"{name:<18} {'(no CSV)':>7}")
            continue
        r = compare_csv(csv_files[name], data_root, args.subsets)
        results[name] = r
        mae_str = f"{r['spacing_mae']:>9.2f}" if r["spacing_mae"] is not None else f"{'N/A':>9}"
        print(f"{name:<18} {r['labeled']:>7} {r['count_rate']*100:>6.1f}% "
              f"{mae_str} {r['mae_n']:>6}")

    # Also print any CSV not in order list
    extras = [s for s in csv_files if s not in order]
    for name in sorted(extras):
        r = compare_csv(csv_files[name], data_root, args.subsets)
        mae_str = f"{r['spacing_mae']:>9.2f}" if r["spacing_mae"] is not None else f"{'N/A':>9}"
        print(f"{name:<18} {r['labeled']:>7} {r['count_rate']*100:>6.1f}% "
              f"{mae_str} {r['mae_n']:>6}")

    print("-" * 52)
    print("N_mae = images where count matched GT and MAE was computed")


if __name__ == "__main__":
    main()
