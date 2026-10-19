"""
Calibrate column-projection peak parameters on a calibration split and evaluate
them on a separate split.

This is intended for a paper-safe workflow:

1. Export train predictions from a fixed checkpoint.
2. Select peak parameters on the train predictions and train labels.
3. Evaluate the selected parameters on the validation predictions.

Do not report validation-selected parameters as final paper results.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_measurement_wholeset as audit  # noqa: E402
from measure import RulerConfig, load_depth_labels  # noqa: E402


def config_grid(quick: bool = False):
    if quick:
        values = [
            (5, 0.05, 0.015, 30, 15),
            (5, 0.08, 0.03, 40, 15),
            (9, 0.08, 0.03, 40, 10),
            (9, 0.08, 0.03, 35, 12),
            (7, 0.08, 0.03, 35, 10),
        ]
        for smooth, height, prom, distance, width in values:
            yield audit.PeakConfig(
                smooth_size=smooth,
                min_height_ratio=height,
                min_prom_ratio=prom,
                min_distance_px=distance,
                min_width_px=width,
            )
        return

    for smooth, height, prom, distance, width in itertools.product(
        [5, 7, 9],
        [0.05, 0.06, 0.08],
        [0.015, 0.02, 0.025, 0.03],
        [25, 30, 35, 40],
        [10, 12, 15, 20],
    ):
        yield audit.PeakConfig(
            smooth_size=smooth,
            min_height_ratio=height,
            min_prom_ratio=prom,
            min_distance_px=distance,
            min_width_px=width,
        )


def cfg_name(cfg: audit.PeakConfig) -> str:
    return (
        f"smooth={cfg.smooth_size},height={cfg.min_height_ratio:g},"
        f"prom={cfg.min_prom_ratio:g},distance={cfg.min_distance_px},"
        f"width={cfg.min_width_px:g}"
    )


def evaluate_cfg(data_root: Path, pred_root: Path, model: str, subset: str, split: str, cfg, center_tol_mm: float):
    depth = load_depth_labels(str(data_root / subset / "depth_labels.txt"))
    ids = audit.read_ids(data_root, subset, split)
    ruler = RulerConfig(hfov_deg=87.0)
    rows = [
        audit.evaluate_one(
            data_root,
            pred_root,
            model,
            subset,
            sid,
            depth,
            ruler,
            cfg,
            center_tol_mm,
        )
        for sid in ids
    ]
    ok = [r for r in rows if r.get("status") == "ok"]
    if len(ok) != len(ids):
        missing = len(ids) - len(ok)
        print(f"[WARN] subset={subset} split={split} model={model}: {missing} records missing or invalid")
    summary = audit.aggregate(rows)
    if not summary:
        return None, rows
    return summary[0], rows


def score_summary(row: dict) -> float:
    """Prefer center recovery and count stability; penalize whole-set spacing drift."""
    center_f1 = float(row["center_f1"])
    count_mae = float(row["count_mae"])
    mean_spacing = float(row["mean_spacing_mae_all_valid_mm"] or 100.0)
    return center_f1 - 0.03 * count_mae - 0.005 * mean_spacing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--calib_pred_root", required=True)
    parser.add_argument("--eval_pred_root", required=True)
    parser.add_argument("--model", default="full")
    parser.add_argument("--subsets", nargs="+", default=["site-1", "site-2"])
    parser.add_argument("--calib_split", default="train", choices=["train", "val"])
    parser.add_argument("--eval_split", default="val", choices=["train", "val"])
    parser.add_argument("--center_tol_mm", type=float, default=20.0)
    parser.add_argument("--out_dir", default="results_clean/peak_calibration")
    parser.add_argument("--quick_grid", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    calib_pred_root = Path(args.calib_pred_root)
    eval_pred_root = Path(args.eval_pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    eval_summary_rows = []
    eval_detail_rows = []
    calib_grid_rows = []

    for subset in args.subsets:
        best = None
        best_cfg = None
        for cfg in config_grid(args.quick_grid):
            summary, _ = evaluate_cfg(
                data_root,
                calib_pred_root,
                args.model,
                subset,
                args.calib_split,
                cfg,
                args.center_tol_mm,
            )
            if summary is None:
                continue
            row = dict(summary)
            row["subset"] = subset
            row["cfg"] = cfg_name(cfg)
            row["score"] = f"{score_summary(summary):.6f}"
            calib_grid_rows.append(row)
            if best is None or score_summary(summary) > score_summary(best):
                best = summary
                best_cfg = cfg

        if best is None or best_cfg is None:
            print(f"[SKIP] no calibration result for subset={subset}")
            continue

        selected.append(
            {
                "subset": subset,
                "cfg": cfg_name(best_cfg),
                "calib_score": f"{score_summary(best):.6f}",
                "calib_count_match": best["count_match"],
                "calib_count_match_rate": best["count_match_rate"],
                "calib_count_mae": best["count_mae"],
                "calib_center_f1": best["center_f1"],
                "calib_mean_spacing_mae": best["mean_spacing_mae_all_valid_mm"],
            }
        )

        eval_summary, eval_rows = evaluate_cfg(
            data_root,
            eval_pred_root,
            args.model,
            subset,
            args.eval_split,
            best_cfg,
            args.center_tol_mm,
        )
        if eval_summary is not None:
            eval_row = dict(eval_summary)
            eval_row["cfg"] = cfg_name(best_cfg)
            eval_summary_rows.append(eval_row)
        eval_detail_rows.extend(dict(r, selected_cfg=cfg_name(best_cfg)) for r in eval_rows)

    for path, rows in [
        (out_dir / "calibration_grid.csv", calib_grid_rows),
        (out_dir / "selected_peak_params.csv", selected),
        (out_dir / "eval_summary.csv", eval_summary_rows),
        (out_dir / "eval_details.csv", eval_detail_rows),
    ]:
        if not rows:
            continue
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote calibration outputs to {out_dir}")


if __name__ == "__main__":
    main()
