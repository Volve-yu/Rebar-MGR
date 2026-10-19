"""
Fast peak-parameter calibration with cached masks and labels.

Compared with scripts/calibrate_peak_params.py, this version reads each PNG and
manual label only once, precomputes smoothed column projections, and prints
progress during the parameter search. It is intended for the AutoDL full-grid
calibration run.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_measurement_wholeset as audit  # noqa: E402
from eval_measure import load_manual_label  # noqa: E402
from measure import RulerConfig, load_depth_labels  # noqa: E402


@dataclass
class Sample:
    subset: str
    image_id: str
    seg: np.ndarray
    gt_xs: List[float]
    mpp: float
    smooth_cols: Dict[int, np.ndarray]


def cfg_name(cfg: audit.PeakConfig) -> str:
    return (
        f"smooth={cfg.smooth_size},height={cfg.min_height_ratio:g},"
        f"prom={cfg.min_prom_ratio:g},distance={cfg.min_distance_px},"
        f"width={cfg.min_width_px:g}"
    )


def config_grid(quick: bool = False) -> List[audit.PeakConfig]:
    if quick:
        values = [
            (5, 0.05, 0.015, 30, 15),
            (5, 0.08, 0.03, 40, 15),
            (9, 0.08, 0.03, 40, 10),
            (9, 0.08, 0.03, 35, 12),
            (7, 0.08, 0.03, 35, 10),
        ]
    else:
        values = list(
            itertools.product(
                [5, 7, 9],
                [0.05, 0.06, 0.08],
                [0.015, 0.02, 0.025, 0.03],
                [25, 30, 35, 40],
                [10, 12, 15, 20],
            )
        )
    return [
        audit.PeakConfig(
            smooth_size=smooth,
            min_height_ratio=height,
            min_prom_ratio=prom,
            min_distance_px=distance,
            min_width_px=width,
        )
        for smooth, height, prom, distance, width in values
    ]


def resolve_seg_path(pred_root: Path, model: str, subset: str, image_id: str) -> Path:
    candidates = [
        pred_root / f"subset_{subset}" / model / subset / "seg" / f"{image_id}.png",
        pred_root / model / subset / "seg" / f"{image_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_scaled_gt_xs(data_root: Path, subset: str, image_id: str, seg_width: int) -> Optional[List[float]]:
    label_path = data_root / subset / "rebar_labels" / f"{image_id}.txt"
    xs = load_manual_label(label_path)
    if not xs:
        return None
    img_path = data_root / subset / "images" / f"{image_id}.png"
    orig_width = Image.open(img_path).width if img_path.exists() else seg_width
    if orig_width != seg_width:
        scale = seg_width / float(orig_width)
        xs = [x * scale for x in xs]
    return sorted(xs)


def load_samples(
    data_root: Path,
    pred_root: Path,
    model: str,
    subset: str,
    split: str,
    smooth_sizes: Iterable[int],
) -> List[Sample]:
    ids = audit.read_ids(data_root, subset, split)
    depth_labels = load_depth_labels(str(data_root / subset / "depth_labels.txt"))
    samples: List[Sample] = []
    missing = []
    invalid = []

    for sid in ids:
        seg_path = resolve_seg_path(pred_root, model, subset, sid)
        if not seg_path.exists():
            missing.append(sid)
            continue
        seg = (np.array(Image.open(seg_path).convert("L")) > 127).astype(np.uint8)
        gt_xs = load_scaled_gt_xs(data_root, subset, sid, seg.shape[1])
        if not gt_xs or sid not in depth_labels:
            invalid.append(sid)
            continue
        cfg = RulerConfig(hfov_deg=87.0, image_width_px=seg.shape[1], image_height_px=seg.shape[0])
        mpp = cfg.mm_per_pixel(depth_labels[sid])
        col = seg.sum(axis=0).astype(float)
        smooth_cols = {
            size: uniform_filter1d(col, size=size)
            for size in smooth_sizes
        }
        samples.append(
            Sample(
                subset=subset,
                image_id=sid,
                seg=seg,
                gt_xs=gt_xs,
                mpp=mpp,
                smooth_cols=smooth_cols,
            )
        )

    if missing or invalid:
        print(
            f"[WARN] subset={subset} split={split}: "
            f"missing_pred={len(missing)} invalid_gt_or_depth={len(invalid)}"
        )
        if missing[:10]:
            print(f"       missing_pred first: {', '.join(missing[:10])}")
        if invalid[:10]:
            print(f"       invalid first: {', '.join(invalid[:10])}")
    print(f"[CACHE] subset={subset} split={split}: loaded {len(samples)}/{len(ids)} samples")
    return samples


def centers_from_sample(sample: Sample, cfg: audit.PeakConfig) -> List[float]:
    seg = sample.seg
    h, w = seg.shape
    col = sample.smooth_cols[cfg.smooth_size]
    if col.max() <= 0:
        return []

    peaks, _ = find_peaks(
        col,
        height=h * cfg.min_height_ratio,
        distance=cfg.min_distance_px,
        prominence=h * cfg.min_prom_ratio,
    )
    if len(peaks) == 0:
        return []

    widths, _, _, _ = peak_widths(col, peaks, rel_height=0.5)
    boundaries = [0]
    boundaries.extend(
        int(peaks[i] + np.argmin(col[peaks[i] : peaks[i + 1]]))
        for i in range(len(peaks) - 1)
    )
    boundaries.append(w)

    xs: List[float] = []
    for idx, width_px in enumerate(widths):
        if width_px < cfg.min_width_px:
            continue
        left = boundaries[idx]
        right = boundaries[idx + 1] - 1
        if right <= left:
            continue
        if int(seg[:, left : right + 1].sum()) < cfg.min_area_px:
            continue
        band = col[left : right + 1]
        if band.sum() <= 0:
            continue
        xs.append(float(left) + float(np.average(np.arange(len(band)), weights=band + 1e-9)))
    return sorted(xs)


def spacings(xs: Sequence[float], mpp: float) -> List[float]:
    return [(xs[i + 1] - xs[i]) * mpp for i in range(len(xs) - 1)]


def greedy_center_match(gt_xs: Sequence[float], pred_xs: Sequence[float], mpp: float, tol_mm: float):
    i = 0
    j = 0
    errors = []
    while i < len(gt_xs) and j < len(pred_xs):
        diff_mm = (pred_xs[j] - gt_xs[i]) * mpp
        if abs(diff_mm) <= tol_mm:
            errors.append(diff_mm)
            i += 1
            j += 1
        elif pred_xs[j] < gt_xs[i]:
            j += 1
        else:
            i += 1
    tp = len(errors)
    fp = len(pred_xs) - tp
    fn = len(gt_xs) - tp
    return tp, fp, fn, errors


def evaluate_samples(samples: Sequence[Sample], cfg: audit.PeakConfig, center_tol_mm: float):
    rows = []
    for sample in samples:
        pred_xs = centers_from_sample(sample, cfg)
        gt_xs = sample.gt_xs
        mpp = sample.mpp
        gt_sp = spacings(gt_xs, mpp)
        pred_sp = spacings(pred_xs, mpp)
        tp, fp, fn, center_errors = greedy_center_match(gt_xs, pred_xs, mpp, center_tol_mm)

        mean_spacing_abs_error = ""
        if gt_sp and pred_sp:
            mean_spacing_abs_error = abs(float(np.mean(pred_sp)) - float(np.mean(gt_sp)))

        conditional_spacing_mae = ""
        if len(gt_xs) == len(pred_xs) and gt_sp and len(gt_sp) == len(pred_sp):
            conditional_spacing_mae = float(np.mean(np.abs(np.array(pred_sp) - np.array(gt_sp))))

        rows.append(
            {
                "model": "full",
                "subset": sample.subset,
                "image_id": sample.image_id,
                "status": "ok",
                "gt_count": len(gt_xs),
                "pred_count": len(pred_xs),
                "count_error": len(pred_xs) - len(gt_xs),
                "count_abs_error": abs(len(pred_xs) - len(gt_xs)),
                "count_match": int(len(pred_xs) == len(gt_xs)),
                "center_tp": tp,
                "center_fp": fp,
                "center_fn": fn,
                "center_mae_mm": float(np.mean(np.abs(center_errors))) if center_errors else "",
                "gt_mean_spacing_mm": float(np.mean(gt_sp)) if gt_sp else "",
                "pred_mean_spacing_mm": float(np.mean(pred_sp)) if pred_sp else "",
                "mean_spacing_abs_error_mm": mean_spacing_abs_error,
                "conditional_spacing_mae_mm": conditional_spacing_mae,
            }
        )
    return rows


def score_summary(row: dict) -> float:
    center_f1 = float(row["center_f1"])
    count_mae = float(row["count_mae"])
    mean_spacing = float(row["mean_spacing_mae_all_valid_mm"] or 100.0)
    return center_f1 - 0.03 * count_mae - 0.005 * mean_spacing


def write_csv(path: Path, rows: Sequence[dict], fields: Optional[List[str]] = None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    parser.add_argument("--out_dir", default="results_clean/peak_calibration_train_to_val_fast")
    parser.add_argument("--quick_grid", action="store_true")
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    calib_pred_root = Path(args.calib_pred_root)
    eval_pred_root = Path(args.eval_pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = config_grid(args.quick_grid)
    smooth_sizes = sorted({cfg.smooth_size for cfg in grid})
    print(f"[INFO] grid size={len(grid)} smooth_sizes={smooth_sizes}")

    calib_cache = {}
    eval_cache = {}
    for subset in args.subsets:
        calib_cache[subset] = load_samples(
            data_root, calib_pred_root, args.model, subset, args.calib_split, smooth_sizes
        )
        eval_cache[subset] = load_samples(
            data_root, eval_pred_root, args.model, subset, args.eval_split, smooth_sizes
        )

    selected = []
    eval_summary_rows = []
    eval_detail_rows = []
    calib_grid_rows = []
    t0 = time.time()

    for subset in args.subsets:
        samples = calib_cache[subset]
        best_summary = None
        best_cfg = None
        for idx, cfg in enumerate(grid, start=1):
            rows = evaluate_samples(samples, cfg, args.center_tol_mm)
            summary = audit.aggregate(rows)
            if not summary:
                continue
            row = dict(summary[0])
            row["subset"] = subset
            row["cfg"] = cfg_name(cfg)
            row["score"] = f"{score_summary(row):.6f}"
            calib_grid_rows.append(row)

            if best_summary is None or score_summary(row) > score_summary(best_summary):
                best_summary = row
                best_cfg = cfg

            if idx == 1 or idx == len(grid) or idx % max(1, args.progress_every) == 0:
                elapsed = time.time() - t0
                print(
                    f"[{subset}] {idx}/{len(grid)} elapsed={elapsed:.1f}s "
                    f"best_score={score_summary(best_summary):.4f} "
                    f"best={cfg_name(best_cfg)}",
                    flush=True,
                )

        if best_summary is None or best_cfg is None:
            print(f"[SKIP] no calibration result for subset={subset}")
            continue

        selected.append(
            {
                "subset": subset,
                "cfg": cfg_name(best_cfg),
                "calib_score": f"{score_summary(best_summary):.6f}",
                "calib_count_match": best_summary["count_match"],
                "calib_count_match_rate": best_summary["count_match_rate"],
                "calib_count_mae": best_summary["count_mae"],
                "calib_center_f1": best_summary["center_f1"],
                "calib_mean_spacing_mae": best_summary["mean_spacing_mae_all_valid_mm"],
            }
        )

        eval_rows = evaluate_samples(eval_cache[subset], best_cfg, args.center_tol_mm)
        eval_summary = audit.aggregate(eval_rows)
        if eval_summary:
            eval_row = dict(eval_summary[0])
            eval_row["cfg"] = cfg_name(best_cfg)
            eval_summary_rows.append(eval_row)
        eval_detail_rows.extend(dict(r, selected_cfg=cfg_name(best_cfg)) for r in eval_rows)

    summary_fields = [
        "model", "subset", "n",
        "count_match", "count_match_rate",
        "count_within_1", "count_acc_pm1",
        "count_within_2", "count_acc_pm2",
        "count_rel20", "count_acc_rel20",
        "count_mae", "count_rmse", "count_bias",
        "center_precision", "center_recall", "center_f1", "center_mae_mm",
        "mean_spacing_mae_all_valid_mm", "mean_spacing_valid_n",
        "conditional_spacing_mae_mm", "conditional_spacing_n",
        "cfg",
    ]
    write_csv(out_dir / "calibration_grid.csv", calib_grid_rows)
    write_csv(out_dir / "selected_peak_params.csv", selected)
    write_csv(out_dir / "eval_summary.csv", eval_summary_rows, summary_fields)
    write_csv(out_dir / "eval_details.csv", eval_detail_rows)

    print(f"Wrote fast calibration outputs to {out_dir}")


if __name__ == "__main__":
    main()
