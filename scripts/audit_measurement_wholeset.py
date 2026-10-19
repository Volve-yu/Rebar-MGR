"""
Whole-set measurement audit for review-risk analysis.

This script evaluates existing prediction masks without retraining. It adds
metrics that do not condition solely on exact count matching:

  - count MAE / RMSE / bias over all validation images
  - bar-center precision / recall / F1 with a metric tolerance
  - center MAE on matched centers
  - mean-spacing MAE on images where both GT and prediction have >= 2 bars
  - conditional spacing MAE on count-matched images, for continuity with paper

It can also run a small one-factor sensitivity check for the deterministic
column-projection peak detector.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from measure import RulerConfig, load_depth_labels  # noqa: E402
from eval_measure import load_manual_label  # noqa: E402


@dataclass(frozen=True)
class PeakConfig:
    smooth_size: int = 5
    min_height_ratio: float = 0.05
    min_prom_ratio: float = 0.015
    min_distance_px: int = 30
    min_width_px: float = 15.0
    min_area_px: int = 50


def read_ids(data_root: Path, subset: str, split: str) -> List[str]:
    split_file = data_root / subset / "gt_sets" / f"{split}.txt"
    if split_file.exists():
        return [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    return []


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


def extract_centers(seg_binary: np.ndarray, cfg: PeakConfig) -> List[float]:
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks, peak_widths

    h, w = seg_binary.shape
    col_proj = seg_binary.sum(axis=0).astype(float)
    col_smooth = uniform_filter1d(col_proj, size=cfg.smooth_size)

    if col_smooth.max() <= 0:
        return []

    peaks, _ = find_peaks(
        col_smooth,
        height=h * cfg.min_height_ratio,
        distance=cfg.min_distance_px,
        prominence=h * cfg.min_prom_ratio,
    )
    if len(peaks) == 0:
        return []

    widths, _, _, _ = peak_widths(col_smooth, peaks, rel_height=0.5)
    boundaries = [0]
    boundaries.extend(
        int(peaks[i] + np.argmin(col_smooth[peaks[i] : peaks[i + 1]]))
        for i in range(len(peaks) - 1)
    )
    boundaries.append(w)

    centers: List[float] = []
    for idx, (peak_x, width_px) in enumerate(zip(peaks, widths)):
        if width_px < cfg.min_width_px:
            continue
        left = boundaries[idx]
        right = boundaries[idx + 1] - 1
        if right <= left:
            continue
        band = col_smooth[left : right + 1]
        if band.sum() <= 0:
            continue
        col_slice = seg_binary[:, left : right + 1]
        if int(col_slice.sum()) < cfg.min_area_px:
            continue
        center = float(left) + float(np.average(np.arange(len(band)), weights=band + 1e-9))
        centers.append(center)
    return sorted(centers)


def spacings(xs: Sequence[float], mpp: float) -> List[float]:
    return [(xs[i + 1] - xs[i]) * mpp for i in range(len(xs) - 1)]


def greedy_center_match(gt_xs: Sequence[float], pred_xs: Sequence[float], mpp: float, tol_mm: float):
    i = 0
    j = 0
    errors: List[float] = []
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


def evaluate_one(
    data_root: Path,
    pred_root: Path,
    model: str,
    subset: str,
    image_id: str,
    depth_labels: dict,
    ruler_cfg: RulerConfig,
    peak_cfg: PeakConfig,
    center_tol_mm: float,
):
    seg_path = resolve_seg_path(pred_root, model, subset, image_id)
    row = {
        "model": model,
        "subset": subset,
        "image_id": image_id,
        "has_seg": int(seg_path.exists()),
    }
    if not seg_path.exists():
        row.update({"status": "missing_seg"})
        return row

    seg_img = np.array(Image.open(seg_path).convert("L"))
    seg_binary = (seg_img > 127).astype(np.uint8)
    gt_xs = load_scaled_gt_xs(data_root, subset, image_id, seg_img.shape[1])
    if not gt_xs or image_id not in depth_labels:
        row.update({"status": "missing_gt_or_depth"})
        return row

    actual_cfg = RulerConfig(
        hfov_deg=ruler_cfg.hfov_deg,
        image_width_px=seg_img.shape[1],
        image_height_px=seg_img.shape[0],
    )
    mpp = actual_cfg.mm_per_pixel(depth_labels[image_id])
    pred_xs = extract_centers(seg_binary, peak_cfg)

    gt_sp = spacings(gt_xs, mpp)
    pred_sp = spacings(pred_xs, mpp)
    gt_count = len(gt_xs)
    pred_count = len(pred_xs)
    tp, fp, fn, center_errors = greedy_center_match(gt_xs, pred_xs, mpp, center_tol_mm)

    mean_spacing_abs_error = ""
    if gt_sp and pred_sp:
        mean_spacing_abs_error = abs(float(np.mean(pred_sp)) - float(np.mean(gt_sp)))

    conditional_spacing_mae = ""
    if gt_count == pred_count and gt_sp and len(gt_sp) == len(pred_sp):
        conditional_spacing_mae = float(np.mean(np.abs(np.array(pred_sp) - np.array(gt_sp))))

    row.update(
        {
            "status": "ok",
            "distance_m": depth_labels[image_id],
            "mpp": mpp,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "count_error": pred_count - gt_count,
            "count_abs_error": abs(pred_count - gt_count),
            "count_match": int(gt_count == pred_count),
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
    return row


def resolve_seg_path(pred_root: Path, model: str, subset: str, image_id: str) -> Path:
    """Resolve both flat report-source and subset-specific ablation layouts."""
    candidates = [
        pred_root / f"subset_{subset}" / model / subset / "seg" / f"{image_id}.png",
        pred_root / model / subset / "seg" / f"{image_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def mean(xs: Iterable[float]) -> str:
    vals = [float(x) for x in xs if x != "" and x is not None and not math.isnan(float(x))]
    return f"{float(np.mean(vals)):.4f}" if vals else ""


def aggregate(rows: Sequence[dict]) -> List[dict]:
    out = []
    keys = sorted({(r["model"], r["subset"]) for r in rows if r.get("status") == "ok"})
    for model, subset in keys:
        rs = [r for r in rows if r.get("status") == "ok" and r["model"] == model and r["subset"] == subset]
        n = len(rs)
        tp = sum(int(r["center_tp"]) for r in rs)
        fp = sum(int(r["center_fp"]) for r in rs)
        fn = sum(int(r["center_fn"]) for r in rs)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        count_errors = np.array([float(r["count_error"]) for r in rs], dtype=float)
        count_abs = np.abs(count_errors)
        count_match = sum(int(r["count_match"]) for r in rs)
        count_within_1 = int(np.sum(count_abs <= 1))
        count_within_2 = int(np.sum(count_abs <= 2))
        count_rel_20 = int(
            np.sum(
                [
                    abs(float(r["count_error"])) / max(1.0, float(r["gt_count"])) <= 0.20
                    for r in rs
                ]
            )
        )
        out.append(
            {
                "model": model,
                "subset": subset,
                "n": n,
                "count_match": count_match,
                "count_match_rate": f"{count_match / n:.4f}" if n else "",
                "count_within_1": count_within_1,
                "count_acc_pm1": f"{count_within_1 / n:.4f}" if n else "",
                "count_within_2": count_within_2,
                "count_acc_pm2": f"{count_within_2 / n:.4f}" if n else "",
                "count_rel20": count_rel_20,
                "count_acc_rel20": f"{count_rel_20 / n:.4f}" if n else "",
                "count_mae": f"{float(np.mean(count_abs)):.4f}" if n else "",
                "count_rmse": f"{float(np.sqrt(np.mean(count_errors ** 2))):.4f}" if n else "",
                "count_bias": f"{float(np.mean(count_errors)):.4f}" if n else "",
                "center_precision": f"{precision:.4f}",
                "center_recall": f"{recall:.4f}",
                "center_f1": f"{f1:.4f}",
                "center_mae_mm": mean(r["center_mae_mm"] for r in rs),
                "mean_spacing_mae_all_valid_mm": mean(r["mean_spacing_abs_error_mm"] for r in rs),
                "mean_spacing_valid_n": sum(1 for r in rs if r["mean_spacing_abs_error_mm"] != ""),
                "conditional_spacing_mae_mm": mean(r["conditional_spacing_mae_mm"] for r in rs),
                "conditional_spacing_n": sum(1 for r in rs if r["conditional_spacing_mae_mm"] != ""),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, summary_rows: Sequence[dict], center_tol_mm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Whole-set measurement audit",
        "",
        f"Center matching tolerance: {center_tol_mm:g} mm.",
        "",
        "These metrics are computed from existing prediction masks and manual center labels.",
        "Conditional spacing MAE is retained for continuity, but count MAE and center F1 are whole-set diagnostics.",
        "",
        "| Model | Subset | N | Count match | Count MAE | Count bias | Center F1 | Center MAE (mm) | Mean-spacing MAE (mm) | Conditional spacing MAE (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            "| {model} | {subset} | {n} | {count_match}/{n} ({rate:.1f}%; $\\pm$1 {pm1:.1f}%; $\\pm$2 {pm2:.1f}%) | {count_mae} | {count_bias} | {center_f1} | {center_mae_mm} | {mean_spacing_mae_all_valid_mm} | {conditional_spacing_mae_mm} |".format(
                model=r["model"],
                subset=r["subset"],
                n=r["n"],
                count_match=r["count_match"],
                rate=float(r["count_match_rate"]) * 100,
                pm1=float(r["count_acc_pm1"]) * 100,
                pm2=float(r["count_acc_pm2"]) * 100,
                count_mae=r["count_mae"],
                count_bias=r["count_bias"],
                center_f1=r["center_f1"],
                center_mae_mm=r["center_mae_mm"],
                mean_spacing_mae_all_valid_mm=r["mean_spacing_mae_all_valid_mm"],
                conditional_spacing_mae_mm=r["conditional_spacing_mae_mm"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def one_factor_peak_configs() -> List[Tuple[str, PeakConfig]]:
    base = PeakConfig()
    configs = [("default", base)]
    for v in [3, 7, 9]:
        configs.append((f"smooth={v}", PeakConfig(smooth_size=v)))
    for v in [0.010, 0.020, 0.025]:
        configs.append((f"prom={v:.3f}", PeakConfig(min_prom_ratio=v)))
    for v in [0.040, 0.060, 0.080]:
        configs.append((f"height={v:.3f}", PeakConfig(min_height_ratio=v)))
    for v in [20, 25, 35, 40]:
        configs.append((f"distance={v}", PeakConfig(min_distance_px=v)))
    for v in [10.0, 20.0, 25.0]:
        configs.append((f"width={v:g}", PeakConfig(min_width_px=v)))
    return configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--pred_root", default="predictions_report_sources")
    ap.add_argument("--out_dir", default="results_clean/review_audit")
    ap.add_argument("--split", default="val")
    ap.add_argument("--subsets", nargs="+", default=["site-1", "site-2"])
    ap.add_argument("--models", nargs="+", default=["ours_full", "deeplab", "gscnn", "mobile_seed", "afenet", "sctransnet"])
    ap.add_argument("--center_tol_mm", type=float, default=10.0)
    ap.add_argument("--hfov", type=float, default=87.0)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--sensitivity_model", default="ours_full")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    ruler_cfg = RulerConfig(hfov_deg=args.hfov)

    all_rows = []
    for subset in args.subsets:
        depth_labels = load_depth_labels(str(data_root / subset / "depth_labels.txt"))
        ids = read_ids(data_root, subset, args.split)
        for model in args.models:
            for image_id in ids:
                all_rows.append(
                    evaluate_one(
                        data_root,
                        pred_root,
                        model,
                        subset,
                        image_id,
                        depth_labels,
                        ruler_cfg,
                        PeakConfig(),
                        args.center_tol_mm,
                    )
                )

    summary_rows = aggregate(all_rows)
    write_csv(out_dir / "whole_set_measurement_per_image.csv", all_rows)
    write_csv(out_dir / "whole_set_measurement_summary.csv", summary_rows)
    write_markdown(out_dir / "whole_set_measurement_audit.md", summary_rows, args.center_tol_mm)

    if args.sensitivity:
        sens_rows = []
        for name, peak_cfg in one_factor_peak_configs():
            cfg_rows = []
            for subset in args.subsets:
                depth_labels = load_depth_labels(str(data_root / subset / "depth_labels.txt"))
                ids = read_ids(data_root, subset, args.split)
                for image_id in ids:
                    row = evaluate_one(
                        data_root,
                        pred_root,
                        args.sensitivity_model,
                        subset,
                        image_id,
                        depth_labels,
                        ruler_cfg,
                        peak_cfg,
                        args.center_tol_mm,
                    )
                    cfg_rows.append(row)
            for row in aggregate(cfg_rows):
                row = dict(row)
                row["peak_config"] = name
                sens_rows.append(row)
        if sens_rows:
            fields = ["peak_config"] + [k for k in sens_rows[0].keys() if k != "peak_config"]
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "peak_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in sens_rows:
                    w.writerow(r)

    print(f"Wrote audit results to {out_dir}")


if __name__ == "__main__":
    main()
