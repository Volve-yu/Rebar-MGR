"""Utilities for the Quality-Aware Evidence Planner (QAEP).

QAEP is a lightweight report-planning layer that consumes the existing
report-generation evidence: predicted segmentation masks, optional RGB images,
and structured measurement records. The helpers here avoid dependencies beyond
NumPy/Pillow so they can run in the same AutoDL environment as the current
report scripts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from design_evidence import is_v2, risk_level, deviation, is_failed
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


FEATURE_NAMES = [
    "distance_m",
    "mpp",
    "rebar_count",
    "num_spacings",
    "mean_spacing_mm",
    "std_spacing_mm",
    "mean_diameter_mm",
    "spacing_cv",
    "max_abs_dev_mm",
    "fail_ratio",
    "mask_foreground_ratio",
    "mask_active_column_ratio",
    "mask_peak_count_norm",
    "mask_vertical_coverage",
    "mask_horizontal_coverage",
    "mask_density_balance",
    "image_brightness",
    "image_contrast",
    "image_laplacian_var_log",
]

TARGET_NAMES = [
    "measurement_reliable",
    "segmentation_complete",
    "image_clear",
]


def read_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def finite_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def load_binary_mask(path: Path) -> Optional[np.ndarray]:
    if not path or not path.exists():
        return None
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def find_gt_mask_path(data_root: Path, subset: str, sid: str) -> Optional[Path]:
    candidates = [
        data_root / subset / "segmentation" / f"{sid}.png",
        data_root / subset / "seg" / f"{sid}.png",
        data_root / subset / "masks" / f"{sid}.png",
        data_root / subset / "mask" / f"{sid}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def mask_features(mask: Optional[np.ndarray]) -> Dict[str, float]:
    if mask is None or mask.size == 0:
        return {
            "mask_foreground_ratio": 0.0,
            "mask_active_column_ratio": 0.0,
            "mask_peak_count_norm": 0.0,
            "mask_vertical_coverage": 0.0,
            "mask_horizontal_coverage": 0.0,
            "mask_density_balance": 0.0,
        }

    h, w = mask.shape[:2]
    fg = mask.astype(bool)
    fg_ratio = float(fg.mean())

    col_sum = fg.sum(axis=0).astype(np.float32)
    row_sum = fg.sum(axis=1).astype(np.float32)
    active_cols = col_sum > max(2.0, 0.02 * h)
    active_rows = row_sum > max(2.0, 0.02 * w)

    if col_sum.max() > 0:
        threshold = max(float(col_sum.mean() + 0.75 * col_sum.std()), 0.08 * h)
        peaks = col_sum > threshold
        peak_count = 0
        in_peak = False
        for flag in peaks:
            if flag and not in_peak:
                peak_count += 1
                in_peak = True
            elif not flag:
                in_peak = False
    else:
        peak_count = 0

    # A coarse balance measure: extremely one-sided masks usually indicate
    # incomplete detections or cropped evidence.
    thirds = np.array_split(col_sum, 3)
    third_mass = np.array([float(x.sum()) for x in thirds], dtype=np.float32)
    total_mass = float(third_mass.sum())
    if total_mass > 0:
        density_balance = float(third_mass.min() / (third_mass.max() + 1e-6))
    else:
        density_balance = 0.0

    return {
        "mask_foreground_ratio": fg_ratio,
        "mask_active_column_ratio": float(active_cols.mean()),
        "mask_peak_count_norm": float(peak_count / max(w / 100.0, 1.0)),
        "mask_vertical_coverage": float(active_rows.mean()),
        "mask_horizontal_coverage": float(active_cols.mean()),
        "mask_density_balance": density_balance,
    }


def segmentation_scores(pred_mask: Optional[np.ndarray], gt_mask: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if pred_mask is None or gt_mask is None:
        return {"seg_iou": None, "seg_f1": None}
    if pred_mask.shape != gt_mask.shape:
        gt_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
        gt_img = gt_img.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
        gt_mask = (np.array(gt_img) > 127).astype(np.uint8)

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    iou = inter / union if union > 0 else 1.0
    f1 = 2.0 * inter / (pred_sum + gt_sum) if pred_sum + gt_sum > 0 else 1.0
    return {"seg_iou": iou, "seg_f1": f1}


def image_quality_features(path: Path) -> Dict[str, float]:
    if not path or not path.exists():
        return {
            "image_brightness": 0.5,
            "image_contrast": 0.0,
            "image_laplacian_var_log": 0.0,
        }

    gray = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    brightness = float(gray.mean())
    contrast = float(gray.std())

    center = gray[1:-1, 1:-1]
    lap = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    lap_var = float(lap.var() * 255.0 * 255.0)
    return {
        "image_brightness": brightness,
        "image_contrast": contrast,
        "image_laplacian_var_log": math.log1p(max(lap_var, 0.0)),
    }


def measurement_features(m: dict) -> Dict[str, float]:
    compliance = m.get("compliance", {}) or {}
    per = compliance.get("per_spacing", []) or []
    spacings = m.get("spacings_mm") or []
    devs = [abs(finite_float(deviation(p))) for p in per]
    fail_count = sum(1 for p in per if is_failed(p))
    max_abs_dev = max(devs) if devs else 0.0
    mean_spacing = finite_float(m.get("mean_spacing_mm"))
    std_spacing = finite_float(m.get("std_spacing_mm"))
    spacing_cv = std_spacing / mean_spacing if mean_spacing > 1e-6 else 0.0
    fail_ratio = fail_count / len(per) if per else 0.0

    return {
        "distance_m": finite_float(m.get("distance_m")),
        "mpp": finite_float(m.get("mpp")),
        "rebar_count": finite_float(m.get("rebar_count")),
        "num_spacings": float(len(spacings)),
        "mean_spacing_mm": mean_spacing,
        "std_spacing_mm": std_spacing,
        "mean_diameter_mm": finite_float(m.get("mean_diameter_mm")),
        "spacing_cv": spacing_cv,
        "max_abs_dev_mm": max_abs_dev,
        "fail_ratio": fail_ratio,
    }


def target_risk_from_measurement(m: dict) -> str:
    if is_v2(m):
        return risk_level(m)
    compliance = m.get("compliance", {}) or {}
    per = compliance.get("per_spacing", []) or []
    status = compliance.get("status")
    if not per:
        if status == "PASS":
            return "ACCEPTABLE"
        if status == "FAIL":
            return "WARNING"
        return "UNKNOWN"

    devs = [abs(finite_float(deviation(p))) for p in per]
    max_dev = max(devs) if devs else 0.0
    fail_ratio = sum(1 for d in devs if d > 20.0) / len(per)
    if fail_ratio == 0 and max_dev <= 5:
        return "SAFE"
    if fail_ratio == 0 and max_dev <= 10:
        return "ACCEPTABLE"
    if fail_ratio <= 0.25 and max_dev <= 40:
        return "MARGINAL"
    if fail_ratio <= 0.5 or max_dev <= 60:
        return "WARNING"
    return "CRITICAL"


def extract_features(rec: dict, data_root: Path, include_gt=True) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    subset = str(rec.get("subset", ""))
    sid = str(rec.get("id", ""))
    seg_path = Path(rec.get("seg_path", ""))
    img_path = Path(rec.get("image_path", ""))
    pred_mask = load_binary_mask(seg_path)
    gt_path = find_gt_mask_path(data_root, subset, sid) if include_gt else None
    gt_mask = load_binary_mask(gt_path) if gt_path else None

    features = {}
    features.update(measurement_features(rec.get("measurement") or {}))
    features.update(mask_features(pred_mask))
    features.update(image_quality_features(img_path))
    features = {name: finite_float(features.get(name)) for name in FEATURE_NAMES}

    quality = segmentation_scores(pred_mask, gt_mask)
    quality["image_available"] = bool(img_path.exists())
    return features, quality


def weak_labels(rec: dict, features: Dict[str, float], quality: Dict[str, Optional[float]]) -> Dict[str, int]:
    m = rec.get("measurement") or {}
    warning = bool(m.get("warning"))
    count_ok = features["rebar_count"] >= 2 and features["num_spacings"] >= 1
    mask_ok = (
        0.005 <= features["mask_foreground_ratio"] <= 0.65
        and features["mask_active_column_ratio"] >= 0.02
        and features["mask_vertical_coverage"] >= 0.10
    )

    if quality.get("seg_iou") is not None:
        segmentation_complete = int(quality["seg_iou"] >= 0.85)
    else:
        segmentation_complete = int(mask_ok and features["mask_peak_count_norm"] > 0)

    if not quality.get("image_available", False):
        # Missing RGB files indicate an incomplete local export, not necessarily
        # poor imaging conditions. Do not train the planner to penalise that.
        image_clear = 1
    else:
        image_clear = int(
            0.08 <= features["image_brightness"] <= 0.92
            and features["image_contrast"] >= 0.04
            and features["image_laplacian_var_log"] >= 1.5
        )

    # Reliability should not punish real non-compliance. It mainly indicates
    # whether the evidence chain is sufficient to support a report.
    measurement_reliable = int(
        not warning
        and count_ok
        and mask_ok
        and features["spacing_cv"] <= 0.50
    )

    return {
        "measurement_reliable": measurement_reliable,
        "segmentation_complete": segmentation_complete,
        "image_clear": image_clear,
    }


def feature_vector(features: Dict[str, float]) -> List[float]:
    return [finite_float(features.get(name)) for name in FEATURE_NAMES]


def qaep_plan(rec: dict, probabilities: Dict[str, float], labels: Dict[str, int]) -> dict:
    m = rec.get("measurement") or {}
    compliance = m.get("compliance", {}) or {}
    per = compliance.get("per_spacing", []) or []
    status = compliance.get("status", "N/A")
    risk = target_risk_from_measurement(m)

    fail_indices = [
        int(p.get("index"))
        for p in per
        if p.get("index") is not None and is_failed(p)
    ]
    devs = [abs(finite_float(deviation(p))) for p in per]
    max_dev = max(devs) if devs else None

    tags = [status, risk]
    constraints = [
        f"State the compliance result as {status}.",
        f"Use the risk level {risk}.",
        "Do not invent measurements that are not present in the structured record.",
    ]

    if max_dev is not None:
        constraints.append(f"Mention the maximum spacing deviation as {max_dev:.1f} mm.")
    if fail_indices:
        idx_text = ", ".join(f"spacing[{idx}]" for idx in fail_indices)
        tags.append("NON_COMPLIANT_SPACING")
        constraints.append(f"Identify the non-compliant spacing positions: {idx_text}.")

    if not labels.get("measurement_reliable", 0):
        tags.append("LIMITED_MEASUREMENT_RELIABILITY")
        constraints.append(
            "Indicate that measurement reliability is limited by incomplete or uncertain evidence."
        )
    if not labels.get("segmentation_complete", 0):
        tags.append("UNCERTAIN_SEGMENTATION")
        constraints.append(
            "Avoid overstating visual certainty because the segmentation evidence is incomplete or uncertain."
        )
    if not labels.get("image_clear", 0):
        tags.append("LOW_IMAGE_QUALITY")
        constraints.append(
            "Mention that image quality may limit visual interpretation if reliability is discussed."
        )

    plan_text = "\n".join(f"- {line}" for line in constraints)
    return {
        "probabilities": probabilities,
        "labels": labels,
        "risk_tags": tags,
        "report_constraints": constraints,
        "plan_text": plan_text,
    }
