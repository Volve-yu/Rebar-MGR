"""Build clean label-derived reference reports for report-generation evaluation.

This script does not call a VLM. It reads manual rebar centerline labels
(`data/{subset}/rebar_labels/{id}.txt`) and shooting distances
(`data/{subset}/depth_labels.txt`), computes spacing measurements with the same
camera-scale rule used by the measurement pipeline, and writes a compact
`reports.json` file that can be passed to `build_report_inputs.py`.

The output should be described as label-derived reference reports rather than
human-written inspection reports.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from measure import RulerConfig, check_compliance, load_depth_labels  # noqa: E402


def load_split_ids(data_root: Path, subset: str, split: str):
    split_dir = data_root / subset / "gt_sets"
    if split == "trainval":
        ids = []
        for name in ("train.txt", "val.txt"):
            path = split_dir / name
            if path.exists():
                ids.extend(x.strip() for x in path.read_text().splitlines() if x.strip())
        return ids
    path = split_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def load_rebar_xs(path: Path):
    xs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        xs.append(float(line))
    return sorted(xs)


def risk_from_compliance(compliance: dict):
    per = compliance.get("per_spacing", []) or []
    status = compliance.get("status")
    if not per:
        return "UNKNOWN", 0.0, 0, 0.0

    devs = [abs(float(p.get("deviation_from_mean_mm", 0.0))) for p in per]
    max_dev = max(devs) if devs else 0.0
    fail_count = sum(1 for p in per if not p.get("within_main_tol", False))
    fail_ratio = fail_count / len(per) if per else 0.0

    if fail_count == 0 and max_dev <= 5:
        level = "SAFE"
    elif fail_count == 0 and max_dev <= 10:
        level = "ACCEPTABLE"
    elif fail_ratio <= 0.25 and max_dev <= 40:
        level = "MARGINAL"
    elif fail_ratio <= 0.5 or max_dev <= 60:
        level = "WARNING"
    elif status == "PASS":
        level = "ACCEPTABLE"
    else:
        level = "CRITICAL"
    return level, round(max_dev, 1), fail_count, fail_ratio


def trend_from_spacings(spacings):
    if len(spacings) < 2:
        return "INSUFFICIENT_DATA", 0.0
    slope = (float(spacings[-1]) - float(spacings[0])) / max(len(spacings) - 1, 1)
    if slope > 2.0:
        trend = "INCREASING"
    elif slope < -2.0:
        trend = "DECREASING"
    else:
        trend = "UNIFORM"
    return trend, round(slope, 1)


def fmt_list(values):
    return ", ".join(f"{v:.1f}" for v in values)


def build_reference(subset: str, sid: str, xs, distance_m: float, cfg: RulerConfig):
    mpp = cfg.mm_per_pixel(distance_m)
    spacings = [round((xs[i + 1] - xs[i]) * mpp, 1) for i in range(len(xs) - 1)]
    mean_spacing = round(float(np.mean(spacings)), 1) if spacings else None
    std_spacing = round(float(np.std(spacings)), 1) if spacings else None
    compliance = check_compliance(spacings, cfg)
    status = compliance.get("status", "N/A")

    risk, max_dev, fail_count, fail_ratio = risk_from_compliance(compliance)
    trend, slope = trend_from_spacings(spacings)
    fail_items = [
        p for p in compliance.get("per_spacing", []) or []
        if not p.get("within_main_tol", False)
    ]

    overview = (
        f"The inspection detected {len(xs)} vertical rebars with "
        f"{len(spacings)} measured longitudinal spacings; the mean spacing was "
        f"{mean_spacing:.1f} mm and the standard deviation was {std_spacing:.1f} mm."
        if mean_spacing is not None and std_spacing is not None
        else f"The inspection detected {len(xs)} vertical rebars, but spacing assessment was incomplete."
    )

    if mean_spacing is None or std_spacing is None:
        report = (
            f"{overview} Fewer than two centerline labels were available, so "
            "longitudinal spacing could not be assessed reliably. Additional "
            "measurement is required before acceptance. Result: N/A."
        )
    elif fail_items:
        fail_desc = "; ".join(
            f"spacing[{int(p['index'])}]={float(p['spacing_mm']):.1f} mm "
            f"({float(p['deviation_from_mean_mm']):+.1f} mm)"
            for p in fail_items
        )
        report = (
            f"{overview} According to the GB 50204-2015 +/-20 mm tolerance, "
            f"the non-compliant interval(s) were {fail_desc}, with a maximum "
            f"deviation of {max_dev:.1f} mm. The risk level is {risk}, the "
            f"non-compliant ratio is {fail_ratio:.3f}, and the spacing trend is "
            f"{trend} (slope={slope:+.1f} mm/gap). Correction and re-inspection "
            f"are recommended before concrete placement. Result: {status}."
        )
    else:
        report = (
            f"{overview} All spacings were within the GB 50204-2015 +/-20 mm "
            f"tolerance for longitudinal bar spacing, with a maximum deviation "
            f"of {max_dev:.1f} mm. The risk level is {risk}, and the spacing trend "
            f"is {trend} (slope={slope:+.1f} mm/gap). The installation can proceed "
            f"to the next construction stage after routine quality confirmation. "
            f"Result: {status}."
        )

    measurement = {
        "distance_m": distance_m,
        "mpp": round(float(mpp), 4) if not math.isnan(mpp) else None,
        "rebar_count": len(xs),
        "spacings_mm": spacings,
        "mean_spacing_mm": mean_spacing,
        "std_spacing_mm": std_spacing,
        "compliance": compliance,
        "risk_level": risk,
        "max_dev_mm": max_dev,
        "fail_count": fail_count,
        "fail_ratio": fail_ratio,
        "trend": trend,
        "slope_mm_per_gap": slope,
    }

    return {
        "subset": subset,
        "id": sid,
        "source_file": f"data/{subset}/rebar_labels/{sid}.txt",
        "heading_type": "label_derived_reference",
        "reference_report": report,
        "measurement": measurement,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Build clean label-derived reference reports."
    )
    p.add_argument("--data_root", default="data", type=Path)
    p.add_argument("--subsets", nargs="+", default=["site-1", "site-2"])
    p.add_argument("--split", default="val", choices=["train", "val", "trainval"])
    p.add_argument("--out_dir", default="groundtruth_label_reference_json", type=Path)
    p.add_argument("--hfov", type=float, default=87.0)
    p.add_argument("--img_w", type=int, default=1280)
    p.add_argument("--img_h", type=int, default=720)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RulerConfig(
        hfov_deg=args.hfov,
        image_width_px=args.img_w,
        image_height_px=args.img_h,
    )

    records = []
    skipped = []
    for subset in args.subsets:
        root = args.data_root / subset
        depth_path = root / "depth_labels.txt"
        depth_labels = load_depth_labels(str(depth_path)) if depth_path.exists() else {}
        for sid in load_split_ids(args.data_root, subset, args.split):
            label_path = root / "rebar_labels" / f"{sid}.txt"
            if not label_path.exists() or sid not in depth_labels:
                skipped.append({"subset": subset, "id": sid})
                continue
            xs = load_rebar_xs(label_path)
            records.append(build_reference(subset, sid, xs, depth_labels[sid], cfg))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports_path = args.out_dir / "reports.json"
    reports_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    per_file_dir = args.out_dir / "by_sample"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = args.out_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        stem = f"{record['subset']}_{record['id']}"
        (per_file_dir / f"{stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (txt_dir / f"{stem}.txt").write_text(
            record["reference_report"] + "\n",
            encoding="utf-8",
        )
    summary = {
        "data_root": args.data_root.as_posix(),
        "split": args.split,
        "num_records": len(records),
        "num_skipped": len(skipped),
        "skipped": skipped,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {reports_path}")
    print(f"Records: {len(records)}  skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
