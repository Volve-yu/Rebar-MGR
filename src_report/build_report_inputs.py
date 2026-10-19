"""
build_report_inputs.py -- assemble report-generation ablation inputs.

Reads:
  - groundtruth_json/reports.json            reference reports
  - data/{subset}/gt_sets/{split}.txt        val IDs per subset
  - data/{subset}/depth_labels.txt           distance labels
  - data/{subset}/images/{id}.png            original images (for size)
  - predictions_measure_guided/{variant}/{subset}/seg/{id}.png
  - predictions_measure_guided/{variant}/{subset}/edge/{id}.png

Writes per variant:
  {out_dir}/{variant}.jsonl
  {out_dir}/summary.json

Usage:
  python src_report/build_report_inputs.py \\
      --references groundtruth_json/reports.json \\
      --data_root data \\
      --pred_root predictions_measure_guided \\
      --variants full seg_only edge_aux measure_only \\
      --split val \\
      --out_dir results_clean/report_generation/inputs
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from measure import RulerConfig, load_depth_labels, measure_image  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUBSETS = ["site-1", "site-2"]


def _safe(v):
    """Replace NaN with None for JSON serialisation."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _load_references(path: Path) -> dict:
    """Return {(subset, id): reference_report}."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {(r["subset"], str(r["id"])): r["reference_report"] for r in records}


def _load_val_ids(data_root: Path, subset: str, split: str) -> list:
    txt = data_root / subset / "gt_sets" / f"{split}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Split file not found: {txt}")
    return [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]


def _image_size(img_path: Path):
    """Return (width, height) from the image file, or (1280, 720) as fallback."""
    if img_path.exists():
        with Image.open(img_path) as im:
            return im.size  # (W, H)
    return (1280, 720)


def _seg_to_binary(seg_path: Path) -> np.ndarray:
    """Load seg PNG and return (H, W) uint8 binary array (0/1)."""
    arr = np.array(Image.open(seg_path).convert("L"))
    return (arr > 127).astype(np.uint8)


def _measurement_dict(result) -> dict:
    return {
        "distance_m":      _safe(result.distance_m),
        "mpp":             _safe(result.mpp),
        "rebar_count":     result.rebar_count,
        "spacings_mm":     result.spacings_mm,
        "mean_spacing_mm": result.mean_spacing_mm,
        "std_spacing_mm":  result.std_spacing_mm,
        "mean_diameter_mm":result.mean_diameter_mm,
        "compliance":      result.compliance,
        "warning":         result.warning,
    }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_variant(
    variant: str,
    data_root: Path,
    pred_root: Path,
    references: dict,
    split: str,
    out_dir: Path,
) -> dict:
    """
    Build {variant}.jsonl for all subsets.
    Returns summary counts for this variant.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{variant}.jsonl"

    n_total        = 0
    n_missing_seg  = 0
    n_missing_ref  = 0
    n_warnings     = 0

    # Cache depth labels per subset
    depth_cache: dict[str, dict] = {}

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for subset in _SUBSETS:
            val_ids = _load_val_ids(data_root, subset, split)

            labels_path = data_root / subset / "depth_labels.txt"
            if subset not in depth_cache:
                depth_cache[subset] = (
                    load_depth_labels(str(labels_path))
                    if labels_path.exists() else {}
                )
            depth_labels = depth_cache[subset]

            for sid in val_ids:
                seg_path  = pred_root / variant / subset / "seg"  / f"{sid}.png"
                edge_path = pred_root / variant / subset / "edge" / f"{sid}.png"
                img_path  = data_root / subset / "images" / f"{sid}.png"

                if not seg_path.exists():
                    n_missing_seg += 1
                    continue

                n_total += 1

                # Reference report
                ref_key = (subset, sid)
                reference_report = references.get(ref_key)
                if reference_report is None:
                    n_missing_ref += 1
                    reference_report = ""

                # Measurement -- use seg prediction's own size (already at orig res)
                w, h = _image_size(seg_path)
                cfg = RulerConfig(image_width_px=w, image_height_px=h)
                seg_binary = _seg_to_binary(seg_path)
                result = measure_image(seg_binary, sid, depth_labels, cfg)

                if result.warning:
                    n_warnings += 1

                record = {
                    "subset":           subset,
                    "id":               sid,
                    "variant":          variant,
                    "image_path":       str(img_path.as_posix()),
                    "seg_path":         str(seg_path.as_posix()),
                    "edge_path":        str(edge_path.as_posix()),
                    "reference_report": reference_report,
                    "measurement":      _measurement_dict(result),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "variant":        variant,
        "n_total":        n_total,
        "n_missing_seg":  n_missing_seg,
        "n_missing_ref":  n_missing_ref,
        "n_warnings":     n_warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build report-generation ablation inputs (no LLM calls)."
    )
    p.add_argument("--references", default="groundtruth_json/reports.json")
    p.add_argument("--data_root",  default="data")
    p.add_argument("--pred_root",  default="predictions_measure_guided")
    p.add_argument("--variants",   nargs="+",
                   default=["full", "seg_only", "edge_aux", "measure_only"])
    p.add_argument("--split",      default="val")
    p.add_argument("--out_dir",    default="results_clean/report_generation/inputs")
    return p.parse_args()


def main():
    args = parse_args()

    references_path = Path(args.references)
    data_root       = Path(args.data_root)
    pred_root       = Path(args.pred_root)
    out_dir         = Path(args.out_dir)

    if not references_path.exists():
        raise FileNotFoundError(f"References not found: {references_path}")

    print(f"Loading references: {references_path}")
    references = _load_references(references_path)
    print(f"  {len(references)} reference entries loaded.")

    summaries = []
    for variant in args.variants:
        print(f"\nBuilding variant: {variant}")
        summary = build_variant(
            variant=variant,
            data_root=data_root,
            pred_root=pred_root,
            references=references,
            split=args.split,
            out_dir=out_dir,
        )
        summaries.append(summary)
        print(f"  n_total={summary['n_total']}  "
              f"missing_seg={summary['n_missing_seg']}  "
              f"missing_ref={summary['n_missing_ref']}  "
              f"warnings={summary['n_warnings']}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
