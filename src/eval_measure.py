"""
Evaluate algorithmic rebar measurement against manual ground-truth labels.

Manual labels: data/{subset}/rebar_labels/{id}.txt
  - One x-pixel per line (integer), sorted left to right.
  - Produced by src/label_rebars.py

Algorithm: measure.py extract_instances() on segmentation masks.
  - Segmentation masks: data/{subset}/segmentation/{id}.png

Usage:
    python src/eval_measure.py --subset site-1
    python src/eval_measure.py --subset site-1 --split val
    python src/eval_measure.py --subset site-1 --visualize --out_dir eval_out

Output:
    Prints per-image comparison table + aggregate MAE stats.
    With --visualize: saves side-by-side images to out_dir/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from measure import (RulerConfig, load_depth_labels,
                     extract_instances, MeasurementResult)


# ---------------------------------------------------------------------------
# Load manual labels
# ---------------------------------------------------------------------------

def load_manual_label(label_path: Path):
    """Return sorted list of x-pixel floats, or None if file missing."""
    if not label_path.exists():
        return None
    xs = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                xs.append(float(line))
            except ValueError:
                pass
    return sorted(xs) if xs else None


def spacings_from_xs(xs: list, mpp: float):
    """Convert sorted x-pixel list to spacing list in mm."""
    return [(xs[i+1] - xs[i]) * mpp for i in range(len(xs) - 1)]


# ---------------------------------------------------------------------------
# Per-image comparison
# ---------------------------------------------------------------------------

def compare_image(image_id: str,
                  seg_path: Path,
                  img_path: Path,
                  label_path: Path,
                  depth_labels: dict,
                  cfg: RulerConfig,
                  min_area_px: int = 50) -> dict:
    """
    Returns a dict with keys:
      image_id, has_label, has_seg,
      gt_count, algo_count, count_match,
      gt_spacings_mm, algo_spacings_mm,
      spacing_mae_mm,          # mean abs error across matched spacings
      spacing_errors_mm,       # per-spacing errors (gt - algo)
      gt_mean_mm, algo_mean_mm,
      mean_error_mm,
      mpp, distance_m
    """
    result = {
        "image_id": image_id,
        "has_label": label_path.exists(),
        "has_seg": seg_path.exists(),
        "gt_count": 0, "algo_count": 0, "count_match": False,
        "gt_spacings_mm": [], "algo_spacings_mm": [],
        "spacing_mae_mm": None, "spacing_errors_mm": [],
        "gt_mean_mm": None, "algo_mean_mm": None,
        "mean_error_mm": None,
        "mpp": None, "distance_m": None,
    }

    if not label_path.exists() or not seg_path.exists():
        return result

    gt_xs = load_manual_label(label_path)
    if gt_xs is None or len(gt_xs) < 2:
        return result

    if image_id not in depth_labels:
        return result

    distance_m = depth_labels[image_id]
    # Use the segmentation image coordinate system for both GT and predictions.
    # Manual labels are annotated in the original image coordinate system, while
    # model predictions are usually exported at the resized training resolution
    # (e.g. 640 px wide). Scale GT x-coordinates into seg_img coordinates before
    # computing spacing; otherwise GT spacing is overestimated by orig_w/seg_w.
    seg_img = np.array(Image.open(seg_path).convert("L"))
    actual_w = seg_img.shape[1]
    if img_path.exists():
        orig_w = Image.open(img_path).width
    else:
        orig_w = actual_w
    if orig_w != actual_w:
        scale = actual_w / float(orig_w)
        gt_xs = [x * scale for x in gt_xs]

    actual_cfg = RulerConfig(hfov_deg=cfg.hfov_deg,
                             image_width_px=actual_w,
                             image_height_px=seg_img.shape[0])
    mpp = actual_cfg.mm_per_pixel(distance_m)
    result["mpp"] = round(mpp, 4)
    result["distance_m"] = distance_m

    # --- Algorithm measurement ---
    seg_binary = (seg_img > 127).astype(np.uint8)
    instances = extract_instances(seg_binary, min_area_px=min_area_px)
    algo_xs = [inst.centroid_px[0] for inst in instances]

    gt_count = len(gt_xs)
    algo_count = len(algo_xs)
    result["gt_count"] = gt_count
    result["algo_count"] = algo_count
    result["count_match"] = (gt_count == algo_count)

    gt_spacings = spacings_from_xs(gt_xs, mpp)
    result["gt_spacings_mm"] = [round(s, 1) for s in gt_spacings]
    result["gt_mean_mm"] = round(float(np.mean(gt_spacings)), 1) if gt_spacings else None

    if len(algo_xs) >= 2:
        algo_spacings = spacings_from_xs(algo_xs, mpp)
        result["algo_spacings_mm"] = [round(s, 1) for s in algo_spacings]
        result["algo_mean_mm"] = round(float(np.mean(algo_spacings)), 1)

    # --- Spacing MAE: only when counts match ---
    if gt_count == algo_count and len(gt_spacings) > 0 and len(result["algo_spacings_mm"]) > 0:
        errors = [g - a for g, a in zip(gt_spacings, result["algo_spacings_mm"])]
        result["spacing_errors_mm"] = [round(e, 1) for e in errors]
        result["spacing_mae_mm"] = round(float(np.mean(np.abs(errors))), 1)
        if result["gt_mean_mm"] and result["algo_mean_mm"]:
            result["mean_error_mm"] = round(result["gt_mean_mm"] - result["algo_mean_mm"], 1)

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_viz(image_id: str,
             seg_path: Path,
             img_path: Path,
             gt_xs: list,
             algo_xs: list,
             gt_spacings: list,
             algo_spacings: list,
             mpp: float,
             out_path: Path):
    """Side-by-side: left=original with GT lines, right=seg with algo lines."""
    orig = Image.open(img_path).convert("RGB") if img_path.exists() else Image.new("RGB", (640, 480), (40, 40, 40))
    seg  = Image.open(seg_path).convert("RGB")

    W, H = orig.size
    canvas = Image.new("RGB", (W * 2 + 4, H + 30), (30, 30, 30))
    canvas.paste(orig, (0, 30))
    canvas.paste(seg,  (W + 4, 30))

    draw = ImageDraw.Draw(canvas)

    # Header
    draw.text((4, 4), f"{image_id}  GT={len(gt_xs)} bars  Algo={len(algo_xs)} bars  mpp={mpp:.3f}mm/px",
              fill=(220, 220, 220))

    # GT lines (red) on left panel
    for x in gt_xs:
        draw.line([(int(x), 30), (int(x), 30 + H)], fill=(255, 60, 60), width=2)
    for i, sp in enumerate(gt_spacings):
        if i < len(gt_xs) - 1:
            mid = int((gt_xs[i] + gt_xs[i+1]) / 2)
            draw.text((mid - 15, 30 + H // 2 - 8), f"{sp:.0f}", fill=(255, 200, 0))

    # Algo lines (cyan) on right panel
    for x in algo_xs:
        draw.line([(W + 4 + int(x), 30), (W + 4 + int(x), 30 + H)], fill=(0, 220, 220), width=2)
    for i, sp in enumerate(algo_spacings):
        if i < len(algo_xs) - 1:
            mid = int((algo_xs[i] + algo_xs[i+1]) / 2)
            draw.text((W + 4 + mid - 15, 30 + H // 2 - 8), f"{sp:.0f}", fill=(100, 255, 100))

    # Labels
    draw.text((4, 30 + H - 16), "RED = GT (manual)", fill=(255, 100, 100))
    draw.text((W + 8, 30 + H - 16), "CYAN = Algo", fill=(0, 220, 220))

    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--subset", default="site-1", choices=["site-1", "site-2"])
    ap.add_argument("--split", default="trainval",
                    choices=["train", "val", "trainval"])
    ap.add_argument("--min_area_px", type=int, default=50)
    ap.add_argument("--hfov", type=float, default=87.0)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--visualize", action="store_true",
                    help="Save side-by-side comparison images")
    ap.add_argument("--out_dir", default="eval_out",
                    help="Output directory for visualizations")
    ap.add_argument("--seg_dir", default=None,
                    help="Override segmentation dir (e.g. model predictions from "
                         "baselines/predict.py). Default: data/{subset}/segmentation/")
    ap.add_argument("--count_mismatch_only", action="store_true",
                    help="Only print images where rebar count differs")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    sub_path  = data_root / args.subset
    img_dir   = sub_path / "images"
    seg_dir   = Path(args.seg_dir) if args.seg_dir else sub_path / "segmentation"
    label_dir = sub_path / "rebar_labels"

    cfg = RulerConfig(hfov_deg=args.hfov, image_width_px=args.img_w,
                      image_height_px=args.img_h)

    depth_file = sub_path / "depth_labels.txt"
    depth_labels = load_depth_labels(str(depth_file)) if depth_file.exists() else {}

    split_file = sub_path / "gt_sets" / f"{args.split}.txt"
    if split_file.exists():
        ids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    else:
        ids = [p.stem for p in sorted(seg_dir.glob("*.png"))]

    if args.visualize:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    for sid in ids:
        r = compare_image(
            sid,
            seg_dir / f"{sid}.png",
            img_dir / f"{sid}.png",
            label_dir / f"{sid}.txt",
            depth_labels, cfg, args.min_area_px,
        )
        if not r["has_label"]:
            skipped += 1
            continue
        rows.append(r)

        if args.count_mismatch_only and r["count_match"]:
            continue

        # Per-image print
        status = "OK " if r["count_match"] else "CNT"
        mae_str = f"MAE={r['spacing_mae_mm']:5.1f}mm" if r["spacing_mae_mm"] is not None else "MAE=  N/A  "
        print(f"[{status}] {sid:>6}  GT={r['gt_count']} Algo={r['algo_count']}  "
              f"{mae_str}  "
              f"GT_mean={r['gt_mean_mm']}  Algo_mean={r['algo_mean_mm']}  "
              f"err={r['mean_error_mm']}")
        if r["spacing_errors_mm"]:
            print(f"         errors(mm): {r['spacing_errors_mm']}")

        if args.visualize and r["has_seg"]:
            gt_xs = load_manual_label(label_dir / f"{sid}.txt") or []
            seg_img = Image.open(seg_dir / f"{sid}.png").convert("L")
            if (img_dir / f"{sid}.png").exists():
                orig_w = Image.open(img_dir / f"{sid}.png").width
            else:
                orig_w = seg_img.width
            if orig_w != seg_img.width:
                scale = seg_img.width / float(orig_w)
                gt_xs = [x * scale for x in gt_xs]
            instances = extract_instances(
                (np.array(seg_img) > 127).astype(np.uint8),
                min_area_px=args.min_area_px,
            )
            algo_xs = [inst.centroid_px[0] for inst in instances]
            viz_cfg = RulerConfig(hfov_deg=cfg.hfov_deg,
                                  image_width_px=seg_img.width,
                                  image_height_px=seg_img.height)
            mpp = viz_cfg.mm_per_pixel(depth_labels.get(sid, 0.4))
            save_viz(
                sid,
                seg_dir / f"{sid}.png",
                img_dir / f"{sid}.png",
                gt_xs, algo_xs,
                spacings_from_xs(gt_xs, mpp),
                spacings_from_xs(algo_xs, mpp),
                mpp,
                out_dir / f"{sid}.png",
            )

    # --- Aggregate stats ---
    labeled = len(rows)
    count_match = sum(1 for r in rows if r["count_match"])
    maes = [r["spacing_mae_mm"] for r in rows if r["spacing_mae_mm"] is not None]
    mean_errors = [r["mean_error_mm"] for r in rows if r["mean_error_mm"] is not None]

    print(f"\n{'='*60}")
    print(f"Subset: {args.subset}  Split: {args.split}")
    print(f"Labeled images evaluated : {labeled}  (skipped unlabeled: {skipped})")
    print(f"Count match (GT==Algo)   : {count_match}/{labeled} "
          f"({100*count_match/labeled:.1f}%)" if labeled else "")
    if maes:
        print(f"Spacing MAE (count-match): {np.mean(maes):.1f} mm  "
              f"(median {np.median(maes):.1f}, max {np.max(maes):.1f})")
    if mean_errors:
        print(f"Mean spacing bias        : {np.mean(mean_errors):+.1f} mm  "
              f"(std {np.std(mean_errors):.1f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
