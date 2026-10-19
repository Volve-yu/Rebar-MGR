"""
End-to-end inference + measurement pipeline.

For each image in the validation set:
  1. Run model inference -> segmentation prediction
  2. Feed prediction to measure.py -> mm-level spacing/diameter + GB 50204 result
  3. Save a summary CSV and per-image visualisations

Usage:
    python src/run_measure.py \
        --checkpoint checkpoints/best.pth \
        --data_root data \
        --subsets site-1 site-2 \
        --out_dir measure_output

Output:
    measure_output/
        results.csv          -- one row per image
        {subset}_{id}.png    -- 3-panel: RGB | seg pred | measurement annotation
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import RebarMTL
from dataset import RebarSplit
from measure import RulerConfig, load_depth_labels, measure_image, extract_instances_from_heatmap

# Baseline models (optional import — only needed when --mode is not 'ours')
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
try:
    from models import build_model, ALL_MODES as BASELINE_MODES
except ImportError:
    BASELINE_MODES = []
    build_model = None

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def denorm(t):
    t = t.clone().cpu()
    for c, (m, s) in enumerate(zip(_MEAN, _STD)):
        t[c] = t[c] * s + m
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def draw_measurement(rgb, seg_pred_np, result, mpp):
    """
    Draw centroid dots and spacing lines on a copy of rgb.
    Returns annotated numpy HWC uint8 image.
    """
    from measure import extract_instances
    instances = extract_instances(seg_pred_np, min_area_px=50)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"id={result.image_id}  dist={result.distance_m}m  "
        f"mm/px={mpp:.3f}  count={result.rebar_count}  "
        f"GB50204: {result.compliance.get('status','N/A')}",
        fontsize=9,
    )

    # Panel 1: original RGB
    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB")

    # Panel 2: seg prediction
    axes[1].imshow(seg_pred_np, cmap="gray")
    axes[1].set_title("Seg Prediction")

    # Panel 3: measurement overlay
    axes[2].imshow(rgb)
    axes[2].set_title("Measurement")

    colors = ["lime" if r.get("within_stirrup_tol", True) else "red"
              for r in result.compliance.get("per_spacing", [])]

    for i, inst in enumerate(instances):
        cx, cy = inst.centroid_px
        axes[2].plot(cx, cy, "o", color="cyan", markersize=5)

        if i < len(instances) - 1:
            cx2, cy2 = instances[i + 1].centroid_px
            c = colors[i] if i < len(colors) else "lime"
            axes[2].plot([cx, cx2], [cy, cy2], "-", color=c, linewidth=1.5)
            mid_x = (cx + cx2) / 2
            mid_y = (cy + cy2) / 2 - 8
            if i < len(result.spacings_mm):
                axes[2].text(mid_x, mid_y, f"{result.spacings_mm[i]:.0f}mm",
                             color="yellow", fontsize=6, ha="center")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    mode = getattr(args, "mode", "ours")

    if mode in ("ours", "no_consistency"):
        model = RebarMTL(img_size=(args.img_h, args.img_w), pretrained=False).to(device)
    else:
        model = build_model(mode, img_size=(args.img_h, args.img_w), pretrained=False).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint [{mode}]  epoch={ckpt.get('epoch','?')}  "
          f"best_seg_IoU={ckpt.get('best_val_iou', 0):.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_h, args.img_w)
    img_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

    cfg = RulerConfig(
        hfov_deg=args.hfov,
        image_width_px=args.img_w,
        image_height_px=args.img_h,
    )

    csv_rows = []

    for sub in args.subsets:
        sub_path = Path(args.data_root) / sub
        if not sub_path.exists():
            print(f"[SKIP] {sub_path} not found")
            continue

        labels_path = sub_path / "depth_labels.txt"
        if not labels_path.exists():
            print(f"[SKIP] {labels_path} not found -- run extract_depth_labels.py first")
            continue
        depth_labels = load_depth_labels(str(labels_path))

        ds = RebarSplit(str(sub_path), "val", img_size)
        ids = ds.ids if args.n <= 0 else ds.ids[: args.n]
        print(f"{sub}: {len(ids)} val images")

        for sid in ids:
            img_pil = Image.open(sub_path / "images" / f"{sid}.png").convert("RGB")
            img_t   = img_tf(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(img_t)
                if isinstance(out, tuple) and len(out) == 3:
                    seg_logit    = out[1]
                    center_logit = out[2]
                elif isinstance(out, tuple):
                    seg_logit    = out[1]
                    center_logit = None
                else:
                    seg_logit    = out
                    center_logit = None
                seg_prob    = torch.sigmoid(seg_logit)[0, 0].cpu().numpy()
                center_prob = (torch.sigmoid(center_logit)[0, 0].cpu().numpy()
                               if center_logit is not None else None)

            seg_pred = (seg_prob > 0.5).astype(np.uint8)

            # Resize predictions back to original image resolution
            orig_w, orig_h = img_pil.size
            if seg_pred.shape != (orig_h, orig_w):
                seg_pred_pil = Image.fromarray(seg_pred * 255).resize(
                    (orig_w, orig_h), Image.NEAREST)
                seg_pred = (np.array(seg_pred_pil) > 127).astype(np.uint8)

            if center_prob is not None and center_prob.shape != (orig_h, orig_w):
                from PIL import Image as _PILImage
                center_pil = _PILImage.fromarray(
                    (center_prob * 255).clip(0, 255).astype(np.uint8)
                ).resize((orig_w, orig_h), Image.BILINEAR)
                center_prob = np.array(center_pil).astype(np.float32) / 255.0

            orig_cfg = RulerConfig(
                hfov_deg=cfg.hfov_deg,
                image_width_px=orig_w,
                image_height_px=orig_h,
            )

            # Use heatmap-based instance detection when center head is available
            if center_prob is not None:
                from measure import extract_instances_from_heatmap
                instances = extract_instances_from_heatmap(
                    center_prob, seg_pred, min_distance=30)
                result = measure_image(seg_pred, sid, depth_labels, orig_cfg,
                                       instances_override=instances)
            else:
                result = measure_image(seg_pred, sid, depth_labels, orig_cfg)

            csv_rows.append({
                "subset":           sub,
                "image_id":         sid,
                "distance_m":       result.distance_m,
                "mpp":              result.mpp,
                "rebar_count":      result.rebar_count,
                "spacings_mm":      ";".join(str(s) for s in result.spacings_mm),
                "mean_spacing_mm":  result.mean_spacing_mm,
                "std_spacing_mm":   result.std_spacing_mm,
                "mean_diameter_mm": result.mean_diameter_mm,
                "compliance":       result.compliance.get("status", "N/A"),
                "warning":          result.warning or "",
            })

            if result.warning:
                print(f"  [{sub}/{sid}] {result.warning}")
                continue

            rgb = denorm(img_t[0].cpu())
            vis = draw_measurement(rgb, seg_pred, result, result.mpp)
            Image.fromarray(vis).save(out_dir / f"{sub}_{sid}.png")

    # Write CSV
    csv_path = out_dir / "results.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV saved -> {csv_path}  ({len(csv_rows)} rows)")

        valid = [r for r in csv_rows if not r["warning"]]
        if valid:
            spacings = [r["mean_spacing_mm"] for r in valid if r["mean_spacing_mm"]]
            diams    = [r["mean_diameter_mm"] for r in valid if r["mean_diameter_mm"]]
            pass_n   = sum(1 for r in valid if r["compliance"] == "PASS")
            print(f"Valid measurements: {len(valid)}/{len(csv_rows)}")
            if spacings:
                print(f"  Mean spacing: {np.mean(spacings):.1f} mm  "
                      f"(std {np.std(spacings):.1f} mm)")
            if diams:
                print(f"  Mean diameter: {np.mean(diams):.1f} mm  "
                      f"(std {np.std(diams):.1f} mm)")
            print(f"  GB 50204 PASS rate: {pass_n}/{len(valid)} "
                  f"({100*pass_n/len(valid):.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--mode",       default="ours",
                   help="Model type: ours, no_consistency, seg_only, edge_only, "
                        "no_coupling, symmetric, gscnn, mobile_seed")
    p.add_argument("--data_root",  default="data")
    p.add_argument("--subsets",    nargs="+", default=["site-1", "site-2"])
    p.add_argument("--out_dir",    default="measure_output")
    p.add_argument("--n",          type=int, default=0,
                   help="Images per subset to process (0 = all val samples)")
    p.add_argument("--hfov",       type=float, default=87.0)
    p.add_argument("--img_h",      type=int, default=480)
    p.add_argument("--img_w",      type=int, default=640)
    run(p.parse_args())
