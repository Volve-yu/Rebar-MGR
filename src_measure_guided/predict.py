"""Export measurement-guided predictions to PNG, CSV, and JSON.

Usage:
    python src_measure_guided/predict.py \
      --checkpoint checkpoints_measure_guided/full/best.pth \
      --pred_dir predictions_measure_guided/full --device cuda
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(1, str(_SRC))

from dataset import RebarSplit  # noqa: E402
from measure import RulerConfig, load_depth_labels, measure_image  # noqa: E402
from model import RebarMTL  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--legacy_measurements", action="store_true",
                   help="Reproduce historical mean-based measurements; not for the physical v2 protocol.")
    p.add_argument("--data_root", default="data")
    p.add_argument("--subsets", nargs="+", default=["site-1", "site-2"])
    p.add_argument("--split", default="val", choices=["train", "val", "test", "trainval"])
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pred_dir", default="predictions_measure_guided/full")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_png(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def to_original_size(pred, image_path):
    if not image_path.exists():
        return pred
    with Image.open(image_path) as img:
        orig_size = img.size
    if pred.shape == (orig_size[1], orig_size[0]):
        return pred
    pred_img = Image.fromarray((pred * 255).astype(np.uint8))
    pred_img = pred_img.resize(orig_size, Image.NEAREST)
    return (np.array(pred_img) > 127).astype(np.uint8)


@torch.no_grad()
def export_subset(model, subset_root, split, img_size, out_dir, device, args):
    """Save seg+edge PNGs and return list of (sid, seg_binary) at original resolution."""
    ds = RebarSplit(str(subset_root), split, img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    n_saved = 0
    seg_records = []  # [(sid, seg_binary_at_orig_res), ...]
    for batch in loader:
        imgs = batch["image"].to(device)
        ids = batch["id"]
        edge_logit, seg_logit = model(imgs)

        for b, sid in enumerate(ids):
            image_path = Path(subset_root) / "images" / f"{sid}.png"

            seg = (torch.sigmoid(seg_logit[b, 0]) > 0.5).cpu().numpy().astype(np.uint8)
            seg = to_original_size(seg, image_path)
            save_png((seg * 255).astype(np.uint8), out_dir / "seg" / f"{sid}.png")
            seg_records.append((sid, seg))

            edge = (torch.sigmoid(edge_logit[b, 0]) > 0.5).cpu().numpy().astype(np.uint8)
            edge = to_original_size(edge, image_path)
            save_png((edge * 255).astype(np.uint8), out_dir / "edge" / f"{sid}.png")
            n_saved += 1

    print(f"  Saved {n_saved} seg+edge predictions -> {out_dir}")
    return seg_records


def _safe_float(v):
    """Return None for NaN so JSON stays valid."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def measure_and_export(subset, seg_records, data_root, out_dir, img_w, img_h):
    """Run measure_image on every prediction and write measurements.csv + measurements.json."""
    labels_path = Path(data_root) / subset / "depth_labels.txt"
    if not labels_path.exists():
        print(f"  [SKIP] depth_labels.txt not found: {labels_path}")
        return 0

    depth_labels = load_depth_labels(str(labels_path))
    cfg = RulerConfig(image_width_px=img_w, image_height_px=img_h)

    results = [
        measure_image(seg_binary, sid, depth_labels, cfg)
        for sid, seg_binary in seg_records
    ]

    # --- CSV (flat) ---
    csv_fields = [
        "subset", "id", "distance_m", "mpp", "rebar_count",
        "spacings_mm", "mean_spacing_mm", "std_spacing_mm",
        "mean_diameter_mm", "compliance", "warning",
    ]
    csv_path = out_dir / "measurements.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "subset": subset,
                "id": r.image_id,
                "distance_m": _safe_float(r.distance_m),
                "mpp": _safe_float(r.mpp),
                "rebar_count": r.rebar_count,
                "spacings_mm": json.dumps(r.spacings_mm),
                "mean_spacing_mm": r.mean_spacing_mm,
                "std_spacing_mm": r.std_spacing_mm,
                "mean_diameter_mm": r.mean_diameter_mm,
                "compliance": r.compliance.get("status", "N/A"),
                "warning": r.warning or "",
            })

    # --- JSON (full compliance detail) ---
    json_path = out_dir / "measurements.json"
    json_rows = [
        {
            "subset": subset,
            "id": r.image_id,
            "distance_m": _safe_float(r.distance_m),
            "mpp": _safe_float(r.mpp),
            "rebar_count": r.rebar_count,
            "spacings_mm": r.spacings_mm,
            "mean_spacing_mm": r.mean_spacing_mm,
            "std_spacing_mm": r.std_spacing_mm,
            "mean_diameter_mm": r.mean_diameter_mm,
            "compliance": r.compliance,
            "warning": r.warning,
        }
        for r in results
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2, ensure_ascii=False)

    print(f"  Measurements: {len(results)} records -> {out_dir}/measurements.{{csv,json}}")
    return len(results)


def main():
    args = parse_args()
    from experiment_snapshot import snapshot, file_sha
    output = Path(args.pred_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh prediction directory; do not mix datasets/splits.')
    prediction_snapshot = snapshot(args.data_root, args.subsets, [args.split])
    device = resolve_device(args.device)
    img_size = (args.img_h, args.img_w)

    model = RebarMTL(img_size=img_size, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for subset in args.subsets:
        subset_root = Path(args.data_root) / subset
        out_dir = Path(args.pred_dir) / subset
        print(f"subset={subset} split={args.split} out={out_dir}")
        seg_records = export_subset(
            model, subset_root, args.split, img_size, out_dir, device, args
        )
        if args.legacy_measurements:
            measure_and_export(subset, seg_records, args.data_root, out_dir, args.img_w, args.img_h)
    (output/'prediction_manifest.json').write_text(json.dumps(dict(
        checkpoint_sha256=file_sha(args.checkpoint), checkpoint=str(args.checkpoint),
        training_snapshot=ckpt.get('training_snapshot'), prediction_snapshot=prediction_snapshot,
        measurement_entry='scripts/run_physical_inspection.py'), indent=2), encoding='utf-8')


if __name__ == "__main__":
    main()
