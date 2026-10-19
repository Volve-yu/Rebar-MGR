"""
Export model predictions to PNG files for spacing MAE evaluation.

Modes and their outputs:
  hed          edge-only   -> predictions/{mode}/{subset}/edge/{id}.png
  deeplab      seg-only    -> predictions/{mode}/{subset}/seg/{id}.png
  sctransnet   seg-only    -> predictions/{mode}/{subset}/seg/{id}.png
  afenet       seg-only    -> predictions/{mode}/{subset}/seg/{id}.png
  mobile_seed  dual-task   -> predictions/{mode}/{subset}/seg/{id}.png
                              predictions/{mode}/{subset}/edge/{id}.png
  gscnn        dual-task   (same as mobile_seed)
  only_e2s     dual-task   (same as mobile_seed)
  only_s2e     dual-task   (same as mobile_seed)

Usage (run from project root):
  python baselines_clean/predict.py --mode deeplab \\
      --checkpoint checkpoints_clean/deeplab/best.pth

  python baselines_clean/predict.py --all

Then compute spacing MAE (seg-output models only):
  python src/eval_measure.py --subset site-1 \\
      --seg_dir predictions/deeplab/site-1/seg --split val
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent.parent
_BC_DIR = Path(__file__).parent
sys.path.insert(0, str(_BC_DIR))
sys.path.insert(1, str(_ROOT / "src"))

from dataset import RebarSplit
from train import CLEAN_MODES, _SEG_ONLY, _EDGE_ONLY, build_model_clean
from experiment_snapshot import snapshot, file_sha


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _to_original_size(pred: np.ndarray, image_path: Path) -> np.ndarray:
    """Resize a binary prediction mask back to the source image resolution."""
    if not image_path.exists():
        return pred
    with Image.open(image_path) as img:
        orig_size = img.size
    if pred.shape == (orig_size[1], orig_size[0]):
        return pred
    pred_img = Image.fromarray((pred * 255).astype(np.uint8))
    pred_img = pred_img.resize(orig_size, Image.NEAREST)
    return (np.array(pred_img) > 127).astype(np.uint8)


def _load_model(mode, img_size, ckpt_path, device):
    model = build_model_clean(mode, img_size=img_size, pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


# ---------------------------------------------------------------------------
# Inference + export
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_predictions(model, mode, subset_root, split, img_size, out_dir, device,
                       num_workers=4, batch_size=2):
    """
    Run inference on one subset split and save binary prediction PNGs.

    seg:  0/255 PNG, saved to out_dir/seg/{id}.png
    edge: 0/255 PNG, saved to out_dir/edge/{id}.png
    """
    ds = RebarSplit(str(subset_root), split, img_size)
    if len(ds) == 0:
        print(f"  [SKIP] No {split} images in {subset_root}")
        return

    loader  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers,
                         pin_memory=(device.type == "cuda"))
    model.eval()

    has_seg  = mode not in _EDGE_ONLY
    has_edge = mode not in _SEG_ONLY

    n_saved = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        ids  = batch["id"]

        if mode in _SEG_ONLY:
            seg_logit  = model(imgs)
            edge_logit = None
        elif mode in _EDGE_ONLY:
            edge_logit = model(imgs)
            seg_logit  = None
        else:
            edge_logit, seg_logit = model(imgs)

        for b in range(imgs.shape[0]):
            sid = ids[b]
            image_path = Path(subset_root) / "images" / f"{sid}.png"
            if seg_logit is not None:
                pred = (torch.sigmoid(seg_logit[b, 0]) > 0.5).cpu().numpy()
                pred = _to_original_size(pred.astype(np.uint8), image_path)
                _save_png(
                    (pred * 255).astype(np.uint8),
                    out_dir / "seg" / f"{sid}.png",
                )
            if edge_logit is not None:
                pred = (torch.sigmoid(edge_logit[b, 0]) > 0.5).cpu().numpy()
                pred = _to_original_size(pred.astype(np.uint8), image_path)
                _save_png(
                    (pred * 255).astype(np.uint8),
                    out_dir / "edge" / f"{sid}.png",
                )
            n_saved += 1

    kind = ("seg+edge" if (has_seg and has_edge)
            else ("seg" if has_seg else "edge"))
    print(f"  Saved {n_saved} {kind} predictions -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Export baselines_clean model predictions to PNG."
    )
    p.add_argument("--mode",       default=None, choices=CLEAN_MODES,
                   help="Model mode (use --all for all modes)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pth checkpoint")
    p.add_argument("--all",        action="store_true",
                   help="Export all modes with default checkpoint paths")
    p.add_argument("--data_root",  default="data")
    p.add_argument("--subsets",    nargs="+",    default=["site-1", "site-2"])
    p.add_argument("--split",      default="val",
                   choices=["train", "val", "trainval"])
    p.add_argument("--img_h",       type=int,     default=480)
    p.add_argument("--img_w",       type=int,     default=640)
    p.add_argument("--save_dir",    default="checkpoints_clean",
                   help="Root checkpoint dir (used with --all)")
    p.add_argument("--pred_dir",    default="predictions",
                   help="Root output dir; files saved to {pred_dir}/{mode}/{subset}/")
    p.add_argument("--num_workers", type=int,     default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    img_size  = (args.img_h, args.img_w)
    data_root = Path(args.data_root)

    if args.all:
        targets = [
            (mode, f"{args.save_dir}/{mode}/best.pth")
            for mode in CLEAN_MODES
        ]
    else:
        if not args.mode or not args.checkpoint:
            print("[ERROR] Provide --mode and --checkpoint, or use --all")
            return
        targets = [(args.mode, args.checkpoint)]

    for mode, ckpt_path in targets:
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(ckpt_path)

        print(f"\n[{mode}] Loading {ckpt_path}")
        payload = torch.load(ckpt_path, map_location='cpu')
        if not payload.get('training_snapshot'):
            raise ValueError('Use a newly trained checkpoint with a data snapshot')
        output = Path(args.pred_dir) / mode
        if output.exists():
            raise FileExistsError(output)
        model = _load_model(mode, img_size, ckpt_path, device)

        for subset in args.subsets:
            subset_root = data_root / subset
            if not subset_root.is_dir():
                raise FileNotFoundError(subset_root)
            out_dir = Path(args.pred_dir) / mode / subset
            print(f"  subset={subset}  split={args.split}  out={out_dir}")
            export_predictions(model, mode, subset_root, args.split,
                               img_size, out_dir, device,
                               num_workers=args.num_workers, batch_size=args.batch_size)
        (output / 'prediction_manifest.json').write_text(json.dumps(dict(
            checkpoint=str(ckpt), checkpoint_sha256=file_sha(ckpt),
            training_snapshot=payload['training_snapshot'],
            prediction_snapshot=snapshot(data_root,args.subsets,[args.split]),
            measurement_entry='scripts/run_physical_inspection.py'),indent=2),encoding='utf-8')

    print("\nDone.")
    print("To compute spacing MAE (seg-output models):")
    print("  python src/eval_measure.py --subset site-1 \\")
    print(f"    --seg_dir {args.pred_dir}/{{mode}}/site-1/seg --split val")


if __name__ == "__main__":
    main()
