"""
Evaluate a baselines_clean checkpoint and print metrics for paper table.

Usage (run from project root):
  python baselines_clean/eval.py --mode deeplab \\
      --checkpoint checkpoints_clean/deeplab/best.pth

  python baselines_clean/eval.py --all

Output:
  Per-model: seg_IoU, seg_F1, edge_F1, Params(M)
  Summary table (copy-paste ready for paper)

Notes:
  - HED (edge-only): seg_IoU and seg_F1 reported as N/A
  - DeepLabV3+, SCTransNet, and AFENet (seg-only): edge_F1 reported as N/A
  - spacing MAE is computed separately via src/eval_measure.py using seg predictions
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent.parent
_BC_DIR = Path(__file__).parent
sys.path.insert(0, str(_BC_DIR))
sys.path.insert(1, str(_ROOT / "src"))

from dataset import build_dataset
from train import (   # re-use mode sets and build_model_clean from train.py
    CLEAN_MODES, _SEG_ONLY, _EDGE_ONLY, build_model_clean,
)


# ---------------------------------------------------------------------------
# Evaluate one checkpoint
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(model, loader, device, mode):
    model.eval()
    seg_iou = seg_miou = seg_f1 = seg_precision = seg_recall = seg_acc = 0.0
    edge_precision = edge_recall = edge_f1 = 0.0
    n = 0
    eps = 1e-6

    for batch in loader:
        img     = batch["image"].to(device)
        edge_gt = batch["edge"].to(device)
        seg_gt  = batch["seg"].to(device)

        if mode in _SEG_ONLY:
            seg_logit, edge_logit = model(img), None
            if isinstance(seg_logit, tuple):
                seg_logit = seg_logit[-1]
        elif mode in _EDGE_ONLY:
            edge_logit, seg_logit = model(img), None
        else:
            edge_logit, seg_logit = model(img)

        B = img.shape[0]
        n += B

        for b in range(B):
            if seg_logit is not None:
                sp = (torch.sigmoid(seg_logit[b, 0]) > 0.5).float()
                sg = seg_gt[b, 0]
                inter = (sp * sg).sum()
                union = sp.sum() + sg.sum() - inter
                seg_iou += ((inter + eps) / (union + eps)).item()
                tp = inter; fp = sp.sum() - inter; fn = sg.sum() - inter
                tn = ((1.0 - sp) * (1.0 - sg)).sum()
                bg_union = (1.0 - sp).sum() + (1.0 - sg).sum() - tn
                bg_iou = (tn + eps) / (bg_union + eps)
                seg_miou += 0.5 * (((inter + eps) / (union + eps)) + bg_iou).item()
                seg_f1  += ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item()
                seg_precision += ((tp + eps) / (tp + fp + eps)).item()
                seg_recall += ((tp + eps) / (tp + fn + eps)).item()
                seg_acc += ((tp + tn + eps) / (tp + tn + fp + fn + eps)).item()
            if edge_logit is not None:
                ep = (torch.sigmoid(edge_logit[b, 0]) > 0.5).float()
                eg = edge_gt[b, 0]
                tp_e = (ep * eg).sum()
                fp_e = ep.sum() - tp_e
                fn_e = eg.sum() - tp_e
                edge_precision += ((tp_e + eps) / (tp_e + fp_e + eps)).item()
                edge_recall += ((tp_e + eps) / (tp_e + fn_e + eps)).item()
                edge_f1 += ((2 * tp_e + eps) / (2 * tp_e + fp_e + fn_e + eps)).item()

    return {
        "seg_iou": round(seg_iou / n, 4) if n else 0.0,
        "seg_miou": round(seg_miou / n, 4) if n else 0.0,
        "seg_f1": round(seg_f1 / n, 4) if n else 0.0,
        "seg_precision": round(seg_precision / n, 4) if n else 0.0,
        "seg_recall": round(seg_recall / n, 4) if n else 0.0,
        "seg_acc": round(seg_acc / n, 4) if n else 0.0,
        "edge_precision": round(edge_precision / n, 4) if n else 0.0,
        "edge_recall": round(edge_recall / n, 4) if n else 0.0,
        "edge_f1": round(edge_f1 / n, 4) if n else 0.0,
    }


def count_params(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)


def load_and_eval(mode, ckpt_path, val_loader, device, img_size):
    model = build_model_clean(mode, img_size=img_size, pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    params  = count_params(model)
    metrics = evaluate_checkpoint(model, val_loader, device, mode)
    return metrics, params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate baselines_clean checkpoints."
    )
    p.add_argument("--mode",       default=None, choices=CLEAN_MODES,
                   help="Single mode (use --all for all modes)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pth checkpoint")
    p.add_argument("--all",        action="store_true",
                   help="Evaluate all modes with default checkpoint paths")
    p.add_argument("--data_root",  default="data")
    p.add_argument("--subsets",    nargs="+",    default=["site-1", "site-2"])
    p.add_argument("--split",      default="val",
                   choices=["train", "val", "trainval"])
    p.add_argument("--img_h",       type=int,     default=480)
    p.add_argument("--img_w",       type=int,     default=640)
    p.add_argument("--save_dir",    default="checkpoints_clean")
    p.add_argument("--num_workers", type=int,     default=4)
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    img_size = (args.img_h, args.img_w)

    val_ds     = build_dataset(args.data_root, args.split, img_size, args.subsets)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
    print(f"Eval set: {args.split}  |  {len(val_ds)} images")

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

    rows = []
    for mode, ckpt_path in targets:
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(ckpt_path)
        try:
            metrics, params = load_and_eval(mode, ckpt_path, val_loader,
                                            device, img_size)
            rows.append((mode, metrics, params))
            print(f"  {mode:16s}  seg_IoU={metrics['seg_iou']:.4f}  "
                  f"seg_mIoU={metrics['seg_miou']:.4f}  "
                  f"seg_F1={metrics['seg_f1']:.4f}  "
                  f"seg_P={metrics['seg_precision']:.4f}  "
                  f"seg_R={metrics['seg_recall']:.4f}  "
                  f"seg_Acc={metrics['seg_acc']:.4f}  "
                  f"edge_P={metrics['edge_precision']:.4f}  "
                  f"edge_R={metrics['edge_recall']:.4f}  "
                  f"edge_F1={metrics['edge_f1']:.4f}  "
                  f"Params={params}M")
        except NotImplementedError as e:
            raise
        except Exception as e:
            raise

    # Paper table
    print(f"\n{'='*72}")
    print(f"{'Method':<22} {'seg_IoU':>8} {'seg_mIoU':>8} {'seg_F1':>8} {'seg_P':>8} {'seg_R':>8} {'edge_F1':>8} {'Params':>8}")
    print(f"{'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for mode, metrics, params in rows:
        if metrics is None:
            print(f"{mode:<22} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        else:
            si = f"{metrics['seg_iou']:.4f}" if mode not in _EDGE_ONLY else "N/A"
            sm = f"{metrics['seg_miou']:.4f}" if mode not in _EDGE_ONLY else "N/A"
            sf = f"{metrics['seg_f1']:.4f}"  if mode not in _EDGE_ONLY else "N/A"
            sp = f"{metrics['seg_precision']:.4f}" if mode not in _EDGE_ONLY else "N/A"
            sr = f"{metrics['seg_recall']:.4f}" if mode not in _EDGE_ONLY else "N/A"
            ef = f"{metrics['edge_f1']:.4f}" if mode not in _SEG_ONLY  else "N/A"
            pm = f"{params}M" if params is not None else "N/A"
            print(f"{mode:<22} {si:>8} {sm:>8} {sf:>8} {sp:>8} {sr:>8} {ef:>8} {pm:>8}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
