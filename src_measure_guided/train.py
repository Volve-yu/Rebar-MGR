"""
Measurement-guided training entrypoint.

This directory is intentionally separate from src/ so previous experiments stay
reproducible. It reuses src/model.py, src/dataset.py, and src/eval.py, but uses
src_measure_guided/losses.py for the new measurement-aware projection loss.

Usage:
    python src_measure_guided/train.py --epochs 50 --device cuda
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(1, str(_SRC))

from dataset import build_dataset  # noqa: E402
from eval import evaluate  # noqa: E402
from model import RebarMTL  # noqa: E402
from losses import MeasurementGuidedLoss  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--subsets", nargs="+", default=["site-1", "site-2"])
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--alpha_edge", type=float, default=0.3,
                   help="Edge auxiliary loss scale.")
    p.add_argument("--beta_proj", type=float, default=0.1,
                   help="Measurement-guided column projection loss scale.")
    p.add_argument("--lambda_c", type=float, default=0.0,
                   help="Optional edge/seg consistency loss scale.")
    p.add_argument("--edge_pos_weight", type=float, default=10.0)
    p.add_argument("--max_edge_to_seg", type=float, default=0.0,
                   help="If >0, cap weighted edge loss to this fraction of seg loss.")
    p.add_argument("--max_proj_to_seg", type=float, default=0.0,
                   help="If >0, cap weighted projection loss to this fraction of seg loss.")
    p.add_argument("--aux_schedule", default="constant",
                   choices=["constant", "warmup_cosine_decay"],
                   help="Schedule for auxiliary edge/projection losses.")
    p.add_argument("--aux_warmup_epochs", type=int, default=0,
                   help="Warm up auxiliary losses for this many epochs.")
    p.add_argument("--aux_min_scale", type=float, default=1.0,
                   help="Final auxiliary scale for warmup_cosine_decay.")
    p.add_argument("--seg_refine_epochs", type=int, default=0,
                   help="Disable auxiliary losses in the last N epochs for segmentation refinement.")

    p.add_argument("--save_dir", default="checkpoints_measure_guided")
    p.add_argument("--log_dir", default="logs_measure_guided")
    p.add_argument("--resume", default=None)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def aux_scale_for_epoch(epoch, epochs, args):
    if args.seg_refine_epochs > 0 and epoch >= epochs - args.seg_refine_epochs:
        return 0.0

    if args.aux_schedule == "constant":
        return 1.0

    warmup = max(0, args.aux_warmup_epochs)
    if warmup > 0 and epoch < warmup:
        return float(epoch + 1) / float(warmup)

    decay_start = warmup
    decay_len = max(1, epochs - decay_start - max(0, args.seg_refine_epochs))
    progress = min(1.0, max(0.0, (epoch - decay_start + 1) / float(decay_len)))
    min_scale = max(0.0, min(1.0, args.aux_min_scale))
    return min_scale + 0.5 * (1.0 - min_scale) * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    from experiment_snapshot import snapshot
    training_snapshot = snapshot(args.data_root, args.subsets, ['train','val'])
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[measure_guided] device={device} seed={args.seed}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM={props.total_memory / 1e9:.1f}GB")

    img_size = (args.img_h, args.img_w)
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    if save_dir.exists() and any(save_dir.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh checkpoint directory for the new dataset.')
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(args.data_root, "train", img_size, args.subsets)
    val_ds = build_dataset(args.data_root, "val", img_size, args.subsets)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = RebarMTL(img_size=img_size, pretrained=True).to(device)
    criterion = MeasurementGuidedLoss(
        lambda_c=args.lambda_c,
        edge_pos_weight=args.edge_pos_weight,
        alpha_edge=args.alpha_edge,
        beta_proj=args.beta_proj,
        max_edge_to_seg=args.max_edge_to_seg,
        max_proj_to_seg=args.max_proj_to_seg,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_iou = 0.0
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get('training_snapshot') != training_snapshot:
            raise ValueError('Resume checkpoint uses another or unrecorded dataset; start a new run.')
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    run_log = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for epoch in range(start_epoch, args.epochs):
        aux_scale = aux_scale_for_epoch(epoch, args.epochs, args)
        criterion.set_aux_scales(edge_scale=aux_scale, proj_scale=aux_scale)
        print(f"Auxiliary loss scale for epoch {epoch:03d}: {aux_scale:.4f}")

        model.train()
        epoch_losses = {}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            edge_gt = batch["edge"].to(device)
            seg_gt = batch["seg"].to(device)
            center_gt = batch.get("center")
            center_gt = center_gt.to(device) if center_gt is not None else None

            edge_logit, seg_logit = model(img)
            losses = criterion(edge_logit, seg_logit, edge_gt, seg_gt, center_gt)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + float(v.item())

            if (step + 1) % args.log_every == 0:
                print(
                    f"  E{epoch}[{step+1}/{len(train_loader)}] "
                    f"total={losses['total'].item():.4f} "
                    f"seg={losses['seg'].item():.4f} "
                    f"edge={losses['edge'].item():.4f} "
                    f"proj={losses['projection'].item():.4f} "
                    f"edge_scale={losses['edge_scale'].item():.3f} "
                    f"proj_scale={losses['projection_scale'].item():.3f} "
                    f"aux_scale={aux_scale:.3f} "
                    f"cons={losses['consistency'].item():.4f}"
                )

        scheduler.step()

        n = len(train_loader)
        avg_losses = {k: v / n for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        loss_str = " | ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
        print(f"Epoch {epoch:03d} [{elapsed:.0f}s] {loss_str}")

        metrics = evaluate(model, val_loader, device)
        print(
            f"  Val -> seg_IoU={metrics['seg_iou']:.4f} "
            f"seg_mIoU={metrics['seg_miou']:.4f} "
            f"seg_F1={metrics['seg_f1']:.4f} edge_F1={metrics['edge_f1']:.4f}"
        )

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "losses": avg_losses,
            "metrics": metrics,
            "elapsed_sec": elapsed,
            "args": vars(args),
        }
        history.append(row)
        run_log.write_text(json.dumps(history, indent=2), encoding="utf-8")

        state = {
            "training_snapshot": training_snapshot,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_iou": best_val_iou,
            "metrics": metrics,
            "args": vars(args),
        }
        torch.save(state, save_dir / "latest.pth")

        if metrics["seg_iou"] > best_val_iou:
            best_val_iou = metrics["seg_iou"]
            state["best_val_iou"] = best_val_iou
            torch.save(state, save_dir / "best.pth")
            print(f"  Saved best (seg_IoU={best_val_iou:.4f})")

    print(f"Training complete. Best val seg IoU: {best_val_iou:.4f}")
    print(f"Checkpoint: {save_dir / 'best.pth'}")
    print(f"Run log:    {run_log}")


if __name__ == "__main__":
    main()
