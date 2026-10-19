"""
Training script for the rebar multi-task system (edge + segmentation).

Usage (local or AutoDL):
    python src/train.py --data_root data --epochs 50 --batch_size 4
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import build_dataset
from model import RebarMTL
from losses import TotalLoss
from eval import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--subsets", nargs="+", default=["site-1", "site-2"])
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_c", type=float, default=0.1)
    p.add_argument("--lambda_h", type=float, default=1.0,
                   help="Weight for center heatmap focal loss (unused, kept for compat)")
    p.add_argument("--edge_pos_weight", type=float, default=10.0,
                   help="Positive-class weight multiplier for edge BCE (edges are sparse)")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--resume", default=None)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram_gb:.1f}GB")
        if vram_gb < 8 and args.batch_size > 4:
            args.batch_size = 4
            print(f"VRAM < 8GB, auto-reduced batch_size to {args.batch_size}")

    img_size = (args.img_h, args.img_w)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # --- Data ---
    train_ds = build_dataset(args.data_root, "train", img_size, args.subsets)
    val_ds   = build_dataset(args.data_root, "val",   img_size, args.subsets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # --- Model ---
    model     = RebarMTL(img_size=img_size, pretrained=True).to(device)
    criterion = TotalLoss(args.lambda_c, args.edge_pos_weight, args.lambda_h)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch  = 0
    best_val_iou = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            img      = batch["image"].to(device)
            edge_gt  = batch["edge"].to(device)
            seg_gt   = batch["seg"].to(device)

            edge_logit, seg_logit = model(img)
            losses = criterion(edge_logit, seg_logit, edge_gt, seg_gt)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            if (step + 1) % args.log_every == 0:
                print(f"  E{epoch} [{step+1}/{len(train_loader)}] "
                      f"total={losses['total'].item():.4f} "
                      f"edge={losses['edge'].item():.4f} "
                      f"seg={losses['seg'].item():.4f} "
                      f"cons={losses['consistency'].item():.4f}")

        scheduler.step()

        n = len(train_loader)
        loss_str = " | ".join(f"{k}={v/n:.4f}" for k, v in epoch_losses.items())
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:03d} [{elapsed:.0f}s] {loss_str}")

        # --- Validation ---
        metrics = evaluate(model, val_loader, device)
        print(f"  Val -> seg_IoU={metrics['seg_iou']:.4f} | seg_F1={metrics['seg_f1']:.4f} | "
              f"edge_F1={metrics['edge_f1']:.4f}")

        if metrics["seg_iou"] > best_val_iou:
            best_val_iou = metrics["seg_iou"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
                "metrics": metrics,
            }, save_dir / "best.pth")
            print(f"  Saved best (seg_IoU={best_val_iou:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
            }, save_dir / f"epoch_{epoch:03d}.pth")

    print(f"Training complete. Best val seg IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
