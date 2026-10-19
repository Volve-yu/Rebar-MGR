"""
Unified training script for baselines_clean/ workspace.

Modes:
  hed          edge-only  (HED-MobileNetV3 adapted)
  deeplab      seg-only   (DeepLabV3+)
  sctransnet   seg-only   (SCTransNet)
  afenet       seg-only   (AFENet)
  mobile_seed  dual-task  (Mobile-Seed adapted)
  gscnn        dual-task  (GSCNN adapted)
  only_e2s     dual-task  (ablation: only Edge->Seg coupling direction)
  only_s2e     dual-task  (ablation: only Seg->Edge coupling direction)

Usage (run from project root):
  python baselines_clean/train.py --mode only_e2s
  python baselines_clean/train.py --mode deeplab
  python baselines_clean/train.py --mode mobile_seed --resume checkpoints_clean/mobile_seed/latest.pth

Checkpoints: {save_dir}/{mode}/best.pth
             {save_dir}/{mode}/latest.pth
JSON log:    {log_dir}/{mode}/run_{timestamp}.json

Hyperparameters match src/train.py for fair comparison.
"""

import argparse
import importlib.util
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dataset import build_dataset
from losses import edge_loss, bce_dice_loss, consistency_loss
from experiment_snapshot import snapshot


# ---------------------------------------------------------------------------
# Mode sets
# ---------------------------------------------------------------------------

CLEAN_MODES      = ["hed", "deeplab", "sctransnet", "afenet",
                    "mobile_seed", "gscnn",
                    "no_coupling", "symmetric",
                    "only_e2s", "only_s2e",
                    "ours_full", "ours_full_no_consistency",
                    "ours_full_edge03", "ours_full_edge15"]
_SEG_ONLY        = {"deeplab", "sctransnet", "afenet"}
_EDGE_ONLY       = {"hed"}
_DUAL            = {"mobile_seed", "gscnn", "no_coupling", "symmetric",
                    "only_e2s", "only_s2e",
                    "ours_full", "ours_full_no_consistency",
                    "ours_full_edge03", "ours_full_edge15"}
# Consistency loss applies only to the full bidirectional model.
# ours_full_no_consistency uses the same architecture but no consistency term,
# isolating the contribution of lambda_c.
_USE_CONSISTENCY = {"ours_full"}
# Per-mode edge loss scale; all modes not listed default to 1.0.
_EDGE_LOSS_SCALE = {
    "ours_full_edge03": 0.3,
    "ours_full_edge15": 1.5,
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_REBAR_ADAPT = _ROOT / "baselines_clean" / "official"


def _load_class(model_py: Path, class_name: str):
    """Import a class from an absolute model.py path without sys.path pollution."""
    spec = importlib.util.spec_from_file_location("_bc_model_tmp", model_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def build_model_clean(mode, img_size=(480, 640), pretrained=True):
    """
    Returns an nn.Module for the given mode.
    Raises NotImplementedError until the corresponding Stage implements it.

    Stage C: only_e2s, only_s2e
    Stage D: hed, deeplab, mobile_seed, gscnn
    Stage E: ours_full
    Stage F: ours_full_no_consistency
    Stage G: ours_full_edge03, ours_full_edge15
    """
    if mode in {"no_coupling", "symmetric"}:
        sys.path.insert(0, str(_ROOT))
        from baselines.models import build_model as build_legacy_model
        return build_legacy_model(mode, img_size=img_size, pretrained=pretrained)

    if mode == "ours_full":
        cls = _load_class(
            _REBAR_ADAPT / "ours_full" / "rebar_adapt" / "model.py",
            "OursFullNet",
        )
        return cls(img_size=img_size, pretrained=pretrained)

    if mode == "ours_full_no_consistency":
        cls = _load_class(
            _REBAR_ADAPT / "ours_full_no_consistency" / "rebar_adapt" / "model.py",
            "OursFullNoConsistencyNet",
        )
        return cls(img_size=img_size, pretrained=pretrained)

    if mode == "ours_full_edge03":
        cls = _load_class(
            _REBAR_ADAPT / "ours_full_edge03" / "rebar_adapt" / "model.py",
            "OursFullEdge03Net",
        )
        return cls(img_size=img_size, pretrained=pretrained)

    if mode == "ours_full_edge15":
        cls = _load_class(
            _REBAR_ADAPT / "ours_full_edge15" / "rebar_adapt" / "model.py",
            "OursFullEdge15Net",
        )
        return cls(img_size=img_size, pretrained=pretrained)

    if mode == "only_e2s":
        cls = _load_class(
            _REBAR_ADAPT / "only_e2s" / "rebar_adapt" / "model.py",
            "OnlyEdgeToSegNet",
        )
        return cls(img_size=img_size, pretrained=pretrained)

    if mode == "only_s2e":
        cls = _load_class(
            _REBAR_ADAPT / "only_s2e" / "rebar_adapt" / "model.py",
            "OnlySegToEdgeNet",
        )
        return cls(img_size=img_size, pretrained=pretrained)

    if mode == "hed":
        raise NotImplementedError(
            "HED not yet implemented. "
            "Add HEDRebar to "
            "baselines_clean/official/HED/rebar_adapt/hed_rebar.py (Stage D)."
        )
    if mode == "deeplab":
        cls = _load_class(
            _REBAR_ADAPT / "DeepLabV3Plus" / "rebar_adapt" / "deeplabv3plus_rebar.py",
            "DeepLabV3PlusRebar",
        )
        return cls(img_size=img_size, pretrained=pretrained)
    if mode == "sctransnet":
        cls = _load_class(
            _REBAR_ADAPT / "SCTransNet" / "rebar_adapt" / "sctransnet_rebar.py",
            "SCTransNetRebar",
        )
        return cls(img_size=img_size, pretrained=pretrained)
    if mode == "afenet":
        cls = _load_class(
            _REBAR_ADAPT / "AFENet" / "rebar_adapt" / "afenet_rebar.py",
            "AFENetRebar",
        )
        return cls(img_size=img_size, pretrained=pretrained)
    if mode == "mobile_seed":
        raise NotImplementedError(
            "Mobile-Seed not yet implemented. "
            "Add MobileSeedRebar to "
            "baselines_clean/official/Mobile-Seed/rebar_adapt/mobile_seed_rebar.py (Stage D)."
        )
    if mode == "gscnn":
        cls = _load_class(
            _REBAR_ADAPT / "GSCNN" / "rebar_adapt" / "gscnn_rebar.py",
            "GSCNNRebar",
        )
        return cls(img_size=img_size, pretrained=pretrained)
    raise ValueError(f"Unknown mode: {mode!r}. Valid modes: {CLEAN_MODES}")


# ---------------------------------------------------------------------------
# Forward + loss
# ---------------------------------------------------------------------------

def forward_and_loss(model, batch, device, mode, edge_pos_weight, lambda_c):
    img     = batch["image"].to(device)
    edge_gt = batch["edge"].to(device)
    seg_gt  = batch["seg"].to(device)

    if mode in _SEG_ONLY:
        seg_logit = model(img)
        if isinstance(seg_logit, tuple):
            l = sum(bce_dice_loss(item, seg_gt) for item in seg_logit)
            seg_logit = seg_logit[-1]
        else:
            l = bce_dice_loss(seg_logit, seg_gt)
        return {"seg": l, "total": l}, None, seg_logit

    if mode in _EDGE_ONLY:
        edge_logit = model(img)
        l = edge_loss(edge_logit, edge_gt, edge_pos_weight)
        return {"edge": l, "total": l}, edge_logit, None

    # Dual-task
    edge_logit, seg_logit = model(img)
    l_edge = edge_loss(edge_logit, edge_gt, edge_pos_weight)
    l_seg  = bce_dice_loss(seg_logit, seg_gt)
    edge_scale = _EDGE_LOSS_SCALE.get(mode, 1.0)
    losses = {"edge": edge_scale * l_edge, "seg": l_seg}

    if mode in _USE_CONSISTENCY and lambda_c > 0:
        losses["consistency"] = lambda_c * consistency_loss(edge_logit, seg_logit)

    losses["total"] = sum(losses.values())
    return losses, edge_logit, seg_logit


# ---------------------------------------------------------------------------
# Inline evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, mode):
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


# ---------------------------------------------------------------------------
# JSON run log
# ---------------------------------------------------------------------------

def _init_log(args, best_key: str, save_dir: Path, pred_dir: Path) -> dict:
    return {
        "mode":              args.mode,
        "started":           datetime.now().isoformat(),
        "args":              vars(args),
        "best_metric_key":   best_key,
        "best_metric_value": None,
        "checkpoint_best":   str(save_dir / "best.pth"),
        "checkpoint_latest": str(save_dir / "latest.pth"),
        "prediction_dir":    str(pred_dir),
        "epochs":            [],
    }


def _append_epoch(log: dict, epoch: int, train_loss: dict,
                  val_metrics: dict, lr: float):
    log["epochs"].append({
        "epoch":        epoch,
        "lr":           round(lr, 8),
        "train_loss":   {k: round(v, 6) for k, v in train_loss.items()},
        "val_metrics":  val_metrics,
    })


def _write_log(log: dict, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a baselines_clean model on rebar data."
    )
    p.add_argument("--mode",            required=True, choices=CLEAN_MODES)
    p.add_argument("--data_root",       default="data",
                   help="Root of the rebar dataset (default: data/)")
    p.add_argument("--subsets",         nargs="+",     default=["site-1", "site-2"])
    p.add_argument("--img_h",           type=int,      default=480)
    p.add_argument("--img_w",           type=int,      default=640)
    p.add_argument("--epochs",          type=int,      default=50)
    p.add_argument("--batch_size",      type=int,      default=4)
    p.add_argument("--lr",              type=float,    default=1e-4)
    p.add_argument("--lambda_c",        type=float,    default=0.1,
                   help="Consistency loss weight (only_e2s / only_s2e only)")
    p.add_argument("--edge_pos_weight", type=float,    default=10.0)
    p.add_argument("--num_workers",     type=int,      default=4,
                   help="DataLoader workers (set 0 for debugging)")
    p.add_argument("--device",          default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Training device (auto: use CUDA if available)")
    p.add_argument("--seed",            type=int,      default=42)
    p.add_argument("--save_dir",        default="checkpoints_clean",
                   help="Root checkpoint directory")
    p.add_argument("--pred_dir",        default="predictions",
                   help="Root prediction directory (logged only; export via predict.py)")
    p.add_argument("--log_dir",         default="logs_clean",
                   help="Root directory for JSON training logs")
    p.add_argument("--resume",          default=None,
                   help="Path to checkpoint to resume from (best.pth or latest.pth)")
    p.add_argument("--log_every",       type=int,      default=10,
                   help="Print step-level loss every N steps")
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)

    print(f"[baselines_clean] mode={args.mode}  device={device}  seed={args.seed}")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram_gb:.1f} GB")

    img_size = (args.img_h, args.img_w)
    save_dir = Path(args.save_dir) / args.mode
    if save_dir.exists() and any(save_dir.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh checkpoint directory or --resume')
    save_dir.mkdir(parents=True, exist_ok=True)
    training_snapshot = snapshot(args.data_root, args.subsets, ['train', 'val'])

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / args.mode / f"run_{run_ts}.json"
    pred_dir = Path(args.pred_dir) / args.mode

    # Dataset + loaders
    train_ds = build_dataset(args.data_root, "train", img_size, args.subsets)
    val_ds   = build_dataset(args.data_root, "val",   img_size, args.subsets)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model    = build_model_clean(args.mode, img_size=img_size, pretrained=True).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params_m:.2f} M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_metric = -1.0
    best_key    = "edge_f1" if args.mode in _EDGE_ONLY else "seg_iou"

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get('training_snapshot') != training_snapshot:
            raise ValueError('Training data changed or checkpoint has no snapshot')
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {start_epoch}  best_{best_key}={best_metric:.4f}")

    log = _init_log(args, best_key, save_dir, pred_dir)
    _write_log(log, log_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            losses, _, _ = forward_and_loss(
                model, batch, device, args.mode,
                args.edge_pos_weight, args.lambda_c,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            if (step + 1) % args.log_every == 0:
                parts = " ".join(
                    f"{k}={v.item():.4f}" for k, v in losses.items() if k != "total"
                )
                print(f"  E{epoch}[{step+1}/{len(train_loader)}] "
                      f"total={losses['total'].item():.4f}  {parts}")

        scheduler.step()
        n_steps     = len(train_loader)
        avg_losses  = {k: v / n_steps for k, v in epoch_losses.items()}
        loss_str    = " | ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
        current_lr  = scheduler.get_last_lr()[0]
        elapsed     = time.time() - t0
        print(f"Epoch {epoch:03d} [{elapsed:.0f}s  lr={current_lr:.2e}] {loss_str}")

        metrics = evaluate(model, val_loader, device, args.mode)
        parts = []
        if args.mode not in _EDGE_ONLY:
            parts.append(
                f"seg_IoU={metrics['seg_iou']:.4f}  "
                f"seg_mIoU={metrics['seg_miou']:.4f}  "
                f"seg_F1={metrics['seg_f1']:.4f}  "
                f"seg_P={metrics['seg_precision']:.4f}  "
                f"seg_R={metrics['seg_recall']:.4f}"
            )
        if args.mode not in _SEG_ONLY:
            parts.append(
                f"edge_P={metrics['edge_precision']:.4f}  "
                f"edge_R={metrics['edge_recall']:.4f}  "
                f"edge_F1={metrics['edge_f1']:.4f}"
            )
        print(f"  Val -> {'  '.join(parts)}")

        # JSON log
        _append_epoch(log, epoch, avg_losses, metrics, current_lr)

        ckpt_payload = {
            "training_snapshot": training_snapshot,
            "args": vars(args),
            "epoch":        epoch,
            "mode":         args.mode,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_metric":  best_metric,
            "best_key":     best_key,
            "metrics":      metrics,
        }

        if metrics[best_key] > best_metric:
            best_metric = metrics[best_key]
            ckpt_payload["best_metric"] = best_metric
            torch.save(ckpt_payload, save_dir / "best.pth")
            log["best_metric_value"] = best_metric
            print(f"  Saved best ({best_key}={best_metric:.4f})")

        torch.save(ckpt_payload, save_dir / "latest.pth")
        _write_log(log, log_path)

    print(f"\nDone. Best {best_key}: {best_metric:.4f}")
    print(f"Checkpoint : {save_dir / 'best.pth'}")
    print(f"Run log    : {log_path}")


if __name__ == "__main__":
    main()
