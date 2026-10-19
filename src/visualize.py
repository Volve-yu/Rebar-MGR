"""
Visualize model predictions on the validation set.

Generates a 4-panel figure per image:
  [RGB input] [edge GT] [edge pred] [seg pred overlaid on RGB]

Usage:
    python src/visualize.py \
        --checkpoint checkpoints/best.pth \
        --data_root data \
        --out_dir vis_output \
        --n 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import RebarMTL
from dataset import RebarSplit


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def denorm(t):
    """Reverse ImageNet normalisation -> numpy HWC uint8."""
    t = t.clone().cpu()
    for c, (m, s) in enumerate(zip(_MEAN, _STD)):
        t[c] = t[c] * s + m
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def overlay_mask(rgb, mask, color=(255, 80, 80), alpha=0.45):
    """Blend a binary mask onto an RGB image."""
    out = rgb.copy().astype(np.float32)
    for c, v in enumerate(color):
        out[..., c] = np.where(mask, out[..., c] * (1 - alpha) + v * alpha, out[..., c])
    return out.clip(0, 255).astype(np.uint8)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = RebarMTL(img_size=(args.img_h, args.img_w), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"best seg_IoU={ckpt.get('best_val_iou', '?'):.4f})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_h, args.img_w)
    img_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

    # Collect val samples from all requested subsets
    samples = []
    for sub in args.subsets:
        path = Path(args.data_root) / sub
        if not path.exists():
            continue
        ds = RebarSplit(str(path), "val", img_size)
        for sid in ds.ids:
            samples.append((path, sid))

    if args.n > 0:
        step = max(1, len(samples) // args.n)
        samples = samples[::step][: args.n]

    print(f"Visualising {len(samples)} samples -> {out_dir}/")

    for path, sid in samples:
        img_pil = Image.open(path / "images" / f"{sid}.png").convert("RGB")
        img_t   = img_tf(img_pil).unsqueeze(0).to(device)

        seg_pil  = Image.open(path / "segmentation" / f"{sid}.png").convert("L")
        seg_pil  = seg_pil.resize((img_size[1], img_size[0]), Image.NEAREST)
        seg_gt   = (np.array(seg_pil) > 127)

        edge_file = path / "edge" / f"{sid}.png"
        if edge_file.exists():
            edge_pil = Image.open(edge_file).convert("L")
            edge_pil = edge_pil.resize((img_size[1], img_size[0]), Image.NEAREST)
            edge_gt  = (np.array(edge_pil) > 127)
        else:
            edge_gt = np.zeros((img_size[0], img_size[1]), dtype=bool)

        with torch.no_grad():
            edge_logit, seg_logit = model(img_t)
            edge_prob = torch.sigmoid(edge_logit)[0, 0].cpu().numpy()
            seg_prob  = torch.sigmoid(seg_logit)[0, 0].cpu().numpy()

        edge_pred = edge_prob > 0.5
        seg_pred  = seg_prob  > 0.5

        rgb = denorm(img_t[0].cpu())

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle(f"{path.name} / {sid}", fontsize=10)

        axes[0].imshow(rgb)
        axes[0].set_title("Input RGB")

        axes[1].imshow(edge_gt, cmap="gray")
        axes[1].set_title("Edge GT")

        axes[2].imshow(edge_pred, cmap="gray")
        axes[2].set_title(f"Edge Pred (thr=0.5)")

        axes[3].imshow(overlay_mask(rgb, seg_pred))
        axes[3].set_title("Seg Pred (overlay)")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(out_dir / f"{path.name}_{sid}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Done. {len(samples)} images saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--data_root",  default="data")
    p.add_argument("--subsets",    nargs="+", default=["site-1", "site-2"])
    p.add_argument("--out_dir",    default="vis_output")
    p.add_argument("--n",          type=int, default=20,
                   help="Number of images to visualise (0 = all val samples)")
    p.add_argument("--img_h",      type=int, default=480)
    p.add_argument("--img_w",      type=int, default=640)
    run(p.parse_args())
