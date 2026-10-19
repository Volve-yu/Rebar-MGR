"""Evaluate a measurement-guided checkpoint.

Usage:
    python src_measure_guided/eval.py \
      --checkpoint checkpoints_measure_guided/full/best.pth --device cuda
"""

import argparse
import json
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(1, str(_SRC))

from dataset import build_dataset  # noqa: E402
from model import RebarMTL  # noqa: E402

_EVAL_SPEC = importlib.util.spec_from_file_location("src_eval_module", _SRC / "eval.py")
_EVAL_MODULE = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(_EVAL_MODULE)
evaluate = _EVAL_MODULE.evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--subsets", nargs="+", default=["site-1", "site-2"])
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out_json", default=None,
                   help="Optional path for writing evaluation metrics as JSON.")
    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    args = parse_args()
    device = resolve_device(args.device)
    img_size = (args.img_h, args.img_w)

    ds = build_dataset(args.data_root, args.split, img_size, args.subsets)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = RebarMTL(img_size=img_size, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    metrics = evaluate(model, loader, device)
    params = count_params(model)

    print(f"Eval set: {args.split} | {len(ds)} images")
    print(
        "measure_guided "
        f"seg_IoU={metrics['seg_iou']:.4f} "
        f"seg_mIoU={metrics['seg_miou']:.4f} "
        f"seg_F1={metrics['seg_f1']:.4f} "
        f"seg_P={metrics['seg_precision']:.4f} "
        f"seg_R={metrics['seg_recall']:.4f} "
        f"seg_Acc={metrics['seg_acc']:.4f} "
        f"edge_P={metrics['edge_precision']:.4f} "
        f"edge_R={metrics['edge_recall']:.4f} "
        f"edge_F1={metrics['edge_f1']:.4f} "
        f"Params={params:.2f}M"
    )

    if args.out_json:
        out = {
            "split": args.split,
            "subsets": args.subsets,
            "n_images": len(ds),
            "params_m": params,
            "metrics": metrics,
        }
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
