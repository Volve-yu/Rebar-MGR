"""
eval_seg_predictions.py -- evaluate segmentation predictions against GT.

Reads exported prediction PNGs and computes binary segmentation metrics per
image and as a dataset average.  Works with any external baseline whose output
has been exported to 0/255 grayscale PNGs.

Usage examples:
  python src/eval_seg_predictions.py \\
      --pred_dir predictions_official/mobile_seed/site-1/seg \\
      --data_root data --subset site-1 --split val

  python src/eval_seg_predictions.py \\
      --pred_dir predictions/deeplab/site-1/seg \\
      --data_root data --subset site-1 --split val

Arguments:
  --pred_dir   Directory containing {id}.png prediction files
  --data_root  Project data root (contains site-1/, site-2/)
  --subset     Dataset subset name (e.g. site-1, site-2)
  --split      Split name matching gt_sets/{split}.txt  (default: val)
  --gt_subdir  Ground-truth subdirectory under the subset (default: segmentation)
  --threshold  Foreground threshold for predictions (default: 0.5 * 255 = 127)
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    Both inputs are bool (H, W).
    Returns a dict of macro per-image metrics.
    """
    eps = 1e-6
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()

    fg_iou = (tp + eps) / (tp + fp + fn + eps)
    bg_iou = (tn + eps) / (tn + fp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return {
        "seg_iou": float(fg_iou),
        "seg_miou": float(0.5 * (fg_iou + bg_iou)),
        "seg_f1": float(f1),
        "seg_precision": float(precision),
        "seg_recall": float(recall),
        "seg_acc": float(acc),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate segmentation predictions against GT.'
    )
    p.add_argument('--pred_dir',  required=True,
                   help='Directory with {id}.png prediction files')
    p.add_argument('--data_root', default='data',
                   help='Project data root (default: data)')
    p.add_argument('--subset',    required=True,
                   help='Subset name, e.g. site-1 or site-2')
    p.add_argument('--split',     default='val',
                   help='Split name (default: val)')
    p.add_argument('--gt_subdir', default='segmentation',
                   help='GT subdirectory under the subset (default: segmentation)')
    p.add_argument('--metric_prefix', default='seg',
                   help='Metric prefix printed in the summary (default: seg)')
    p.add_argument('--threshold', type=int, default=127,
                   help='Pixel threshold for foreground (default: 127)')
    p.add_argument('--verbose',   action='store_true',
                   help='Print per-image metrics')
    return p.parse_args()


def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    data_root = Path(args.data_root)
    subset_dir = data_root / args.subset
    gt_seg_dir = subset_dir / args.gt_subdir
    split_file = subset_dir / 'gt_sets' / f'{args.split}.txt'

    if not split_file.exists():
        raise FileNotFoundError(f'Split file not found: {split_file}')
    if not pred_dir.exists():
        raise FileNotFoundError(f'Prediction directory not found: {pred_dir}')

    ids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    print(f'Subset: {args.subset}  Split: {args.split}  IDs: {len(ids)}')
    print(f'GT dir  : {gt_seg_dir}')
    print(f'Pred dir: {pred_dir}')
    print()

    metric_rows = []
    missing_pred = 0
    missing_gt   = 0

    for sid in ids:
        gt_path   = gt_seg_dir / f'{sid}.png'
        pred_path = pred_dir   / f'{sid}.png'

        if not gt_path.exists():
            missing_gt += 1
            continue

        gt_arr = np.array(Image.open(gt_path).convert('L'))
        if not pred_path.exists():
            missing_pred += 1
            pred_arr = np.zeros_like(gt_arr)
            if args.verbose:
                print(f'  [WARN] missing prediction: {sid}')
        else:
            pred_arr = np.array(Image.open(pred_path).convert('L'))

        # Resize prediction to GT size if they differ
        if gt_arr.shape != pred_arr.shape:
            pred_img  = Image.open(pred_path).convert('L').resize(
                (gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST
            )
            pred_arr = np.array(pred_img)

        gt_bin   = gt_arr   > 0               # 0/255 or 0/1 GT -> bool
        pred_bin = pred_arr > args.threshold  # threshold predictions -> bool

        metrics = compute_binary_metrics(pred_bin, gt_bin)
        metric_rows.append(metrics)

        if args.verbose:
            print(
                f"  {sid:30s}  IoU={metrics['seg_iou']:.4f}  "
                f"mIoU={metrics['seg_miou']:.4f}  F1={metrics['seg_f1']:.4f}  "
                f"P={metrics['seg_precision']:.4f}  R={metrics['seg_recall']:.4f}"
            )

    print('-' * 50)
    if missing_gt:
        print(f'[WARN] {missing_gt} GT file(s) missing -- skipped')
    if missing_pred:
        print(f'[WARN] {missing_pred} prediction file(s) missing -- treated as zero recall')

    n = len(metric_rows)
    if n == 0:
        print('No valid predictions found.')
        return

    means = {
        k: float(np.mean([row[k] for row in metric_rows]))
        for k in metric_rows[0]
    }
    print(f'N evaluated : {n}')
    prefix = args.metric_prefix
    print(f"{prefix}_IoU     : {means['seg_iou']:.4f}")
    print(f"{prefix}_mIoU    : {means['seg_miou']:.4f}")
    print(f"{prefix}_F1      : {means['seg_f1']:.4f}")
    print(f"{prefix}_P       : {means['seg_precision']:.4f}")
    print(f"{prefix}_R       : {means['seg_recall']:.4f}")
    print(f"{prefix}_Acc     : {means['seg_acc']:.4f}")


if __name__ == '__main__':
    main()
