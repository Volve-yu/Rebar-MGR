"""
export_rebar.py -- export Mobile-Seed predictions for rebar evaluation.

Runs inference on the val split using the official EncoderDecoderRefine model
and saves binary segmentation PNGs per subset.

Output layout:
  {out_dir}/mobile_seed/{subset}/seg/{id}.png   -- 0/255 binary PNG
  {out_dir}/mobile_seed/{subset}/edge/{id}.png  -- 0/255 binary PNG (optional)

NOTE on inference path:
  We call model.whole_inference(img_tensor, img_meta, rescale=True) directly,
  bypassing simple_test / encode_decode.  This is necessary because:
    - simple_test calls seg_logit.argmax(dim=1), which always returns 0 for a
      single-channel (num_classes=1) output.
    - whole_inference returns (seg_logit, bound_logit) as raw tensors before
      argmax, allowing sigmoid thresholding for binary segmentation.
  This is confirmed from encoder_decoder_refine.py whole_inference source.

Usage (run from inside repo/):
  python tools/export_rebar.py \\
      configs/rebar/MS_tiny_rebar.py \\
      work_dirs/ms_tiny_rebar/best.pth \\
      --out-dir ../../../../predictions_official \\
      --data-root data_ms \\
      --split val \\
      --device cuda:0
"""

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import load_checkpoint

# Ensure repo mmseg is importable when running from repo/
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mmseg.models import build_segmentor
from mmseg.datasets.pipelines import Compose


# ---------------------------------------------------------------------------
# Minimal single-image pipeline for inference (mirrors inference_segmentor
# but without requiring img_meta['CLASSES'] in checkpoint meta)
# ---------------------------------------------------------------------------

class _LoadImage:
    def __call__(self, results):
        img = mmcv.imread(results['img'])
        results['filename'] = results.get('filename', None)
        results['ori_filename'] = results.get('ori_filename', None)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def _build_model(config_path, checkpoint_path, device):
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu',
                    revise_keys=[(r'^module\.', '')])
    # Set CLASSES manually; checkpoint meta may not have it for custom datasets
    model.CLASSES = ('background', 'rebar')
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model, cfg


def _run_inference(model, img_path, device):
    """
    Run whole_inference on a single image file.

    Returns:
        seg_logit  : torch.Tensor (1, 1, H_orig, W_orig)
        bound_logit: torch.Tensor (1, 1, H_orig, W_orig)
    """
    cfg = model.cfg
    # Build test pipeline: replace LoadImage step with our minimal loader
    test_pipeline = [_LoadImage()] + list(cfg.data.test.pipeline[1:])
    test_pipeline = Compose(test_pipeline)

    data = dict(img=img_path)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        data = scatter(data, [device.index if device.type == 'cuda' else 0])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    img = data['img']
    img_meta = data['img_metas']

    if isinstance(img, DataContainer):
        img = img.data
    if isinstance(img_meta, DataContainer):
        img_meta = img_meta.data

    # Collated test data is usually [[tensor]] / [[meta]] after scatter.
    while isinstance(img, (list, tuple)):
        img = img[0]
    while isinstance(img_meta, (list, tuple)) and len(img_meta) == 1 and isinstance(img_meta[0], (list, tuple)):
        img_meta = img_meta[0]

    with torch.no_grad():
        seg_logit, bound_logit = model.whole_inference(img, img_meta, rescale=True)

    return seg_logit, bound_logit


def _logit_to_binary_png(logit_tensor):
    """
    Convert a (1, 1, H, W) logit tensor to a 0/255 uint8 numpy array (H, W).
    """
    prob = torch.sigmoid(logit_tensor[0, 0])   # (H, W)
    binary = (prob > 0.5).cpu().numpy().astype(np.uint8) * 255
    return binary


def _find_project_root(data_root):
    """
    Find the gangjing project root from repo/data_ms or an absolute data_ms path.

    Expected project layout:
      <project>/data/site-1/images/
      <project>/data/site-2/images/
      <project>/baselines_clean/official/Mobile-Seed/repo/data_ms/
    """
    start = Path(data_root).resolve()
    for p in [start] + list(start.parents):
        if (p / 'data' / 'site-1' / 'images').exists():
            return p
    raise FileNotFoundError(
        'Could not find project root containing data/site-1/images. '
        f'Started from: {start}'
    )


def parse_args():
    p = argparse.ArgumentParser(
        description='Export Mobile-Seed predictions for rebar evaluation.'
    )
    p.add_argument('config',     help='Path to configs/rebar/MS_tiny_rebar.py')
    p.add_argument('checkpoint', help='Path to trained checkpoint (.pth)')
    p.add_argument('--out-dir',  default='../../../../predictions_official',
                   help='Root output directory (default: ../../../../predictions_official)')
    p.add_argument('--data-root', default='data_ms',
                   help='data_ms root relative to repo/ (default: data_ms)')
    p.add_argument('--split',    default='val',
                   choices=['train', 'val'],
                   help='Which split to export (default: val)')
    p.add_argument('--device',   default='cuda:0',
                   help='Inference device (default: cuda:0)')
    p.add_argument('--save-edge', action='store_true',
                   help='Also save boundary/edge predictions (optional)')
    p.add_argument('--subset', required=True, choices=['site-1', 'site-2'])
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}')

    # Build model
    config_path = osp.abspath(args.config)
    checkpoint_path = osp.abspath(args.checkpoint)
    model, cfg = _build_model(config_path, checkpoint_path, device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model loaded. Params: {params_m:.2f}M')

    # Resolve data paths
    data_root = osp.abspath(args.data_root)
    split_file = osp.join(data_root, 'splits', f'{args.split}.txt')
    if not osp.exists(split_file):
        raise FileNotFoundError(f'Split file not found: {split_file}')

    # Read sample IDs and their source subsets.
    # prepare_rebar_data.py merges site-1 and site-2 IDs into a single split;
    # we need to know which subset each ID came from to route output correctly.
    # Strategy: for each id, check which of the original subset image dirs it
    # belongs to by looking at the project data/ directories.
    ids = [l.strip() for l in open(split_file) if l.strip()]
    print(f'Loaded {len(ids)} IDs from {args.split} split.')

    # Map id -> subset by checking the original project data/ tree.
    project_root = _find_project_root(data_root)
    subsets = [args.subset]
    id_to_subset = {}
    for sid in ids:
        found = False
        for ss in subsets:
            candidate = project_root / 'data' / ss / 'images' / f'{sid}.png'
            if candidate.exists():
                id_to_subset[sid] = ss
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f'Could not map id {sid} to site-1 or site-2 under '
                f'{project_root / "data"}'
            )

    out_root = Path(osp.abspath(args.out_dir)) / 'mobile_seed'

    total = len(ids)
    for i, sid in enumerate(ids):
        subset = id_to_subset[sid]
        img_path = osp.join(data_root, 'images', f'{sid}.png')
        if not osp.exists(img_path):
            raise FileNotFoundError(img_path)

        seg_logit, bound_logit = _run_inference(model, img_path, device)

        # Save segmentation
        seg_arr = _logit_to_binary_png(seg_logit)
        seg_out = out_root / subset / 'seg' / f'{sid}.png'
        seg_out.parent.mkdir(parents=True, exist_ok=True)
        mmcv.imwrite(seg_arr, str(seg_out))

        # Save edge (optional)
        if args.save_edge:
            edge_arr = _logit_to_binary_png(bound_logit)
            edge_out = out_root / subset / 'edge' / f'{sid}.png'
            edge_out.parent.mkdir(parents=True, exist_ok=True)
            mmcv.imwrite(edge_arr, str(edge_out))

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f'  [{i+1}/{total}] {subset}/{sid}.png')

    print(f'\nDone. Predictions saved to: {out_root}')
    if args.save_edge:
        print(f'  seg  : {out_root}/{{subset}}/seg/{{id}}.png')
        print(f'  edge : {out_root}/{{subset}}/edge/{{id}}.png')
    else:
        print(f'  seg  : {out_root}/{{subset}}/seg/{{id}}.png')


if __name__ == '__main__':
    main()
