"""
LoadRebarAnnotations -- pipeline for binary rebar seg + edge GT.

Replaces LoadAnnotations for RebarDataset_boundary.

seg GT  (segmentation/{id}.png): 0=background, 255=rebar  -> stored as 0/1 int64 (H,W)
edge GT (edge/{id}.png):         0=background, 255=edge   -> stored as 0/1 float32 (H,W,1)

gt_semantic_sebound shape (H,W,1) is compatible with EncoderDecoderRefine.forward_train,
which calls gt_semantic_sebound.permute(0,3,1,2) on a batched (B,H,W,C=1) tensor.

np.unpackbits is NOT used: our edge PNGs are direct binary images (0/255),
not bit-packed multi-label Cityscapes boundary maps.
"""

import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadRebarAnnotations(object):
    """Load binary seg and edge annotations for rebar data.

    Expected on-disk format:
      seg  : uint8 grayscale PNG with values 0 (background) or 255 (rebar)
      edge : uint8 grayscale PNG with values 0 (background) or 255 (edge)

    Outputs stored in results:
      gt_semantic_seg    : np.int64 (H, W)   values in {0, 1}
      gt_semantic_sebound: np.float32 (H, W, 1) values in {0.0, 1.0}

    Args:
        file_client_args (dict): Args for mmcv.FileClient. Default: disk.
        imdecode_backend (str): Default: 'cv2'.
    """

    def __init__(
        self,
        file_client_args=dict(backend='disk'),
        imdecode_backend='cv2',
    ):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def _load_binary_png(self, path: str) -> np.ndarray:
        """Load a 0/255 grayscale PNG and return a 0/1 uint8 array (H, W)."""
        img_bytes = self.file_client.get(path)
        arr = mmcv.imfrombytes(
            img_bytes, flag='unchanged', backend=self.imdecode_backend
        ).squeeze()
        # handle both 0/255 and already 0/1 encodings robustly
        return (arr > 0).astype(np.uint8)

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        ann_info = results['ann_info']

        # --- segmentation GT ---
        seg_prefix = results.get('seg_prefix', None)
        seg_path = (
            osp.join(seg_prefix, ann_info['seg_map'])
            if seg_prefix
            else ann_info['seg_map']
        )
        gt_seg = self._load_binary_png(seg_path).astype(np.int64)  # (H, W)
        results['gt_semantic_seg'] = gt_seg
        results['seg_fields'].append('gt_semantic_seg')

        # --- edge GT ---
        if 'sebound_map' in ann_info:
            sebound_prefix = results.get('sebound_prefix', None)
            edge_path = (
                osp.join(sebound_prefix, ann_info['sebound_map'])
                if sebound_prefix
                else ann_info['sebound_map']
            )
            gt_edge = self._load_binary_png(edge_path)  # (H, W) uint8 0/1
            # shape (H, W, 1): DefaultFormatBundle will NOT touch this key
            # (only gt_semantic_seg gets [None,...] expand); EncoderDecoderRefine
            # expects (B, H, W, C) after DataLoader stacks the batch dimension.
            gt_edge = gt_edge[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
            results['gt_semantic_sebound'] = gt_edge
            results['seg_fields'].append('gt_semantic_sebound')

        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(imdecode_backend={self.imdecode_backend!r})'
        )


@PIPELINES.register_module()
class RebarPad(object):
    """Pad RGB image, binary segmentation, and binary edge maps.

    Mobile-Seed's original Pad assumes ``gt_semantic_sebound`` remains
    ``(H, W, C)`` after resize.  Some mmcv versions squeeze a one-channel
    edge map to ``(H, W)``.  This Rebar-specific pad restores the channel
    dimension for edge maps and otherwise mirrors the original behavior.
    """

    def __init__(
        self,
        size=None,
        size_divisor=None,
        pad_val=0,
        seg_pad_val=0,
        sebound_pad_val=0,
    ):
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.sebound_pad_val = sebound_pad_val

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'],
                shape=self.size,
                pad_val=self.pad_val,
                padding_mode='reflect',
            )
        else:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val
            )
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            if key == 'gt_semantic_sebound':
                edge = results[key]
                if edge.ndim == 2:
                    edge = edge[:, :, np.newaxis]
                result_temp = np.zeros(
                    (*results['pad_shape'][:2], edge.shape[-1]),
                    dtype=edge.dtype,
                )
                for i in range(edge.shape[-1]):
                    result_temp[..., i] = mmcv.impad(
                        edge[..., i],
                        shape=results['pad_shape'][:2],
                        pad_val=self.sebound_pad_val,
                        padding_mode='reflect',
                    )
                results[key] = result_temp
            else:
                results[key] = mmcv.impad(
                    results[key],
                    shape=results['pad_shape'][:2],
                    pad_val=self.seg_pad_val,
                    padding_mode='reflect',
                )

    def __call__(self, results):
        self._pad_img(results)
        self._pad_seg(results)
        return results
