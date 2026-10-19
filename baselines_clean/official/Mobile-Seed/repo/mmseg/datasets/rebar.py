"""
RebarDataset_boundary -- Mobile-Seed dataset adapter for binary rebar data.

Directory layout expected under data_root:
  images/{id}.png         -- RGB input images
  segmentation/{id}.png   -- binary seg GT (0=bg, 255=rebar)
  edge/{id}.png           -- binary edge GT (0=bg, 255=edge)
  splits/train.txt        -- one bare id per line (no suffix)
  splits/val.txt

Differences from CityscapesDataset_boundary:
  - ann_dir  = 'segmentation'  (seg GT)
  - sebound_dir = 'edge'       (edge GT, separate directory)
  - img_suffix = seg_map_suffix = sebound_map_suffix = '.png'
  - pre_pipeline sets 'sebound_prefix' so LoadRebarAnnotations can
    find edge files without ann_dir pointing at them
  - split file format: bare id per line (no suffix), matching prepare_rebar_data.py output

Official model files (AFFormer, BoundaryHead, RefineHead, AFD) are NOT modified.
"""

import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RebarDataset_boundary(CustomDataset):
    """Binary rebar segmentation + edge dataset for Mobile-Seed.

    Args:
        sebound_dir (str): Directory containing edge GT PNGs, relative to
            data_root. Default: 'edge'.
        All other args forwarded to CustomDataset.
    """

    CLASSES = ('background', 'rebar')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(
        self,
        sebound_dir='edge',
        img_suffix='.png',
        seg_map_suffix='.png',
        sebound_map_suffix='.png',
        **kwargs,
    ):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            sebound_map_suffix=sebound_map_suffix,
            **kwargs,
        )
        # Resolve sebound_dir against data_root (mirrors how CustomDataset
        # resolves ann_dir in __init__).
        data_root = kwargs.get('data_root', None)
        if data_root is not None and not osp.isabs(sebound_dir):
            self.sebound_dir = osp.join(data_root, sebound_dir)
        else:
            self.sebound_dir = sebound_dir
        from .pipelines.rebar_loading import LoadRebarAnnotations
        self.gt_seg_map_loader = LoadRebarAnnotations()

    def pre_pipeline(self, results):
        """Extend CustomDataset.pre_pipeline to supply sebound_prefix."""
        super().pre_pipeline(results)
        # LoadRebarAnnotations reads this to locate edge PNG files.
        results['sebound_prefix'] = self.sebound_dir
