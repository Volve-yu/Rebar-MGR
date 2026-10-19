"""
RebarDataset: loads RGB image + edge map + binary segmentation mask.

Directory layout expected:
  data/
    site-1/
      images/       {id}.png   RGB
      edge/         {id}.png   binary edge map (0/255), derived from seg masks
      segmentation/ {id}.png   binary 0/255
      gt_sets/      train.txt, val.txt  (one image id per line, no extension)
    site-2/
      ...

Edge images are loaded as binary float tensors in [0, 1].
If the edge file is missing for a sample, a zero tensor is returned so the
rest of the pipeline still runs (edge loss will be zero for that sample).
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image


class RebarSplit(Dataset):
    def __init__(self, root: str, split: str, img_size=(480, 640)):
        """
        root    : path to one subset folder, e.g. data/site-1
        split   : 'train' or 'val'
        img_size: (H, W) to resize everything to
        """
        self.root = Path(root)
        self.img_size = img_size  # (H, W)

        split_file = self.root / "gt_sets" / f"{split}.txt"
        with open(split_file) as f:
            raw_ids = [line.strip() for line in f if line.strip()]
        img_dir = self.root / "images"
        self.ids = [sid for sid in raw_ids if (img_dir / f"{sid}.png").exists()]

        self.edge_path   = self.root / "edge"
        self.label_path  = self.root / "rebar_labels"

        self.img_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _make_center_heatmap(self, sid: str, orig_w: int) -> torch.Tensor:
        """
        Build a (1, H, W) soft heatmap from rebar x-coordinates.
        Each rebar gets a 2-D Gaussian: narrow in x (sigma~10px at 640w),
        tall in y (sigma=H, nearly flat vertically).
        Multiple rebars are combined with element-wise max.
        Returns zeros when rebar_labels file is absent.
        """
        H, W = self.img_size
        label_file = self.label_path / f"{sid}.txt"
        heatmap = np.zeros((H, W), dtype=np.float32)

        if not label_file.exists():
            return torch.from_numpy(heatmap).unsqueeze(0)

        xs_orig = []
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    xs_orig.append(float(line))
                except ValueError:
                    pass
        if not xs_orig:
            return torch.from_numpy(heatmap).unsqueeze(0)

        scale   = W / orig_w
        sigma_x = max(8.0, W / 64.0)   # ~10 px at 640-wide
        sigma_y = float(H)              # very tall: nearly uniform along y

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cy = H / 2.0
        for cx_orig in xs_orig:
            cx = cx_orig * scale
            g  = np.exp(-0.5 * ((xx - cx) ** 2 / sigma_x ** 2
                               + (yy - cy) ** 2 / sigma_y ** 2))
            heatmap = np.maximum(heatmap, g)

        return torch.from_numpy(heatmap).unsqueeze(0)   # (1, H, W)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        # RGB
        img = Image.open(self.root / "images" / f"{sid}.png").convert("RGB")
        orig_w = img.width
        img_t = self.img_tf(img)  # (3, H, W)

        # Segmentation: 0/255 binary -> 0/1 float
        seg = Image.open(self.root / "segmentation" / f"{sid}.png").convert("L")
        seg = seg.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        seg_t = torch.from_numpy(np.array(seg) / 255.0).float().unsqueeze(0)

        # Edge: 0/255 binary -> 0/1 float
        edge_file = self.edge_path / f"{sid}.png"
        if edge_file.exists():
            edge = Image.open(edge_file).convert("L")
            edge = edge.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
            edge_t = torch.from_numpy(
                (np.array(edge) > 127).astype(np.float32)
            ).unsqueeze(0)
        else:
            edge_t = torch.zeros(1, *self.img_size)

        # Center heatmap: Gaussian blobs at rebar x-positions (zero if no labels)
        center_t = self._make_center_heatmap(sid, orig_w)   # (1, H, W)

        return {
            "image":  img_t,      # (3, H, W)
            "edge":   edge_t,     # (1, H, W)  0/1 binary
            "seg":    seg_t,      # (1, H, W)  0/1 binary
            "center": center_t,   # (1, H, W)  soft [0,1] heatmap
            "id":     sid,
        }


def build_dataset(data_root: str, split: str, img_size=(480, 640),
                  subsets=("site-1", "site-2")):
    """Concatenate multiple subset splits into one dataset."""
    datasets = []
    for sub in subsets:
        path = os.path.join(data_root, sub)
        if os.path.isdir(path):
            datasets.append(RebarSplit(path, split, img_size))
    if not datasets:
        raise FileNotFoundError(f"No valid subsets found under {data_root}")
    return ConcatDataset(datasets)
