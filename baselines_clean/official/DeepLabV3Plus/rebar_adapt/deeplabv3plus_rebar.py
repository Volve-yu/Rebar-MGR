"""
DeepLabV3PlusRebar -- Table 2 baseline.

Table name: DeepLabV3+ (ResNet-101)

Uses the official VainF/DeepLabV3Plus-Pytorch implementation unchanged:
  _load_model('deeplabv3plus', 'resnet101', num_classes=1,
              output_stride=8, pretrained_backbone=True)
This is the canonical DeepLabV3+ configuration from the repo's README.

Adaptation to binary rebar segmentation (only changes allowed):
  - num_classes=1 instead of 21
  - pretrained_backbone=True when pretrained=True, False otherwise
  - forward() wraps the model output; _SimpleSegmentationModel already
    upsamples to input resolution internally, so no extra resize needed.

Official repo files: NOT modified.
Backbone family: ResNet-101 (preserved from official repo default).
Core architecture: DeepLabHeadV3Plus + ASPP (preserved unchanged).

Shape trace for (B, 3, 480, 640), output_stride=8:
  ResNet-101 layer1 (low_level): (B, 256, 120, 160)   stride 4
  ResNet-101 layer4 dilated (out): (B, 2048, 60, 80)  stride 8
  ASPP -> (B, 256, 60, 80) -> upsample -> (B, 256, 120, 160)
  project(low_level): (B, 48, 120, 160)
  concat: (B, 304, 120, 160) -> classifier -> (B, 1, 120, 160)
  internal interpolate to input -> (B, 1, 480, 640)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Insert repo/ into sys.path so relative imports inside modeling.py resolve.
# No file under repo/ is modified.
_REPO = Path(__file__).parent.parent / "repo"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from network.modeling import _load_model  # noqa: E402


class DeepLabV3PlusRebar(nn.Module):
    """
    DeepLabV3+ ResNet-101 adapted for binary rebar segmentation.
    Seg-only baseline; returns seg_logit (B, 1, H, W) at input resolution.
    _SimpleSegmentationModel.forward already interpolates to input size.
    """

    def __init__(self, img_size=(480, 640), pretrained=True):
        super().__init__()
        self.model = _load_model(
            arch_type="deeplabv3plus",
            backbone="resnet101",
            num_classes=1,
            output_stride=8,
            pretrained_backbone=pretrained,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DeepLabV3PlusRebar(img_size=(480, 640), pretrained=False).to(device)
    dummy  = torch.zeros(1, 3, 480, 640, device=device)
    with torch.no_grad():
        s = model(dummy)
    assert s.shape == (1, 1, 480, 640), f"seg shape wrong: {s.shape}"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[deeplab] SMOKE TEST PASSED  seg={tuple(s.shape)}  params={params:.2f}M")
