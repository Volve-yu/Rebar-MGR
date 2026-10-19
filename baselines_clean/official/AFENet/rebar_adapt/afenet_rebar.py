"""
AFENetRebar -- external segmentation baseline adapter.

Table name: AFENet

Uses the official AFENet implementation under:
  baselines_clean/official/AFENet/

AFENet is a remote-sensing semantic segmentation network.  The official code
uses a multi-class Lightning training pipeline; for a fair binary rebar
comparison we only adapt the model interface:
  1. num_classes is set to 1 for foreground/background rebar segmentation;
  2. pretrained=False is used by default to avoid hidden network downloads on
     AutoDL when timm weights are not cached;
  3. the model returns a single segmentation logit map at input resolution.

Official repository files are not modified.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).parent.parent
_OFFICIAL_ROOT = _REPO.parent
if str(_OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_ROOT))

from AFENet.model.AFENet import AFENet  # noqa: E402


class AFENetRebar(nn.Module):
    """
    Binary rebar segmentation wrapper for AFENet.

    The official model already returns raw segmentation logits.  We keep that
    convention so baselines_clean can train it with the same BCE+Dice objective
    used for other binary segmentation baselines.
    """

    def __init__(self, img_size=(480, 640), pretrained=True):
        super().__init__()
        self.model = AFENet(
            backbone_name="swsl_resnet18", pretrained=pretrained, num_classes=1,
        )
        self.img_size = img_size
        # Pin timm in the upload environment; never substitute another backbone.

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFENetRebar(img_size=(480, 640), pretrained=False).to(device)
    dummy = torch.zeros(1, 3, 480, 640, device=device)
    with torch.no_grad():
        s = model(dummy)
    assert s.shape == (1, 1, 480, 640), f"seg shape wrong: {s.shape}"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[afenet] SMOKE TEST PASSED  seg={tuple(s.shape)}  params={params:.2f}M")
