"""
SCTransNetRebar -- external segmentation baseline adapter.

Table name: SCTransNet

Uses the official SCTransNet implementation under:
  baselines_clean/official/SCTransNet/

SCTransNet was originally designed for infrared small-target detection and its
official code predicts a single binary mask.  This adapter keeps the network
body and only changes the task interface:
  1. input channels are set to 3 so the model can consume our RGB rebar images;
  2. the final sigmoid probability map is converted back to a logit, matching
     the BCE+Dice loss and prediction utilities used by baselines_clean;
  3. only segmentation is exported, because SCTransNet has no edge branch.

Official repository files are not modified.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).parent.parent


def _load_symbol(module_path: Path, module_name: str, symbol_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, symbol_name)


get_SCTrans_config = _load_symbol(
    _REPO / "model" / "Config.py",
    "_official_sctrans_config",
    "get_SCTrans_config",
)
SCTransNet = _load_symbol(
    _REPO / "model" / "SCTransNet.py",
    "_official_sctrans_model",
    "SCTransNet",
)


class SCTransNetRebar(nn.Module):
    """
    Binary rebar segmentation wrapper for SCTransNet.

    The official model returns sigmoid probabilities.  Our shared training and
    evaluation code expects raw logits, so forward() applies an inverse sigmoid.
    """

    def __init__(self, img_size=(480, 640), pretrained=True):
        super().__init__()
        cfg = get_SCTrans_config()
        self.model = SCTransNet(
            cfg,
            n_channels=3,
            n_classes=1,
            img_size=img_size[0],
            mode="train",
            deepsuper=True,
        )
        self.img_size = img_size
        # pretrained is kept for the common baseline API.  The official repo
        # does not provide a local ImageNet-style pretraining path here.

    def forward(self, x):
        self.model.mode = 'train' if self.training else 'test'
        prob = self.model(x)
        if isinstance(prob, tuple):
            return tuple(torch.logit(p.clamp(1e-6, 1.0-1e-6)) for p in prob)
        prob = prob.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(prob / (1.0 - prob))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCTransNetRebar(img_size=(480, 640), pretrained=False).to(device)
    dummy = torch.zeros(1, 3, 480, 640, device=device)
    with torch.no_grad():
        s = model(dummy)
    assert s.shape == (1, 1, 480, 640), f"seg shape wrong: {s.shape}"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[sctransnet] SMOKE TEST PASSED  seg={tuple(s.shape)}  params={params:.2f}M")
