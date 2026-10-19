"""
GSCNNRebar -- Table 2 baseline adapter.

Table name: GSCNN (WideResNet-38)

Uses the official nv-tlabs/GSCNN implementation unchanged.
Official repo: baselines_clean/official/GSCNN/repo/  (dirty=0, commit edd16c5d)
License: CC BY-NC-SA 4.0

Architecture preserved (unchanged from repo/network/gscnn.py):
  Backbone:   wider_resnet38_a2 (classes=1000, dilation=True)  -- random init
              mod1-mod7, pool2/pool3 extracted from DataParallel wrapper
  Shape stream: GatedSpatialConv2d gates (gate1/gate2/gate3) on res1/res2/res3 blocks
  Canny:      cv2.Canny applied per image on raw uint8 input (non-differentiable; preserved)
  ASPP:       _AtrousSpatialPyramidPoolingModule(4096, 256, output_stride=8)
              takes (m7, acts) where acts fuses edge_out + Canny
  Decoder:    bot_fine (m2 skip) + bot_aspp + final_seg (conv 256+48 -> num_classes)
  pretrained: random init (repo provides no pretrained backbone weights)

Adaptations (interface only):
  1. num_classes=1  (binary rebar segmentation)
  2. Device compatibility: official forward hardcodes .cuda(); adapter's forward
     replicates the full forward body with canny.to(device) instead -- this is
     documented as a device-compatibility adaptation, not an architecture change.
     All nn layers (mod1-mod7, gates, ASPP, etc.) are inherited unchanged from GSCNN.
  3. forward() always returns (edge_logit, seg_logit) regardless of train/eval mode,
     bypassing the criterion branch.  Criterion is never instantiated.
  4. Output: (edge_logit, seg_logit) both (B,1,H,W) at input resolution.

Official repo files: NOT modified.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).parent.parent / "repo"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# config must be imported before network.gscnn so that cfg.MODEL.BNFUNC is set.
# Default value (torch.nn.BatchNorm2d) is assigned at module level in config.py
# without calling assert_and_infer_cfg -- safe for our use.
from config import cfg  # noqa: F401, E402 (side-effect: sets cfg.MODEL.BNFUNC)
from network.gscnn import GSCNN  # noqa: E402


class GSCNNRebar(nn.Module):
    """
    GSCNN adapter for binary rebar segmentation.

    Inherits all layers from GSCNN(num_classes=1, criterion=None).
    Overrides forward() to:
      a) replace .cuda() with .to(device) for CPU/multi-GPU compatibility
      b) always return (edge_logit, seg_logit) -- never calls criterion
    All mod1-mod7, gates, ASPP, and decoder layers are inherited unchanged.
    """

    def __init__(self, img_size=(480, 640), pretrained=True):
        super().__init__()
        # criterion=None: the criterion attribute exists on GSCNN but is never
        # invoked because our forward override bypasses that branch entirely.
        self._gscnn = GSCNN(num_classes=1, criterion=None)
        self.img_size = img_size
        # pretrained arg kept for API uniformity; wider_resnet38_a2 has no
        # pretrained weights in the official repo (random init is correct).

    def forward(self, x):
        """
        Replicates repo/network/gscnn.py GSCNN.forward() with two changes only:
          1. canny tensor moved to x.device instead of .cuda()
          2. always returns (edge_logit, seg_logit) -- no criterion branch

        Every line that computes features is copied verbatim from the official
        forward and delegates to self._gscnn.<layer>.
        Device compatibility adaptation: lines marked [DCA].
        """
        g = self._gscnn  # shorthand; all layers live on g
        x_size = x.size()

        # --- Backbone (WideResNet-38) ---
        m1 = g.mod1(x)
        m2 = g.mod2(g.pool2(m1))
        m3 = g.mod3(g.pool3(m2))
        m4 = g.mod4(m3)
        m5 = g.mod5(m4)
        m6 = g.mod6(m5)
        m7 = g.mod7(m6)

        # --- Deep supervision signals for shape stream ---
        s3 = F.interpolate(g.dsn3(m3), x_size[2:], mode='bilinear', align_corners=True)
        s4 = F.interpolate(g.dsn4(m4), x_size[2:], mode='bilinear', align_corners=True)
        s7 = F.interpolate(g.dsn7(m7), x_size[2:], mode='bilinear', align_corners=True)

        m1f = F.interpolate(m1, x_size[2:], mode='bilinear', align_corners=True)

        # --- Canny (cpu numpy round-trip, preserved as-is) ---
        # [DCA] original: torch.from_numpy(canny).cuda().float()
        #       adapted:  torch.from_numpy(canny).to(x.device).float()
        # RebarSplit provides ImageNet-normalized tensors; Canny needs RGB bytes.
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        rgb = ((x.detach()*std+mean).clamp(0, 1)*255).round()
        im_arr = rgb.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).to(x.device).float()  # [DCA]

        # --- Shape stream (GatedSpatialConv2d gates) ---
        cs = g.res1(m1f)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        cs = g.d1(cs)
        cs = g.gate1(cs, s3)
        cs = g.res2(cs)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        cs = g.d2(cs)
        cs = g.gate2(cs, s4)
        cs = g.res3(cs)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        cs = g.d3(cs)
        cs = g.gate3(cs, s7)
        cs = g.fuse(cs)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = g.sigmoid(cs)

        cat  = torch.cat((edge_out, canny), dim=1)
        acts = g.sigmoid(g.cw(cat))

        # --- ASPP + decoder ---
        x_aspp  = g.aspp(m7, acts)
        dec0_up = g.bot_aspp(x_aspp)

        dec0_fine = g.bot_fine(m2)
        dec0_up   = g.interpolate(dec0_up, m2.size()[2:], mode='bilinear', align_corners=True)
        dec0      = torch.cat([dec0_fine, dec0_up], dim=1)
        dec1      = g.final_seg(dec0)

        seg_out = g.interpolate(dec1, x_size[2:], mode='bilinear')

        # Return convention: (edge_logit, seg_logit)
        # edge_out is already in [0,1] (post-sigmoid); we return raw logits for
        # loss compatibility.  Convert back: logit = log(p / (1-p)).
        # However forward_and_loss in train.py applies sigmoid internally via
        # edge_loss which expects raw logits.  Use inverse sigmoid (logit).
        edge_logit = torch.log(edge_out.clamp(1e-6, 1 - 1e-6) /
                               (1 - edge_out.clamp(1e-6, 1 - 1e-6)))

        return edge_logit, seg_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GSCNNRebar(img_size=(480, 640), pretrained=False).to(device)
    model.train()
    dummy  = torch.zeros(1, 3, 480, 640, device=device)
    with torch.no_grad():
        e, s = model(dummy)
    assert e.shape == (1, 1, 480, 640), f"edge shape wrong: {e.shape}"
    assert s.shape == (1, 1, 480, 640), f"seg  shape wrong: {s.shape}"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[gscnn] SMOKE TEST PASSED  edge={tuple(e.shape)}  seg={tuple(s.shape)}  params={params:.2f}M")
