"""
Asymmetric Bidirectional Coupling Decoder for joint edge detection + segmentation.

Architecture:
  Shared encoder: MobileNetV3-Large (lightweight, Colab/AutoDL-friendly)
                  swap for Swin-L / ViT-L when GPU memory allows
  Edge decoder:   FPN-style upsampling -> binary edge map
  Seg decoder:    FPN-style upsampling -> binary segmentation mask
  Coupling:       AsymmetricCouplingBlock at each decoder scale

  Direction 1 (edge -> seg):
    Sobel gradient of edge features -> boundary attention -> refine seg features.
    Physical meaning: strong edge responses mark rebar boundaries; segmentation
    should be sharper at those locations.

  Direction 2 (seg -> edge):
    Coarse seg soft-mask -> gate edge features to suppress false edges inside
    homogeneous foreground/background regions.
    Physical meaning: within a solid rebar region, spurious texture edges should
    be suppressed; only true structural boundaries matter.

Difference from Mobile-Seed (IEEE RAL 2024):
  Mobile-Seed uses a generic ADTI module for robotics scenes.
  Here the two directions are physically grounded in the rebar inspection task
  and tied to the joint energy function E(D,S) in idea.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision.models as _tv_models
except ImportError:
    raise ImportError("torchvision is required: pip install torchvision")


# ---------------------------------------------------------------------------
# Sobel gradient operator (depthwise, works on feature maps)
# ---------------------------------------------------------------------------
class SobelGradient(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.channels = channels

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.ky, padding=1, groups=self.channels)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


# ---------------------------------------------------------------------------
# Asymmetric coupling block (one decoder scale)
# ---------------------------------------------------------------------------
class AsymmetricCouplingBlock(nn.Module):
    """
    F_e : edge decoder features   (B, C, H, W)
    F_s : seg  decoder features   (B, C, H, W)

    Direction 1 (edge -> seg):
      gradient of F_e -> boundary attention A_e -> F_s_out = F_s + F_s * A_e
      Adds spatial emphasis at locations where edge features have high gradient
      magnitude, sharpening segmentation boundaries.

    Direction 2 (seg -> edge):
      coarse seg soft-mask M -> gate F_e to suppress intra-region false edges
      F_e_out = F_e * (1 - alpha * M_interior)
      where M_interior = M * (1 - boundary(M)) suppresses the interior of
      foreground blobs, leaving only boundary-adjacent features active.
    """

    def __init__(self, channels):
        super().__init__()
        # Direction 1: edge -> seg
        self.sobel = SobelGradient(channels)
        self.boundary_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
        # Direction 2: seg -> edge
        # Coarse seg head: single-channel soft mask
        self.seg_head_coarse = nn.Conv2d(channels, 1, 1)
        # Learned suppression weight per channel
        self.suppression = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def _interior_mask(self, soft_mask):
        """
        Erode soft_mask slightly to get the interior (away from boundaries).
        Uses average pooling as a differentiable approximation of erosion.
        soft_mask: (B, 1, H, W) in [0, 1]
        """
        interior = F.avg_pool2d(soft_mask, kernel_size=7, stride=1, padding=3)
        # Pixels where the neighbourhood is mostly foreground = interior
        return (interior > 0.6).float().detach()

    def forward(self, F_e, F_s):
        # --- Direction 1: edge -> seg ---
        grad = self.sobel(F_e)
        A_e = self.boundary_attn(grad)          # (B, C, H, W) in [0, 1]
        F_s_out = F_s + F_s * A_e

        # --- Direction 2: seg -> edge ---
        soft_mask = torch.sigmoid(self.seg_head_coarse(F_s.detach()))  # (B,1,H,W)
        interior = self._interior_mask(soft_mask)                       # (B,1,H,W)
        # Suppress edge features in the interior of foreground regions
        suppress_w = self.suppression(F_e)      # (B, C, H, W) in [0, 1]
        F_e_out = F_e * (1.0 - suppress_w * interior)

        return F_e_out, F_s_out


# ---------------------------------------------------------------------------
# Lightweight FPN decoder block
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class RebarMTL(nn.Module):
    """
    Multi-task model: shared MobileNetV3 encoder +
    asymmetric bidirectional coupling decoder ->
    edge map logit (1 ch) + segmentation logit (1 ch, binary).

    Both outputs are raw logits; apply sigmoid for probabilities.
    """

    def __init__(self, img_size=(480, 640), pretrained=True):
        super().__init__()
        H, W = img_size

        # --- Encoder ---
        # MobileNet_V3_Large_Weights was added in torchvision 0.13;
        # fall back to the deprecated pretrained= flag for older installs.
        try:
            _Weights = _tv_models.MobileNet_V3_Large_Weights
            _w = _Weights.IMAGENET1K_V1 if pretrained else None
            backbone = _tv_models.mobilenet_v3_large(weights=_w)
        except AttributeError:
            backbone = _tv_models.mobilenet_v3_large(pretrained=pretrained)
        features = backbone.features

        self.enc0 = features[:2]    # stride 2  -> 1/2
        self.enc1 = features[2:4]   # stride 4  -> 1/4
        self.enc2 = features[4:7]   # stride 8  -> 1/8
        self.enc3 = features[7:13]  # stride 16 -> 1/16
        self.enc4 = features[13:]   # stride 32 -> 1/32

        # Channel sizes for MobileNetV3-Large
        c0, c1, c2, c3, c4 = 16, 24, 40, 112, 960

        # Lateral 1x1 projections to uniform 128 channels
        proj_ch = 128
        self.lat0 = nn.Conv2d(c0, proj_ch, 1)
        self.lat1 = nn.Conv2d(c1, proj_ch, 1)
        self.lat2 = nn.Conv2d(c2, proj_ch, 1)
        self.lat3 = nn.Conv2d(c3, proj_ch, 1)
        self.lat4 = nn.Conv2d(c4, proj_ch, 1)

        # --- Edge decoder ---
        self.e_dec3 = DecoderBlock(proj_ch, proj_ch, proj_ch)
        self.e_dec2 = DecoderBlock(proj_ch, proj_ch, proj_ch)
        self.e_dec1 = DecoderBlock(proj_ch, proj_ch, 64)
        self.e_dec0 = DecoderBlock(64, proj_ch, 32)
        self.e_head = nn.Conv2d(32, 1, 3, padding=1)  # logit, BCE loss

        # --- Seg decoder ---
        self.s_dec3 = DecoderBlock(proj_ch, proj_ch, proj_ch)
        self.s_dec2 = DecoderBlock(proj_ch, proj_ch, proj_ch)
        self.s_dec1 = DecoderBlock(proj_ch, proj_ch, 64)
        self.s_dec0 = DecoderBlock(64, proj_ch, 32)
        self.s_head = nn.Conv2d(32, 1, 3, padding=1)  # logit, BCE loss

        # --- Coupling blocks (channels must match decoder output at each scale) ---
        self.couple3 = AsymmetricCouplingBlock(proj_ch)  # 1/16: 128 ch
        self.couple2 = AsymmetricCouplingBlock(proj_ch)  # 1/8:  128 ch
        self.couple1 = AsymmetricCouplingBlock(64)        # 1/4:  64 ch
        self.couple0 = AsymmetricCouplingBlock(32)        # 1/2:  32 ch

        self.img_size = img_size

    def forward(self, x):
        # --- Encode ---
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Lateral projections
        p0 = self.lat0(e0)
        p1 = self.lat1(e1)
        p2 = self.lat2(e2)
        p3 = self.lat3(e3)
        p4 = self.lat4(e4)

        # --- Decode scale 1/16 ---
        Fe = self.e_dec3(p4, p3)
        Fs = self.s_dec3(p4, p3)
        Fe, Fs = self.couple3(Fe, Fs)

        # --- Decode scale 1/8 ---
        Fe = self.e_dec2(Fe, p2)
        Fs = self.s_dec2(Fs, p2)
        Fe, Fs = self.couple2(Fe, Fs)

        # --- Decode scale 1/4 ---
        Fe = self.e_dec1(Fe, p1)
        Fs = self.s_dec1(Fs, p1)
        Fe, Fs = self.couple1(Fe, Fs)

        # --- Decode scale 1/2 ---
        Fe = self.e_dec0(Fe, p0)
        Fs = self.s_dec0(Fs, p0)
        Fe, Fs = self.couple0(Fe, Fs)

        # --- Heads ---
        edge_logit = self.e_head(Fe)
        seg_logit  = self.s_head(Fs)

        # Upsample to input resolution
        edge_logit = F.interpolate(edge_logit, size=self.img_size,
                                   mode="bilinear", align_corners=False)
        seg_logit  = F.interpolate(seg_logit,  size=self.img_size,
                                   mode="bilinear", align_corners=False)

        return edge_logit, seg_logit
