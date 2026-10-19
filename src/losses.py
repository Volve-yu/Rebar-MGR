"""
Loss functions for the multi-task rebar detection system (edge + segmentation).

L_total = L_edge + L_seg + lambda_c * L_consistency

L_edge       : weighted BCE + Dice for binary edge detection
               (class-balanced weighting because edges are sparse ~2-5% pixels)
L_seg        : BCE + Dice for binary segmentation
L_consistency: coupling regulariser -- encourages edge predictions to align
               with segmentation boundaries and suppresses intra-region edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Edge detection loss
# ---------------------------------------------------------------------------

def edge_loss(logit, target, pos_weight_factor=10.0):
    """
    Weighted BCE + Dice for sparse binary edge maps.

    logit  : (B, 1, H, W) raw logit
    target : (B, 1, H, W) float 0/1  (edge pixels ~2-5% of image)
    pos_weight_factor: upweight positive (edge) pixels to counter class imbalance
    """
    # Adaptive positive weight: ratio of neg/pos pixels per batch
    pos = target.sum().clamp(min=1.0)
    neg = (1.0 - target).sum().clamp(min=1.0)
    pw = torch.tensor([neg / pos * pos_weight_factor], device=logit.device)
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)

    prob = torch.sigmoid(logit)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()


# ---------------------------------------------------------------------------
# Segmentation loss
# ---------------------------------------------------------------------------

def bce_dice_loss(logit, target, smooth=1.0):
    """
    logit : (B, 1, H, W) raw logit
    target: (B, 1, H, W) float 0/1
    """
    bce = F.binary_cross_entropy_with_logits(logit, target)
    prob = torch.sigmoid(logit)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)
    return bce + dice.mean()


# ---------------------------------------------------------------------------
# Consistency loss (edge-seg coupling, no extra annotations needed)
# ---------------------------------------------------------------------------

def heatmap_focal_loss(pred_logit, gt_heatmap, alpha=2, beta=4):
    """
    CenterNet-style penalty-reduced focal loss for soft heatmap targets.

    pred_logit  : (B, 1, H, W) raw logit
    gt_heatmap  : (B, 1, H, W) soft target in [0, 1], peaks = 1.0
    alpha, beta : focal loss exponents (CenterNet defaults: 2, 4)

    For peak pixels (gt == 1):   -(1-p)^alpha * log(p)
    For off-peak pixels:         -(1-gt)^beta * p^alpha * log(1-p)
    Normalised by number of peak pixels (prevents zero-label images dominating).
    """
    pred = torch.sigmoid(pred_logit).clamp(1e-6, 1.0 - 1e-6)

    peak_mask = (gt_heatmap == 1.0).float()
    off_mask  = 1.0 - peak_mask

    pos_loss  = -peak_mask * (1.0 - pred).pow(alpha) * torch.log(pred)
    neg_loss  = -off_mask  * (1.0 - gt_heatmap).pow(beta) * pred.pow(alpha) * torch.log(1.0 - pred)

    num_peaks = peak_mask.sum().clamp(min=1.0)
    return (pos_loss + neg_loss).sum() / num_peaks


def consistency_loss(edge_logit, seg_logit):
    """
    Boundary alignment term: edge predictions should be strong where the
    segmentation prediction has high-gradient boundaries.

    L_align = -mean(edge_prob * seg_boundary_strength)

    Interior suppression is handled architecturally by AsymmetricCouplingBlock
    (feature-level seg->edge gating) and is not duplicated here.
    """
    edge_prob = torch.sigmoid(edge_logit)   # (B, 1, H, W)
    seg_prob  = torch.sigmoid(seg_logit)    # (B, 1, H, W)

    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                       dtype=seg_prob.dtype, device=seg_prob.device).view(1, 1, 3, 3)
    ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                       dtype=seg_prob.dtype, device=seg_prob.device).view(1, 1, 3, 3)
    gx = F.conv2d(seg_prob.detach(), kx, padding=1)
    gy = F.conv2d(seg_prob.detach(), ky, padding=1)
    seg_boundary = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    return -(edge_prob * seg_boundary).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    def __init__(self, lambda_c=0.1, edge_pos_weight=10.0, lambda_h=1.0):
        """
        lambda_c        : weight for consistency loss
        edge_pos_weight : positive-class weight multiplier for edge BCE
        lambda_h        : weight for center heatmap focal loss
        """
        super().__init__()
        self.lambda_c = lambda_c
        self.edge_pos_weight = edge_pos_weight
        self.lambda_h = lambda_h

    def forward(self, edge_logit, seg_logit, edge_gt, seg_gt,
                center_logit=None, center_gt=None):
        losses = {}

        losses["edge"] = edge_loss(edge_logit, edge_gt, self.edge_pos_weight)
        losses["seg"]  = bce_dice_loss(seg_logit, seg_gt)
        losses["consistency"] = self.lambda_c * consistency_loss(
            edge_logit, seg_logit)

        # Center heatmap loss: only on images that have GT labels (non-zero heatmap)
        if center_logit is not None and center_gt is not None:
            has_labels = (center_gt.sum(dim=(1, 2, 3)) > 0)   # (B,) bool
            if has_labels.any():
                losses["center"] = self.lambda_h * heatmap_focal_loss(
                    center_logit[has_labels], center_gt[has_labels])

        losses["total"] = sum(losses.values())
        return losses
