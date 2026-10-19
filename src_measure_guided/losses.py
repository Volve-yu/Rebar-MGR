"""
Measurement-guided losses for the rebar multi-task model.

This module leaves the original src/ losses untouched. It adds a differentiable
geometry surrogate for downstream measurement:

  - column projection of the predicted segmentation mask
  - column projection of the GT segmentation mask
  - optional centerline heatmap projection from rebar_labels

The loss does not directly optimise non-differentiable count/spacing metrics.
Instead it aligns the horizontal rebar distribution that those metrics depend on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_loss(logit, target, pos_weight_factor=10.0):
    """Weighted BCE + Dice for sparse binary edge maps."""
    pos = target.sum().clamp(min=1.0)
    neg = (1.0 - target).sum().clamp(min=1.0)
    pw = torch.tensor([neg / pos * pos_weight_factor], device=logit.device)
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)

    prob = torch.sigmoid(logit)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()


def bce_dice_loss(logit, target, smooth=1.0):
    """Binary segmentation BCE + Dice."""
    bce = F.binary_cross_entropy_with_logits(logit, target)
    prob = torch.sigmoid(logit)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)
    return bce + dice.mean()


def consistency_loss(edge_logit, seg_logit):
    """Encourage edge predictions to align with segmentation boundaries."""
    edge_prob = torch.sigmoid(edge_logit)
    seg_prob = torch.sigmoid(seg_logit)

    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                      dtype=seg_prob.dtype, device=seg_prob.device).view(1, 1, 3, 3)
    ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                      dtype=seg_prob.dtype, device=seg_prob.device).view(1, 1, 3, 3)
    gx = F.conv2d(seg_prob.detach(), kx, padding=1)
    gy = F.conv2d(seg_prob.detach(), ky, padding=1)
    seg_boundary = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    return -(edge_prob * seg_boundary).mean()


def _normalise_columns(col, eps=1e-6):
    """Normalise each sample's 1-D column signal to [0, 1]-like scale."""
    max_v = col.amax(dim=-1, keepdim=True).clamp(min=eps)
    return col / max_v


def projection_geometry_loss(seg_logit, seg_gt, center_gt=None):
    """
    Differentiable measurement surrogate.

    seg_logit : (B, 1, H, W)
    seg_gt    : (B, 1, H, W)
    center_gt : optional (B, 1, H, W) Gaussian x-location heatmap from labels

    Returns an L1 loss on horizontal rebar distributions. This encourages the
    predicted mask to preserve bar count/position cues that drive spacing
    measurement, without putting non-differentiable post-processing in training.
    """
    pred_col = torch.sigmoid(seg_logit).mean(dim=2)  # (B, 1, W)
    gt_col = seg_gt.float().mean(dim=2)              # (B, 1, W)

    pred_col_n = _normalise_columns(pred_col)
    gt_col_n = _normalise_columns(gt_col)
    loss = F.l1_loss(pred_col_n, gt_col_n)

    if center_gt is not None:
        has_center = center_gt.sum(dim=(1, 2, 3)) > 0
        if has_center.any():
            center_col = center_gt.float().amax(dim=2)  # (B, 1, W)
            center_col_n = _normalise_columns(center_col)
            loss = loss + F.l1_loss(
                pred_col_n[has_center],
                center_col_n[has_center],
            )

    return loss


class MeasurementGuidedLoss(nn.Module):
    """
    Total loss for measurement-guided training.

    L = seg + alpha_edge * edge + lambda_c * consistency
        + beta_proj * projection_geometry

    Optional max_*_to_seg ratios cap auxiliary losses relative to the primary
    segmentation loss. This keeps edge/projection supervision as structural
    guidance and reduces negative transfer on small validation splits.
    """

    def __init__(
        self,
        lambda_c=0.0,
        edge_pos_weight=10.0,
        alpha_edge=0.3,
        beta_proj=0.1,
        max_edge_to_seg=0.0,
        max_proj_to_seg=0.0,
    ):
        super().__init__()
        self.lambda_c = lambda_c
        self.edge_pos_weight = edge_pos_weight
        self.alpha_edge = alpha_edge
        self.beta_proj = beta_proj
        self.max_edge_to_seg = max_edge_to_seg
        self.max_proj_to_seg = max_proj_to_seg
        self.edge_schedule_scale = 1.0
        self.proj_schedule_scale = 1.0

    def set_aux_scales(self, edge_scale=1.0, proj_scale=1.0):
        self.edge_schedule_scale = float(edge_scale)
        self.proj_schedule_scale = float(proj_scale)

    @staticmethod
    def _cap_to_seg(aux_loss, seg_loss, max_ratio, eps=1e-6):
        if max_ratio <= 0:
            return aux_loss, aux_loss.new_tensor(1.0)
        budget = max_ratio * seg_loss.detach()
        scale = (budget / (aux_loss.detach() + eps)).clamp(max=1.0)
        return aux_loss * scale, scale.detach()

    def forward(self, edge_logit, seg_logit, edge_gt, seg_gt, center_gt=None):
        seg = bce_dice_loss(seg_logit, seg_gt)
        raw_edge = edge_loss(edge_logit, edge_gt, self.edge_pos_weight)
        edge, edge_scale = self._cap_to_seg(
            self.alpha_edge * self.edge_schedule_scale * raw_edge,
            seg,
            self.max_edge_to_seg,
        )
        raw_projection = projection_geometry_loss(seg_logit, seg_gt, center_gt)
        projection, projection_scale = self._cap_to_seg(
            self.beta_proj * self.proj_schedule_scale * raw_projection,
            seg,
            self.max_proj_to_seg,
        )

        losses = {
            "edge": edge,
            "edge_raw": raw_edge.detach(),
            "edge_scale": edge_scale,
            "edge_schedule_scale": edge.new_tensor(self.edge_schedule_scale),
            "seg": seg,
            "projection": projection,
            "projection_raw": raw_projection.detach(),
            "projection_scale": projection_scale,
            "projection_schedule_scale": projection.new_tensor(
                self.proj_schedule_scale
            ),
        }

        if self.lambda_c > 0:
            losses["consistency"] = self.lambda_c * consistency_loss(
                edge_logit, seg_logit
            )
        else:
            losses["consistency"] = seg_logit.new_tensor(0.0)

        losses["total"] = (
            losses["edge"]
            + losses["seg"]
            + losses["projection"]
            + losses["consistency"]
        )
        return losses
