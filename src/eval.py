"""
Evaluation metrics for edge detection and binary segmentation.
"""

import torch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_seg_iou = 0.0
    total_seg_miou = 0.0
    total_seg_f1 = 0.0
    total_seg_precision = 0.0
    total_seg_recall = 0.0
    total_seg_acc = 0.0
    total_edge_precision = 0.0
    total_edge_recall = 0.0
    total_edge_f1 = 0.0
    n = 0
    eps = 1e-6

    for batch in loader:
        img     = batch["image"].to(device)
        edge_gt = batch["edge"].to(device)
        seg_gt  = batch["seg"].to(device)

        edge_logit, seg_logit = model(img)[:2]
        seg_pred  = (torch.sigmoid(seg_logit)  > 0.5).float()
        edge_pred = (torch.sigmoid(edge_logit) > 0.5).float()

        B = img.shape[0]
        n += B

        for b in range(B):
            # Segmentation IoU and F1
            sp = seg_pred[b, 0]
            sg = seg_gt[b, 0]
            inter = (sp * sg).sum()
            union = sp.sum() + sg.sum() - inter
            total_seg_iou += ((inter + eps) / (union + eps)).item()

            tp = inter
            fp = sp.sum() - inter
            fn = sg.sum() - inter
            tn = ((1.0 - sp) * (1.0 - sg)).sum()
            bg_union = (1.0 - sp).sum() + (1.0 - sg).sum() - tn
            bg_iou = (tn + eps) / (bg_union + eps)

            total_seg_miou += 0.5 * (((inter + eps) / (union + eps)) + bg_iou).item()
            total_seg_f1 += ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item()
            total_seg_precision += ((tp + eps) / (tp + fp + eps)).item()
            total_seg_recall += ((tp + eps) / (tp + fn + eps)).item()
            total_seg_acc += ((tp + tn + eps) / (tp + tn + fp + fn + eps)).item()

            # Edge F1
            ep = edge_pred[b, 0]
            eg = edge_gt[b, 0]
            tp_e = (ep * eg).sum()
            fp_e = ep.sum() - tp_e
            fn_e = eg.sum() - tp_e
            total_edge_precision += ((tp_e + eps) / (tp_e + fp_e + eps)).item()
            total_edge_recall += ((tp_e + eps) / (tp_e + fn_e + eps)).item()
            total_edge_f1 += ((2 * tp_e + eps) / (2 * tp_e + fp_e + fn_e + eps)).item()

    return {
        "seg_iou": total_seg_iou / n,
        "seg_miou": total_seg_miou / n,
        "seg_f1": total_seg_f1 / n,
        "seg_precision": total_seg_precision / n,
        "seg_recall": total_seg_recall / n,
        "seg_acc": total_seg_acc / n,
        "edge_precision": total_edge_precision / n,
        "edge_recall": total_edge_recall / n,
        "edge_f1": total_edge_f1 / n,
    }
