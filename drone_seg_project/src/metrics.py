"""
metrics.py

Segmentation evaluation metrics: mean IoU, per-class IoU, and two flavours of
Dice coefficient.

`kaggle_dice_per_image` deliberately mirrors the metric used by the Kaggle
leaderboard for this competition (a pixel-agreement-based Dice score, computed
per image and then averaged) — it is what produced the 0.57 leaderboard score
reported in this project's README.
"""

import torch
import torch.nn.functional as F


def mean_iou(predictions, ground_truths, num_classes=2, dims=(1, 2)):
    """Classwise mean IoU, averaged over classes present in each image and then
    over the batch.

    Args:
        predictions: predicted class-id mask, shape [B, H, W]
        ground_truths: ground-truth class-id mask, shape [B, H, W]

    Returns:
        scalar tensor: batch-mean IoU
    """
    ground_truths = F.one_hot(ground_truths, num_classes=num_classes)
    predictions = F.one_hot(predictions, num_classes=num_classes)

    intersection = (predictions * ground_truths).sum(dim=dims)
    summation = predictions.sum(dim=dims) + ground_truths.sum(dim=dims)
    union = summation - intersection

    iou = intersection / union
    iou = torch.nan_to_num(iou, nan=0.0)

    num_classes_present = torch.count_nonzero(summation, dim=1)
    iou = iou.sum(dim=1) / num_classes_present

    return iou.mean()


def iou_per_class(predictions, ground_truths, num_classes):
    """Per-class IoU, shape [B, num_classes]. Used to track which classes the
    model struggles with across training."""
    ground_truths = F.one_hot(ground_truths, num_classes=num_classes)
    predictions = F.one_hot(predictions, num_classes=num_classes)

    intersection = (predictions & ground_truths).sum(dim=(1, 2))
    union = (predictions | ground_truths).sum(dim=(1, 2))

    return intersection.float() / (union.float() + 1e-8)


def dice_coef(predictions, ground_truths, num_classes=2, dims=(1, 2)):
    """Standard multi-class Dice coefficient (all classes included, including
    background), averaged over classes and batch. Used as a training-time
    metric (not the leaderboard metric)."""
    ground_truths = F.one_hot(ground_truths, num_classes=num_classes)
    predictions = F.one_hot(predictions, num_classes=num_classes)

    intersection = (predictions * ground_truths).sum(dim=dims)
    summation = predictions.sum(dim=dims) + ground_truths.sum(dim=dims)

    dice_score = 2 * intersection / summation
    dice_score = torch.nan_to_num(dice_score, nan=1.0)

    return dice_score.mean()


def kaggle_dice_per_image(pred_mask, true_mask, smooth=1e-8):
    """Per-image Dice score matching the competition's leaderboard metric.

    Args:
        pred_mask: [H, W] integer class-id mask (numpy array)
        true_mask: [H, W] integer class-id mask (numpy array)

    Returns:
        float: 2 * (#agreeing pixels) / (total pixels in pred + total pixels in target)
    """
    pred_flat = pred_mask.reshape(-1)
    true_flat = true_mask.reshape(-1)

    intersection = (pred_flat == true_flat).sum()

    if (true_flat.sum() == 0) and (pred_flat.sum() == 0):
        return 1.0

    return (2.0 * intersection + smooth) / (pred_flat.size + true_flat.size + smooth)
