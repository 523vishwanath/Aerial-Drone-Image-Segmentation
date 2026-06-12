"""
losses.py

Combined Dice + Cross-Entropy loss used for training the segmentation model.
"""

import torch.nn.functional as F


def dice_coef_loss(predictions, ground_truths, num_classes=2, dims=(1, 2), smooth=1e-8):
    """Combined Dice + Cross-Entropy loss.

    Args:
        predictions: model output logits, shape [B, num_classes, H, W]
        ground_truths: integer class-id mask, shape [B, H, W]
        dims: spatial dimensions over which intersection/union are summed
        smooth: smoothing constant to avoid division by zero

    The Dice term is computed on softmax probabilities, one-hot ground truth,
    and the **background class (index 0) is excluded** from the Dice average —
    this prevents the (usually large) background region from dominating the
    Dice score and lets the loss focus on the foreground classes that matter
    for this task. Cross-entropy is computed over all classes as usual.

    Returns:
        scalar tensor: (1 - mean_dice_over_foreground_classes) + cross_entropy
    """
    # [B, H, W] -> [B, H, W, num_classes]
    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)

    # [B, num_classes, H, W] -> [B, H, W, num_classes]
    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)

    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
    summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (summation + smooth)

    # Drop background class (index 0) before averaging.
    dice = dice[:, 1:]
    dice_mean = dice.mean()

    ce = F.cross_entropy(predictions, ground_truths)

    return (1.0 - dice_mean) + ce
