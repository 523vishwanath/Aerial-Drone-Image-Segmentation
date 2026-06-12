"""
visualization.py

Helper functions for converting between class-id masks and RGB visualisations,
overlaying predictions on images, and denormalising image tensors for display.
"""

import cv2
import numpy as np
import torch

from src.config import ID2COLOR, REV_ID2COLOR, DatasetConfig


def rgb_to_grayscale(rgb_arr, color_map=REV_ID2COLOR, num_classes=DatasetConfig.NUM_CLASSES):
    """Collapses an (H, W, 3) RGB mask into an (H, W) class-id mask using `color_map`."""
    reshaped = rgb_arr.reshape((-1, 3))
    unique_pixels, inverse = np.unique(reshaped, axis=0, return_inverse=True)
    grayscale_map = np.array([color_map.get(tuple(p), 0) for p in unique_pixels])[inverse]
    return grayscale_map.reshape(rgb_arr.shape[:2])


def num_to_rgb(num_arr, color_map=ID2COLOR):
    """Converts an (H, W) class-id mask into an (H, W, 3) float RGB image in [0, 1]."""
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))

    for class_id, color in color_map.items():
        output[single_layer == class_id] = color

    return np.float32(output) / 255.0


def image_overlay(image, segmented_image):
    """Blends a colour segmentation map on top of an RGB image.

    Args:
        image: (H, W, 3) float RGB image in [0, 1]
        segmented_image: (H, W, 3) float RGB segmentation map in [0, 1]

    Returns:
        (H, W, 3) float RGB image in [0, 1]
    """
    alpha, beta, gamma = 1.0, 0.7, 0.0

    segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    blended = cv2.addWeighted(image_bgr, alpha, segmented_bgr, beta, gamma, image_bgr)
    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return np.clip(blended, 0.0, 1.0)


def denormalize(tensors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Reverses ImageNet normalisation on a batch of image tensors (in-place)."""
    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, min=0.0, max=1.0)
