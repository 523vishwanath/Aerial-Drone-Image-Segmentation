"""
visualize_predictions.py

Loads a trained checkpoint and saves side-by-side comparison images
(original | ground-truth mask | predicted mask | overlay) for the validation
set, and (original | predicted mask | overlay) for the test set.

These are the images used in the project README's "Results" section.

Usage:
    python -m scripts.visualize_predictions \
        --checkpoint model_checkpoint/DeepLabv3_CamVid_Dice_loss/version_0/ckpt.tar \
        --split valid \
        --num_batches 2 \
        --output_dir assets/predictions
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch

from src.config import DatasetConfig, TrainingConfig, ModelConfig, InferenceConfig, ID2COLOR
from src.dataset import get_dataloaders
from src.model import prepare_model
from src.visualization import denormalize, num_to_rgb, image_overlay
from scripts.train import get_default_device


@torch.inference_mode()
def visualize_validation(model, loader, device, output_dir, num_batches=2):
    """Saves (image | GT mask | predicted mask | overlay) figures for the validation set."""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for batch_idx, (batch_img, batch_mask) in enumerate(loader):
        if batch_idx == num_batches:
            break

        pred_all = model(batch_img.to(device))["out"].cpu().argmax(dim=1).numpy()
        display_imgs = denormalize(batch_img.cpu()).permute(0, 2, 3, 1).numpy()

        for i in range(len(display_imgs)):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(display_imgs[i])
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            gt_rgb = num_to_rgb(batch_mask[i].numpy(), color_map=ID2COLOR)
            axes[1].imshow(gt_rgb)
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis("off")

            pred_rgb = num_to_rgb(pred_all[i], color_map=ID2COLOR)
            axes[2].imshow(pred_rgb)
            axes[2].set_title("Predicted Mask")
            axes[2].axis("off")

            overlay = image_overlay(display_imgs[i], pred_rgb)
            axes[3].imshow(overlay)
            axes[3].set_title("Predicted Overlay")
            axes[3].axis("off")

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"valid_sample_{saved}.png")
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")
            saved += 1


@torch.inference_mode()
def visualize_test(model, loader, device, output_dir, num_batches=2):
    """Saves (image | predicted mask | overlay) figures for the test set (no GT available)."""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for batch_idx, batch_img in enumerate(loader):
        if batch_idx == num_batches:
            break

        pred_all = model(batch_img.to(device))["out"].cpu().argmax(dim=1).numpy()
        display_imgs = denormalize(batch_img.cpu()).permute(0, 2, 3, 1).numpy()

        for i in range(len(display_imgs)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(display_imgs[i])
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            pred_rgb = num_to_rgb(pred_all[i], color_map=ID2COLOR)
            axes[1].imshow(pred_rgb)
            axes[1].set_title("Predicted Mask")
            axes[1].axis("off")

            overlay = image_overlay(display_imgs[i], pred_rgb)
            axes[2].imshow(overlay)
            axes[2].set_title("Predicted Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"test_sample_{saved}.png")
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")
            saved += 1


def main(args):
    device, _ = get_default_device()

    dataset_cfg = DatasetConfig(
        IMAGE_DIR=args.image_dir, MASK_DIR=args.mask_dir,
        TRAIN_CSV=args.train_csv, TEST_CSV=args.test_csv,
    )
    training_cfg = TrainingConfig()

    model = prepare_model(
        model_name=ModelConfig.MODEL_NAME,
        use_pretrained=ModelConfig.USE_PRETRAINED,
        num_classes=dataset_cfg.NUM_CLASSES,
    )
    _ = model(torch.randn(2, 3, dataset_cfg.IMG_HEIGHT, dataset_cfg.IMG_WIDTH))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.to(device)
    model.eval()

    train_loader, valid_loader, test_loader, _ = get_dataloaders(
        dataset_cfg, training_cfg,
        batch_size=InferenceConfig.BATCH_SIZE, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    if args.split in ("valid", "both"):
        visualize_validation(model, valid_loader, device, args.output_dir, args.num_batches)
    if args.split in ("test", "both"):
        visualize_test(model, test_loader, device, args.output_dir, args.num_batches)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate qualitative prediction figures")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", default=DatasetConfig.IMAGE_DIR)
    parser.add_argument("--mask_dir", default=DatasetConfig.MASK_DIR)
    parser.add_argument("--train_csv", default=DatasetConfig.TRAIN_CSV)
    parser.add_argument("--test_csv", default=DatasetConfig.TEST_CSV)
    parser.add_argument("--split", choices=["valid", "test", "both"], default="both")
    parser.add_argument("--num_batches", type=int, default=InferenceConfig.NUM_BATCHES)
    parser.add_argument("--output_dir", default="assets/predictions")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
