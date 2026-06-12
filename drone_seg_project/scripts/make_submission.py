"""
make_submission.py

Runs the trained model over the test set and writes a Kaggle-format submission
CSV using run-length encoding (RLE), one row per (image, class) pair.

Usage:
    python -m scripts.make_submission \
        --checkpoint model_checkpoint/DeepLabv3_CamVid_Dice_loss/version_0/ckpt.tar \
        --output submission.csv
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import DatasetConfig, TrainingConfig, ModelConfig, InferenceConfig
from src.dataset import get_dataloaders
from src.model import prepare_model
from scripts.train import get_default_device


def rle_encode(mask):
    """Run-length encodes a binary (H, W) mask in column-major order, matching
    the encoding expected by the competition's submission format."""
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


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

    _, _, test_loader, test_df = get_dataloaders(
        dataset_cfg, training_cfg,
        batch_size=InferenceConfig.BATCH_SIZE, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    image_ids = test_df["ImageID"].values
    image_counter = 0
    submission = []

    with torch.no_grad():
        for batch_img in tqdm(test_loader, desc="Generating submission"):
            batch_img = batch_img.to(device)
            output = model(batch_img)["out"]
            pred_all = output.argmax(dim=1).cpu().numpy()

            for i in range(pred_all.shape[0]):
                pred_mask = pred_all[i]
                image_id = image_ids[image_counter]

                original_image_path = os.path.join(dataset_cfg.IMAGE_DIR, f"{image_id}.jpg")
                original_img = cv2.imread(original_image_path)

                if original_img is None or original_img.size == 0:
                    print(f"Warning: could not load {original_image_path}, skipping.")
                    image_counter += 1
                    continue

                orig_h, orig_w = original_img.shape[:2]
                resized_pred = cv2.resize(
                    pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST,
                )

                for class_id in range(dataset_cfg.NUM_CLASSES):
                    class_mask = (resized_pred == class_id).astype(np.uint8)
                    rle = np.nan if class_mask.sum() == 0 else rle_encode(class_mask)

                    submission.append({"ImageID": f"{image_id}_{class_id}", "EncodedPixels": rle})

                image_counter += 1

    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(args.output, index=False)

    print(f"Submission written to {args.output} ({len(submission_df)} rows)")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission CSV (RLE-encoded)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", default=DatasetConfig.IMAGE_DIR)
    parser.add_argument("--mask_dir", default=DatasetConfig.MASK_DIR)
    parser.add_argument("--train_csv", default=DatasetConfig.TRAIN_CSV)
    parser.add_argument("--test_csv", default=DatasetConfig.TEST_CSV)
    parser.add_argument("--output", default="submission.csv")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
