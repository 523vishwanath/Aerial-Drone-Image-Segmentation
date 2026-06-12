"""
dataset.py

Custom PyTorch Dataset for the drone semantic segmentation task, plus a factory
function that builds train/validation/test DataLoaders from the competition CSVs.
"""

import os

import cv2
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from src.config import DatasetConfig, TrainingConfig


class CustomSegDataset(Dataset):
    """Loads (image, mask) pairs for training/validation, or images only for the test set."""

    def __init__(self, *, df, image_size, num_classes, image_dir, mask_dir=None, is_train=False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size  # (width, height) for cv2.resize
        self.image_dir = image_dir
        self.mask_dir = mask_dir  # None for the test set (images only)
        self.num_classes = num_classes
        self.is_train = is_train

        self.transforms = self._build_transforms()

    def __len__(self):
        return len(self.df)

    def _build_transforms(self):
        transforms = []

        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ])

        # Normalisation uses ImageNet statistics, as required by the pretrained
        # torchvision DeepLabV3 backbone.
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),  # (H, W, C) -> (C, H, W)
        ])

        return A.Compose(transforms)

    def _load_file(self, file_path, interpolation=cv2.INTER_NEAREST, is_mask=False):
        if not is_mask:
            file = cv2.imread(file_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
        else:
            file = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        file = cv2.resize(file, self.image_size, interpolation=interpolation)
        return file

    def __getitem__(self, index):
        image_path = self.df.loc[index, "image_path"]
        image = self._load_file(image_path, interpolation=cv2.INTER_CUBIC, is_mask=False)

        if self.mask_dir is not None:  # Train / validation
            mask_path = self.df.loc[index, "mask_path"]
            mask = self._load_file(mask_path, interpolation=cv2.INTER_NEAREST, is_mask=True)

            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].to(__import__("torch").long)
            return image, mask

        # Test set: image only
        transformed = self.transforms(image=image)
        return transformed["image"]


def load_dataframes(cfg: DatasetConfig = DatasetConfig()):
    """Reads train/test CSVs and attaches resolved image/mask paths."""
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    test_df = pd.read_csv(cfg.TEST_CSV)

    train_df["image_path"] = train_df["ImageID"].apply(
        lambda x: os.path.join(cfg.IMAGE_DIR, f"{x}.jpg")
    )
    train_df["mask_path"] = train_df["ImageID"].apply(
        lambda x: os.path.join(cfg.MASK_DIR, f"{x}.png")
    )
    test_df["image_path"] = test_df["ImageID"].apply(
        lambda x: os.path.join(cfg.IMAGE_DIR, f"{x}.jpg")
    )

    return train_df, test_df


def get_dataloaders(
    dataset_cfg: DatasetConfig = DatasetConfig(),
    training_cfg: TrainingConfig = TrainingConfig(),
    batch_size=None,
    num_workers=None,
    pin_memory=False,
):
    """Builds train / validation / test DataLoaders.

    `batch_size` and `num_workers` default to the values in `training_cfg` but can
    be overridden (e.g. a smaller batch size for inference).
    """
    batch_size = batch_size or training_cfg.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else training_cfg.NUM_WORKERS

    train_df, test_df = load_dataframes(dataset_cfg)

    train_df, valid_df = train_test_split(
        train_df, test_size=training_cfg.VAL_SPLIT, random_state=training_cfg.SEED
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    image_size = (dataset_cfg.IMG_WIDTH, dataset_cfg.IMG_HEIGHT)  # cv2 expects (W, H)

    train_dataset = CustomSegDataset(
        df=train_df, image_dir=dataset_cfg.IMAGE_DIR, mask_dir=dataset_cfg.MASK_DIR,
        is_train=True, num_classes=dataset_cfg.NUM_CLASSES, image_size=image_size,
    )
    valid_dataset = CustomSegDataset(
        df=valid_df, image_dir=dataset_cfg.IMAGE_DIR, mask_dir=dataset_cfg.MASK_DIR,
        is_train=False, num_classes=dataset_cfg.NUM_CLASSES, image_size=image_size,
    )
    test_dataset = CustomSegDataset(
        df=test_df, image_dir=dataset_cfg.IMAGE_DIR, mask_dir=None,
        is_train=False, num_classes=dataset_cfg.NUM_CLASSES, image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=num_workers, drop_last=True, shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=num_workers, shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=num_workers, shuffle=False,
    )

    return train_loader, valid_loader, test_loader, test_df
