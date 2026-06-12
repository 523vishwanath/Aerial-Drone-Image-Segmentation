"""
config.py

Central configuration for the drone-image semantic segmentation project.
All dataclasses are frozen (immutable) to avoid accidental mutation during training.
"""

import os
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 12
    IMG_WIDTH: int = 512
    IMG_HEIGHT: int = 512

    # Root paths — override via CLI args / environment if your layout differs.
    IMAGE_DIR: str = "data/imgs/imgs"
    MASK_DIR: str = "data/masks/masks"
    TRAIN_CSV: str = "data/train.csv"
    TEST_CSV: str = "data/test.csv"


@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 32
    EPOCHS: int = 80
    LEARNING_RATE: float = 0.0004
    CHECKPOINT_DIR: str = os.path.join("model_checkpoint", "DeepLabv3_CamVid_Dice_loss")
    NUM_WORKERS: int = os.cpu_count() or 4
    VAL_SPLIT: float = 0.2
    SEED: int = 41


@dataclass
class ModelConfig:
    MODEL_NAME: str = "deeplabv3_resnet101"
    USE_PRETRAINED: bool = True
    AUX_LOSS: Union[bool, None] = True
    AUX_WEIGHT: float = 0.5


@dataclass(frozen=True)
class InferenceConfig:
    BATCH_SIZE: int = 4
    NUM_BATCHES: int = 2


# ---------------------------------------------------------------------------
# Class <-> Color mapping (drone semantic segmentation classes)
# ---------------------------------------------------------------------------
ID2COLOR = {
    0: (0, 0, 0),          # Background
    1: (181, 117, 117),    # Person
    2: (21, 116, 0),       # Bike
    3: (237, 90, 237),     # Car
    4: (114, 11, 8),       # Drone
    5: (12, 2, 118),       # Boat
    6: (181, 18, 119),     # Animal
    7: (179, 18, 12),      # Obstacle
    8: (181, 118, 0),      # Construction
    9: (10, 56, 0),        # Vegetation
    10: (115, 117, 0),     # Road
    11: (24, 116, 117),    # Sky
}

CLASS_NAMES = [
    "Background", "Person", "Bike", "Car", "Drone", "Boat",
    "Animal", "Obstacle", "Construction", "Vegetation", "Road", "Sky",
]

REV_ID2COLOR = {value: key for key, value in ID2COLOR.items()}
