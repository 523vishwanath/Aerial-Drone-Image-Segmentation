"""
train.py

Trains DeepLabV3-ResNet101 on the drone semantic segmentation dataset using a
combined Dice + Cross-Entropy loss (plus an auxiliary cross-entropy term from
DeepLabV3's auxiliary classifier head).

Usage:
    python -m scripts.train \
        --image_dir data/imgs/imgs \
        --mask_dir data/masks/masks \
        --train_csv data/train.csv \
        --test_csv data/test.csv \
        --epochs 80 \
        --batch_size 32 \
        --lr 4e-4
"""

import argparse
import gc
import os

import numpy as np
import torch
from torch.cuda import amp
from torch.optim import Adam
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from src.config import DatasetConfig, TrainingConfig, ModelConfig
from src.dataset import get_dataloaders
from src.losses import dice_coef_loss
from src.metrics import mean_iou, dice_coef, iou_per_class
from src.model import prepare_model


def get_default_device():
    gpu_available = torch.cuda.is_available()
    return torch.device("cuda" if gpu_available else "cpu"), gpu_available


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_checkpoint_dir(checkpoint_dir):
    """Creates a new `version_N` subdirectory under `checkpoint_dir` so that
    previous training runs are never overwritten."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    try:
        num_versions = [int(d.split("_")[-1]) for d in os.listdir(checkpoint_dir) if "version" in d]
        version_num = max(num_versions) + 1
    except ValueError:
        version_num = 0

    version_dir = os.path.join(checkpoint_dir, f"version_{version_num}")
    os.makedirs(version_dir)

    print(f"Checkpoint directory: {version_dir}")
    return version_dir


def train_one_epoch(model, loader, optimizer, scaler, num_classes, device, epoch_idx, total_epochs):
    model.train()

    loss_record = MeanMetric()
    iou_record = MeanMetric()
    dice_record = MeanMetric()
    acc_record = MulticlassAccuracy(num_classes=num_classes, average="micro")

    with tqdm(total=len(loader), ncols=122) as tq:
        tq.set_description(f"Train :: Epoch: {epoch_idx}/{total_epochs}")

        for data, target in loader:
            tq.update(1)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            with amp.autocast():
                output_dict = model(data)
                cls_out = output_dict["out"]

                loss = dice_coef_loss(cls_out, target, num_classes=num_classes)

                if ModelConfig.AUX_LOSS:
                    aux_out = output_dict["aux"]
                    aux_loss = torch.nn.functional.cross_entropy(aux_out, target)
                    loss = loss + ModelConfig.AUX_WEIGHT * aux_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cls_out = cls_out.detach()
            pred_idx = cls_out.argmax(dim=1)

            batch_iou = mean_iou(pred_idx, target, num_classes=num_classes)
            batch_dice = dice_coef(pred_idx, target, num_classes=num_classes)

            acc_record.update(pred_idx.cpu(), target.cpu())
            loss_record.update(loss.detach().cpu(), weight=data.shape[0])
            iou_record.update(batch_iou.cpu(), weight=data.shape[0])
            dice_record.update(batch_dice.cpu(), weight=data.shape[0])

            tq.set_postfix_str(
                f"Loss: {loss_record.compute():.4f}, "
                f"IoU: {iou_record.compute():.4f}, "
                f"DiceCoef: {dice_record.compute():.4f}, "
                f"Acc: {acc_record.compute():.4f}"
            )

    return (
        loss_record.compute().item(),
        iou_record.compute().item(),
        dice_record.compute().item(),
        acc_record.compute().item(),
    )


def validate(model, loader, device, num_classes, epoch_idx, total_epochs):
    model.eval()

    loss_record = MeanMetric()
    acc_record = MulticlassAccuracy(num_classes=num_classes, average="micro")

    iou_record = torch.zeros(num_classes, device=device)
    dice_record = 0.0
    total_batches = 0

    with tqdm(total=len(loader), ncols=122) as tq:
        tq.set_description(f"Valid :: Epoch: {epoch_idx}/{total_epochs}")

        for data, target in loader:
            tq.update(1)
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output_dict = model(data)

            cls_out = output_dict["out"]
            loss = dice_coef_loss(cls_out, target, num_classes=num_classes)

            pred_idx = cls_out.argmax(dim=1)

            batch_iou = iou_per_class(pred_idx, target, num_classes)
            iou_record += batch_iou.mean(dim=0)

            pred_np = pred_idx.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            from src.metrics import kaggle_dice_per_image
            batch_dice_scores = [
                kaggle_dice_per_image(pred_np[i], target_np[i]) for i in range(pred_np.shape[0])
            ]
            dice_record += torch.tensor(batch_dice_scores, device=device).mean()

            acc_record.update(pred_idx.cpu(), target.cpu())
            loss_record.update(loss.cpu())
            total_batches += 1

    mean_iou_per_class = iou_record / total_batches
    mean_iou_value = mean_iou_per_class.mean()
    mean_dice_value = dice_record / total_batches

    return (
        loss_record.compute().item(),
        mean_iou_value.item(),
        mean_dice_value.item(),
        acc_record.compute().item(),
        mean_iou_per_class.detach().cpu(),
    )


def main(args):
    dataset_cfg = DatasetConfig(
        NUM_CLASSES=args.num_classes,
        IMG_WIDTH=args.img_size,
        IMG_HEIGHT=args.img_size,
        IMAGE_DIR=args.image_dir,
        MASK_DIR=args.mask_dir,
        TRAIN_CSV=args.train_csv,
        TEST_CSV=args.test_csv,
    )
    training_cfg = TrainingConfig(
        BATCH_SIZE=args.batch_size,
        EPOCHS=args.epochs,
        LEARNING_RATE=args.lr,
        CHECKPOINT_DIR=args.checkpoint_dir,
        SEED=args.seed,
    )

    seed_everything(training_cfg.SEED)
    device, gpu_available = get_default_device()

    ckpt_dir = create_checkpoint_dir(training_cfg.CHECKPOINT_DIR)

    model = prepare_model(
        model_name=ModelConfig.MODEL_NAME,
        use_pretrained=ModelConfig.USE_PRETRAINED,
        num_classes=dataset_cfg.NUM_CLASSES,
    )
    model.to(device)

    # Dummy forward pass to materialise LazyConv2d parameters before optimizer creation.
    _ = model(torch.randn((2, 3, dataset_cfg.IMG_HEIGHT, dataset_cfg.IMG_WIDTH), device=device))

    optimizer = Adam(model.parameters(), lr=training_cfg.LEARNING_RATE, amsgrad=True, fused=gpu_available)

    train_loader, valid_loader, _, _ = get_dataloaders(
        dataset_cfg, training_cfg, pin_memory=gpu_available,
    )

    scaler = amp.GradScaler(enabled=gpu_available)
    best_dice = 0.0

    for epoch in range(training_cfg.EPOCHS):
        torch.cuda.empty_cache()
        gc.collect()

        train_loss, train_iou, train_dice, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler,
            dataset_cfg.NUM_CLASSES, device, epoch + 1, training_cfg.EPOCHS,
        )

        valid_loss, valid_iou, valid_dice, valid_acc, per_class_iou = validate(
            model, valid_loader, device,
            dataset_cfg.NUM_CLASSES, epoch + 1, training_cfg.EPOCHS,
        )

        print(
            f"Epoch {epoch + 1}/{training_cfg.EPOCHS} | "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={valid_loss:.4f} val_iou={valid_iou:.4f} val_dice={valid_dice:.4f}"
        )

        if valid_dice > best_dice:
            best_dice = valid_dice
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "dice": valid_dice,
                "per_class_iou": per_class_iou,
            }, os.path.join(ckpt_dir, "ckpt.tar"))
            print(f"  -> New best model saved (val_dice={valid_dice:.4f})")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train DeepLabV3-ResNet101 for drone image segmentation")
    parser.add_argument("--image_dir", default=DatasetConfig.IMAGE_DIR)
    parser.add_argument("--mask_dir", default=DatasetConfig.MASK_DIR)
    parser.add_argument("--train_csv", default=DatasetConfig.TRAIN_CSV)
    parser.add_argument("--test_csv", default=DatasetConfig.TEST_CSV)
    parser.add_argument("--checkpoint_dir", default=TrainingConfig.CHECKPOINT_DIR)
    parser.add_argument("--num_classes", type=int, default=DatasetConfig.NUM_CLASSES)
    parser.add_argument("--img_size", type=int, default=DatasetConfig.IMG_WIDTH)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TrainingConfig.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=TrainingConfig.SEED)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
