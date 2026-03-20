#!/usr/bin/env python3
"""Train ResNet-50 CNN baseline on Gravity Spy O3 spectrograms.

Usage:
    python scripts/09_train_cnn_baseline.py

Loads config from configs/cnn_baseline.yaml.
Saves best model to checkpoints/02-cnn-baseline/best_model.pt.
Saves training log to results/02-cnn-baseline/training_log.json.
Saves training curves to results/02-cnn-baseline/training_curves.png.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy

import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import GravitySpyDataset
from src.data.transforms import train_transforms, eval_transforms
from src.losses.focal_loss import FocalLoss
from src.models.resnet_baseline import build_resnet50_baseline
from src.training.train_cnn import train_one_epoch, validate, CosineWarmupScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Load config
    cfg_path = Path("configs/cnn_baseline.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")

    # Create output directories
    output_dir = Path(cfg["output_dir"])
    ckpt_dir = Path(cfg["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    logger.info("Loading datasets...")
    train_ds = GravitySpyDataset(
        cfg["data"]["train_manifest"],
        image_root=cfg["data"]["image_root"],
        duration=cfg["data"]["duration"],
        transform=train_transforms(cfg["data"]["image_size"]),
    )
    val_ds = GravitySpyDataset(
        cfg["data"]["val_manifest"],
        image_root=cfg["data"]["image_root"],
        duration=cfg["data"]["duration"],
        transform=eval_transforms(cfg["data"]["image_size"]),
    )
    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    logger.info(f"Classes: {train_ds.num_classes}")

    # Build weighted sampler
    sampler = train_ds.get_sampler()

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # Build model
    logger.info("Loading model...")
    model, model_id, pretrain_source = build_resnet50_baseline(
        num_classes=cfg["model"]["num_classes"],
        pretrained=True,
    )
    model = model.to(device)
    logger.info(f"Model: {model_id} ({pretrain_source})")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params / 1e6:.1f}M")

    # Loss function
    alpha_weights = train_ds.class_weights().to(device)
    criterion = FocalLoss(
        gamma=cfg["loss"]["gamma"],
        alpha=alpha_weights,
        reduction="mean",
        label_smoothing=cfg["training"]["label_smoothing"],
    )

    # Optimizer (single param group, lr=1e-3 for CNN per protocol)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg["training"]["warmup_epochs"],
        total_epochs=cfg["training"]["epochs"],
        steps_per_epoch=steps_per_epoch,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]
    max_epochs = cfg["training"]["epochs"]

    training_log = {
        "config": cfg,
        "model_id": model_id,
        "pretrain_source": pretrain_source,
        "total_params": total_params,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "seed": seed,
        "epochs": [],
    }

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train (scheduler stepped per-batch inside train_one_epoch)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            num_classes=cfg["model"]["num_classes"],
            gradient_clip_max_norm=cfg["training"]["gradient_clip_max_norm"],
            scheduler=scheduler,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            num_classes=cfg["model"]["num_classes"],
        )

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_lr()

        # Log
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_overall_accuracy": val_metrics["overall_accuracy"],
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        }
        training_log["epochs"].append(epoch_record)

        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['overall_accuracy']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # RED FLAG checks
        if np.isnan(train_metrics["loss"]):
            logger.error("NaN training loss detected! Stopping.")
            break

        # Early stopping on val macro-F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            # Save best checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_metrics["macro_f1"],
                "val_overall_accuracy": val_metrics["overall_accuracy"],
                "val_per_class_f1": val_metrics["per_class_f1"],
                "config": cfg,
                "model_id": model_id,
                "pretrain_source": pretrain_source,
                "label_to_idx": train_ds.label_to_idx,
                "idx_to_label": train_ds.idx_to_label,
                "seed": seed,
            }
            torch.save(checkpoint, ckpt_dir / "best_model.pt")
            logger.info(f"  -> New best! Saved checkpoint (macro-F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    total_time = time.time() - start_time
    training_log["total_time_seconds"] = total_time
    training_log["best_epoch"] = best_epoch
    training_log["best_val_macro_f1"] = best_val_f1
    training_log["stopped_at_epoch"] = epoch

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Training log saved to {output_dir / 'training_log.json'}")

    # Plot training curves
    epochs_list = [e["epoch"] for e in training_log["epochs"]]
    train_losses = [e["train_loss"] for e in training_log["epochs"]]
    val_losses = [e["val_loss"] for e in training_log["epochs"]]
    val_f1s = [e["val_macro_f1"] for e in training_log["epochs"]]
    val_accs = [e["val_overall_accuracy"] for e in training_log["epochs"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(epochs_list, train_losses, label="Train Loss", color="tab:blue")
    axes[0].plot(epochs_list, val_losses, label="Val Loss", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Macro-F1 curve (PRIMARY METRIC)
    axes[1].plot(epochs_list, val_f1s, label="Val Macro-F1", color="tab:green", linewidth=2)
    axes[1].axhline(y=best_val_f1, color="tab:red", linestyle="--", alpha=0.5,
                     label=f"Best: {best_val_f1:.4f} (epoch {best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Validation Macro-F1 (PRIMARY METRIC)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Overall accuracy (SANITY CHECK ONLY)
    axes[2].plot(epochs_list, val_accs, label="Val Accuracy", color="tab:purple")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Val Overall Accuracy (SANITY CHECK ONLY)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"CNN Baseline Training (ResNet-50, seed={seed})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {output_dir / 'training_curves.png'}")

    # Final summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best val macro-F1: {best_val_f1:.4f} (PRIMARY METRIC)")
    logger.info(f"  Best val overall accuracy: {training_log['epochs'][best_epoch - 1]['val_overall_accuracy']:.4f} (SANITY CHECK)")
    logger.info(f"  Total time: {total_time / 60:.1f} minutes")
    logger.info(f"  Checkpoint: {ckpt_dir / 'best_model.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
