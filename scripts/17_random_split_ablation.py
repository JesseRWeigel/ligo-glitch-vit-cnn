#!/usr/bin/env python3
"""Random-split CNN ablation: retrain CNN on stratified random split to isolate
the effect of temporal vs random splitting on classification accuracy.

This script:
  1. Creates stratified random train/val/test splits (70/15/15%)
  2. Trains the SAME CNN architecture with IDENTICAL hyperparameters
  3. Evaluates with bootstrap CIs on the random-split test set
  4. Compares accuracy to temporal-split baseline (91.81%) and published
     Gravity Spy benchmarks (95-99%, Zevin et al. 2017)

Usage:
    python scripts/17_random_split_ablation.py
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy, bootstrap_resamples>=10000

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
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import GravitySpyDataset
from src.data.transforms import train_transforms, eval_transforms
from src.losses.focal_loss import FocalLoss
from src.models.resnet_baseline import build_resnet50_baseline
from src.training.train_cnn import train_one_epoch, validate, CosineWarmupScheduler
from src.evaluation.evaluate import run_inference, compute_metrics
from src.evaluation.bootstrap_ci import bootstrap_macro_f1, bootstrap_metric

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Reference values (hardcoded from Phase 2 / STATE.md) ---
TEMPORAL_SPLIT_ACCURACY = 0.9181
TEMPORAL_SPLIT_MACRO_F1 = 0.6786
PUBLISHED_BENCHMARK_RANGE = [0.95, 0.99]  # Zevin et al. 2017


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_random_split_manifests(master_csv, output_dir, seed=42):
    """Create stratified random train/val/test manifests from master metadata.

    Two-stage split: 70/30 (train vs rest), then 50/50 on rest (val vs test).
    Image paths are constructed from gravityspy_id and ml_label.

    Returns (train_df, val_df, test_df) with manifest columns.
    """
    logger.info(f"Loading master metadata from {master_csv}...")
    df = pd.read_csv(master_csv)
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Unique classes: {df['ml_label'].nunique()}")

    # Construct image path columns (same pattern as existing manifests)
    for dur in ["0.5s", "1.0s", "2.0s", "4.0s"]:
        df[f"image_path_{dur}"] = (
            "data/spectrograms/" + df["ml_label"] + "/" + df["gravityspy_id"] + f"_{dur}.png"
        )

    # Stage 1: 70% train, 30% rest (stratified)
    train_df, rest_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["ml_label"]
    )
    # Stage 2: 50/50 on rest -> 15% val, 15% test (stratified)
    val_df, test_df = train_test_split(
        rest_df, test_size=0.50, random_state=seed, stratify=rest_df["ml_label"]
    )

    # Add split column
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Select manifest columns (matching existing temporal-split manifests)
    manifest_cols = [
        "gravityspy_id", "event_time", "ifo", "ml_label", "ml_confidence", "snr",
        "image_path_0.5s", "image_path_1.0s", "image_path_2.0s", "image_path_4.0s", "split"
    ]
    train_df = train_df[manifest_cols]
    val_df = val_df[manifest_cols]
    test_df = test_df[manifest_cols]

    # --- CRITICAL CHECKS ---
    # 1. No sample ID overlap
    train_ids = set(train_df["gravityspy_id"])
    val_ids = set(val_df["gravityspy_id"])
    test_ids = set(test_df["gravityspy_id"])
    assert len(train_ids & val_ids) == 0, "Train/val overlap detected!"
    assert len(train_ids & test_ids) == 0, "Train/test overlap detected!"
    assert len(val_ids & test_ids) == 0, "Val/test overlap detected!"
    logger.info("  [CHECK] No sample overlap between splits: PASSED")

    # 2. All 23 classes in each split
    assert train_df["ml_label"].nunique() == 23, f"Train missing classes: {23 - train_df['ml_label'].nunique()}"
    assert val_df["ml_label"].nunique() == 23, f"Val missing classes: {23 - val_df['ml_label'].nunique()}"
    assert test_df["ml_label"].nunique() == 23, f"Test missing classes: {23 - test_df['ml_label'].nunique()}"
    logger.info("  [CHECK] All 23 classes in each split: PASSED")

    # 3. Split proportions within 1% of targets
    total = len(df)
    train_pct = len(train_df) / total
    val_pct = len(val_df) / total
    test_pct = len(test_df) / total
    assert abs(train_pct - 0.70) < 0.01, f"Train proportion off: {train_pct:.4f}"
    assert abs(val_pct - 0.15) < 0.01, f"Val proportion off: {val_pct:.4f}"
    assert abs(test_pct - 0.15) < 0.01, f"Test proportion off: {test_pct:.4f}"
    logger.info(f"  [CHECK] Split proportions: train={train_pct:.4f}, val={val_pct:.4f}, test={test_pct:.4f}: PASSED")

    # Total rows preserved
    assert len(train_df) + len(val_df) + len(test_df) == total, "Sample count mismatch!"

    # Save manifests
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "random_split_train_manifest.csv"
    val_path = output_dir / "random_split_val_manifest.csv"
    test_path = output_dir / "random_split_test_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"  Saved: {train_path} ({len(train_df)} rows)")
    logger.info(f"  Saved: {val_path} ({len(val_df)} rows)")
    logger.info(f"  Saved: {test_path} ({len(test_df)} rows)")

    # Log per-class counts
    logger.info("  Per-class test counts:")
    for lbl in sorted(test_df["ml_label"].unique()):
        n = (test_df["ml_label"] == lbl).sum()
        logger.info(f"    {lbl}: {n}")

    return train_df, val_df, test_df


def main():
    start_time = time.time()

    # Load config
    cfg_path = Path("configs/cnn_random_split.yaml")
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

    # =========================================================================
    # Step 1: Create random-split manifests
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Creating stratified random-split manifests")
    logger.info("=" * 60)

    master_csv = "data/metadata/gravity_spy_o3_filtered.csv"
    manifest_dir = "data/metadata"
    train_df, val_df, test_df = create_random_split_manifests(
        master_csv, manifest_dir, seed=seed
    )

    # =========================================================================
    # Step 2: Train CNN on random-split data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Training CNN on random-split data")
    logger.info("=" * 60)

    # Create output dirs
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

    # Weighted sampler
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
        batch_size=cfg["training"]["batch_size"] * 2,
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

    # Optimizer
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
        "split_method": "stratified_random",
        "epochs": [],
    }

    train_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            num_classes=cfg["model"]["num_classes"],
            gradient_clip_max_norm=cfg["training"]["gradient_clip_max_norm"],
            scheduler=scheduler,
        )

        val_metrics = validate(
            model, val_loader, criterion, device,
            num_classes=cfg["model"]["num_classes"],
        )

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_lr()

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

        # RED FLAG: NaN loss
        if np.isnan(train_metrics["loss"]):
            logger.error("NaN training loss detected! Stopping.")
            break

        # Early stopping on val macro-F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
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

    training_time = time.time() - train_start
    training_log["training_time_seconds"] = training_time
    training_log["best_epoch"] = best_epoch
    training_log["best_val_macro_f1"] = best_val_f1
    training_log["stopped_at_epoch"] = epoch

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # =========================================================================
    # Step 3: Evaluate on random-split test set with bootstrap CIs
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Evaluating on random-split test set")
    logger.info("=" * 60)

    # Load best checkpoint
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    idx_to_label = ckpt["idx_to_label"]
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    # Build test loader
    test_ds = GravitySpyDataset(
        cfg["data"]["test_manifest"],
        image_root=cfg["data"]["image_root"],
        duration=cfg["data"]["duration"],
        transform=eval_transforms(cfg["data"]["image_size"]),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # Run inference
    preds, labels = run_inference(model, test_loader, device)
    metrics = compute_metrics(preds, labels, class_names)

    # Bootstrap CIs
    logger.info("Computing bootstrap CIs (10K resamples)...")

    # Macro-F1
    macro_f1_point, macro_f1_lo, macro_f1_hi = bootstrap_macro_f1(
        preds, labels, n_resamples=10000, confidence=0.95, seed=42
    )

    # Overall accuracy
    def accuracy_fn(p, l):
        return float(np.mean(p == l))

    acc_point, acc_lo, acc_hi, _ = bootstrap_metric(
        preds, labels, accuracy_fn, n_resamples=10000, confidence=0.95, seed=42
    )

    # Per-class F1
    per_class_f1 = {}
    for i, name in enumerate(class_names):
        if name in metrics["per_class"]:
            per_class_f1[name] = metrics["per_class"][name]["f1"]

    # =========================================================================
    # Step 4: Compile results
    # =========================================================================
    random_split_accuracy = metrics["overall_accuracy"]
    accuracy_gap_pp = (random_split_accuracy - TEMPORAL_SPLIT_ACCURACY) * 100

    # RED FLAG check
    if random_split_accuracy < TEMPORAL_SPLIT_ACCURACY:
        logger.error("=" * 60)
        logger.error("RED FLAG: Random-split accuracy BELOW temporal-split!")
        logger.error(f"  Random-split: {random_split_accuracy:.4f}")
        logger.error(f"  Temporal-split: {TEMPORAL_SPLIT_ACCURACY:.4f}")
        logger.error("  This suggests a PIPELINE ISSUE, not a split effect.")
        logger.error("  Results saved but investigation required!")
        logger.error("=" * 60)

    results = {
        "overall_accuracy": float(random_split_accuracy),
        "overall_accuracy_ci": [float(acc_lo), float(acc_hi)],
        "macro_f1": float(macro_f1_point),
        "macro_f1_ci": [float(macro_f1_lo), float(macro_f1_hi)],
        "per_class_f1": per_class_f1,
        "temporal_split_accuracy": TEMPORAL_SPLIT_ACCURACY,
        "temporal_split_macro_f1": TEMPORAL_SPLIT_MACRO_F1,
        "accuracy_gap_pp": float(accuracy_gap_pp),
        "published_benchmark_range": PUBLISHED_BENCHMARK_RANGE,
        "split_method": "stratified_random",
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "seed": seed,
        "best_epoch": best_epoch,
        "training_time_seconds": float(training_time),
        "bootstrap_resamples": 10000,
        "confidence_level": 0.95,
    }

    results_path = Path("results/06-computation-statistical-analysis/random_split_ablation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("RANDOM-SPLIT ABLATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Random-split overall accuracy: {random_split_accuracy:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
    logger.info(f"  Random-split macro-F1:         {macro_f1_point:.4f} [{macro_f1_lo:.4f}, {macro_f1_hi:.4f}]")
    logger.info(f"  Temporal-split accuracy:        {TEMPORAL_SPLIT_ACCURACY:.4f}")
    logger.info(f"  Temporal-split macro-F1:        {TEMPORAL_SPLIT_MACRO_F1:.4f}")
    logger.info(f"  Accuracy gap (pp):              {accuracy_gap_pp:+.2f}")
    logger.info(f"  Published benchmark range:      {PUBLISHED_BENCHMARK_RANGE}")
    logger.info(f"  Best epoch:                     {best_epoch}")
    logger.info(f"  Training time:                  {training_time / 60:.1f} min")
    logger.info(f"  Total time:                     {total_time / 60:.1f} min")

    if random_split_accuracy >= 0.948:
        logger.info("  CONCLUSION: Random-split accuracy >= 3pp above temporal-split.")
        logger.info("  The temporal split (not the pipeline) explains the accuracy gap.")
    elif random_split_accuracy >= TEMPORAL_SPLIT_ACCURACY:
        logger.info("  NOTE: Random-split accuracy is above temporal-split but < 3pp gap.")
        logger.info("  The split method contributes but may not fully explain the gap.")
    else:
        logger.error("  RED FLAG: Random-split accuracy BELOW temporal-split!")
        logger.error("  Pipeline investigation required before proceeding.")


if __name__ == "__main__":
    main()
