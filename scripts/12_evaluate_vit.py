#!/usr/bin/env python3
"""Evaluate ViT-B/16 and CNN baseline on Gravity Spy test set with paired bootstrap.

Plan 03-02 Task 1: Produces metrics.json, per_class_f1.csv, predictions.npz,
paired_bootstrap_results.json for both models.

Convention: macro-F1 is PRIMARY METRIC. Overall accuracy is SANITY CHECK ONLY.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import GravitySpyDataset
from src.data.transforms import eval_transforms
from src.evaluation.evaluate import run_inference, compute_metrics, verify_metric_consistency
from src.evaluation.bootstrap_ci import (
    bootstrap_metric,
    bootstrap_macro_f1,
    bootstrap_per_class_f1,
    bootstrap_per_class_recall,
)
from src.evaluation.paired_bootstrap import (
    paired_bootstrap_rare_f1,
    paired_bootstrap_metric,
    _macro_f1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RARE_THRESHOLD = 200
RARE_CLASSES = ["Chirp", "Wandering_Line", "Helix", "Light_Modulation"]

# Phase 2 reference values for CNN re-evaluation check
CNN_REF_MACRO_F1 = 0.6786466698931387
CNN_REF_ACCURACY = 0.918108301770908


def hash_file(path):
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path, model_builder_fn, device):
    """Load a model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}
    num_classes = len(label_to_idx)

    model, model_id, source = model_builder_fn(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, model_id, label_to_idx, idx_to_label, num_classes, ckpt


def main():
    # ==================== CONFIGURATION ====================
    with open("configs/vit_rare_class.yaml") as f:
        vit_cfg = yaml.safe_load(f)
    with open("configs/cnn_baseline.yaml") as f:
        cnn_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_boot = vit_cfg["evaluation"]["bootstrap_resamples"]
    conf = vit_cfg["evaluation"]["confidence_level"]
    seed = vit_cfg["seed"]

    # ==================== VERIFY TEST SET IDENTITY ====================
    test_manifest_path = vit_cfg["data"]["test_manifest"]
    test_hash = hash_file(test_manifest_path)
    logger.info(f"Test manifest SHA-256: {test_hash}")
    assert vit_cfg["data"]["test_manifest"] == cnn_cfg["data"]["test_manifest"], \
        "ViT and CNN configs use different test manifests!"

    # ==================== LOAD TEST DATASET ====================
    test_ds = GravitySpyDataset(
        test_manifest_path,
        image_root=vit_cfg["data"]["image_root"],
        duration=vit_cfg["data"]["duration"],
        transform=eval_transforms(vit_cfg["data"]["image_size"]),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=vit_cfg["training"]["batch_size"] * 2,
        shuffle=False, num_workers=vit_cfg["data"]["num_workers"], pin_memory=True,
    )
    class_names = [test_ds.idx_to_label[i] for i in range(test_ds.num_classes)]
    label_to_idx = test_ds.label_to_idx
    num_classes = test_ds.num_classes
    logger.info(f"Test set: {len(test_ds)} samples, {num_classes} classes")

    # Train counts for rare class verification and comparison table
    train_ds = GravitySpyDataset(
        vit_cfg["data"]["train_manifest"],
        image_root=vit_cfg["data"]["image_root"],
        duration=vit_cfg["data"]["duration"],
    )
    train_counts = train_ds.class_counts()
    test_counts = test_ds.class_counts()

    # Verify rare classes match
    computed_rare = sorted([c for c, n in train_counts.items() if n < RARE_THRESHOLD])
    expected_rare = sorted(RARE_CLASSES)
    assert computed_rare == expected_rare, \
        f"Rare class mismatch: computed={computed_rare}, expected={expected_rare}"
    logger.info(f"Rare classes confirmed: {RARE_CLASSES}")

    rare_class_indices = [label_to_idx[c] for c in RARE_CLASSES]

    # ==================== CNN RE-EVALUATION ====================
    logger.info("=" * 60)
    logger.info("STEP 1: CNN Re-Evaluation")
    logger.info("=" * 60)

    from src.models.resnet_baseline import build_resnet50_baseline
    cnn_model, cnn_model_id, cnn_l2i, cnn_i2l, _, cnn_ckpt = load_model(
        "checkpoints/02-cnn-baseline/best_model.pt", build_resnet50_baseline, device
    )
    assert cnn_l2i == label_to_idx, "CNN label_to_idx doesn't match test dataset!"

    logger.info("Running CNN inference...")
    cnn_preds, cnn_labels = run_inference(cnn_model, test_loader, device)
    cnn_metrics = compute_metrics(cnn_preds, cnn_labels, class_names)

    cnn_macro_f1 = cnn_metrics["macro_f1"]
    cnn_accuracy = cnn_metrics["overall_accuracy"]
    logger.info(f"  CNN Macro-F1 (PRIMARY): {cnn_macro_f1:.10f}")
    logger.info(f"  CNN Overall accuracy (SANITY CHECK): {cnn_accuracy:.10f}")

    # CRITICAL: Verify CNN re-evaluation reproduces Phase 2 metrics
    f1_diff = abs(cnn_macro_f1 - CNN_REF_MACRO_F1)
    acc_diff = abs(cnn_accuracy - CNN_REF_ACCURACY)
    logger.info(f"  Macro-F1 diff from Phase 2: {f1_diff:.2e} (threshold: 1e-4)")
    logger.info(f"  Accuracy diff from Phase 2: {acc_diff:.2e} (threshold: 1e-4)")
    assert f1_diff < 1e-4, f"CNN macro-F1 mismatch: {cnn_macro_f1} vs ref {CNN_REF_MACRO_F1} (diff={f1_diff:.2e})"
    assert acc_diff < 1e-4, f"CNN accuracy mismatch: {cnn_accuracy} vs ref {CNN_REF_ACCURACY} (diff={acc_diff:.2e})"
    logger.info("  CNN re-evaluation: PASS (matches Phase 2 exactly)")

    # CNN metric consistency
    cnn_sk_f1, cnn_tm_f1, cnn_match, cnn_diff = verify_metric_consistency(cnn_preds, cnn_labels, num_classes)
    logger.info(f"  CNN metric consistency: diff={cnn_diff:.2e} {'PASS' if cnn_match else 'FAIL'}")

    del cnn_model
    torch.cuda.empty_cache()

    # ==================== ViT EVALUATION ====================
    logger.info("=" * 60)
    logger.info("STEP 2: ViT Evaluation")
    logger.info("=" * 60)

    from src.models.vit_classifier import build_vit_classifier
    vit_model, vit_model_id, vit_l2i, vit_i2l, _, vit_ckpt = load_model(
        "checkpoints/03-vit-rare-class/best_model.pt", build_vit_classifier, device
    )
    assert vit_l2i == label_to_idx, "ViT label_to_idx doesn't match test dataset!"

    logger.info("Running ViT inference...")
    vit_preds, vit_labels = run_inference(vit_model, test_loader, device)
    vit_metrics = compute_metrics(vit_preds, vit_labels, class_names)

    logger.info(f"  ViT Macro-F1 (PRIMARY): {vit_metrics['macro_f1']:.4f}")
    logger.info(f"  ViT Overall accuracy (SANITY CHECK): {vit_metrics['overall_accuracy']:.4f}")

    # ViT metric consistency
    vit_sk_f1, vit_tm_f1, vit_match, vit_diff = verify_metric_consistency(vit_preds, vit_labels, num_classes)
    logger.info(f"  ViT metric consistency: sklearn={vit_sk_f1:.8f}, torchmetrics={vit_tm_f1:.8f}, diff={vit_diff:.2e} {'PASS' if vit_match else 'FAIL'}")
    assert vit_match, f"ViT metric consistency FAILED: diff={vit_diff:.2e} > 1e-6"

    del vit_model
    torch.cuda.empty_cache()

    # Verify labels identical
    y_true = vit_labels
    assert np.array_equal(y_true, cnn_labels), "Labels differ between ViT and CNN runs!"

    # ==================== BOOTSTRAP CIs ====================
    logger.info("=" * 60)
    logger.info("STEP 3: Bootstrap CIs and Paired Tests")
    logger.info("=" * 60)

    # ViT overall macro-F1 CI
    vit_mf1_pt, vit_mf1_lo, vit_mf1_hi = bootstrap_macro_f1(
        vit_preds, y_true, n_resamples=n_boot, confidence=conf, seed=seed
    )
    logger.info(f"  ViT macro-F1: {vit_mf1_pt:.4f} [{vit_mf1_lo:.4f}, {vit_mf1_hi:.4f}]")

    # CNN overall macro-F1 CI
    cnn_mf1_pt, cnn_mf1_lo, cnn_mf1_hi = bootstrap_macro_f1(
        cnn_preds, y_true, n_resamples=n_boot, confidence=conf, seed=seed
    )
    logger.info(f"  CNN macro-F1: {cnn_mf1_pt:.4f} [{cnn_mf1_lo:.4f}, {cnn_mf1_hi:.4f}]")

    # Rare-class macro-F1 with bootstrap CI for both
    from sklearn.metrics import f1_score as skf1

    def rare_f1_fn(p, l):
        return float(skf1(l, p, average="macro", labels=rare_class_indices, zero_division=0))

    vit_rare_pt, vit_rare_lo, vit_rare_hi, _ = bootstrap_metric(
        vit_preds, y_true, rare_f1_fn, n_resamples=n_boot, confidence=conf, seed=seed + 999
    )
    cnn_rare_pt, cnn_rare_lo, cnn_rare_hi, _ = bootstrap_metric(
        cnn_preds, y_true, rare_f1_fn, n_resamples=n_boot, confidence=conf, seed=seed + 999
    )
    logger.info(f"  ViT rare-class macro-F1: {vit_rare_pt:.4f} [{vit_rare_lo:.4f}, {vit_rare_hi:.4f}]")
    logger.info(f"  CNN rare-class macro-F1: {cnn_rare_pt:.4f} [{cnn_rare_lo:.4f}, {cnn_rare_hi:.4f}]")

    # Paired bootstrap: rare-class macro-F1
    logger.info(f"  Running paired bootstrap on rare-class macro-F1 ({n_boot} resamples)...")
    rare_boot = paired_bootstrap_rare_f1(
        y_true, vit_preds, cnn_preds,
        rare_class_indices=rare_class_indices,
        n_resamples=n_boot, seed=seed,
    )
    logger.info(f"  Rare-class F1 diff (ViT-CNN): {rare_boot['point_estimate_difference']:.4f} [{rare_boot['ci_lower']:.4f}, {rare_boot['ci_upper']:.4f}]")
    logger.info(f"  p-value (H0: ViT <= CNN): {rare_boot['p_value']:.6f}")

    # Paired bootstrap: overall macro-F1
    logger.info(f"  Running paired bootstrap on overall macro-F1 ({n_boot} resamples)...")
    overall_boot = paired_bootstrap_metric(
        y_true, vit_preds, cnn_preds,
        metric_fn=_macro_f1,
        n_resamples=n_boot, seed=seed + 1,
    )
    logger.info(f"  Overall macro-F1 diff (ViT-CNN): {overall_boot['point_estimate_difference']:.4f} [{overall_boot['ci_lower']:.4f}, {overall_boot['ci_upper']:.4f}]")
    logger.info(f"  p-value (H0: ViT <= CNN): {overall_boot['p_value']:.6f}")

    # ==================== PER-CLASS BOOTSTRAP CIs ====================
    logger.info("Computing per-class bootstrap CIs...")

    vit_per_class_f1 = {}
    vit_per_class_recall = {}
    cnn_per_class_f1_boot = {}
    cnn_per_class_recall_boot = {}

    for i, name in enumerate(class_names):
        # ViT
        f1_pt, f1_lo, f1_hi = bootstrap_per_class_f1(vit_preds, y_true, i, n_resamples=n_boot, confidence=conf, seed=seed + i)
        rec_pt, rec_lo, rec_hi = bootstrap_per_class_recall(vit_preds, y_true, i, n_resamples=n_boot, confidence=conf, seed=seed + i + 100)
        vit_pc = vit_metrics["per_class"].get(name, {})
        vit_per_class_f1[name] = {"f1": f1_pt, "ci_lower": f1_lo, "ci_upper": f1_hi, "n_train": train_counts.get(name, 0), "n_test": test_counts.get(name, 0)}
        vit_per_class_recall[name] = {"recall": rec_pt, "ci_lower": rec_lo, "ci_upper": rec_hi}

        # CNN
        f1_pt_c, f1_lo_c, f1_hi_c = bootstrap_per_class_f1(cnn_preds, y_true, i, n_resamples=n_boot, confidence=conf, seed=seed + i)
        rec_pt_c, rec_lo_c, rec_hi_c = bootstrap_per_class_recall(cnn_preds, y_true, i, n_resamples=n_boot, confidence=conf, seed=seed + i + 100)
        cnn_pc = cnn_metrics["per_class"].get(name, {})
        cnn_per_class_f1_boot[name] = {"f1": f1_pt_c, "ci_lower": f1_lo_c, "ci_upper": f1_hi_c}
        cnn_per_class_recall_boot[name] = {"recall": rec_pt_c, "ci_lower": rec_lo_c, "ci_upper": rec_hi_c}

    # ==================== SAVE ALL RESULTS ====================
    logger.info("=" * 60)
    logger.info("STEP 4: Saving Results")
    logger.info("=" * 60)

    output_dir = Path(vit_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common class avg F1
    common_classes = sorted([c for c, n in train_counts.items() if n >= 1000])
    vit_common_f1 = float(np.mean([vit_per_class_f1[c]["f1"] for c in common_classes]))
    vit_rare_f1_from_pc = float(np.mean([vit_per_class_f1[c]["f1"] for c in RARE_CLASSES]))

    # 1. ViT metrics.json
    vit_metrics_json = {
        "primary_metric": "macro_f1",
        "macro_f1": {
            "value": vit_mf1_pt, "ci_lower": vit_mf1_lo, "ci_upper": vit_mf1_hi,
            "bootstrap_resamples": n_boot, "confidence_level": conf,
        },
        "overall_accuracy": {
            "value": vit_metrics["overall_accuracy"],
            "note": "SANITY CHECK ONLY -- not the primary metric (fp-overall-accuracy)",
        },
        "rare_class_macro_f1": {
            "value": vit_rare_pt, "ci_lower": vit_rare_lo, "ci_upper": vit_rare_hi,
            "rare_classes": RARE_CLASSES, "rare_threshold_train": RARE_THRESHOLD,
        },
        "common_class_avg_f1": {"value": vit_common_f1, "common_classes": common_classes},
        "rare_class_gap": {
            "common_avg_f1": vit_common_f1, "rare_avg_f1": vit_rare_f1_from_pc,
            "gap_pp": (vit_common_f1 - vit_rare_f1_from_pc) * 100,
        },
        "per_class_f1": vit_per_class_f1,
        "per_class_recall": vit_per_class_recall,
        "model": {
            "architecture": vit_model_id,
            "pretrained": "AugReg ImageNet-21k fine-tuned on ImageNet-1k",
            "best_epoch": vit_ckpt["epoch"], "seed": seed,
        },
        "metric_consistency": {
            "sklearn_macro_f1": vit_sk_f1, "torchmetrics_macro_f1": vit_tm_f1,
            "difference": vit_diff, "match": vit_match,
        },
        "test_manifest_hash": test_hash,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(vit_metrics_json, f, indent=2)
    logger.info(f"Saved: {output_dir / 'metrics.json'}")

    # 2. ViT per_class_f1.csv
    per_class_rows = []
    for name in class_names:
        vit_pc = vit_metrics["per_class"].get(name, {})
        f1_data = vit_per_class_f1[name]
        rec_data = vit_per_class_recall[name]
        per_class_rows.append({
            "class": name, "n_train": train_counts.get(name, 0), "n_val": 0,
            "n_test": test_counts.get(name, 0),
            "f1": f1_data["f1"], "f1_ci_lower": f1_data["ci_lower"], "f1_ci_upper": f1_data["ci_upper"],
            "recall": rec_data["recall"], "recall_ci_lower": rec_data["ci_lower"], "recall_ci_upper": rec_data["ci_upper"],
            "precision": vit_pc.get("precision", 0.0), "support": test_counts.get(name, 0),
        })
    vit_pc_df = pd.DataFrame(per_class_rows).sort_values("n_train").reset_index(drop=True)
    vit_pc_df.to_csv(output_dir / "per_class_f1.csv", index=False)
    logger.info(f"Saved: {output_dir / 'per_class_f1.csv'}")

    # 3. predictions.npz
    np.savez(output_dir / "predictions.npz", y_true=y_true, vit_preds=vit_preds, cnn_preds=cnn_preds)
    saved_npz = np.load(output_dir / "predictions.npz")
    assert saved_npz["y_true"].shape[0] == len(test_ds)
    assert saved_npz["vit_preds"].shape[0] == len(test_ds)
    assert saved_npz["cnn_preds"].shape[0] == len(test_ds)
    logger.info(f"Saved: {output_dir / 'predictions.npz'} ({len(test_ds)} samples)")

    # 4. paired_bootstrap_results.json
    paired_results = {
        "rare_class_macro_f1": {
            "point_estimate_difference": rare_boot["point_estimate_difference"],
            "ci_lower": rare_boot["ci_lower"], "ci_upper": rare_boot["ci_upper"],
            "p_value": rare_boot["p_value"], "n_resamples": rare_boot["n_resamples"],
            "vit_rare_f1": rare_boot["rare_f1_a"], "cnn_rare_f1": rare_boot["rare_f1_b"],
            "rare_classes": RARE_CLASSES, "rare_class_indices": rare_class_indices,
            "h0": "ViT rare-class macro-F1 <= CNN rare-class macro-F1",
            "test_type": "one-sided paired percentile bootstrap",
        },
        "overall_macro_f1": {
            "point_estimate_difference": overall_boot["point_estimate_difference"],
            "ci_lower": overall_boot["ci_lower"], "ci_upper": overall_boot["ci_upper"],
            "p_value": overall_boot["p_value"], "n_resamples": overall_boot["n_resamples"],
            "vit_macro_f1": overall_boot["metric_a"], "cnn_macro_f1": overall_boot["metric_b"],
        },
        "n_test_samples": int(len(y_true)),
        "test_manifest_hash": test_hash, "seed": seed,
    }
    with open(output_dir / "paired_bootstrap_results.json", "w") as f:
        json.dump(paired_results, f, indent=2)
    logger.info(f"Saved: {output_dir / 'paired_bootstrap_results.json'}")

    # Save confusion matrices and class names for Task 2
    np.save(output_dir / "vit_confusion_matrix.npy", vit_metrics["confusion_matrix"])
    np.save(output_dir / "cnn_confusion_matrix.npy", cnn_metrics["confusion_matrix"])
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    # Save CNN per-class bootstrap data for comparison table in Task 2
    cnn_boot_data = {"per_class_f1": cnn_per_class_f1_boot, "per_class_recall": cnn_per_class_recall_boot}
    with open(output_dir / "cnn_per_class_bootstrap.json", "w") as f:
        json.dump(cnn_boot_data, f, indent=2)

    # ==================== FINAL SUMMARY ====================
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Test set: {len(y_true)} samples, hash={test_hash[:16]}...")
    logger.info(f"  CNN re-eval: macro-F1={cnn_macro_f1:.4f}, accuracy={cnn_accuracy:.4f} -- MATCHES Phase 2")
    logger.info(f"  ViT:         macro-F1={vit_mf1_pt:.4f}, accuracy={vit_metrics['overall_accuracy']:.4f}")
    logger.info(f"  ---")
    logger.info(f"  PRIMARY: Rare-class macro-F1")
    logger.info(f"    CNN: {cnn_rare_pt:.4f} [{cnn_rare_lo:.4f}, {cnn_rare_hi:.4f}]")
    logger.info(f"    ViT: {vit_rare_pt:.4f} [{vit_rare_lo:.4f}, {vit_rare_hi:.4f}]")
    logger.info(f"    Diff (ViT-CNN): {rare_boot['point_estimate_difference']:.4f} [{rare_boot['ci_lower']:.4f}, {rare_boot['ci_upper']:.4f}]")
    logger.info(f"    p-value: {rare_boot['p_value']:.6f}")
    logger.info(f"  SECONDARY: Overall macro-F1")
    logger.info(f"    CNN: {cnn_mf1_pt:.4f} [{cnn_mf1_lo:.4f}, {cnn_mf1_hi:.4f}]")
    logger.info(f"    ViT: {vit_mf1_pt:.4f} [{vit_mf1_lo:.4f}, {vit_mf1_hi:.4f}]")
    logger.info(f"    Diff (ViT-CNN): {overall_boot['point_estimate_difference']:.4f} [{overall_boot['ci_lower']:.4f}, {overall_boot['ci_upper']:.4f}]")
    logger.info(f"    p-value: {overall_boot['p_value']:.6f}")
    logger.info(f"  Overall accuracy (SANITY CHECK): CNN={cnn_accuracy:.4f}, ViT={vit_metrics['overall_accuracy']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
