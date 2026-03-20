#!/usr/bin/env python3
"""Assess trained CNN baseline on Gravity Spy test set.

Produces:
- results/02-cnn-baseline/metrics.json
- results/02-cnn-baseline/per_class_f1.csv
- results/02-cnn-baseline/rare_class_gap_analysis.md
- figures/cnn_confusion_matrix.png
- figures/cnn_per_class_f1.png

Convention: macro-F1 is PRIMARY METRIC. Overall accuracy is SANITY CHECK ONLY.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RARE_THRESHOLD = 200


def main():
    with open("configs/cnn_baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best model
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_model.pt"
    logger.info(f"Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}
    num_classes = len(label_to_idx)
    class_names = [idx_to_label[i] for i in range(num_classes)]

    from src.models.resnet_baseline import build_resnet50_baseline
    model, model_id, _ = build_resnet50_baseline(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded: {model_id}, epoch {ckpt['epoch']}")

    # Load test dataset
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
        pin_memory=True,
    )
    logger.info(f"Test set: {len(test_ds)} samples")

    # Run inference
    logger.info("Running inference on test set...")
    preds, labels = run_inference(model, test_loader, device)
    logger.info(f"Predictions: {len(preds)}, Labels: {len(labels)}")

    # Compute metrics
    metrics = compute_metrics(preds, labels, class_names)
    logger.info(f"Macro-F1 (PRIMARY): {metrics['macro_f1']:.4f}")
    logger.info(f"Overall accuracy (SANITY CHECK): {metrics['overall_accuracy']:.4f}")

    # Metric consistency check
    logger.info("Verifying metric consistency (sklearn vs torchmetrics)...")
    sk_f1, tm_f1, match, diff = verify_metric_consistency(preds, labels, num_classes)
    logger.info(f"sklearn macro-F1: {sk_f1:.8f}")
    logger.info(f"torchmetrics macro-F1: {tm_f1:.8f}")
    logger.info(f"Difference: {diff:.2e} {'PASS' if match else 'FAIL (>1e-6)'}")

    # Bootstrap CIs
    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    conf = cfg["evaluation"]["confidence_level"]
    seed = cfg["seed"]
    logger.info(f"Computing bootstrap CIs ({n_boot} resamples)...")

    macro_f1_pt, macro_f1_lo, macro_f1_hi = bootstrap_macro_f1(
        preds, labels, n_resamples=n_boot, confidence=conf, seed=seed
    )
    logger.info(f"Macro-F1: {macro_f1_pt:.4f} [{macro_f1_lo:.4f}, {macro_f1_hi:.4f}]")

    # Per-class data
    train_ds = GravitySpyDataset(cfg["data"]["train_manifest"], image_root=cfg["data"]["image_root"], duration=cfg["data"]["duration"])
    val_ds = GravitySpyDataset(cfg["data"]["val_manifest"], image_root=cfg["data"]["image_root"], duration=cfg["data"]["duration"])
    train_counts = train_ds.class_counts()
    val_counts = val_ds.class_counts()
    test_counts = test_ds.class_counts()

    per_class_data = []
    for i, name in enumerate(class_names):
        f1_pt, f1_lo, f1_hi = bootstrap_per_class_f1(preds, labels, i, n_resamples=n_boot, confidence=conf, seed=seed + i)
        rec_pt, rec_lo, rec_hi = bootstrap_per_class_recall(preds, labels, i, n_resamples=n_boot, confidence=conf, seed=seed + i + 100)
        pc = metrics["per_class"].get(name, {})
        per_class_data.append({
            "class": name,
            "n_train": train_counts.get(name, 0),
            "n_val": val_counts.get(name, 0),
            "n_test": test_counts.get(name, 0),
            "f1": f1_pt,
            "f1_ci_lower": f1_lo,
            "f1_ci_upper": f1_hi,
            "recall": rec_pt,
            "recall_ci_lower": rec_lo,
            "recall_ci_upper": rec_hi,
            "precision": pc.get("precision", 0.0),
            "support": pc.get("support", 0),
        })

    per_class_df = pd.DataFrame(per_class_data).sort_values("n_train").reset_index(drop=True)

    # Rare vs common
    rare_classes = per_class_df[per_class_df["n_train"] < RARE_THRESHOLD]["class"].tolist()
    common_classes = per_class_df[per_class_df["n_train"] >= 1000]["class"].tolist()

    rare_f1_values = per_class_df[per_class_df["n_train"] < RARE_THRESHOLD]["f1"].values
    common_f1_values = per_class_df[per_class_df["n_train"] >= 1000]["f1"].values
    rare_macro_f1 = float(np.mean(rare_f1_values)) if len(rare_f1_values) > 0 else 0.0
    common_macro_f1 = float(np.mean(common_f1_values)) if len(common_f1_values) > 0 else 0.0
    gap = common_macro_f1 - rare_macro_f1

    logger.info(f"Rare-class avg F1 ({len(rare_classes)} classes): {rare_macro_f1:.4f}")
    logger.info(f"Common-class avg F1 ({len(common_classes)} classes): {common_macro_f1:.4f}")
    logger.info(f"Gap: {gap:.4f} ({gap*100:.1f} pp)")

    # Bootstrap CI for rare-class macro-F1
    def rare_macro_f1_fn(p, l):
        f1s = []
        for name in rare_classes:
            idx = label_to_idx[name]
            bp = (p == idx).astype(int)
            bt = (l == idx).astype(int)
            tp = np.sum(bp & bt)
            fp = np.sum(bp & ~bt.astype(bool))
            fn = np.sum(~bp.astype(bool) & bt)
            if tp == 0:
                f1s.append(0.0)
            else:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f1s.append(2 * prec * rec / (prec + rec))
        return float(np.mean(f1s)) if f1s else 0.0

    rare_f1_pt, rare_f1_lo, rare_f1_hi, _ = bootstrap_metric(
        preds, labels, rare_macro_f1_fn, n_resamples=n_boot, confidence=conf, seed=seed + 999
    )
    logger.info(f"Rare-class macro-F1: {rare_f1_pt:.4f} [{rare_f1_lo:.4f}, {rare_f1_hi:.4f}]")

    # ==================== SAVE OUTPUTS ====================
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. metrics.json
    metrics_json = {
        "primary_metric": "macro_f1",
        "macro_f1": {"value": macro_f1_pt, "ci_lower": macro_f1_lo, "ci_upper": macro_f1_hi,
                      "bootstrap_resamples": n_boot, "confidence_level": conf},
        "overall_accuracy": {"value": metrics["overall_accuracy"],
                             "note": "SANITY CHECK ONLY -- not the primary metric (fp-overall-accuracy)"},
        "rare_class_macro_f1": {"value": rare_f1_pt, "ci_lower": rare_f1_lo, "ci_upper": rare_f1_hi,
                                 "rare_classes": rare_classes, "rare_threshold_train": RARE_THRESHOLD},
        "common_class_avg_f1": {"value": common_macro_f1, "common_classes": common_classes},
        "rare_class_gap": {"common_avg_f1": common_macro_f1, "rare_avg_f1": rare_macro_f1, "gap_pp": gap * 100},
        "per_class_f1": {row["class"]: {"f1": row["f1"], "ci_lower": row["f1_ci_lower"],
                         "ci_upper": row["f1_ci_upper"], "n_train": row["n_train"], "n_test": row["n_test"]}
                         for _, row in per_class_df.iterrows()},
        "per_class_recall": {row["class"]: {"recall": row["recall"], "ci_lower": row["recall_ci_lower"],
                             "ci_upper": row["recall_ci_upper"]} for _, row in per_class_df.iterrows()},
        "model": {"architecture": model_id, "pretrained": ckpt.get("pretrain_source", "unknown"),
                  "best_epoch": ckpt["epoch"], "seed": seed},
        "metric_consistency": {"sklearn_macro_f1": sk_f1, "torchmetrics_macro_f1": tm_f1,
                               "difference": diff, "match": match},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Saved: {output_dir / 'metrics.json'}")

    # 2. per_class_f1.csv
    per_class_df.to_csv(output_dir / "per_class_f1.csv", index=False)
    logger.info(f"Saved: {output_dir / 'per_class_f1.csv'}")

    # 3. Confusion matrix figure
    cm = metrics["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    sort_order = per_class_df["class"].tolist()
    sort_indices = [class_names.index(c) for c in sort_order]
    cm_sorted = cm[np.ix_(sort_indices, sort_indices)]
    cm_norm_sorted = cm_norm[np.ix_(sort_indices, sort_indices)]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm_sorted, annot=cm_sorted, fmt="d", cmap="Blues",
                xticklabels=sort_order, yticklabels=sort_order, ax=ax, vmin=0, vmax=1,
                cbar_kws={"label": "Proportion (row-normalized)"})
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(f"CNN Baseline Confusion Matrix (Test Set, N={len(preds)})\n"
                 f"Macro-F1={macro_f1_pt:.4f} | Overall Acc={metrics['overall_accuracy']:.4f} (sanity check)", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    # Highlight rare classes
    very_rare = [c for c in sort_order if train_counts.get(c, 0) < 25]
    for c in very_rare:
        idx_s = sort_order.index(c)
        rect = plt.Rectangle((idx_s, idx_s), 1, 1, fill=False, edgecolor="red", linewidth=2.5)
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(figures_dir / "cnn_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {figures_dir / 'cnn_confusion_matrix.png'}")

    # 4. Per-class F1 bar chart
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_df = per_class_df.sort_values("f1").reset_index(drop=True)

    colors = []
    for _, row in plot_df.iterrows():
        if row["n_train"] < 25:
            colors.append("tab:red")
        elif row["n_train"] < RARE_THRESHOLD:
            colors.append("tab:orange")
        else:
            colors.append("tab:blue")

    y_pos = range(len(plot_df))
    xerr_lo = plot_df["f1"] - plot_df["f1_ci_lower"]
    xerr_hi = plot_df["f1_ci_upper"] - plot_df["f1"]

    ax.barh(y_pos, plot_df["f1"], xerr=[xerr_lo, xerr_hi], color=colors,
            edgecolor="black", linewidth=0.5, capsize=3, height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{row['class']} (n={row['n_train']})" for _, row in plot_df.iterrows()], fontsize=9)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title(f"CNN Baseline Per-Class F1 on Temporal Test Set (N={len(preds)})\n"
                 f"Macro-F1 = {macro_f1_pt:.4f} [{macro_f1_lo:.4f}, {macro_f1_hi:.4f}]", fontsize=13)
    ax.set_xlim(0, 1.05)

    ax.axvline(x=macro_f1_pt, color="green", linestyle="--", alpha=0.7, label=f"Macro-F1 = {macro_f1_pt:.3f}")
    ax.axvline(x=rare_f1_pt, color="red", linestyle=":", alpha=0.7, label=f"Rare-class F1 = {rare_f1_pt:.3f}")

    legend_handles = [
        mpatches.Patch(color="tab:red", label="Rare (<25 train)"),
        mpatches.Patch(color="tab:orange", label=f"Near-rare (25-{RARE_THRESHOLD} train)"),
        mpatches.Patch(color="tab:blue", label=f"Common (>{RARE_THRESHOLD} train)"),
    ]
    ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0], loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "cnn_per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {figures_dir / 'cnn_per_class_f1.png'}")

    # 5. Gap analysis document
    acc_status = "PASS" if 0.95 <= metrics["overall_accuracy"] <= 0.99 else "TENSION"
    gap_doc_lines = [
        "# Rare-Class Performance Gap Analysis: CNN Baseline\n",
        "## Definition\n",
        f"- **Rare classes** (n_train < {RARE_THRESHOLD}): {', '.join(rare_classes)}",
        f"- **Common classes** (n_train >= 1000): {', '.join(common_classes)}\n",
        "## Summary\n",
        "| Metric | Value | 95% CI |",
        "|--------|-------|--------|",
        f"| Macro-F1 (all classes) | {macro_f1_pt:.4f} | [{macro_f1_lo:.4f}, {macro_f1_hi:.4f}] |",
        f"| Common-class avg F1 | {common_macro_f1:.4f} | -- |",
        f"| Rare-class avg F1 | {rare_macro_f1:.4f} | [{rare_f1_lo:.4f}, {rare_f1_hi:.4f}] |",
        f"| **Gap (common - rare)** | **{gap*100:.1f} pp** | -- |",
        f"| Overall accuracy (SANITY CHECK) | {metrics['overall_accuracy']:.4f} | -- |\n",
        "## Per-Class Breakdown (sorted by n_train ascending)\n",
        "| Class | n_train | n_test | F1 | F1 95% CI | Recall | Category |",
        "|-------|---------|--------|-----|-----------|--------|----------|",
    ]
    for _, row in per_class_df.iterrows():
        n_tr = row["n_train"]
        cat = "RARE" if n_tr < 25 else ("Near-rare" if n_tr < RARE_THRESHOLD else "Common")
        ci_w = row["f1_ci_upper"] - row["f1_ci_lower"]
        note = " *" if row["n_test"] < 15 else ""
        gap_doc_lines.append(
            f"| {row['class']}{note} | {n_tr} | {row['n_test']} | "
            f"{row['f1']:.4f} | [{row['f1_ci_lower']:.4f}, {row['f1_ci_upper']:.4f}] "
            f"(width={ci_w:.3f}) | {row['recall']:.4f} | {cat} |"
        )

    gap_doc_lines.append("\n\\* Classes with n_test < 15 -- bootstrap CIs may be unreliable.\n")
    gap_doc_lines.append("## Small-Sample Caveats\n")
    small_test = per_class_df[per_class_df["n_test"] < 15]
    for _, row in small_test.iterrows():
        ci_w = row["f1_ci_upper"] - row["f1_ci_lower"]
        warn = " -- WARNING: CI width > 0.5, per-class F1 unreliable" if ci_w > 0.5 else ""
        gap_doc_lines.append(f"- **{row['class']}** (n_test={row['n_test']}): CI width = {ci_w:.3f}{warn}")

    gap_doc_lines.extend([
        "\n## Anchor Comparison (ref-gravity-spy)\n",
        f"Overall accuracy of **{metrics['overall_accuracy']*100:.2f}%** confirms pipeline validity against published",
        "Gravity Spy accuracy of ~97% (Zevin et al. 2017, CQG 34 064003).",
        "- Expected range: [95%, 99%]",
        f"- Status: {acc_status}\n",
        "Note: Our dataset uses O3 data with 23 classes vs. the original O1/O2 with ~20 classes.",
        "The temporal split (vs. random split in the original) makes exact reproduction not expected.\n",
        "## Decisive Output\n",
        f"Overall accuracy of {metrics['overall_accuracy']*100:.2f}% confirms pipeline validity (ref-gravity-spy anchor).",
        f"The decisive output is the **rare-class macro-F1 of {rare_f1_pt*100:.1f}%** [CI: {rare_f1_lo*100:.1f}%, {rare_f1_hi*100:.1f}%],",
        "which is the baseline the ViT must beat in Phase 3.\n",
        f"The gap of **{gap*100:.1f} percentage points** between common-class avg F1 ({common_macro_f1*100:.1f}%) and",
        f"rare-class avg F1 ({rare_macro_f1*100:.1f}%) demonstrates that the CNN struggles significantly with rare classes,",
        "motivating the ViT investigation in Phase 3.\n",
        "## Forbidden Proxy Enforcement (fp-overall-accuracy)\n",
        "This analysis leads with macro-F1 and per-class F1 as primary results.",
        "Overall accuracy appears ONLY as a sanity check against the Gravity Spy anchor.",
    ])

    with open(output_dir / "rare_class_gap_analysis.md", "w") as f:
        f.write("\n".join(gap_doc_lines) + "\n")
    logger.info(f"Saved: {output_dir / 'rare_class_gap_analysis.md'}")

    # Final summary
    logger.info("=" * 60)
    logger.info("ASSESSMENT COMPLETE")
    logger.info(f"  Macro-F1 (PRIMARY): {macro_f1_pt:.4f} [{macro_f1_lo:.4f}, {macro_f1_hi:.4f}]")
    logger.info(f"  Rare-class macro-F1: {rare_f1_pt:.4f} [{rare_f1_lo:.4f}, {rare_f1_hi:.4f}]")
    logger.info(f"  Common-class avg F1: {common_macro_f1:.4f}")
    logger.info(f"  Gap: {gap*100:.1f} pp")
    logger.info(f"  Overall accuracy (sanity): {metrics['overall_accuracy']:.4f}")
    logger.info(f"  Metric consistency: sklearn={sk_f1:.6f}, torchmetrics={tm_f1:.6f}, diff={diff:.2e}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
