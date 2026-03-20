#!/usr/bin/env python3
"""
Generate split quality visualization: grouped bar chart of per-class
sample counts across train/val/test splits.

Convention: SI units, GPS time in seconds, macro-F1 primary metric.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATS_PATH = Path("data/metadata/split_statistics.json")
OUTPUT_PATH = Path("figures/split_class_distribution.png")

RARE_THRESHOLD = 25


def main():
    with open(STATS_PATH) as f:
        stats = json.load(f)

    pcs = stats["per_class_per_split"]

    # Sort classes by total count (descending)
    classes = sorted(pcs.keys(), key=lambda c: sum(pcs[c].values()), reverse=True)

    train_counts = [pcs[c]["train"] for c in classes]
    val_counts = [pcs[c]["val"] for c in classes]
    test_counts = [pcs[c]["test"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))

    bars_train = ax.bar(x - width, train_counts, width, label="Train", color="#4477AA", alpha=0.85)
    bars_val = ax.bar(x, val_counts, width, label="Val", color="#44AA77", alpha=0.85)
    bars_test = ax.bar(x + width, test_counts, width, label="Test", color="#DD7733", alpha=0.85)

    # Rare-class threshold line
    ax.axhline(y=RARE_THRESHOLD, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"Rare threshold (N={RARE_THRESHOLD})")

    ax.set_yscale("log")
    ax.set_xlabel("Glitch Class", fontsize=12)
    ax.set_ylabel("Sample Count (log scale)", fontsize=12)
    ax.set_title("Per-Class Sample Distribution Across Temporal Splits", fontsize=14, fontweight="bold")

    # Format class names for display
    display_names = [c.replace("_", " ") for c in classes]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=55, ha="right", fontsize=9)

    ax.legend(fontsize=11, loc="upper right")

    # Annotation: total per split
    total_train = sum(train_counts)
    total_val = sum(val_counts)
    total_test = sum(test_counts)
    annotation = (
        f"Train: {total_train:,} ({total_train/(total_train+total_val+total_test):.0%})\n"
        f"Val: {total_val:,} ({total_val/(total_train+total_val+total_test):.0%})\n"
        f"Test: {total_test:,} ({total_test/(total_train+total_val+total_test):.0%})"
    )
    ax.text(0.02, 0.97, annotation, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))

    ax.set_ylim(bottom=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
