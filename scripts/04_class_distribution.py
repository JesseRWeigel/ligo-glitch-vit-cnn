#!/usr/bin/env python3
"""
Compute per-class distribution from filtered metadata and downloaded spectrograms.

Produces:
    data/metadata/class_distribution_raw.json   -- Per-class counts, rare flags, coverage
    figures/class_distribution_o3.png            -- Bar chart with color-coded rarity

Modes:
    --metadata-only   Use metadata counts directly (before download completes)
    (default)         Cross-reference with download_progress.json for coverage

Usage:
    python scripts/04_class_distribution.py [--metadata-only]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "gravity_spy_o3_filtered.csv"
SPECTROGRAMS_DIR = PROJECT_ROOT / "data" / "spectrograms"
PROGRESS_FILE = SPECTROGRAMS_DIR / "download_progress.json"
DISTRIBUTION_JSON = PROJECT_ROOT / "data" / "metadata" / "class_distribution_raw.json"
FIGURE_PATH = PROJECT_ROOT / "figures" / "class_distribution_o3.png"

RARE_THRESHOLD = 25
HIGH_RISK_THRESHOLD = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_completed_ids() -> set:
    """Get set of completed gravityspy_ids from download progress JSON."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        return set(progress.get("completed_ids", []))
    return set()


def compute_distribution(df: pd.DataFrame, metadata_only: bool = False) -> dict:
    """Compute per-class distribution with optional spectrogram coverage."""

    total_meta = len(df)
    all_classes = sorted(df["ml_label"].unique())
    meta_counts = df["ml_label"].value_counts().to_dict()
    per_class_meta_counts = {c: meta_counts.get(c, 0) for c in all_classes}

    if metadata_only:
        # Use metadata counts as the usable counts (assume all will be downloaded)
        logger.info("Metadata-only mode: using metadata counts as usable counts")
        completed_ids = set(df["gravityspy_id"])
        usable_df = df
        coverage_pct = 100.0
        note = "metadata_only_mode: spectrogram coverage not yet verified"
    else:
        # Cross-reference with download progress
        completed_ids = get_completed_ids()
        logger.info(f"Download progress: {len(completed_ids)} completed IDs")
        usable_df = df[df["gravityspy_id"].isin(completed_ids)]
        coverage_pct = len(usable_df) / total_meta * 100 if total_meta > 0 else 0
        note = None

    total_with_specs = len(usable_df)
    logger.info(f"Usable samples: {total_with_specs}/{total_meta} ({coverage_pct:.1f}%)")

    usable_counts = usable_df["ml_label"].value_counts().to_dict()
    per_class_counts = {c: usable_counts.get(c, 0) for c in all_classes}

    # Per-detector counts
    per_detector_counts = {}
    for (ifo, label), group in usable_df.groupby(["ifo", "ml_label"]):
        per_detector_counts[f"{ifo}_{label}"] = int(len(group))

    # Rare classes (based on usable counts)
    rare_classes = [c for c, n in per_class_counts.items() if 0 < n < RARE_THRESHOLD]
    high_risk_classes = [c for c, n in per_class_counts.items() if 0 < n < HIGH_RISK_THRESHOLD]
    zero_classes = [c for c, n in per_class_counts.items() if n == 0]

    distribution = {
        "per_class_counts": per_class_counts,
        "per_class_meta_counts": per_class_meta_counts,
        "total_samples": int(total_with_specs),
        "total_metadata_samples": total_meta,
        "per_detector_counts": per_detector_counts,
        "rare_class_threshold": RARE_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "rare_classes": sorted(rare_classes),
        "high_risk_classes": sorted(high_risk_classes),
        "zero_classes": sorted(zero_classes),
        "spectrogram_coverage": round(coverage_pct, 2),
        "n_failed_glitches": total_meta - int(total_with_specs),
        "n_classes": len(all_classes),
        "per_detector_totals": {
            "H1": int(usable_df[usable_df["ifo"] == "H1"].shape[0]),
            "L1": int(usable_df[usable_df["ifo"] == "L1"].shape[0]),
        },
    }
    if note:
        distribution["note"] = note

    return distribution


def make_bar_chart(distribution: dict, save_path: Path):
    """Generate color-coded class distribution bar chart."""

    counts = distribution["per_class_counts"]
    rare = set(distribution["rare_classes"])
    high_risk = set(distribution["high_risk_classes"])

    # Sort descending
    sorted_classes = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
    sorted_counts = [counts[c] for c in sorted_classes]

    # Color coding
    colors = []
    for c in sorted_classes:
        if c in high_risk:
            colors.append("#d62728")  # red
        elif c in rare:
            colors.append("#ff7f0e")  # orange
        else:
            colors.append("#2ca02c")  # green

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(range(len(sorted_classes)), sorted_counts, color=colors,
                  edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Number of Samples (log scale)", fontsize=13)
    ax.set_xlabel("Glitch Class", fontsize=13)
    ax.set_title(
        "Gravity Spy O3 Class Distribution\n(ml_confidence > 0.9, None_of_the_Above excluded)",
        fontsize=14, fontweight="bold",
    )

    ax.set_xticks(range(len(sorted_classes)))
    ax.set_xticklabels(sorted_classes, rotation=45, ha="right", fontsize=10)

    # Count labels on bars
    for bar, count in zip(bars, sorted_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                    f"{count:,}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label=f"Common (>= {RARE_THRESHOLD})"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label=f"Rare (< {RARE_THRESHOLD})"),
        Patch(facecolor="#d62728", edgecolor="black", label=f"High Risk (< {HIGH_RISK_THRESHOLD})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    # Annotation box
    total = distribution["total_samples"]
    n_rare = len(distribution["rare_classes"])
    n_high_risk = len(distribution["high_risk_classes"])
    cov = distribution["spectrogram_coverage"]
    n_classes = distribution["n_classes"]

    annotation = (
        f"Total usable samples: {total:,}\n"
        f"Classes: {n_classes}\n"
        f"Rare classes (< {RARE_THRESHOLD}): {n_rare}\n"
        f"High-risk classes (< {HIGH_RISK_THRESHOLD}): {n_high_risk}\n"
        f"Spectrogram coverage: {cov:.1f}%"
    )
    ax.text(0.98, 0.55, annotation, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.5, len(sorted_classes) - 0.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Bar chart saved to {save_path}")


def print_class_summary(distribution: dict):
    """Print per-class summary sorted ascending (rare classes first)."""

    counts = distribution["per_class_counts"]
    rare = set(distribution["rare_classes"])
    high_risk = set(distribution["high_risk_classes"])
    zero = set(distribution["zero_classes"])

    sorted_classes = sorted(counts.keys(), key=lambda c: counts[c])

    print("\n" + "=" * 60)
    print("Per-Class Distribution (sorted ascending)")
    print("=" * 60)

    for c in sorted_classes:
        n = counts[c]
        flag = ""
        if c in zero:
            flag = " *** CRITICAL: ZERO SAMPLES ***"
        elif c in high_risk:
            flag = " ** HIGH RISK **"
        elif c in rare:
            flag = " * RARE *"
        print(f"  {c:30s}: {n:>8,}{flag}")

    print("-" * 60)
    print(f"  {'TOTAL':30s}: {distribution['total_samples']:>8,}")
    print(f"  Coverage: {distribution['spectrogram_coverage']:.1f}%")
    print(f"  H1: {distribution['per_detector_totals']['H1']:,}, "
          f"L1: {distribution['per_detector_totals']['L1']:,}")
    print("=" * 60)

    if zero:
        logger.critical(f"ZERO-sample classes: {zero}")
    if high_risk:
        logger.warning(f"High-risk classes (< {HIGH_RISK_THRESHOLD}): {high_risk}")
    if rare:
        logger.warning(f"Rare classes (< {RARE_THRESHOLD}): {rare}")


def main():
    parser = argparse.ArgumentParser(description="Compute class distribution")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Use metadata counts only (skip spectrogram coverage check)")
    args = parser.parse_args()

    logger.info("Loading metadata...")
    df = pd.read_csv(METADATA_CSV)
    logger.info(f"Loaded {len(df)} glitches across {df['ml_label'].nunique()} classes")

    # Verify 23 classes
    expected_classes = 23
    actual_classes = df["ml_label"].nunique()
    assert actual_classes == expected_classes, (
        f"Expected {expected_classes} classes, got {actual_classes}"
    )

    distribution = compute_distribution(df, metadata_only=args.metadata_only)

    # Verify sum
    total_from_counts = sum(distribution["per_class_counts"].values())
    assert total_from_counts == distribution["total_samples"], (
        f"Sum mismatch: {total_from_counts} != {distribution['total_samples']}"
    )

    # Save JSON
    DISTRIBUTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DISTRIBUTION_JSON, "w") as f:
        json.dump(distribution, f, indent=2)
    logger.info(f"Distribution saved to {DISTRIBUTION_JSON}")

    # Generate bar chart
    make_bar_chart(distribution, FIGURE_PATH)

    # Print summary
    print_class_summary(distribution)

    # Verify required fields
    required_fields = [
        "per_class_counts", "total_samples", "rare_class_threshold",
        "rare_classes", "spectrogram_coverage",
    ]
    for field in required_fields:
        assert field in distribution, f"Missing required field: {field}"

    logger.info("Class distribution computation complete.")


if __name__ == "__main__":
    main()
