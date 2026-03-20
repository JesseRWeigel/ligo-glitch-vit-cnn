#!/usr/bin/env python3
"""
Temporal train/val/test split for Gravity Spy O3 glitch classification.

Splits by GPS event_time with >= 60s gaps between train/val and val/test
to prevent temporal data leakage from correlated instrument states.

Input:  data/metadata/gravity_spy_o3_filtered.csv (Plan 01-01 output)
Output: data/metadata/{train,val,test}_manifest.csv
        data/metadata/split_statistics.json

Convention: SI units, GPS time in seconds, macro-F1 primary metric.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GAP_SECONDS = 60  # minimum temporal gap between splits
TRAIN_FRAC_TARGET = 0.70
VAL_FRAC_TARGET = 0.15
# test gets the remainder (~0.15)

# Acceptable ranges for split fractions
TRAIN_RANGE = (0.60, 0.80)
VAL_RANGE = (0.08, 0.25)
TEST_RANGE = (0.08, 0.25)

METADATA_CSV = Path("data/metadata/gravity_spy_o3_filtered.csv")
OUTPUT_DIR = Path("data/metadata")

# Duration view mapping: url1=0.5s, url2=1.0s, url3=2.0s, url4=4.0s
DURATION_MAP = {
    "url1": "0.5",
    "url2": "1.0",
    "url3": "2.0",
    "url4": "4.0",
}


def build_image_path(row: pd.Series, duration_key: str) -> str:
    """Construct the local spectrogram path for a given duration view."""
    label = row["ml_label"]
    gid = row["gravityspy_id"]
    dur = DURATION_MAP[duration_key]
    return f"data/spectrograms/{label}/{gid}_{dur}s.png"


def compute_split_boundaries(event_times: np.ndarray):
    """
    Compute GPS split boundaries with >= 60s gaps.

    Strategy:
    1. Sort all event times.
    2. Find the 70th and 85th percentile times.
    3. Place train_end just before the 70th percentile boundary.
    4. Place val_start at first event > train_end + GAP_SECONDS.
    5. Place val_end just before the 85th percentile boundary.
    6. Place test_start at first event > val_end + GAP_SECONDS.
    """
    sorted_times = np.sort(event_times)
    n = len(sorted_times)

    # Percentile boundaries
    idx_70 = int(n * TRAIN_FRAC_TARGET) - 1
    idx_85 = int(n * (TRAIN_FRAC_TARGET + VAL_FRAC_TARGET)) - 1

    boundary_70 = sorted_times[idx_70]
    boundary_85 = sorted_times[idx_85]

    # Train: all events <= boundary_70
    train_end = boundary_70

    # Val start: first event that is > train_end + GAP
    val_candidates = sorted_times[sorted_times > train_end + GAP_SECONDS]
    if len(val_candidates) == 0:
        raise ValueError("No events found after train_end + gap. Data too sparse.")
    val_start = val_candidates[0]

    # Val end: last event <= boundary_85
    val_end = boundary_85

    # If val_end < val_start, we need to extend val_end
    if val_end < val_start:
        # Push val_end forward to include at least some val samples
        val_mask = (sorted_times >= val_start)
        available = sorted_times[val_mask]
        # Take enough to approximate 15% of total
        target_val_count = int(n * VAL_FRAC_TARGET)
        if len(available) < target_val_count:
            val_end = available[-1]
        else:
            val_end = available[target_val_count - 1]

    # Test start: first event > val_end + GAP
    test_candidates = sorted_times[sorted_times > val_end + GAP_SECONDS]
    if len(test_candidates) == 0:
        raise ValueError("No events found after val_end + gap. Data too sparse.")
    test_start = test_candidates[0]

    return {
        "train_end_gps": float(train_end),
        "val_start_gps": float(val_start),
        "val_end_gps": float(val_end),
        "test_start_gps": float(test_start),
    }


def assign_splits(df: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
    """Assign split labels based on GPS boundaries. Samples in gaps are excluded."""
    t = df["event_time"].values

    train_mask = t <= boundaries["train_end_gps"]
    val_mask = (t >= boundaries["val_start_gps"]) & (t <= boundaries["val_end_gps"])
    test_mask = t >= boundaries["test_start_gps"]

    df = df.copy()
    df["split"] = "gap_excluded"
    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"

    return df


def check_class_coverage(df: pd.DataFrame, all_classes: list) -> dict:
    """Check per-class representation across splits. Return coverage info."""
    per_class_per_split = {}
    for cls in sorted(all_classes):
        counts = {}
        for split in ["train", "val", "test"]:
            counts[split] = int(((df["split"] == split) & (df["ml_label"] == cls)).sum())
        per_class_per_split[cls] = counts

    zero_test = [cls for cls, c in per_class_per_split.items() if c["test"] == 0]
    lt5_test = [cls for cls, c in per_class_per_split.items() if 0 < c["test"] < 5]

    return {
        "per_class_per_split": per_class_per_split,
        "classes_with_zero_test": zero_test,
        "classes_with_lt5_test": lt5_test,
    }


def attempt_boundary_fix_for_missing_classes(
    df: pd.DataFrame, boundaries: dict, missing_classes: list
) -> dict:
    """
    If a class has 0 test samples, try shifting val_end/test_start to capture at least 1.
    Returns updated boundaries.
    """
    if not missing_classes:
        return boundaries

    print(f"\n[BOUNDARY FIX] Attempting to fix {len(missing_classes)} classes with 0 test samples: {missing_classes}")

    for cls in missing_classes:
        cls_times = df.loc[df["ml_label"] == cls, "event_time"].values
        max_cls_time = np.max(cls_times)

        if max_cls_time <= boundaries["val_end_gps"]:
            print(f"  {cls}: all samples before val_end ({max_cls_time:.1f} <= {boundaries['val_end_gps']:.1f})")
            print(f"  -> Cannot fix by boundary shift; class is temporally clustered before test window.")
            continue

        # Check if any samples exist after current test_start
        in_test = cls_times[cls_times >= boundaries["test_start_gps"]]
        if len(in_test) > 0:
            print(f"  {cls}: {len(in_test)} samples already in test window -- something else is wrong")
            continue

        # Samples exist between val_end and test_start (in the gap)
        in_gap = cls_times[
            (cls_times > boundaries["val_end_gps"])
            & (cls_times < boundaries["test_start_gps"])
        ]
        if len(in_gap) > 0:
            print(f"  {cls}: {len(in_gap)} samples in val-test gap. Adjusting test_start downward.")
            new_test_start = np.min(in_gap)
            # Ensure gap from val_end is still >= 60s
            if new_test_start - boundaries["val_end_gps"] >= GAP_SECONDS:
                boundaries["test_start_gps"] = float(new_test_start)
                print(f"  -> New test_start: {new_test_start:.1f} (gap: {new_test_start - boundaries['val_end_gps']:.1f}s)")
            else:
                print(f"  -> Cannot shift: would violate 60s gap requirement")

    return boundaries


def main():
    print("=" * 70)
    print("TEMPORAL SPLIT: Gravity Spy O3 Glitch Classification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    df = pd.read_csv(METADATA_CSV)
    n_total = len(df)
    all_classes = sorted(df["ml_label"].unique().tolist())
    n_classes = len(all_classes)
    print(f"\nLoaded {n_total:,} samples across {n_classes} classes")
    print(f"GPS range: [{df['event_time'].min():.2f}, {df['event_time'].max():.2f}]")

    # ------------------------------------------------------------------
    # Compute boundaries
    # ------------------------------------------------------------------
    boundaries = compute_split_boundaries(df["event_time"].values)
    print(f"\nInitial boundaries:")
    for k, v in boundaries.items():
        print(f"  {k}: {v:.2f}")

    # ------------------------------------------------------------------
    # Assign splits
    # ------------------------------------------------------------------
    df = assign_splits(df, boundaries)

    # ------------------------------------------------------------------
    # Check class coverage
    # ------------------------------------------------------------------
    coverage = check_class_coverage(df, all_classes)
    missing = coverage["classes_with_zero_test"]

    if missing:
        print(f"\nWARNING: {len(missing)} classes have 0 test samples: {missing}")
        boundaries = attempt_boundary_fix_for_missing_classes(df, boundaries, missing)
        df = assign_splits(df, boundaries)
        coverage = check_class_coverage(df, all_classes)
        missing = coverage["classes_with_zero_test"]
        if missing:
            print(f"STILL MISSING after fix attempt: {missing}")

    # ------------------------------------------------------------------
    # Split statistics
    # ------------------------------------------------------------------
    split_counts = {}
    for split in ["train", "val", "test", "gap_excluded"]:
        split_counts[split] = int((df["split"] == split).sum())

    n_assigned = split_counts["train"] + split_counts["val"] + split_counts["test"]
    split_fracs = {
        "train": split_counts["train"] / n_assigned,
        "val": split_counts["val"] / n_assigned,
        "test": split_counts["test"] / n_assigned,
    }

    print(f"\nSplit counts (excluding {split_counts['gap_excluded']} gap-excluded):")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {split_counts[split]:>8,} ({split_fracs[split]:.1%})")
    print(f"  gap_excluded: {split_counts['gap_excluded']:>8,}")
    print(f"  total: {n_total:>8,}")

    # ------------------------------------------------------------------
    # Validate split ratios
    # ------------------------------------------------------------------
    ratio_ok = (
        TRAIN_RANGE[0] <= split_fracs["train"] <= TRAIN_RANGE[1]
        and VAL_RANGE[0] <= split_fracs["val"] <= VAL_RANGE[1]
        and TEST_RANGE[0] <= split_fracs["test"] <= TEST_RANGE[1]
    )
    if not ratio_ok:
        print(f"\nWARNING: Split ratios outside acceptable range!")
        print(f"  train: {split_fracs['train']:.3f} (range: {TRAIN_RANGE})")
        print(f"  val:   {split_fracs['val']:.3f} (range: {VAL_RANGE})")
        print(f"  test:  {split_fracs['test']:.3f} (range: {TEST_RANGE})")

    # ------------------------------------------------------------------
    # Add image path columns
    # ------------------------------------------------------------------
    for url_col, dur in DURATION_MAP.items():
        col_name = f"image_path_{dur}s"
        df[col_name] = df.apply(lambda row: build_image_path(row, url_col), axis=1)

    # ------------------------------------------------------------------
    # Save manifests
    # ------------------------------------------------------------------
    manifest_cols = [
        "gravityspy_id", "event_time", "ifo", "ml_label", "ml_confidence", "snr",
        "image_path_0.5s", "image_path_1.0s", "image_path_2.0s", "image_path_4.0s",
        "split",
    ]

    for split in ["train", "val", "test"]:
        split_df = df.loc[df["split"] == split, manifest_cols].copy()
        out_path = OUTPUT_DIR / f"{split}_manifest.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}: {len(split_df):,} rows")

    # ------------------------------------------------------------------
    # Save split statistics JSON
    # ------------------------------------------------------------------
    rare_test_coverage = {
        "classes_with_zero_test": coverage["classes_with_zero_test"],
        "classes_with_lt5_test": coverage["classes_with_lt5_test"],
        "mitigation": (
            "Classes with < 5 test samples will have per-class metrics reported "
            "but excluded from rare-class macro-F1 with documentation."
            if coverage["classes_with_lt5_test"]
            else "All classes have >= 5 test samples."
        ),
    }

    # We will fill verification_results after running 07_verify_split.py
    stats = {
        "split_boundaries": boundaries,
        "gap_seconds": GAP_SECONDS,
        "samples_excluded_in_gaps": split_counts["gap_excluded"],
        "total_samples_input": n_total,
        "total_samples_assigned": n_assigned,
        "split_counts": {k: v for k, v in split_counts.items() if k != "gap_excluded"},
        "split_fractions": {k: round(v, 6) for k, v in split_fracs.items()},
        "per_class_per_split": coverage["per_class_per_split"],
        "rare_class_test_coverage": rare_test_coverage,
        "verification_results": {},  # populated by 07_verify_split.py
    }

    stats_path = OUTPUT_DIR / "split_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved {stats_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SPLIT COMPLETE")
    print(f"  Train: {split_counts['train']:,} | Val: {split_counts['val']:,} | Test: {split_counts['test']:,}")
    print(f"  Gap-excluded: {split_counts['gap_excluded']:,}")
    print(f"  Classes with 0 test: {coverage['classes_with_zero_test']}")
    print(f"  Classes with <5 test: {coverage['classes_with_lt5_test']}")
    print(f"{'=' * 70}")

    if coverage["classes_with_zero_test"]:
        print("\nERROR: Some classes have 0 test samples. Manual review needed.")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
