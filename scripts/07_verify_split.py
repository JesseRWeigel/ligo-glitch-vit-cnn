#!/usr/bin/env python3
"""
Programmatic verification of temporal train/val/test split.

Runs 5 mandatory checks:
1. Temporal gap: all test samples >= 60s from nearest train sample
2. Temporal gap: all val samples >= 60s from nearest train sample
3. No ID overlap across splits
4. No GPS time range overlap (with 60s gap)
5. Class coverage: all 23 classes have >= 1 test sample
6. Split ratio within acceptable bounds

Uses binary search (O(N log N)) for gap verification.

Convention: SI units, GPS time in seconds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GAP_SECONDS = 60
EXPECTED_CLASSES = 23

TRAIN_RANGE = (0.60, 0.80)
VAL_RANGE = (0.08, 0.25)
TEST_RANGE = (0.08, 0.25)

MANIFEST_DIR = Path("data/metadata")
STATS_PATH = MANIFEST_DIR / "split_statistics.json"


def check_temporal_gap(times_a: np.ndarray, times_b: np.ndarray, label_a: str, label_b: str) -> dict:
    """
    Check that every sample in times_a is >= GAP_SECONDS from nearest sample in times_b.
    Uses sorted array + binary search for O(N log N).

    Returns dict with pass/fail, min_gap, n_violations.
    """
    sorted_b = np.sort(times_b)
    n = len(times_a)

    min_gaps = np.empty(n)
    for i, t in enumerate(times_a):
        idx = np.searchsorted(sorted_b, t)
        # Check neighbors
        candidates = []
        if idx > 0:
            candidates.append(abs(t - sorted_b[idx - 1]))
        if idx < len(sorted_b):
            candidates.append(abs(t - sorted_b[idx]))
        min_gaps[i] = min(candidates) if candidates else float("inf")

    violations = int(np.sum(min_gaps < GAP_SECONDS))
    min_gap = float(np.min(min_gaps)) if len(min_gaps) > 0 else float("inf")

    passed = violations == 0
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Temporal gap ({label_a} vs {label_b}): "
          f"min_gap={min_gap:.2f}s, violations={violations}/{n}")

    return {
        "status": status,
        "min_gap_seconds": round(min_gap, 4),
        "violations": violations,
        "total_checked": n,
    }


def check_no_id_overlap(train_ids, val_ids, test_ids) -> dict:
    """Check that no gravityspy_id appears in more than one split."""
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    n_overlap = len(tv) + len(tt) + len(vt)
    passed = n_overlap == 0
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] No ID overlap: "
          f"train-val={len(tv)}, train-test={len(tt)}, val-test={len(vt)}")

    return {
        "status": status,
        "train_val_overlap": len(tv),
        "train_test_overlap": len(tt),
        "val_test_overlap": len(vt),
    }


def check_no_time_range_overlap(train_times, val_times, test_times) -> dict:
    """Check that GPS time ranges don't overlap (with 60s gap)."""
    train_max = float(np.max(train_times))
    val_min = float(np.min(val_times))
    val_max = float(np.max(val_times))
    test_min = float(np.min(test_times))

    gap_train_val = val_min - train_max
    gap_val_test = test_min - val_max

    passed = (gap_train_val >= GAP_SECONDS) and (gap_val_test >= GAP_SECONDS)
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] No time range overlap: "
          f"train_max={train_max:.2f}, val_min={val_min:.2f} (gap={gap_train_val:.2f}s), "
          f"val_max={val_max:.2f}, test_min={test_min:.2f} (gap={gap_val_test:.2f}s)")

    return {
        "status": status,
        "train_max_gps": train_max,
        "val_min_gps": val_min,
        "gap_train_val_seconds": round(gap_train_val, 4),
        "val_max_gps": val_max,
        "test_min_gps": test_min,
        "gap_val_test_seconds": round(gap_val_test, 4),
    }


def check_class_coverage(train_df, val_df, test_df) -> dict:
    """Check that all 23 classes have >= 1 sample in each split."""
    train_classes = set(train_df["ml_label"].unique())
    val_classes = set(val_df["ml_label"].unique())
    test_classes = set(test_df["ml_label"].unique())
    all_classes = train_classes | val_classes | test_classes

    missing_test = sorted(all_classes - test_classes)
    missing_val = sorted(all_classes - val_classes)
    missing_train = sorted(all_classes - train_classes)

    n_test_classes = len(test_classes)
    passed = (n_test_classes >= EXPECTED_CLASSES) and len(missing_test) == 0
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] Class coverage: "
          f"train={len(train_classes)}, val={len(val_classes)}, test={n_test_classes} classes")
    if missing_test:
        print(f"    Missing from test: {missing_test}")
    if missing_val:
        print(f"    Missing from val: {missing_val}")
    if missing_train:
        print(f"    Missing from train: {missing_train}")

    return {
        "status": status,
        "n_train_classes": len(train_classes),
        "n_val_classes": len(val_classes),
        "n_test_classes": n_test_classes,
        "missing_from_test": missing_test,
        "missing_from_val": missing_val,
        "missing_from_train": missing_train,
    }


def check_split_ratios(n_train, n_val, n_test) -> dict:
    """Check split fractions are within acceptable bounds."""
    total = n_train + n_val + n_test
    fracs = {
        "train": n_train / total,
        "val": n_val / total,
        "test": n_test / total,
    }

    train_ok = TRAIN_RANGE[0] <= fracs["train"] <= TRAIN_RANGE[1]
    val_ok = VAL_RANGE[0] <= fracs["val"] <= VAL_RANGE[1]
    test_ok = TEST_RANGE[0] <= fracs["test"] <= TEST_RANGE[1]

    passed = train_ok and val_ok and test_ok
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] Split ratios: "
          f"train={fracs['train']:.3f} [{TRAIN_RANGE}], "
          f"val={fracs['val']:.3f} [{VAL_RANGE}], "
          f"test={fracs['test']:.3f} [{TEST_RANGE}]")

    return {
        "status": status,
        "fractions": {k: round(v, 6) for k, v in fracs.items()},
        "counts": {"train": n_train, "val": n_val, "test": n_test, "total_assigned": total},
    }


def main():
    print("=" * 70)
    print("SPLIT VERIFICATION: Temporal Gap & Coverage Checks")
    print("=" * 70)

    # Load manifests
    train_df = pd.read_csv(MANIFEST_DIR / "train_manifest.csv")
    val_df = pd.read_csv(MANIFEST_DIR / "val_manifest.csv")
    test_df = pd.read_csv(MANIFEST_DIR / "test_manifest.csv")

    print(f"\nLoaded: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    train_times = train_df["event_time"].values
    val_times = val_df["event_time"].values
    test_times = test_df["event_time"].values

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------
    print("\nRunning verification checks...\n")

    # Check 1: Temporal gap test vs train
    gap_test_train = check_temporal_gap(test_times, train_times, "test", "train")

    # Check 2: Temporal gap val vs train
    gap_val_train = check_temporal_gap(val_times, train_times, "val", "train")

    # Check 3: No ID overlap
    id_overlap = check_no_id_overlap(
        train_df["gravityspy_id"], val_df["gravityspy_id"], test_df["gravityspy_id"]
    )

    # Check 4: No time range overlap
    time_overlap = check_no_time_range_overlap(train_times, val_times, test_times)

    # Check 5: Class coverage
    class_cov = check_class_coverage(train_df, val_df, test_df)

    # Check 6: Split ratios
    ratio_check = check_split_ratios(len(train_df), len(val_df), len(test_df))

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    verification_results = {
        "temporal_gap_test_vs_train": gap_test_train["status"],
        "temporal_gap_val_vs_train": gap_val_train["status"],
        "no_id_overlap": id_overlap["status"],
        "no_time_range_overlap": time_overlap["status"],
        "class_coverage": class_cov["status"],
        "split_ratio": ratio_check["status"],
    }

    all_pass = all(v == "PASS" for v in verification_results.values())

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    for k, v in verification_results.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Update split_statistics.json with verification results
    # ------------------------------------------------------------------
    with open(STATS_PATH) as f:
        stats = json.load(f)

    stats["verification_results"] = verification_results
    stats["verification_details"] = {
        "temporal_gap_test_vs_train": gap_test_train,
        "temporal_gap_val_vs_train": gap_val_train,
        "no_id_overlap": id_overlap,
        "no_time_range_overlap": time_overlap,
        "class_coverage": class_cov,
        "split_ratio": ratio_check,
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nUpdated {STATS_PATH} with verification results")

    # Also verify total count consistency
    original_csv = pd.read_csv("data/metadata/gravity_spy_o3_filtered.csv")
    n_original = len(original_csv)
    n_assigned = len(train_df) + len(val_df) + len(test_df)
    n_gap = n_original - n_assigned
    print(f"\nSample accounting: {n_original:,} total = {n_assigned:,} assigned + {n_gap:,} gap-excluded")

    if n_gap != stats.get("samples_excluded_in_gaps", -1):
        print(f"  WARNING: gap count mismatch with split_statistics.json "
              f"({n_gap} vs {stats.get('samples_excluded_in_gaps')})")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
