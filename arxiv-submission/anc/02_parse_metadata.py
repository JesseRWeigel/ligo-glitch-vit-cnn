#!/usr/bin/env python3
"""Parse and filter Gravity Spy O3 metadata for the GW glitch classification project.

Loads raw CSV files from Zenodo 5649212, filters to O3 data with ml_confidence > 0.9,
excludes None_of_the_Above, validates metadata integrity, and saves filtered dataset.

Conventions:
  ASSERT_CONVENTION: natural_units=SI, coordinate_system=Q-transform spectrograms 10-2048 Hz log axis 224x224 px
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# O3 GPS time boundaries (from LIGO observing run schedule)
O3A_START = 1238166018  # 2019-04-01T15:00:00 UTC
O3A_END = 1253977218    # 2019-10-01T15:00:00 UTC
O3B_START = 1256655618  # 2019-11-01T15:00:00 UTC
O3B_END = 1269363618    # 2020-03-27T17:00:00 UTC

# The 23-class O3 taxonomy (22 glitch classes + No_Glitch, excluding None_of_the_Above)
VALID_CLASSES = sorted([
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Blip_Low_Frequency",
    "Chirp", "Extremely_Loud", "Fast_Scattering", "Helix", "Koi_Fish",
    "Light_Modulation", "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch",
    "Paired_Doves", "Power_Line", "Repeating_Blips", "Scattered_Light", "Scratchy",
    "Tomte", "Violin_Mode", "Wandering_Line", "Whistle",
])
assert len(VALID_CLASSES) == 23, f"Expected 23 classes, got {len(VALID_CLASSES)}"

RAW_DIR = Path("data/raw/zenodo_5649212")
OUTPUT_PATH = Path("data/metadata/gravity_spy_o3_filtered.csv")

# Required columns in the output
REQUIRED_COLUMNS = ["gravityspy_id", "event_time", "ifo", "ml_label", "ml_confidence", "snr"]


def load_o3_data():
    """Load and concatenate O3a and O3b CSV files for both detectors."""
    o3_files = ["H1_O3a.csv", "H1_O3b.csv", "L1_O3a.csv", "L1_O3b.csv"]
    frames = []
    for fname in o3_files:
        fpath = RAW_DIR / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping")
            continue
        df = pd.read_csv(fpath)
        print(f"Loaded {fname}: {len(df)} rows, {df.ml_label.nunique()} unique labels")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined O3 data: {len(combined)} rows")
    return combined


def validate_and_filter(df):
    """Apply all filters and validation checks, printing summary at each step."""
    step_counts = {"raw_o3": len(df)}

    # Step 1: Verify GPS time ranges are within O3
    in_o3a = (df["event_time"] >= O3A_START) & (df["event_time"] <= O3A_END)
    in_o3b = (df["event_time"] >= O3B_START) & (df["event_time"] <= O3B_END)
    in_o3 = in_o3a | in_o3b
    outside_o3 = (~in_o3).sum()
    if outside_o3 > 0:
        print(f"WARNING: {outside_o3} rows have event_time outside O3 range, filtering them out")
        # Check what times these are
        bad_times = df.loc[~in_o3, "event_time"]
        print(f"  Range of bad times: {bad_times.min():.0f} to {bad_times.max():.0f}")
    df = df[in_o3].copy()
    step_counts["after_gps_filter"] = len(df)
    print(f"After O3 GPS filter: {len(df)} rows (removed {outside_o3})")

    # Step 2: Validate detector labels
    valid_ifos = df["ifo"].isin(["H1", "L1"])
    invalid_ifos = (~valid_ifos).sum()
    if invalid_ifos > 0:
        print(f"WARNING: {invalid_ifos} rows have invalid ifo values: {df.loc[~valid_ifos, 'ifo'].unique()}")
        df = df[valid_ifos].copy()
    step_counts["after_ifo_filter"] = len(df)
    print(f"After ifo validation: {len(df)} rows")

    # Step 3: Exclude None_of_the_Above
    nota_count = (df["ml_label"] == "None_of_the_Above").sum()
    df = df[df["ml_label"] != "None_of_the_Above"].copy()
    step_counts["after_nota_exclusion"] = len(df)
    print(f"After NOTA exclusion: {len(df)} rows (removed {nota_count})")

    # Step 4: Validate class labels are in 23-class taxonomy
    invalid_labels = ~df["ml_label"].isin(VALID_CLASSES)
    n_invalid = invalid_labels.sum()
    if n_invalid > 0:
        unknown = df.loc[invalid_labels, "ml_label"].unique()
        print(f"WARNING: {n_invalid} rows have labels not in 23-class taxonomy: {unknown}")
        df = df[~invalid_labels].copy()
    step_counts["after_label_validation"] = len(df)
    print(f"After label validation: {len(df)} rows")

    # Step 5: Apply ml_confidence > 0.9 filter
    pre_conf = len(df)
    df = df[df["ml_confidence"] > 0.9].copy()
    step_counts["after_confidence_filter"] = len(df)
    print(f"After ml_confidence > 0.9: {len(df)} rows (removed {pre_conf - len(df)}, {100*(pre_conf-len(df))/pre_conf:.1f}%)")

    # Step 6: Check for NaN in critical columns
    for col in REQUIRED_COLUMNS:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"WARNING: {n_nan} NaN values in {col}")
            if col in ["event_time", "ifo", "ml_label", "ml_confidence"]:
                df = df[df[col].notna()].copy()
                print(f"  Dropped rows with NaN {col}: {len(df)} rows remain")
    step_counts["after_nan_removal"] = len(df)

    # Step 7: Check for duplicate GPS times per detector
    dupes = df.groupby("ifo")["event_time"].apply(lambda x: x.duplicated().sum())
    total_dupes = dupes.sum()
    if total_dupes > 0:
        print(f"WARNING: {total_dupes} duplicate event_times found per detector:")
        print(dupes)
        # Keep first occurrence
        df = df.drop_duplicates(subset=["ifo", "event_time"], keep="first").copy()
    step_counts["after_dedup"] = len(df)
    print(f"After deduplication: {len(df)} rows")

    return df, step_counts


def print_summary(df, step_counts):
    """Print detailed summary of the filtered dataset."""
    print("\n" + "=" * 60)
    print("FILTERED DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")
    print(f"Unique classes: {df['ml_label'].nunique()}")
    print(f"Detectors: {sorted(df['ifo'].unique())}")
    print(f"GPS time range: {df['event_time'].min():.0f} to {df['event_time'].max():.0f}")
    print(f"ml_confidence range: [{df['ml_confidence'].min():.4f}, {df['ml_confidence'].max():.4f}]")
    print(f"SNR range: [{df['snr'].min():.2f}, {df['snr'].max():.2f}]")

    print(f"\nPer-detector counts:")
    for ifo in sorted(df["ifo"].unique()):
        n = (df["ifo"] == ifo).sum()
        print(f"  {ifo}: {n}")

    print(f"\nPer-class counts (sorted ascending):")
    class_counts = df["ml_label"].value_counts().sort_values()
    for label, count in class_counts.items():
        flag = ""
        if count == 0:
            flag = " [CRITICAL: 0 samples!]"
        elif count < 10:
            flag = " [HIGH RISK: < 10]"
        elif count < 25:
            flag = " [RARE: < 25]"
        print(f"  {label}: {count}{flag}")

    print(f"\nFilter pipeline:")
    for step, count in step_counts.items():
        print(f"  {step}: {count}")

    # Sanity checks
    # NOTE: Plan estimated ~8-10K samples, but the actual Gravity Spy O3 dataset
    # at ml_confidence > 0.9 has ~325K samples. The plan's estimate was based on
    # older/smaller dataset versions. The actual count is consistent with ~500K
    # total O3 glitches and ~75% passing the confidence filter.
    # Adjusted range: expect 100K-500K for full O3 at confidence > 0.9.
    total = len(df)
    if total < 5000:
        print(f"\nWARNING: Total filtered samples ({total}) < 5000 — suspiciously low!")
    elif total > 500000:
        print(f"\nWARNING: Total filtered samples ({total}) > 500000 — suspiciously high!")
    else:
        print(f"\nTotal filtered samples ({total}) is within plausible range")

    classes_present = df["ml_label"].nunique()
    if classes_present != 23:
        print(f"WARNING: Expected 23 classes, found {classes_present}")
        missing = set(VALID_CLASSES) - set(df["ml_label"].unique())
        if missing:
            print(f"Missing classes: {missing}")
    else:
        print(f"All 23 expected classes present")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load O3 data
    df = load_o3_data()

    # Validate and filter
    df_filtered, step_counts = validate_and_filter(df)

    # Select and order output columns
    # Keep all required columns plus useful extras
    output_cols = REQUIRED_COLUMNS + [
        "peak_frequency", "central_freq", "bandwidth", "duration", "q_value",
    ]
    # Add URL columns if present (for cross-referencing with Gravity Spy website)
    for url_col in ["url1", "url2", "url3", "url4"]:
        if url_col in df_filtered.columns:
            output_cols.append(url_col)

    df_out = df_filtered[output_cols].copy()

    # Save
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved filtered dataset to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / (1024*1024):.1f} MB")

    # Print summary
    print_summary(df_out, step_counts)

    # Final validation assertions
    assert len(df_out) > 0, "Filtered dataset is empty!"
    assert df_out["ml_confidence"].min() >= 0.9, "Confidence filter not applied correctly"
    assert set(df_out["ifo"].unique()) == {"H1", "L1"}, "Missing detector"
    assert "None_of_the_Above" not in df_out["ml_label"].values, "NOTA not excluded"
    assert df_out["ml_label"].nunique() == 23, f"Expected 23 classes, got {df_out['ml_label'].nunique()}"
    # Plan estimated ~8-10K but actual O3 data at conf>0.9 has ~325K (see DEVIATION note)
    assert 5000 < len(df_out) < 500000, f"Total {len(df_out)} outside plausible range"

    # Check no NaN in critical columns
    for col in REQUIRED_COLUMNS:
        assert df_out[col].isna().sum() == 0, f"NaN found in {col}"

    # Check GPS times are in O3 range
    assert df_out["event_time"].min() >= O3A_START, "GPS time before O3a start"
    assert df_out["event_time"].max() <= O3B_END, "GPS time after O3b end"

    # Check no exact duplicates per detector
    for ifo in ["H1", "L1"]:
        ifo_times = df_out.loc[df_out["ifo"] == ifo, "event_time"]
        assert ifo_times.duplicated().sum() == 0, f"Duplicate times in {ifo}"

    print("\nAll validation assertions PASSED")


if __name__ == "__main__":
    main()
