#!/usr/bin/env python3
"""
Fetch O4 Gravity Spy subjects from Zooniverse subject sets.

Subject sets are organized by class and observing run, e.g.:
  "Blip 'O4 (A)' 0.998 0.85" -> ML confidence 0.85-0.998
  "Blip 'O4 (B)' 1.0 0.998"  -> ML confidence 0.998-1.0

We fetch subjects from O4 (A) and (B) sets (high confidence)
to build the evaluation dataset.
"""

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "o4" / "metadata"
SPECTROGRAMS_DIR = PROJECT_ROOT / "data" / "o4" / "spectrograms"

BASE_URL = "https://www.zooniverse.org/api"
HEADERS = {
    "Accept": "application/vnd.api+json; version=1",
    "Content-Type": "application/json",
}

O3_CLASSES = sorted([
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip",
    "Blip_Low_Frequency", "Chirp", "Extremely_Loud", "Fast_Scattering",
    "Helix", "Koi_Fish", "Light_Modulation", "Low_Frequency_Burst",
    "Low_Frequency_Lines", "No_Glitch", "Paired_Doves", "Power_Line",
    "Repeating_Blips", "Scattered_Light", "Scratchy", "Tomte",
    "Violin_Mode", "Wandering_Line", "Whistle",
])

# Map display names to O3 class names
NAME_MAP = {
    "1080lines": "1080Lines",
    "1400ripples": "1400Ripples",
    "air compressor": "Air_Compressor",
    "blip": "Blip",
    "blip low frequency": "Blip_Low_Frequency",
    "chirp": "Chirp",
    "extremely loud": "Extremely_Loud",
    "fast scattering": "Fast_Scattering",
    "helix": "Helix",
    "koi fish": "Koi_Fish",
    "light modulation": "Light_Modulation",
    "low frequency burst": "Low_Frequency_Burst",
    "low frequency lines": "Low_Frequency_Lines",
    "no glitch": "No_Glitch",
    "paired doves": "Paired_Doves",
    "power line": "Power_Line",
    "repeating blips": "Repeating_Blips",
    "scattered light": "Scattered_Light",
    "scratchy": "Scratchy",
    "tomte": "Tomte",
    "violin mode": "Violin_Mode",
    "wandering line": "Wandering_Line",
    "whistle": "Whistle",
}

TARGET_SIZE = (224, 224)
PROGRESS_FILE = METADATA_DIR / "o4_fetch_sets_progress.json"


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"fetched_sets": [], "subjects": []}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, default=str))


def get_o4_subject_sets():
    """Get all O4 subject sets from Gravity Spy project."""
    all_sets = []
    page = 1

    while True:
        r = requests.get(f"{BASE_URL}/subject_sets",
                        params={"project_id": 1104, "page": page, "page_size": 100},
                        headers=HEADERS, timeout=30)
        if r.status_code != 200:
            break

        data = r.json()
        sets = data.get("subject_sets", [])
        if not sets:
            break

        all_sets.extend(sets)
        meta = data.get("meta", {}).get("subject_sets", {})
        if page >= meta.get("page_count", page):
            break
        page += 1
        time.sleep(0.3)

    # Filter to O4 sets
    o4_sets = []
    for s in all_sets:
        name = s.get("display_name", "")
        if "O4" in name and ("(A)" in name or "(B)" in name):
            # Parse class name from display_name
            # Format: "Blip 'O4 (A)' 0.998 0.85"
            class_part = name.split("'")[0].strip()
            class_name = NAME_MAP.get(class_part.lower(), None)

            if class_name:
                o4_sets.append({
                    "set_id": s["id"],
                    "display_name": name,
                    "class_name": class_name,
                    "n_subjects": s.get("set_member_subjects_count", 0),
                })

    return o4_sets


def fetch_subjects_from_set(set_id, max_pages=50):
    """Fetch all subjects from a subject set."""
    subjects = []
    page = 1

    while page <= max_pages:
        r = requests.get(f"{BASE_URL}/subjects",
                        params={"subject_set_id": set_id, "page": page, "page_size": 100},
                        headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} at page {page}")
            break

        data = r.json()
        page_subjects = data.get("subjects", [])
        if not page_subjects:
            break

        for subj in page_subjects:
            meta = subj.get("metadata", {})
            locs = subj.get("locations", [])

            urls = {}
            for i, loc in enumerate(locs):
                for mime, url in loc.items():
                    urls[f"url{i+1}"] = url

            gps = None
            for key in ["event_time", "#event_time", "peakGPS"]:
                if key in meta:
                    try:
                        gps = float(meta[key])
                    except:
                        pass
                    break

            subjects.append({
                "subject_id": subj.get("id"),
                "gravityspy_id": meta.get("gravityspy_id", meta.get("#gravityspy_id", "")),
                "event_time": gps,
                "ifo": meta.get("ifo", meta.get("#ifo", meta.get("Detector", ""))),
                "ml_label": meta.get("ml_label", meta.get("#Label", "")),
                "ml_confidence": meta.get("ml_confidence", meta.get("ml_posterior", None)),
                **urls,
            })

        meta_page = data.get("meta", {}).get("subjects", {})
        total_pages = meta_page.get("page_count", page)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)

    return subjects


def download_and_crop_spectrogram(url, save_path):
    """Download composite image, crop 1.0s view, resize to 224x224."""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return False

        img = Image.open(io.BytesIO(r.content))

        # Composite is 1210x400 with 4 duration views side by side
        # Views: 0.5s, 1.0s, 2.0s, 4.0s
        # Each view is approximately 1210/4 = 302.5 px wide
        # But the exact layout may vary. Let's check.
        w, h = img.size

        if w > 800:  # Composite image
            # The 4 views are arranged horizontally
            # Crop the second quarter (1.0s view)
            view_w = w // 4
            left = view_w  # Start of 1.0s view
            right = 2 * view_w
            crop = img.crop((left, 0, right, h))
        else:
            # Single view image
            crop = img

        # Convert to RGB and resize
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        crop = crop.resize(TARGET_SIZE, Image.BILINEAR)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(save_path, format="PNG")
        return True

    except Exception as e:
        return False


def main():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    SPECTROGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress()
    fetched_sets = set(progress["fetched_sets"])
    all_subjects = progress["subjects"]

    # Step 1: Get O4 subject sets
    print("Fetching O4 subject sets...")
    o4_sets = get_o4_subject_sets()
    print(f"Found {len(o4_sets)} O4 subject sets:")
    total_subjects = 0
    for s in o4_sets:
        print(f"  [{s['set_id']}] {s['display_name']:50s} -> {s['class_name']:25s} ({s['n_subjects']} subjects)")
        total_subjects += s["n_subjects"]
    print(f"Total O4 subjects across sets: {total_subjects}")

    # Step 2: Fetch subjects from each set
    print("\nFetching subjects from O4 sets...")
    for s in o4_sets:
        if s["set_id"] in fetched_sets:
            print(f"  Skipping {s['display_name']} (already fetched)")
            continue

        print(f"  Fetching {s['display_name']} ({s['n_subjects']} subjects)...")
        subjects = fetch_subjects_from_set(s["set_id"])

        # Add class label from set if not in metadata
        for subj in subjects:
            if not subj["ml_label"]:
                subj["ml_label"] = s["class_name"]
            subj["source_set"] = s["display_name"]
            subj["source_set_id"] = s["set_id"]

        all_subjects.extend(subjects)
        fetched_sets.add(s["set_id"])

        # Save progress
        progress["fetched_sets"] = list(fetched_sets)
        progress["subjects"] = all_subjects
        save_progress(progress)

        print(f"    Got {len(subjects)} subjects (total: {len(all_subjects)})")

    # Step 3: Create manifest
    print(f"\nTotal subjects collected: {len(all_subjects)}")

    if not all_subjects:
        print("No subjects collected!")
        sys.exit(1)

    df = pd.DataFrame(all_subjects)

    # Assign ml_label from source set if missing
    df["ml_label"] = df["ml_label"].replace("", pd.NA)
    # If ml_label is missing, use class from set name
    if "source_set" in df.columns:
        for idx, row in df[df["ml_label"].isna()].iterrows():
            set_name = row.get("source_set", "")
            for key, val in NAME_MAP.items():
                if key in set_name.lower():
                    df.at[idx, "ml_label"] = val
                    break

    # Deduplicate by subject_id
    n_before = len(df)
    df = df.drop_duplicates(subset="subject_id", keep="first")
    print(f"After dedup: {n_before} -> {len(df)}")

    # Filter to O3 classes
    df_filtered = df[df["ml_label"].isin(O3_CLASSES)].copy()
    print(f"After O3 filter: {len(df_filtered)}")

    # Save raw metadata
    df_filtered.to_csv(METADATA_DIR / "o4_subjects_raw.csv", index=False)
    print(f"Saved raw metadata: {METADATA_DIR / 'o4_subjects_raw.csv'}")

    # Print class distribution
    print("\nO4 class distribution:")
    for cls, count in df_filtered["ml_label"].value_counts().items():
        print(f"  {cls:30s}: {count:6d}")

    # Print URL stats
    url_cols = [c for c in df_filtered.columns if c.startswith("url")]
    print(f"\nURL columns: {url_cols}")
    for col in url_cols:
        print(f"  {col}: {df_filtered[col].notna().sum()} non-null")

    # GPS stats
    n_gps = df_filtered["event_time"].notna().sum()
    print(f"\nGPS time: {n_gps}/{len(df_filtered)} non-null")

    # Step 4: Download and crop spectrograms
    print(f"\nDownloading spectrograms (1.0s view cropped to 224x224)...")

    n_downloaded = 0
    n_failed = 0
    n_skipped = 0

    for i, (_, row) in enumerate(df_filtered.iterrows()):
        sid = row.get("gravityspy_id") or row["subject_id"]
        detector = row.get("ifo", "UNK")
        label = row["ml_label"]
        url = row.get("url1")

        if pd.isna(url):
            n_skipped += 1
            continue

        save_path = SPECTROGRAMS_DIR / label / f"{sid}_1.0s.png"

        if save_path.exists():
            n_downloaded += 1
            continue

        if download_and_crop_spectrogram(url, save_path):
            n_downloaded += 1
        else:
            n_failed += 1

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(df_filtered)}, "
                  f"downloaded={n_downloaded}, failed={n_failed}, skipped={n_skipped}")

        time.sleep(0.05)  # Rate limit

    print(f"\nDownload complete: {n_downloaded} downloaded, {n_failed} failed, {n_skipped} skipped")

    # Step 5: Create evaluation manifest
    print("\nCreating evaluation manifest...")

    manifest_rows = []
    for _, row in df_filtered.iterrows():
        sid = row.get("gravityspy_id") or row["subject_id"]
        label = row["ml_label"]
        img_path = f"data/o4/spectrograms/{label}/{sid}_1.0s.png"

        if (PROJECT_ROOT / img_path).exists():
            manifest_rows.append({
                "gravityspy_id": str(sid),
                "event_time": row.get("event_time"),
                "ifo": row.get("ifo", "UNK"),
                "ml_label": label,
                "ml_confidence": row.get("ml_confidence"),
                "image_path_1.0s": img_path,
                "split": "o4_test",
            })

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_path = METADATA_DIR / "o4_evaluation_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Evaluation manifest: {len(df_manifest)} samples -> {manifest_path}")

    # Class distribution
    print("\nManifest class distribution:")
    class_dist = df_manifest["ml_label"].value_counts().to_dict()
    for cls, count in sorted(class_dist.items()):
        print(f"  {cls:30s}: {count:6d}")

    # Save class distribution JSON
    with open(METADATA_DIR / "o4_class_distribution.json", "w") as f:
        json.dump({
            "per_class_counts": class_dist,
            "total_samples": len(df_manifest),
            "num_classes": len(class_dist),
            "classes_present": sorted(class_dist.keys()),
            "classes_missing_from_o3": sorted(set(O3_CLASSES) - set(class_dist.keys())),
        }, f, indent=2)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
