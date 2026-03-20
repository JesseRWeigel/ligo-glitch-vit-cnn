#!/usr/bin/env python3
"""
Batch fetch O4 Gravity Spy subject metadata from Zooniverse API.

Fetches subjects in large batches using the ?id=X,Y,Z parameter.
Zooniverse API allows up to ~100 IDs per request.

Targets: get ~5000-10000 subjects with image URLs for evaluation.
We prioritize diversity across classes over total volume.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "o4" / "metadata"

BASE_URL = "https://www.zooniverse.org/api/subjects"
HEADERS = {
    "Accept": "application/vnd.api+json; version=1",
    "Content-Type": "application/json",
}

BATCH_SIZE = 50  # IDs per API request
DELAY = 0.5  # seconds between requests
TARGET_PER_CLASS = 500  # target subjects per class


def fetch_batch(subject_ids):
    """Fetch metadata for a batch of subject IDs."""
    ids_str = ",".join(str(s) for s in subject_ids)
    try:
        r = requests.get(BASE_URL, params={"id": ids_str, "page_size": len(subject_ids)},
                        headers=HEADERS, timeout=60)
        if r.status_code == 200:
            data = r.json()
            results = []
            for subj in data.get("subjects", []):
                meta = subj.get("metadata", {})
                locs = subj.get("locations", [])

                urls = {}
                for i, loc in enumerate(locs):
                    for mime, url in loc.items():
                        urls[f"url{i+1}"] = url

                # GPS time
                gps = None
                for key in ["event_time", "#event_time", "peakGPS"]:
                    if key in meta:
                        try:
                            gps = float(meta[key])
                        except:
                            pass
                        break

                results.append({
                    "subject_id": subj.get("id"),
                    "event_time": gps,
                    "ifo": meta.get("ifo", meta.get("#ifo", meta.get("Detector", ""))),
                    "ml_label_api": meta.get("ml_label", meta.get("#Label", "")),
                    "ml_confidence": meta.get("ml_confidence", meta.get("ml_posterior", None)),
                    **urls,
                })
            return results
        elif r.status_code == 429:
            print("Rate limited, sleeping 60s")
            time.sleep(60)
            return fetch_batch(subject_ids)  # retry
        else:
            print(f"HTTP {r.status_code}")
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def main():
    # Load consensus labels
    df = pd.read_csv(METADATA_DIR / "o4_consensus_labels.csv")
    print(f"Total O4 subjects: {len(df)}")

    # Sample subjects stratified by class for diversity
    sampled = []
    for label, group in df.groupby("label"):
        n = min(len(group), TARGET_PER_CLASS)
        sampled.append(group.sample(n=n, random_state=42))
    df_sample = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df_sample)} subjects across {df_sample['label'].nunique()} classes")

    # Check for existing progress
    output_csv = METADATA_DIR / "o4_subjects_metadata.csv"
    existing_ids = set()
    all_rows = []
    if output_csv.exists():
        df_existing = pd.read_csv(output_csv)
        existing_ids = set(df_existing["subject_id"].astype(str).tolist())
        all_rows = df_existing.to_dict("records")
        print(f"Existing: {len(existing_ids)} subjects")

    # Filter to un-fetched IDs
    remaining_ids = [sid for sid in df_sample["Subject_id"].tolist() if str(sid) not in existing_ids]
    print(f"Remaining to fetch: {len(remaining_ids)}")

    if not remaining_ids:
        print("All subjects already fetched!")
        return

    t_start = time.time()
    n_fetched = 0

    for i in range(0, len(remaining_ids), BATCH_SIZE):
        batch = remaining_ids[i:i + BATCH_SIZE]
        results = fetch_batch(batch)
        all_rows.extend(results)
        n_fetched += len(results)

        if (i // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t_start
            rate = n_fetched / elapsed if elapsed > 0 else 0
            remaining = len(remaining_ids) - i - len(batch)
            eta = remaining / (rate * BATCH_SIZE) * DELAY if rate > 0 else 0
            print(f"Batch {i//BATCH_SIZE}: fetched {n_fetched} total, "
                  f"rate={rate:.1f}/s, remaining={remaining}, ETA={eta/60:.1f}min")

        # Save periodically
        if (i // BATCH_SIZE) % 50 == 0 and i > 0:
            df_out = pd.DataFrame(all_rows)
            df_out.to_csv(output_csv, index=False)
            print(f"Checkpoint saved: {len(df_out)} rows")

        time.sleep(DELAY)

    # Final save
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(output_csv, index=False)
    print(f"\nDone! Saved {len(df_out)} subjects to {output_csv}")

    # Check URL coverage
    url_cols = [c for c in df_out.columns if c.startswith("url")]
    for col in url_cols:
        n_valid = df_out[col].notna().sum()
        print(f"  {col}: {n_valid}/{len(df_out)} valid")

    # Check GPS coverage
    n_gps = df_out["event_time"].notna().sum()
    print(f"  event_time: {n_gps}/{len(df_out)} valid")


if __name__ == "__main__":
    main()
