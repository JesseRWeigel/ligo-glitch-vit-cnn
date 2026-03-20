#!/usr/bin/env python3
"""
Fetch O4 Gravity Spy subject metadata (image URLs, GPS times) from Zooniverse API.

Reads o4_consensus_labels.csv and queries subjects in batches.
Saves complete O4 metadata with image URLs to o4_subjects_metadata.csv.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "o4" / "metadata"
CONSENSUS_CSV = METADATA_DIR / "o4_consensus_labels.csv"
OUTPUT_CSV = METADATA_DIR / "o4_subjects_metadata.csv"
PROGRESS_FILE = METADATA_DIR / "fetch_progress.json"

# Zooniverse API
BASE_URL = "https://www.zooniverse.org/api/subjects"
HEADERS = {
    "Accept": "application/vnd.api+json; version=1",
    "Content-Type": "application/json",
}

BATCH_SIZE = 10  # Zooniverse API allows querying multiple subjects
RATE_LIMIT_DELAY = 0.3  # seconds between requests


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_ids": [], "rows": []}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, default=str))


def fetch_subjects(subject_ids):
    """Fetch metadata for a list of subject IDs from Zooniverse API."""
    results = []

    # Zooniverse API supports comma-separated IDs
    ids_str = ",".join(str(s) for s in subject_ids)
    params = {"id": ids_str}

    try:
        r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            for subj in data.get("subjects", []):
                meta = subj.get("metadata", {})
                locs = subj.get("locations", [])

                # Extract image URLs from locations
                urls = {}
                for i, loc in enumerate(locs):
                    for mime, url in loc.items():
                        urls[f"url{i+1}"] = url

                # Extract GPS time
                gps_time = None
                for key in ["event_time", "#event_time", "peakGPS", "Peak GPS"]:
                    if key in meta:
                        try:
                            gps_time = float(meta[key])
                        except (ValueError, TypeError):
                            pass
                        break

                record = {
                    "subject_id": subj.get("id"),
                    "event_time": gps_time,
                    "ifo": meta.get("ifo", meta.get("Detector", meta.get("#ifo", ""))),
                    "ml_label": meta.get("ml_label", meta.get("#Label", meta.get("label", ""))),
                    "ml_confidence": meta.get("ml_confidence", meta.get("ml_posterior", "")),
                    **urls,
                }
                results.append(record)
        elif r.status_code == 429:
            print(f"Rate limited! Sleeping 30s...")
            time.sleep(30)
        else:
            print(f"HTTP {r.status_code} for subjects {ids_str[:50]}...")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return results


def main():
    df_consensus = pd.read_csv(CONSENSUS_CSV)
    all_ids = df_consensus["Subject_id"].unique().tolist()
    print(f"Total subjects to fetch: {len(all_ids)}")

    # Load progress
    progress = load_progress()
    completed = set(progress["completed_ids"])
    rows = progress["rows"]

    remaining = [sid for sid in all_ids if sid not in completed]
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    # We'll fetch enough subjects for evaluation -- aiming for ~5000-10000
    # since we need per-class coverage not total volume
    # Prioritize by getting a representative sample across classes

    # To speed things up: fetch subjects in batches of 10
    total_fetched = len(completed)
    target = min(len(remaining), 15000)  # Cap at 15K for time

    t_start = time.time()

    for i in range(0, min(len(remaining), target), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        results = fetch_subjects(batch)

        for r in results:
            rows.append(r)
            completed.add(int(r["subject_id"]))

        total_fetched = len(completed)

        # Progress
        if (i // BATCH_SIZE) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + BATCH_SIZE) / elapsed if elapsed > 0 else 0
            eta = (target - i - BATCH_SIZE) / rate / 60 if rate > 0 else 0
            print(f"Progress: {total_fetched}/{target} subjects "
                  f"({total_fetched/target*100:.1f}%), "
                  f"rate={rate:.1f}/s, ETA={eta:.1f} min")

        # Save checkpoint periodically
        if (i // BATCH_SIZE) % 100 == 0 and i > 0:
            progress["completed_ids"] = list(completed)
            progress["rows"] = rows
            save_progress(progress)

        time.sleep(RATE_LIMIT_DELAY)

    # Final save
    progress["completed_ids"] = list(completed)
    progress["rows"] = rows
    save_progress(progress)

    # Save as CSV
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(df_out)} subjects to {OUTPUT_CSV}")
        print(f"Columns: {list(df_out.columns)}")
        print(f"URL columns present: {[c for c in df_out.columns if 'url' in c]}")
        print(f"GPS time coverage: {df_out['event_time'].notna().sum()}/{len(df_out)}")
    else:
        print("No subjects fetched!")


if __name__ == "__main__":
    main()
