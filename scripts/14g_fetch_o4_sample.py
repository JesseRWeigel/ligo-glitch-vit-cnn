#!/usr/bin/env python3
"""Fetch a sample of actual O4 Gravity Spy subjects and check images."""

import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load O4 consensus labels
df = pd.read_csv("data/o4/metadata/o4_consensus_labels.csv")

# Sample 5 per class for quick check (max 21 classes * 5 = 105)
sample = df.groupby("label").apply(lambda g: g.head(3), include_groups=False).reset_index()
print(f"Sampling {len(sample)} O4 subjects across {df['label'].nunique()} classes")
sample = sample.merge(df[['Subject_id', 'label']], left_on='level_1', right_index=True, how='left') if 'Subject_id' not in sample.columns else sample
# Fix: just re-do the sampling simply
sample = pd.concat([g.head(3) for _, g in df.groupby("label")]).reset_index(drop=True)
print(f"Sampling {len(sample)} O4 subjects across {sample['label'].nunique()} classes")

# Fetch subject metadata from Zooniverse API
BASE_URL = "https://www.zooniverse.org/api/subjects"
HEADERS = {
    "Accept": "application/vnd.api+json; version=1",
    "Content-Type": "application/json",
}

results = []
for i in range(0, min(len(sample), 60), 10):
    batch_ids = sample["Subject_id"].iloc[i:i+10].tolist()
    ids_str = ",".join(str(s) for s in batch_ids)

    r = requests.get(BASE_URL, params={"id": ids_str, "page_size": 10},
                    headers=HEADERS, timeout=30)
    if r.status_code == 200:
        data = r.json()
        for subj in data.get("subjects", []):
            meta = subj.get("metadata", {})
            locs = subj.get("locations", [])
            urls = {}
            for j, loc in enumerate(locs):
                for mime, url in loc.items():
                    urls[f"url{j+1}"] = url

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
                "ifo": meta.get("ifo", meta.get("#ifo", "")),
                "ml_label_api": meta.get("ml_label", meta.get("#Label", "")),
                "ml_confidence": meta.get("ml_confidence", meta.get("ml_posterior", "")),
                "n_urls": len(urls),
                **urls,
            })
    else:
        print(f"HTTP {r.status_code}")

    time.sleep(0.5)

print(f"\nFetched {len(results)} subjects")

# Analyze
df_results = pd.DataFrame(results)
print(f"GPS non-null: {df_results['event_time'].notna().sum()}")
print(f"ml_label non-empty: {(df_results['ml_label_api'] != '').sum()}")
print(f"URL columns: {[c for c in df_results.columns if c.startswith('url')]}")
print(f"n_urls distribution: {df_results['n_urls'].value_counts().to_dict()}")

# Download and check first 3 images
out_dir = Path("data/o4/spectrograms/samples")
out_dir.mkdir(parents=True, exist_ok=True)

for _, row in df_results.head(3).iterrows():
    if "url1" in row and pd.notna(row.get("url1")):
        url = row["url1"]
        sid = row["subject_id"]
        print(f"\nSubject {sid}: {url[:80]}...")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content))
            print(f"  Size: {img.size}, Mode: {img.mode}")
            outpath = out_dir / f"o4_subject_{sid}.png"
            img.save(outpath)
            print(f"  Saved: {outpath}")

# Save results
df_results.to_csv("data/o4/metadata/o4_sample_subjects.csv", index=False)
print(f"\nSaved sample to data/o4/metadata/o4_sample_subjects.csv")

# Print metadata keys from first subject for debugging
if results:
    print(f"\nFirst subject full metadata check:")
    sid = results[0]["subject_id"]
    r = requests.get(f"{BASE_URL}/{sid}", headers=HEADERS, timeout=30)
    if r.status_code == 200:
        data = r.json()
        subj = data.get("subjects", [{}])[0]
        meta = subj.get("metadata", {})
        print(f"  All metadata keys: {list(meta.keys())}")
        print(f"  Metadata (first 500 chars): {json.dumps(meta)[:500]}")
