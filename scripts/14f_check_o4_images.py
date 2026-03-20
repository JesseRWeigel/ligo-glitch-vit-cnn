#!/usr/bin/env python3
"""
Check what image format O4 Gravity Spy subjects have on Zooniverse.

Download a few sample images and examine their format.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

df = pd.read_csv("data/o4/metadata/o4_subjects_metadata.csv")

# Sample 5 subjects with URLs
sample = df[df["url1"].notna()].head(5)

for _, row in sample.iterrows():
    url = row["url1"]
    sid = row["subject_id"]
    print(f"Subject {sid}: {url[:60]}...")

    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        img = Image.open(io.BytesIO(r.content))
        print(f"  Size: {img.size}, Mode: {img.mode}")

        # Save to inspect
        out = Path(f"data/o4/spectrograms/samples/subject_{sid}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"  Saved: {out}")
    else:
        print(f"  HTTP {r.status_code}")
    print()
