#!/usr/bin/env python3
"""Download Gravity Spy ML classification metadata from Zenodo record 5649212.

Uses the Zenodo REST API to discover file URLs and download the classification
CSV files for O1-O3b data. Implements resume capability via file size checking
and optional checksum verification.

Conventions:
  ASSERT_CONVENTION: natural_units=SI, coordinate_system=Q-transform spectrograms 10-2048 Hz log axis 224x224 px
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "5649212"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
OUTPUT_DIR = Path("data/raw/zenodo_5649212")


def get_zenodo_files():
    """Query Zenodo API to get file metadata for the record."""
    print(f"Querying Zenodo API for record {ZENODO_RECORD_ID}...")
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    record = resp.json()

    files = record.get("files", [])
    print(f"Found {len(files)} files in record:")
    for f in files:
        size_mb = f["size"] / (1024 * 1024)
        print(f"  - {f['key']}: {size_mb:.1f} MB (checksum: {f.get('checksum', 'N/A')})")

    return files


def download_file(file_meta, output_dir):
    """Download a single file with resume capability and checksum verification."""
    filename = file_meta["key"]
    url = file_meta["links"]["self"]
    expected_size = file_meta["size"]
    checksum_str = file_meta.get("checksum", "")

    output_path = output_dir / filename

    # Resume capability: skip if file exists with correct size
    if output_path.exists():
        current_size = output_path.stat().st_size
        if current_size == expected_size:
            print(f"  Skipping {filename} (already complete: {current_size} bytes)")
            return output_path
        else:
            print(f"  Resuming {filename} (have {current_size}/{expected_size} bytes)")
            # For simplicity, re-download if sizes differ
            output_path.unlink()

    print(f"  Downloading {filename} ({expected_size / (1024*1024):.1f} MB)...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    with open(output_path, "wb") as f:
        with tqdm(total=expected_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Verify size
    actual_size = output_path.stat().st_size
    assert actual_size == expected_size, (
        f"Size mismatch for {filename}: expected {expected_size}, got {actual_size}"
    )

    # Verify checksum if available (Zenodo uses md5:hash format)
    if checksum_str.startswith("md5:"):
        expected_md5 = checksum_str.split(":", 1)[1]
        actual_md5 = hashlib.md5(output_path.read_bytes()).hexdigest()
        assert actual_md5 == expected_md5, (
            f"Checksum mismatch for {filename}: expected {expected_md5}, got {actual_md5}"
        )
        print(f"  Checksum verified: {actual_md5}")

    return output_path


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get file listing from Zenodo
    files = get_zenodo_files()

    # Download CSV files (the ML classification data)
    csv_files = [f for f in files if f["key"].endswith(".csv")]
    if not csv_files:
        # Some Zenodo records use .csv.gz or other formats
        csv_files = [f for f in files if "csv" in f["key"].lower() or "classification" in f["key"].lower()]

    if not csv_files:
        print("WARNING: No CSV files found. Downloading all files...")
        csv_files = files

    downloaded = []
    for f in csv_files:
        path = download_file(f, OUTPUT_DIR)
        downloaded.append(str(path))

    # Save download manifest
    manifest = {
        "zenodo_record": ZENODO_RECORD_ID,
        "files_downloaded": downloaded,
        "total_files_in_record": len(files),
        "all_file_names": [f["key"] for f in files],
    }
    manifest_path = OUTPUT_DIR / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDownload complete. {len(downloaded)} files saved to {OUTPUT_DIR}/")
    print(f"Manifest written to {manifest_path}")

    return downloaded


if __name__ == "__main__":
    main()
