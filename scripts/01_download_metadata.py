#!/usr/bin/env python3
"""Download Gravity Spy ML classification metadata from Zenodo record 5649212.

Uses the Zenodo REST API to discover file URLs and download the classification
CSV files for O1-O3b data. Implements resume capability via file size checking
and checksum verification before using an existing or new file.

Conventions:
  ASSERT_CONVENTION: natural_units=SI, coordinate_system=Q-transform spectrograms 10-2048 Hz log axis 224x224 px
"""

import hashlib
import json
import os
import tempfile
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


def verify_file(path, expected_size, checksum_str):
    """Raise ValueError unless the file matches the advertised integrity metadata."""
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Size mismatch for {path.name}: expected {expected_size}, got {actual_size}")

    if checksum_str:
        algorithm, separator, expected_digest = checksum_str.partition(":")
        if not separator or not expected_digest:
            raise ValueError(f"Invalid Zenodo checksum for {path.name}: {checksum_str!r}")
        with path.open("rb") as f:
            actual_digest = hashlib.file_digest(f, algorithm).hexdigest()
        if actual_digest != expected_digest.lower():
            raise ValueError(
                f"Checksum mismatch for {path.name}: expected {expected_digest}, got {actual_digest}"
            )


def download_file(file_meta, output_dir):
    """Reuse verified files or replace them only after a complete, verified download."""
    filename = file_meta["key"]
    url = file_meta["links"]["self"]
    expected_size = file_meta["size"]
    checksum_str = file_meta.get("checksum") or ""
    output_path = output_dir / filename

    if output_path.exists():
        try:
            verify_file(output_path, expected_size, checksum_str)
        except ValueError as exc:
            print(f"  {exc}. Re-downloading {filename}.")
        else:
            print(f"  Skipping {filename} (verified {expected_size} bytes)")
            return output_path

    print(f"  Downloading {filename} ({expected_size / (1024*1024):.1f} MB)...")
    temp_path = None
    try:
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".zenodo-", delete=False) as f:
                temp_path = Path(f.name)
                with tqdm(total=expected_size, unit="B", unit_scale=True, desc=filename) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        verify_file(temp_path, expected_size, checksum_str)
        temp_path.replace(output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

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
