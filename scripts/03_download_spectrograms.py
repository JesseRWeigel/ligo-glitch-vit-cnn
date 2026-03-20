#!/usr/bin/env python3
"""
Download pre-made Gravity Spy spectrogram images from Zooniverse CDN.

Reads the filtered metadata CSV and downloads all 4 duration views (url1-url4)
for each glitch, organized by class label.

Output structure:
    data/spectrograms/{ml_label}/{gravityspy_id}_{duration_idx}.png

Where duration_idx maps to the 4 Gravity Spy time windows:
    url1 -> 0.5s view
    url2 -> 1.0s view
    url3 -> 2.0s view
    url4 -> 4.0s view

Images are resized to 224x224 RGB if needed.

Usage:
    python scripts/03_download_spectrograms.py [--workers 50] [--batch-size 500]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "gravity_spy_o3_filtered.csv"
SPECTROGRAMS_DIR = PROJECT_ROOT / "data" / "spectrograms"
PROGRESS_FILE = SPECTROGRAMS_DIR / "download_progress.json"
FAILURES_FILE = SPECTROGRAMS_DIR / "download_failures.json"

# Gravity Spy URL columns and their duration labels
URL_COLUMNS = {
    "url1": "0.5",
    "url2": "1.0",
    "url3": "2.0",
    "url4": "4.0",
}

TARGET_SIZE = (224, 224)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "data" / "spectrograms_download.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    """Load download progress from JSON checkpoint file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_ids": [], "failed_ids": [], "stats": {}}


def save_progress(progress: dict):
    """Save download progress to JSON checkpoint file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def save_failures(failures: list):
    """Save detailed failure info."""
    with open(FAILURES_FILE, "w") as f:
        json.dump(failures, f, indent=2)


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

async def download_and_save_image(
    session: aiohttp.ClientSession,
    url: str,
    save_path: Path,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> tuple[bool, str]:
    """Download a single image, resize to 224x224 RGB, save as PNG.

    Returns (success: bool, error_msg: str).
    """
    if save_path.exists():
        return True, ""

    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status != 200:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return False, f"HTTP {resp.status}"
                    data = await resp.read()

            # Process image in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _process_and_save, data, save_path)
            return True, ""

        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return False, "timeout"
        except aiohttp.ClientError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return False, str(e)
        except Exception as e:
            return False, f"processing error: {e}"

    return False, "max retries exceeded"


def _process_and_save(data: bytes, save_path: Path):
    """Process raw image bytes: resize to 224x224 RGB, save as PNG."""
    img = Image.open(io.BytesIO(data))

    # Convert to RGB if needed (some may be RGBA or grayscale)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if needed
    if img.size != TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.BILINEAR)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path, format="PNG")


async def download_batch(
    session: aiohttp.ClientSession,
    batch: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[list[str], list[dict]]:
    """Download a batch of glitches (all 4 views each).

    Returns (completed_ids, failures).
    """
    completed = []
    failures = []

    tasks = []
    task_meta = []  # Track which glitch/url each task belongs to

    for row in batch:
        gspy_id = row["gravityspy_id"]
        label = row["ml_label"]
        class_dir = SPECTROGRAMS_DIR / label

        all_exist = True
        for url_col, dur_label in URL_COLUMNS.items():
            fname = f"{gspy_id}_{dur_label}s.png"
            save_path = class_dir / fname
            if not save_path.exists():
                all_exist = False
                url = row[url_col]
                tasks.append(download_and_save_image(session, url, save_path, semaphore))
                task_meta.append({"id": gspy_id, "url_col": url_col, "url": url})

        if all_exist:
            completed.append(gspy_id)

    if not tasks:
        return completed, failures

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Group results by glitch ID
    id_results = {}
    for meta, result in zip(task_meta, results):
        gspy_id = meta["id"]
        if gspy_id not in id_results:
            id_results[gspy_id] = {"success": True, "errors": []}

        if isinstance(result, Exception):
            id_results[gspy_id]["success"] = False
            id_results[gspy_id]["errors"].append(
                {"url_col": meta["url_col"], "error": str(result)}
            )
        else:
            success, err = result
            if not success:
                id_results[gspy_id]["success"] = False
                id_results[gspy_id]["errors"].append(
                    {"url_col": meta["url_col"], "error": err}
                )

    for gspy_id, info in id_results.items():
        if info["success"]:
            completed.append(gspy_id)
        else:
            failures.append({"gravityspy_id": gspy_id, "errors": info["errors"]})

    return completed, failures


async def run_download(
    df: pd.DataFrame,
    workers: int = 50,
    batch_size: int = 500,
    checkpoint_interval: int = 5000,
):
    """Main download loop with progress tracking and checkpointing."""

    progress = load_progress()
    completed_set = set(progress["completed_ids"])
    all_failures = []

    # Filter out already-completed IDs
    remaining = df[~df["gravityspy_id"].isin(completed_set)]
    total = len(df)
    already_done = len(completed_set)

    logger.info(f"Total glitches: {total}")
    logger.info(f"Already completed: {already_done}")
    logger.info(f"Remaining: {len(remaining)}")
    logger.info(f"Workers: {workers}, Batch size: {batch_size}")

    if len(remaining) == 0:
        logger.info("All downloads already complete!")
        return progress

    # Create class directories
    for label in df["ml_label"].unique():
        (SPECTROGRAMS_DIR / label).mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(workers)

    connector = aiohttp.TCPConnector(
        limit=workers,
        limit_per_host=workers,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    headers = {
        "User-Agent": "GW-Research-Pipeline/1.0 (academic research; gravitational wave glitch classification)",
    }

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        rows = remaining.to_dict("records")
        n_batches = (len(rows) + batch_size - 1) // batch_size

        t_start = time.time()
        batch_completed_count = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            batch_num = i // batch_size + 1

            completed_ids, failures = await download_batch(session, batch, semaphore)

            completed_set.update(completed_ids)
            batch_completed_count += len(completed_ids)
            all_failures.extend(failures)

            # Progress stats
            elapsed = time.time() - t_start
            total_completed = len(completed_set)
            rate = batch_completed_count / elapsed if elapsed > 0 else 0
            remaining_count = total - total_completed
            eta = remaining_count / rate / 60 if rate > 0 else float("inf")

            logger.info(
                f"Batch {batch_num}/{n_batches}: "
                f"{total_completed}/{total} complete ({total_completed/total*100:.1f}%), "
                f"{len(failures)} failures this batch, "
                f"rate={rate:.1f} glitches/s, ETA={eta:.1f} min"
            )

            # Checkpoint
            if total_completed % checkpoint_interval < batch_size or batch_num == n_batches:
                progress["completed_ids"] = list(completed_set)
                progress["failed_ids"] = [f["gravityspy_id"] for f in all_failures]
                progress["stats"] = {
                    "total": total,
                    "completed": total_completed,
                    "failed": len(all_failures),
                    "coverage_pct": total_completed / total * 100,
                    "elapsed_seconds": elapsed,
                    "rate_per_second": rate,
                }
                save_progress(progress)
                if all_failures:
                    save_failures(all_failures)

    # Final save
    progress["completed_ids"] = list(completed_set)
    progress["failed_ids"] = [f["gravityspy_id"] for f in all_failures]
    elapsed = time.time() - t_start
    progress["stats"] = {
        "total": total,
        "completed": len(completed_set),
        "failed": len(all_failures),
        "coverage_pct": len(completed_set) / total * 100,
        "elapsed_seconds": elapsed,
        "rate_per_second": batch_completed_count / elapsed if elapsed > 0 else 0,
    }
    save_progress(progress)
    if all_failures:
        save_failures(all_failures)

    logger.info(f"Download complete: {len(completed_set)}/{total} glitches "
                f"({len(completed_set)/total*100:.1f}% coverage)")
    logger.info(f"Failures: {len(all_failures)}")
    logger.info(f"Total time: {elapsed/60:.1f} min")

    return progress


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_downloads(df: pd.DataFrame, n_samples: int = 100) -> dict:
    """Validate downloaded spectrograms: check dimensions, mode, coverage."""
    import random

    logger.info(f"Validating {n_samples} random images...")

    # Check overall coverage
    total = len(df)
    complete_count = 0
    partial_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        existing = 0
        for url_col, dur_label in URL_COLUMNS.items():
            fpath = SPECTROGRAMS_DIR / row["ml_label"] / f"{row['gravityspy_id']}_{dur_label}s.png"
            if fpath.exists():
                existing += 1
        if existing == 4:
            complete_count += 1
        elif existing > 0:
            partial_count += 1
        else:
            missing_count += 1

    logger.info(f"Coverage: {complete_count}/{total} complete ({complete_count/total*100:.1f}%)")
    logger.info(f"  Partial: {partial_count}, Missing: {missing_count}")

    # Sample validation
    sample_ids = random.sample(list(df["gravityspy_id"]), min(n_samples, len(df)))
    sample_df = df[df["gravityspy_id"].isin(sample_ids)]

    dim_ok = 0
    dim_fail = 0
    dim_errors = []

    for _, row in sample_df.iterrows():
        for url_col, dur_label in URL_COLUMNS.items():
            fpath = SPECTROGRAMS_DIR / row["ml_label"] / f"{row['gravityspy_id']}_{dur_label}s.png"
            if fpath.exists():
                try:
                    img = Image.open(fpath)
                    if img.size == TARGET_SIZE and img.mode == "RGB":
                        dim_ok += 1
                    else:
                        dim_fail += 1
                        dim_errors.append(
                            f"{fpath.name}: size={img.size}, mode={img.mode}"
                        )
                except Exception as e:
                    dim_fail += 1
                    dim_errors.append(f"{fpath.name}: {e}")

    logger.info(f"Dimension check: {dim_ok} OK, {dim_fail} FAIL out of {dim_ok+dim_fail} checked")
    if dim_errors:
        for err in dim_errors[:10]:
            logger.warning(f"  {err}")

    return {
        "total_glitches": total,
        "complete_coverage": complete_count,
        "partial_coverage": partial_count,
        "missing_coverage": missing_count,
        "coverage_pct": complete_count / total * 100,
        "sample_checked": dim_ok + dim_fail,
        "sample_dim_ok": dim_ok,
        "sample_dim_fail": dim_fail,
        "dim_errors": dim_errors[:20],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download Gravity Spy spectrogram images")
    parser.add_argument("--workers", type=int, default=50,
                        help="Number of concurrent downloads (default: 50)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size for progress reporting (default: 500)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation, skip download")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
                        help="Save progress every N glitches (default: 5000)")
    args = parser.parse_args()

    # Ensure output dirs exist
    SPECTROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    logger.info("Loading metadata...")
    df = pd.read_csv(METADATA_CSV)
    logger.info(f"Loaded {len(df)} glitches across {df['ml_label'].nunique()} classes")

    if not args.validate_only:
        logger.info("Starting download...")
        progress = asyncio.run(
            run_download(
                df,
                workers=args.workers,
                batch_size=args.batch_size,
                checkpoint_interval=args.checkpoint_interval,
            )
        )
        logger.info(f"Download stats: {json.dumps(progress.get('stats', {}), indent=2)}")

    logger.info("Running validation...")
    validation = validate_downloads(df)
    logger.info(f"Validation results: {json.dumps(validation, indent=2)}")

    # Save validation results
    val_path = SPECTROGRAMS_DIR / "validation_results.json"
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    logger.info(f"Validation saved to {val_path}")

    # Summary
    cov = validation["coverage_pct"]
    if cov >= 95:
        logger.info(f"SUCCESS: Coverage {cov:.1f}% >= 95% threshold")
    elif cov >= 90:
        logger.warning(f"ACCEPTABLE: Coverage {cov:.1f}% >= 90% but < 95%")
    else:
        logger.error(f"INSUFFICIENT: Coverage {cov:.1f}% < 90% threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
