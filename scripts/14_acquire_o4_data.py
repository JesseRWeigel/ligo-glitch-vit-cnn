#!/usr/bin/env python3
"""
Acquire O4a Gravity Spy evaluation dataset from Zooniverse.

Phase 4, Plan 01, Task 1.

Strategy:
1. Query Zooniverse for all O4 Gravity Spy subject sets (organized by class + confidence tier)
2. For each O4 subject set with ML confidence > 0.9 (tiers A and B),
   fetch subject metadata (gravityspy_id, ml_label, ml_confidence, ifo, image URLs)
3. Download the 1.0s duration spectrogram (locations[1]) for each subject
4. Map O4 ML labels to O3 23-class taxonomy; exclude new O4-only classes
5. Create evaluation manifest matching O3 format

ASSERT_CONVENTION: primary_metric=macro_f1, input_format=224x224_RGB_PNG_0to1
"""

import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
from PIL import Image
from panoptes_client import Panoptes, SubjectSet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/o4/o4_acquisition_task1.log"),
    ],
)
log = logging.getLogger(__name__)

# === CONSTANTS ===

O3_CLASSES = sorted([
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip",
    "Blip_Low_Frequency", "Chirp", "Extremely_Loud", "Fast_Scattering",
    "Helix", "Koi_Fish", "Light_Modulation", "Low_Frequency_Burst",
    "Low_Frequency_Lines", "No_Glitch", "Paired_Doves", "Power_Line",
    "Repeating_Blips", "Scattered_Light", "Scratchy", "Tomte",
    "Violin_Mode", "Wandering_Line", "Whistle",
])

# O4 Zooniverse subject set names -> O3 class names
O4_TO_O3_CLASS_MAP = {
    "1080Lines": "1080Lines",
    "1400Ripples": "1400Ripples",
    "Air Compressor": "Air_Compressor",
    "Blip": "Blip",
    "Blip Low Frequency": "Blip_Low_Frequency",
    "Chirp": "Chirp",
    "Extremely Loud": "Extremely_Loud",
    "Fast Scattering": "Fast_Scattering",
    "Helix": "Helix",
    "Koi Fish": "Koi_Fish",
    "Light Modulation": "Light_Modulation",
    "Low Frequency Burst": "Low_Frequency_Burst",
    "Low Frequency Lines": "Low_Frequency_Lines",
    "No Glitch": "No_Glitch",
    "Paired Doves": "Paired_Doves",
    "Power Line": "Power_Line",
    "Repeating Blips": "Repeating_Blips",
    "Scattered Light": "Scattered_Light",
    "Scratchy": "Scratchy",
    "Tomte": "Tomte",
    "Violin Mode": "Violin_Mode",
    "Wandering Line": "Wandering_Line",
    "Whistle": "Whistle",
}

GRAVITY_SPY_PROJECT_ID = 1104
MIN_ML_CONFIDENCE = 0.9
MAX_PER_CLASS = 3000
DOWNLOAD_CONCURRENCY = 20

DATA_DIR = Path("data/o4")
SPEC_DIR = DATA_DIR / "spectrograms"
META_DIR = DATA_DIR / "metadata"
MANIFEST_PATH = META_DIR / "o4_evaluation_manifest.csv"
CLASS_DIST_PATH = META_DIR / "o4_class_distribution.json"


def parse_class_from_set_name(set_name: str) -> tuple:
    """Extract O3 class name and confidence tier from subject set display name.

    Returns (o3_class_name or None, tier letter).
    """
    parts = set_name.split("'O4")
    if len(parts) < 2:
        return None, "?"

    class_part = parts[0].strip()
    tier_part = parts[1].strip()

    tier = "?"
    if "(A)" in tier_part:
        tier = "A"
    elif "(B)" in tier_part:
        tier = "B"
    elif "(M)" in tier_part:
        tier = "M"

    o3_class = O4_TO_O3_CLASS_MAP.get(class_part)
    return o3_class, tier


def fetch_o4_subject_sets() -> list:
    """Fetch all O4 subject sets from Gravity Spy project."""
    log.info("Fetching O4 subject sets from Gravity Spy (project 1104)...")
    Panoptes.connect()

    o4_sets = []
    for ss in SubjectSet.where(project_id=GRAVITY_SPY_PROJECT_ID):
        name = ss.display_name
        count = ss.raw.get("set_member_subjects_count", 0)
        if "O4" in name and count > 0:
            o3_class, tier = parse_class_from_set_name(name)
            o4_sets.append({
                "set_id": int(ss.id),
                "name": name,
                "count": count,
                "o3_class": o3_class,
                "tier": tier,
            })

    log.info(f"Found {len(o4_sets)} O4 subject sets, "
             f"{sum(s['count'] for s in o4_sets)} total subjects")
    return o4_sets


def select_subject_sets(o4_sets: list) -> list:
    """Select high-confidence subject sets with O3 class mapping."""
    selected = []
    excluded_new = []

    for s in o4_sets:
        if s["tier"] == "M":
            continue
        if s["o3_class"] is None:
            excluded_new.append(s)
            continue
        selected.append(s)

    log.info(f"Selected {len(selected)} subject sets (A/B tiers, O3 classes)")
    if excluded_new:
        log.info(f"Excluded {len(excluded_new)} sets (not in O3 taxonomy):")
        for s in excluded_new:
            log.info(f"  {s['name']}: {s['count']} subjects")

    return selected, excluded_new


def fetch_subjects_via_api(set_id: int, page_size: int = 100,
                          max_subjects: int = 0) -> list:
    """Fetch subjects from a subject set using direct HTTP API (more robust).

    Args:
        max_subjects: Stop fetching after this many subjects (0 = no limit).
    """
    import requests
    import time

    headers = {
        "Accept": "application/vnd.api+json; version=1",
        "Content-Type": "application/json",
    }
    url = "https://www.zooniverse.org/api/subjects"
    all_subjects = []
    page = 1

    while True:
        params = {
            "subject_set_id": set_id,
            "page_size": page_size,
            "page": page,
        }
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=30)
                if r.status_code == 200:
                    break
                elif r.status_code == 429:
                    time.sleep(2 ** attempt)
                else:
                    log.warning(f"API HTTP {r.status_code} for set {set_id} page {page}")
                    return all_subjects
            except requests.RequestException as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    log.warning(f"API error for set {set_id}: {e}")
                    return all_subjects

        data = r.json()
        subjects = data.get("subjects", [])
        if not subjects:
            break

        all_subjects.extend(subjects)

        # Early exit if we have enough
        if max_subjects > 0 and len(all_subjects) >= max_subjects:
            break

        meta_info = data.get("meta", {}).get("subjects", {})
        page_count = meta_info.get("page_count", 1)
        if page >= page_count:
            break
        page += 1
        time.sleep(0.3)  # Rate limiting

    return all_subjects


def parse_api_subject(raw: dict, o3_class: str, tier: str) -> dict | None:
    """Parse a raw API subject response into our format."""
    meta = raw.get("metadata", {})
    locations = raw.get("locations", [])

    gravityspy_id = meta.get("subject_id", "")
    ml_confidence_raw = meta.get("#ml_top_confidence", "")
    filename1 = meta.get("Filename1", "")
    date_str = meta.get("date", "")

    # Parse IFO
    ifo = ""
    if filename1.startswith("H1_"):
        ifo = "H1"
    elif filename1.startswith("L1_"):
        ifo = "L1"

    # Get 1.0s URL (locations[1])
    url_1s = ""
    if len(locations) >= 2:
        loc = locations[1]
        if isinstance(loc, dict):
            url_1s = list(loc.values())[0]
        elif isinstance(loc, str):
            url_1s = loc

    try:
        conf = float(ml_confidence_raw)
    except (ValueError, TypeError):
        conf = 0.0

    if conf < MIN_ML_CONFIDENCE:
        return None

    return {
        "gravityspy_id": gravityspy_id,
        "zooniverse_id": int(raw.get("id", 0)),
        "label": o3_class,
        "ml_confidence": conf,
        "ifo": ifo,
        "date": date_str,
        "url_1s": url_1s,
        "tier": tier,
    }


def fetch_subjects_from_sets(selected_sets: list, progress_path: Path) -> list:
    """Fetch subject metadata from selected subject sets with resumption support."""
    progress = {}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    all_subjects = []
    per_class_count = Counter()

    # Load cache
    cache_path = META_DIR / "o4_subjects_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            all_subjects = json.load(f)
        for s in all_subjects:
            per_class_count[s["label"]] += 1
        log.info(f"Loaded {len(all_subjects)} cached subjects")

    completed_ids = set(progress.get("completed_sets", []))

    for i, sset in enumerate(selected_sets):
        set_id = sset["set_id"]
        o3_class = sset["o3_class"]

        if set_id in completed_ids:
            log.info(f"  [{i+1}/{len(selected_sets)}] {sset['name']}: already fetched")
            continue

        if per_class_count[o3_class] >= MAX_PER_CLASS:
            log.info(f"  [{i+1}/{len(selected_sets)}] {sset['name']}: "
                     f"{o3_class} at cap ({per_class_count[o3_class]})")
            completed_ids.add(set_id)
            continue

        log.info(f"  [{i+1}/{len(selected_sets)}] Fetching {sset['name']} "
                 f"({sset['count']} subjects)...")

        # Fetch only enough pages to fill the class cap
        remaining = MAX_PER_CLASS - per_class_count[o3_class]
        # Fetch ~2x what we need to account for confidence filtering
        fetch_limit = min(sset["count"], remaining * 2)
        raw_subjects = fetch_subjects_via_api(set_id, max_subjects=fetch_limit)
        set_subjects = []
        for raw in raw_subjects:
            if per_class_count[o3_class] >= MAX_PER_CLASS:
                break
            parsed = parse_api_subject(raw, o3_class, sset["tier"])
            if parsed:
                set_subjects.append(parsed)
                per_class_count[o3_class] += 1

        all_subjects.extend(set_subjects)
        completed_ids.add(set_id)
        log.info(f"    -> {len(set_subjects)} subjects "
                 f"(class total: {per_class_count[o3_class]})")

        # Save progress
        progress["completed_sets"] = list(completed_ids)
        with open(progress_path, "w") as f:
            json.dump(progress, f)
        with open(cache_path, "w") as f:
            json.dump(all_subjects, f)

    log.info(f"\nTotal: {len(all_subjects)} subjects")
    for cls in sorted(per_class_count):
        log.info(f"  {cls}: {per_class_count[cls]}")

    return all_subjects


async def download_spectrogram(session, url, save_path, sem, retries=3):
    """Download a single spectrogram with retry."""
    if save_path.exists():
        return True
    async with sem:
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_bytes(data)
                        return True
                    elif resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return False
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < retries - 1:
                    await asyncio.sleep(1)
        return False


async def download_all(subjects):
    """Download 1.0s spectrograms for all subjects."""
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for subj in subjects:
            url = subj["url_1s"]
            if not url:
                continue
            path = SPEC_DIR / subj["label"] / f"{subj['gravityspy_id']}_1.0s.png"
            tasks.append(download_spectrogram(session, url, path, sem))

        log.info(f"Downloading {len(tasks)} spectrograms...")
        results = await asyncio.gather(*tasks)
        ok = sum(results)
        log.info(f"Downloaded: {ok}/{len(tasks)} success")
        return ok, len(tasks) - ok


def verify_spectrograms(subjects, n=50):
    """Verify sample of downloaded spectrograms."""
    rng = np.random.RandomState(42)
    idx = rng.choice(len(subjects), min(n, len(subjects)), replace=False)

    valid = 0
    sizes = []
    modes = Counter()
    issues = []

    for i in idx:
        subj = subjects[i]
        path = SPEC_DIR / subj["label"] / f"{subj['gravityspy_id']}_1.0s.png"
        if not path.exists():
            issues.append(f"{subj['gravityspy_id']}: missing")
            continue
        try:
            img = Image.open(path)
            sizes.append(img.size)
            modes[img.mode] += 1
            valid += 1
        except Exception as e:
            issues.append(f"{subj['gravityspy_id']}: {e}")

    unique_sizes = set(sizes)
    log.info(f"Verification: {valid}/{len(idx)} valid")
    log.info(f"  Sizes: {unique_sizes}")
    log.info(f"  Modes: {dict(modes)}")
    if issues:
        log.warning(f"  Issues ({len(issues)}): {issues[:5]}")

    return {"valid": valid, "checked": len(idx), "sizes": list(unique_sizes),
            "modes": dict(modes), "issues": issues}


def create_manifest(subjects):
    """Create O4 evaluation manifest."""
    rows = []
    for subj in subjects:
        gid = subj["gravityspy_id"]
        label = subj["label"]
        path = SPEC_DIR / label / f"{gid}_1.0s.png"
        if not path.exists():
            continue
        rows.append({
            "gravityspy_id": gid,
            "image_path": str(path),
            "label": label,
            "detector": subj["ifo"],
            "ml_confidence": subj["ml_confidence"],
            "date": subj["date"],
            "duration": "1.0",
        })
    return pd.DataFrame(rows)


def create_class_distribution(manifest, excluded):
    """Create class distribution summary."""
    dist = manifest["label"].value_counts().to_dict()
    conf = manifest["ml_confidence"]

    o3_train = {
        "1080Lines": 341, "1400Ripples": 2428, "Air_Compressor": 1361,
        "Blip": 7156, "Blip_Low_Frequency": 13659, "Chirp": 11,
        "Extremely_Loud": 13469, "Fast_Scattering": 34555, "Helix": 33,
        "Koi_Fish": 11950, "Light_Modulation": 142, "Low_Frequency_Burst": 19834,
        "Low_Frequency_Lines": 2853, "No_Glitch": 11568, "Paired_Doves": 216,
        "Power_Line": 1582, "Repeating_Blips": 1061, "Scattered_Light": 68160,
        "Scratchy": 558, "Tomte": 30403, "Violin_Mode": 274,
        "Wandering_Line": 30, "Whistle": 6299,
    }

    return {
        "total_o4_evaluation": int(len(manifest)),
        "per_class_o4": {k: int(v) for k, v in dist.items()},
        "per_class_o3_train": o3_train,
        "classes_in_o4": sorted(dist.keys()),
        "classes_missing_from_o4": sorted(c for c in O3_CLASSES if c not in dist),
        "n_classes_in_o4": len(dist),
        "n_classes_missing": len(O3_CLASSES) - len(dist),
        "excluded_non_o3_classes": [
            {"name": s["name"], "count": s["count"]} for s in excluded
        ],
        "excluded_fraction": (
            sum(s["count"] for s in excluded) /
            max(1, sum(s["count"] for s in excluded) + len(manifest))
        ),
        "ml_confidence_stats": {
            "mean": float(conf.mean()),
            "median": float(conf.median()),
            "min": float(conf.min()),
            "std": float(conf.std()),
        },
        "detector_distribution": manifest["detector"].value_counts().to_dict(),
    }


def main():
    log.info("=" * 60)
    log.info("O4a Gravity Spy Evaluation Dataset Acquisition")
    log.info("=" * 60)

    META_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch O4 subject sets
    o4_sets = fetch_o4_subject_sets()
    with open(META_DIR / "o4_subject_sets.json", "w") as f:
        json.dump(o4_sets, f, indent=2)

    # Step 2: Select high-confidence sets
    selected, excluded = select_subject_sets(o4_sets)

    # Step 3: Fetch subject metadata
    progress_path = META_DIR / "o4_fetch_progress_task1.json"
    subjects = fetch_subjects_from_sets(selected, progress_path)
    if not subjects:
        log.error("No subjects fetched!")
        sys.exit(1)

    # Step 4: Download spectrograms
    log.info("\nDownloading 1.0s spectrograms...")
    success, failed = asyncio.run(download_all(subjects))

    # Step 5: Verify
    log.info("\nVerifying spectrograms...")
    verify = verify_spectrograms(subjects)

    # Step 6: Create manifest
    manifest = create_manifest(subjects)
    manifest.to_csv(MANIFEST_PATH, index=False)
    log.info(f"Manifest: {MANIFEST_PATH} ({len(manifest)} entries)")

    # Step 7: Class distribution
    class_dist = create_class_distribution(manifest, excluded)
    with open(CLASS_DIST_PATH, "w") as f:
        json.dump(class_dist, f, indent=2)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Total samples: {len(manifest)}")
    log.info(f"Classes: {class_dist['n_classes_in_o4']}/{len(O3_CLASSES)}")
    log.info(f"Missing: {class_dist['classes_missing_from_o4']}")
    log.info(f"Confidence: mean={class_dist['ml_confidence_stats']['mean']:.3f}")
    log.info(f"Verified: {verify['valid']}/{verify['checked']}")

    assert len(manifest) >= 1000, f"Too few samples: {len(manifest)}"
    assert class_dist["n_classes_in_o4"] >= 15, \
        f"Too few classes: {class_dist['n_classes_in_o4']}"
    log.info("All checks PASSED.")


if __name__ == "__main__":
    main()
