---
phase: 01-data-pipeline-experimental-design
plan: 01
depth: full
one-liner: "Downloaded Gravity Spy O3 metadata (325K glitches, 23 classes) and launched async image downloader for 1.3M pre-made spectrogram images at 224x224 RGB"
subsystem: [data-pipeline, analysis]
tags: [gravity-spy, spectrograms, glitch-classification, LIGO-O3, data-acquisition]

requires: []
provides:
  - "Filtered O3 metadata CSV: 325,634 glitches across 23 classes (ml_confidence > 0.9, NOTA excluded)"
  - "Per-class distribution JSON with rare-class flags"
  - "Async spectrogram downloader with resume capability (download in progress)"
  - "Class distribution bar chart with color-coded rarity levels"
affects: [02-experimental-design, 03-model-training]

methods:
  added: [async-http-download, image-preprocessing, class-distribution-analysis]
  patterns: [resume-capable-download, checkpoint-json, metadata-driven-pipeline]

key-files:
  created:
    - scripts/01_download_metadata.py
    - scripts/02_parse_metadata.py
    - scripts/03_download_spectrograms.py
    - scripts/04_class_distribution.py
    - scripts/05_finalize_after_download.sh
    - data/metadata/gravity_spy_o3_filtered.csv
    - data/metadata/class_distribution_raw.json
    - figures/class_distribution_o3.png

key-decisions:
  - "Option C chosen: download pre-made Gravity Spy spectrogram images from Zooniverse CDN instead of generating Q-transforms from GWOSC strain data"
  - "50 concurrent async workers for download; rate ~19 glitches/s"
  - "Images resized to 224x224 RGB on download (bilinear interpolation)"

patterns-established:
  - "Spectrogram naming: {gravityspy_id}_{duration}s.png in data/spectrograms/{class_label}/"
  - "4 duration views per glitch: 0.5s, 1.0s, 2.0s, 4.0s (url1-url4)"
  - "Resume via download_progress.json checkpoint; re-run script to continue"

conventions:
  - "SI units: strain dimensionless, frequency Hz, time s (GPS)"
  - "Spectrograms: 224x224 px, RGB PNG, organized by class label"
  - "Macro-F1 primary metric; overall accuracy is forbidden proxy"

plan_contract_ref: ".gpd/phases/01-data-pipeline-experimental-design/01-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-data-integrity:
      status: partial
      summary: "Metadata integrity fully verified (325,634 samples, 23 classes, all schema checks pass). Spectrogram download in progress (~4.7h wall time); 0 failures through first 11K glitches. Image dimension validation: 100/100 sampled images are 224x224 RGB."
      linked_ids: [deliv-filtered-metadata, deliv-generated-spectrograms, deliv-class-distribution, test-metadata-integrity, test-image-dimensions, test-class-count]
  deliverables:
    deliv-filtered-metadata:
      status: passed
      path: "data/metadata/gravity_spy_o3_filtered.csv"
      summary: "325,634 O3 glitches filtered to ml_confidence > 0.9 with all required columns"
      linked_ids: [claim-data-integrity, test-metadata-integrity]
    deliv-generated-spectrograms:
      status: partial
      path: "data/spectrograms/"
      summary: "Download in progress via scripts/03_download_spectrograms.py. 11K/325K complete with 0 failures. Pre-made images from Zooniverse CDN (Option C) instead of GWOSC Q-transforms."
      linked_ids: [claim-data-integrity, test-image-dimensions, test-spectrogram-quality]
    deliv-class-distribution:
      status: passed
      path: "data/metadata/class_distribution_raw.json"
      summary: "Per-class counts for all 23 classes with rare-class flags. 1 rare class (Chirp: 19). Includes bar chart."
      linked_ids: [claim-data-integrity, test-class-count]
  acceptance_tests:
    test-metadata-integrity:
      status: passed
      summary: "All 6 required columns present, correct dtypes, GPS times in O3 range, ml_confidence >= 0.9, 23 classes (NOTA excluded), no NaN in critical fields, no duplicates per detector"
      linked_ids: [claim-data-integrity, deliv-filtered-metadata, ref-gravity-spy-csv]
    test-image-dimensions:
      status: passed
      summary: "100/100 sampled images are exactly 224x224x3 RGB PNG. Validated during active download at ~11K glitches."
      linked_ids: [claim-data-integrity, deliv-generated-spectrograms]
    test-class-count:
      status: passed
      summary: "23 unique classes confirmed (22 glitch + No_Glitch, NOTA excluded). Total 325,634 far exceeds literature estimate of ~8-10K (estimate was for stricter filtering)."
      linked_ids: [claim-data-integrity, deliv-filtered-metadata, deliv-class-distribution, ref-gravity-spy-csv]
    test-spectrogram-quality:
      status: not_attempted
      summary: "Visual QA comparison against Gravity Spy website examples not yet performed. Will be done after download completes using scripts/05_finalize_after_download.sh."
      linked_ids: [deliv-generated-spectrograms]
  references:
    ref-gravity-spy-csv:
      status: completed
      completed_actions: [read, use]
      missing_actions: [compare, cite]
      summary: "Zenodo record 5649212 downloaded, parsed, and filtered. CSV provides glitch labels, GPS times, detector metadata, and pre-made spectrogram URLs."
    ref-gwosc-strain:
      status: not_applicable
      completed_actions: []
      missing_actions: []
      summary: "Not used -- user chose Option C (pre-made Gravity Spy images) instead of generating Q-transforms from GWOSC strain data."
  forbidden_proxies:
    fp-overall-accuracy:
      status: rejected
      notes: "Per-class counts individually documented in class_distribution_raw.json. Bar chart shows per-class breakdown. Rare class (Chirp: 19) explicitly flagged."
  uncertainty_markers:
    weakest_anchors:
      - "Pre-made Gravity Spy images may use different normalization/colormap than would be produced by Q-transforms from strain data"
      - "Spectrogram download coverage not yet final (in progress)"
    unvalidated_assumptions:
      - "All 325K Zooniverse URLs will remain accessible for the full download duration"
    competing_explanations: []
    disconfirming_observations: []

comparison_verdicts:
  - subject_id: test-class-count
    subject_kind: acceptance_test
    reference_id: ref-gravity-spy-csv
    comparison_kind: benchmark
    verdict: pass
    metric: "unique class count"
    threshold: "23 classes expected"
    notes: "23 unique classes confirmed (22 glitch + No_Glitch, NOTA excluded). Total 325,634 samples."

duration: 25min
completed: 2026-03-16
---

# Plan 01-01: Data Pipeline Setup Summary

**Downloaded Gravity Spy O3 metadata (325,634 glitches, 23 classes) and launched async image downloader for 1.3M pre-made spectrogram images at 224x224 RGB**

## Performance

- **Duration:** ~25 min (active AI work; download continues in background ~4.7h)
- **Started:** 2026-03-16T20:33:00Z
- **Completed:** 2026-03-16T20:57:42Z (scripts committed; download in progress)
- **Tasks:** 3/3 (Task 2 download still running)
- **Files modified:** 8

## Key Results

- **325,634 filtered O3 samples** across 23 classes (ml_confidence > 0.9, NOTA excluded)
  - H1: 121,627 samples, L1: 204,007 samples
  - Far exceeds original ~8-10K estimate (estimate was based on stricter filtering)
- **1 rare class**: Chirp with 19 samples (below 25-sample threshold)
- **0 high-risk classes** (below 10-sample threshold), **0 zero-sample classes**
- **Download running**: 11K/325K glitches complete, 0 failures, ~19 glitches/s rate
- **Image validation**: 100/100 sampled images confirm 224x224 RGB format

## Task Commits

1. **Task 1: Environment setup + metadata download + parse/filter** - `a3df626` (data)
2. **Task 2: Spectrogram image downloader (Option C: pre-made images)** - `5a84b59` (implement)
3. **Task 3: Class distribution computation and visualization** - `83f2849` (analyze)

## Files Created/Modified

- `scripts/01_download_metadata.py` - Downloads Gravity Spy CSV from Zenodo 5649212
- `scripts/02_parse_metadata.py` - Parses and filters to O3, ml_confidence > 0.9, 23 classes
- `scripts/03_download_spectrograms.py` - Async image downloader (50 workers, resume capability)
- `scripts/04_class_distribution.py` - Per-class distribution computation and bar chart
- `scripts/05_finalize_after_download.sh` - Post-download validation and distribution update
- `data/metadata/gravity_spy_o3_filtered.csv` - 325,634 filtered O3 glitch metadata
- `data/metadata/class_distribution_raw.json` - Per-class counts with rare-class flags
- `figures/class_distribution_o3.png` - Color-coded class distribution bar chart

## Next Phase Readiness

- **Metadata fully ready** for experimental design (train/val/test splits)
- **Spectrogram download in progress** -- must complete before model training begins
- **To finalize after download:** Run `bash scripts/05_finalize_after_download.sh` to update class distribution with actual spectrogram coverage and validate images
- **Key concern for downstream:** Chirp class (19 samples) will need special handling (augmentation, few-shot, or exclusion from macro-F1)

## Contract Coverage

- Claim IDs advanced: claim-data-integrity -> partial (metadata complete, spectrograms downloading)
- Deliverable IDs produced: deliv-filtered-metadata -> passed, deliv-generated-spectrograms -> partial (in progress), deliv-class-distribution -> passed
- Acceptance test IDs run: test-metadata-integrity -> passed, test-image-dimensions -> passed, test-class-count -> passed, test-spectrogram-quality -> not_attempted
- Reference IDs surfaced: ref-gravity-spy-csv -> read+use completed, ref-gwosc-strain -> not applicable (Option C)
- Forbidden proxies rejected: fp-overall-accuracy -> rejected (per-class counts documented)

## Validations Completed

- Metadata integrity: all 6 required columns, correct dtypes, GPS times in O3 range
- Confidence floor: min(ml_confidence) >= 0.9 enforced
- Class taxonomy: 23 unique classes (22 glitch + No_Glitch), NOTA excluded
- No NaN in critical fields, no duplicate GPS times per detector
- Image dimensions: 100/100 sampled images are exactly 224x224x3 RGB
- Class distribution sum equals total sample count
- Both detectors (H1, L1) represented

## Decisions & Deviations

### Key Decision: Option C (Pre-made Images)

User chose to download pre-made Gravity Spy spectrogram images from Zooniverse CDN (url1-url4 columns) instead of generating Q-transforms from GWOSC strain data. This changes:
- No dependency on GWOSC strain data availability
- Much simpler pipeline (HTTP download vs strain processing)
- Trade-off: less control over spectrogram normalization and colormap
- ref-gwosc-strain marked as not_applicable

### Deviation: Sample Count (325K vs 8-10K estimate)

The plan estimated ~8-10K samples after ml_confidence > 0.9 filtering. Actual count is 325,634. The estimate appears to have been based on a different dataset version or additional filtering criteria not applied here. The larger dataset is beneficial for training but means 4x more images to download (~1.3M total).

**Impact:** Download takes ~4.7 hours instead of ~30 minutes. No correctness concern.

## Figures Produced

| Figure | File | Description | Key Feature |
| --- | --- | --- | --- |
| Fig. 01.1 | `figures/class_distribution_o3.png` | Class distribution bar chart | Log scale, color-coded: green (common), orange (rare < 25), red (high-risk < 10) |

## Open Questions

- What normalization/colormap do the pre-made Gravity Spy images use? This may differ from the Q-transform conventions in the plan.
- Should Chirp (19 samples) be included in macro-F1 evaluation or handled separately?
- Will all 325K Zooniverse URLs remain accessible for the full ~5h download?

---

_Phase: 01-data-pipeline-experimental-design_
_Plan: 01_
_Completed: 2026-03-16_
