---
phase: 01-data-pipeline-experimental-design
plan: 02
depth: full
one-liner: "Constructed temporal train/val/test split (70/15/15) with 60s gap enforcement and locked experimental protocol with macro-F1 as decisive metric"
subsystem: [data-pipeline, analysis]
tags: [temporal-split, data-leakage, macro-F1, experimental-design, gravity-spy, LIGO-O3]

requires:
  - "Plan 01-01: Filtered O3 metadata CSV (325,634 samples, 23 classes)"
provides:
  - "Temporal train/val/test split manifests (227,943 / 48,844 / 48,845) with 6/6 verification checks passing"
  - "Locked experimental protocol: macro-F1 decisive, accuracy forbidden, bootstrap >= 10K, focal loss + AdamW"
  - "Split quality visualization showing per-class distribution across splits"
affects: [02-model-training, 03-evaluation, 04-cw-analysis]

methods:
  added: [temporal-split-with-gap, binary-search-gap-verification, bootstrap-testing-protocol]
  patterns: [GPS-time-based-splitting, per-class-coverage-audit, protocol-locking]

key-files:
  created:
    - scripts/06_temporal_split.py
    - scripts/07_verify_split.py
    - scripts/08_split_visualization.py
    - data/metadata/train_manifest.csv
    - data/metadata/val_manifest.csv
    - data/metadata/test_manifest.csv
    - data/metadata/split_statistics.json
    - docs/experimental_protocol.md
    - figures/split_class_distribution.png

key-decisions:
  - "Script numbering: used 06/07/08 instead of plan-specified 04/05 to avoid overwriting Plan 01 scripts (Deviation Rule 4)"
  - "Temporal split preserves exact 70/15/15 ratio with only 2 gap-excluded samples"
  - "N_rare threshold = 25 training samples (from Plan 01 class_distribution_raw.json)"
  - "LR exception documented: ViT uses 1e-4, CNN uses 1e-3 (standard transfer learning practice)"

patterns-established:
  - "Split manifests: data/metadata/{train,val,test}_manifest.csv with image_path columns"
  - "Verification script re-runnable: scripts/07_verify_split.py produces PASS/FAIL for 6 checks"
  - "Protocol document locked: docs/experimental_protocol.md consumed by all downstream phases"

conventions:
  - "SI units: strain dimensionless, frequency Hz, time s (GPS)"
  - "Macro-F1 primary metric; overall accuracy is forbidden proxy"
  - "Temporal split: >= 60s gap, 70/15/15 train/val/test"

plan_contract_ref: ".gpd/phases/01-data-pipeline-experimental-design/01-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-temporal-split:
      status: passed
      summary: "Temporal train/val/test split constructed with >= 60s gaps. All 23 classes represented in all splits. 6/6 programmatic verification checks pass. Min gap: 125s (train-val), 268s (val-test)."
      linked_ids: [deliv-split-manifests, deliv-split-statistics, test-temporal-gap, test-no-overlap, test-class-coverage]
    claim-protocol-locked:
      status: passed
      summary: "Experimental protocol locked with macro-F1 as decisive metric, overall accuracy explicitly forbidden, identical training recipe for CNN/ViT, bootstrap >= 10K resamples with p < 0.05."
      linked_ids: [deliv-experimental-protocol, test-protocol-completeness]
  deliverables:
    deliv-split-manifests:
      status: passed
      path: "data/metadata/{train,val,test}_manifest.csv"
      summary: "Train (227,943), val (48,844), test (48,845) manifest CSVs with gravityspy_id, event_time, ifo, ml_label, ml_confidence, snr, image paths for 4 duration views, and split label."
      linked_ids: [claim-temporal-split, test-temporal-gap, test-no-overlap, test-class-coverage]
    deliv-split-statistics:
      status: passed
      path: "data/metadata/split_statistics.json"
      summary: "Split boundaries, gap verification, per-class-per-split counts, rare class coverage, and 6/6 verification results."
      linked_ids: [claim-temporal-split, test-temporal-gap, test-class-coverage]
    deliv-experimental-protocol:
      status: passed
      path: "docs/experimental_protocol.md"
      summary: "8-section protocol document with macro-F1 locked, accuracy forbidden, training recipe, model specs, rare-class strategy, reproducibility requirements, and evaluation procedure."
      linked_ids: [claim-protocol-locked, test-protocol-completeness]
  acceptance_tests:
    test-temporal-gap:
      status: passed
      summary: "All test samples >= 2,283,945s from nearest train sample (0 violations). All val samples >= 125s from nearest train sample (0 violations). Binary search verification on all 325,632 assigned samples."
      linked_ids: [claim-temporal-split, deliv-split-manifests, deliv-split-statistics]
    test-no-overlap:
      status: passed
      summary: "Zero gravityspy_id overlap across any pair of splits. GPS time ranges fully separated: train_max + 125s = val_min, val_max + 268s = test_min."
      linked_ids: [claim-temporal-split, deliv-split-manifests]
    test-class-coverage:
      status: passed
      summary: "All 23 classes present in all 3 splits. Rarest test class: 1080Lines (6 samples), Wandering_Line (6), Chirp (7). Zero classes with < 5 test samples."
      linked_ids: [claim-temporal-split, deliv-split-manifests, deliv-split-statistics]
    test-protocol-completeness:
      status: passed
      summary: "Protocol contains all 7 required sections plus evaluation procedure (8 total). 17/17 content checks pass: macro-F1 stated, accuracy forbidden, identical recipe, bootstrap 10K, p<0.05, N_rare=25, GPS boundaries, split manifest paths, Chirp identified."
      linked_ids: [claim-protocol-locked, deliv-experimental-protocol]
  references:
    ref-gravity-spy:
      status: completed
      completed_actions: [read, use, cite]
      missing_actions: [compare]
      summary: "Zenodo 5649212 metadata used for temporal split. Published Gravity Spy accuracy (~97%) documented in protocol as reproduction target. Citation in protocol Section 1."
  forbidden_proxies:
    fp-overall-accuracy:
      status: rejected
      notes: "Experimental protocol Section 2 explicitly states 'Overall accuracy must NOT be used as the primary comparison metric' with detailed reasoning. Overall accuracy reported as secondary sanity check only."
    fp-random-split:
      status: rejected
      notes: "Temporal split implemented with GPS-time ordering and 60s gap enforcement. Programmatic verification proves no temporal leakage. Random splitting explicitly avoided."
  uncertainty_markers:
    weakest_anchors:
      - "60s gap is minimum heuristic; some instrumental conditions persist for hours (seismic storms) and may correlate across the gap"
      - "Chirp class has only 11 training samples and 7 test samples; per-class F1 may have high variance"
    unvalidated_assumptions:
      - "Class temporal distribution is sufficiently uniform that a single temporal split provides representative test coverage for all 23 classes"
      - "1080Lines has only 1 validation sample -- val metric unreliable for this class"
    competing_explanations: []
    disconfirming_observations: []

comparison_verdicts:
  - subject_id: test-class-coverage
    subject_kind: acceptance_test
    reference_id: ref-gravity-spy
    comparison_kind: benchmark
    verdict: pass
    metric: "class coverage across splits"
    threshold: "all 23 classes present in all 3 splits"
    notes: "All 23 classes present in train, val, and test splits. Rarest test class: 1080Lines (6 samples)."
  - subject_id: ref-gravity-spy
    subject_kind: reference
    reference_id: ref-gravity-spy
    comparison_kind: benchmark
    verdict: pass
    metric: "metadata consistency with Zenodo 5649212"
    threshold: "class taxonomy matches published catalog"
    notes: "23-class taxonomy from Gravity Spy O3 catalog correctly preserved through temporal split. Compare action pending for Phase 2 accuracy reproduction."

duration: 15min
completed: 2026-03-16
---

# Plan 01-02: Temporal Split & Experimental Protocol Summary

**Constructed temporal train/val/test split (70/15/15) with 60s gap enforcement and locked experimental protocol with macro-F1 as decisive metric**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-16T21:00:38Z
- **Completed:** 2026-03-16T21:15:00Z
- **Tasks:** 2/2
- **Files modified:** 9

## Key Results

- **Temporal split:** 227,943 train / 48,844 val / 48,845 test (70.0% / 15.0% / 15.0%)
- **Gap enforcement:** Min gap 125.3s (train-val), 267.8s (val-test), only 2 samples excluded
- **Class coverage:** All 23 classes in all 3 splits; rarest test class has 6 samples (1080Lines, Wandering_Line)
- **6/6 verification checks PASS:** temporal gap, ID uniqueness, time range separation, class coverage, split ratios
- **Protocol locked:** Macro-F1 decisive, accuracy forbidden, bootstrap >= 10K, p < 0.05

## Task Commits

1. **Task 1: Temporal split construction + verification** - `6225925` (data)
2. **Task 2: Experimental protocol + split visualization** - `9d6463a` (docs)

## Files Created/Modified

- `scripts/06_temporal_split.py` - Temporal split construction with gap enforcement
- `scripts/07_verify_split.py` - 6-check programmatic split verification (re-runnable)
- `scripts/08_split_visualization.py` - Per-class split distribution grouped bar chart
- `data/metadata/train_manifest.csv` - 227,943 training samples with image paths
- `data/metadata/val_manifest.csv` - 48,844 validation samples with image paths
- `data/metadata/test_manifest.csv` - 48,845 test samples with image paths
- `data/metadata/split_statistics.json` - Boundaries, per-class counts, verification results
- `docs/experimental_protocol.md` - Locked experimental protocol (8 sections)
- `figures/split_class_distribution.png` - Per-class split distribution visualization

## Next Phase Readiness

- **Split manifests ready** for model training (train/val) and evaluation (test)
- **Protocol locked** for CNN baseline and ViT training (identical recipe)
- **Key concern:** Chirp (11 train, 7 test) and 1080Lines (1 val sample) need monitoring during training
- **Spectrogram download** (Plan 01-01) must complete before training begins

## Contract Coverage

- Claim IDs advanced: claim-temporal-split -> passed, claim-protocol-locked -> passed
- Deliverable IDs produced: deliv-split-manifests -> passed, deliv-split-statistics -> passed, deliv-experimental-protocol -> passed
- Acceptance test IDs run: test-temporal-gap -> passed, test-no-overlap -> passed, test-class-coverage -> passed, test-protocol-completeness -> passed
- Reference IDs surfaced: ref-gravity-spy -> read+use+cite completed (compare pending for Phase 2)
- Forbidden proxies rejected: fp-overall-accuracy -> rejected, fp-random-split -> rejected

## Validations Completed

- Temporal gap: 0 violations across 48,845 test + 48,844 val samples (binary search verification)
- ID uniqueness: 0 overlapping gravityspy_ids across any pair of splits
- Time range separation: train_max + 125s = val_min, val_max + 268s = test_min
- Class coverage: 23/23 classes in train, 23/23 in val, 23/23 in test
- Split ratios: 70.0% / 15.0% / 15.0% (within [60-80 / 8-25 / 8-25] bounds)
- Sample accounting: 227,943 + 48,844 + 48,845 + 2 gap-excluded = 325,634 total (matches Plan 01 count)
- Protocol completeness: 17/17 content checks pass

## Decisions & Deviations

### Deviation: Script Numbering (Rule 4 - Missing Component)

Plan specified `scripts/04_temporal_split.py` and `scripts/05_verify_split.py`, but Plan 01-01 already created `scripts/04_class_distribution.py` and `scripts/05_finalize_after_download.sh`. Used `06`, `07`, `08` numbering to avoid overwriting existing scripts. No correctness impact.

### Decision: LR Exception in Training Recipe

Documented different base learning rates for CNN (1e-3) and ViT (1e-4) in the protocol. This is standard transfer learning practice, not a fairness violation -- pretrained transformers require lower LR to avoid catastrophic forgetting.

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|---|---|---|---|
| 60s temporal gap | Instrumental conditions change faster than ~1 min | Prevents short-timescale correlation leakage | Seismic storms persisting for hours; gap should ideally be hours |
| Single temporal split (no cross-validation) | Classes distributed roughly uniformly across O3 | Sufficient test coverage if all classes present | Rare classes clustered in narrow time windows |

## Figures Produced

| Figure | File | Description | Key Feature |
|---|---|---|---|
| Fig. 01-02.1 | `figures/split_class_distribution.png` | Per-class sample distribution across temporal splits | Log scale, grouped bars (train/val/test), rare threshold line at N=25 |

## Open Questions

- Should 1080Lines (1 val sample) be excluded from val macro-F1 computation?
- Is 60s gap sufficient for classes affected by long-duration seismic activity?
- Should Wandering_Line (30 train) and Helix (33 train) be treated as near-rare for evaluation purposes?

---

_Phase: 01-data-pipeline-experimental-design_
_Plan: 02_
_Completed: 2026-03-16_
