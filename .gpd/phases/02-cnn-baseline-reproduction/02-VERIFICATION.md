---
phase: 02-cnn-baseline-reproduction
plan_contract_ref: ".gpd/phases/02-cnn-baseline-reproduction/02-01-PLAN.md#/contract"
verified: 2026-03-17T06:15:00Z
status: gaps_found
score: 4/5 contract targets verified
consistency_score: 18/19 checks passed
independently_confirmed: 14/19 checks independently confirmed
confidence: medium
gaps:
  - subject_kind: acceptance_test
    subject_id: test-accuracy-reproduction
    expectation: "Overall accuracy in [95%, 99%] per contract pass_condition"
    expected_check: "Benchmark reproduction within anchor range"
    status: failed
    category: literature_agreement
    reason: "Overall accuracy 91.81% is below the [95%, 99%] range. The executor attributes this to O3 temporal split vs O1/O2 random split, 23 vs ~20 classes. This explanation is plausible but not conclusively validated -- no ablation (e.g., random split on same data) was performed to confirm the split method is the cause vs a pipeline issue."
    computation_evidence: "Recomputed accuracy from recall*support = 0.91810830, matches reported value exactly. Excluding 3 anomalous common classes (Air_Compressor, Power_Line, Paired_Doves) only raises accuracy to 91.96%, so class confusion is NOT the sole explanation for the gap."
    artifacts:
      - path: "results/02-cnn-baseline/metrics.json"
        issue: "overall_accuracy.value = 0.9181 < 0.95 contract threshold"
    missing:
      - "Ablation study: random split on same O3 data to isolate split effect vs dataset effect"
      - "Alternative: citation of published results on O3 data showing similar accuracy drop"
    severity: significant
  - subject_kind: acceptance_test
    subject_id: test-accuracy-reproduction
    expectation: "Rare threshold consistency between protocol and implementation"
    expected_check: "Protocol Section 6 defines N_rare=25 (Chirp only); implementation uses 200 (4 classes)"
    status: partial
    category: math_consistency
    reason: "The experimental protocol defines N_rare=25 training samples (making only Chirp rare), but the plan and implementation use rare_threshold_train=200 (making Chirp, Wandering_Line, Helix, Light_Modulation all rare). The plan explicitly overrides this with its own threshold. Both the protocol-consistent metric (Chirp-only F1=0.47) and the plan metric (4-class rare F1=0.30) are computed. The phase goal says 'rare-class performance gap' without specifying the threshold, so using 200 is defensible but creates a discrepancy with the locked protocol."
    computation_evidence: "With threshold=25: rare macro-F1 = 0.4706 (Chirp only). With threshold=200: rare macro-F1 = 0.3028 (4 classes). Both are valid baselines but downstream Phase 3 must use the SAME threshold."
    artifacts:
      - path: "docs/experimental_protocol.md"
        issue: "Section 6 says N_rare=25"
      - path: "configs/cnn_baseline.yaml"
        issue: "rare_threshold_train: 200"
    missing:
      - "Reconcile protocol N_rare=25 with plan threshold=200, or document the override"
    severity: minor
comparison_verdicts:
  - subject_kind: acceptance_test
    subject_id: test-accuracy-reproduction
    reference_id: ref-gravity-spy
    comparison_kind: benchmark
    verdict: tension
    metric: overall_accuracy
    threshold: "[0.95, 0.99]"
    notes: "91.81% vs ~97% published. Plausible explanation (O3 temporal split) but not conclusively validated."
  - subject_kind: acceptance_test
    subject_id: test-rare-class-gap
    reference_id: ref-gravity-spy
    comparison_kind: benchmark
    verdict: pass
    metric: rare_class_gap_pp
    threshold: "> 5pp"
    notes: "Gap = 50.5pp far exceeds 5pp threshold."
  - subject_kind: acceptance_test
    subject_id: test-focal-loss-implementation
    reference_id: null
    comparison_kind: other
    verdict: pass
    metric: relative_error
    threshold: "< 1e-6"
    notes: "All 3 analytic tests pass: rel errors 8.4e-7, 5.3e-8, 7.1e-8."
  - subject_kind: acceptance_test
    subject_id: test-metric-consistency
    reference_id: null
    comparison_kind: other
    verdict: pass
    metric: sklearn_vs_torchmetrics_diff
    threshold: "< 1e-6"
    notes: "Diff = 7.34e-08."
suggested_contract_checks:
  - check: "Ablation: random split on same O3 data to isolate temporal-split accuracy penalty"
    reason: "The 5.2pp accuracy gap from anchor could be pipeline or split -- only an ablation can distinguish"
    suggested_subject_kind: acceptance_test
    suggested_subject_id: test-accuracy-reproduction-ablation
    evidence_path: "results/02-cnn-baseline/metrics.json"
---

# Phase 02 Verification: CNN Baseline Reproduction

**Phase goal:** A fair CNN baseline is established using the same modern training recipe that will be applied to the ViT, reproducing published Gravity Spy accuracy within 2% and providing per-class F1 breakdown that quantifies the rare-class performance gap.

**Verified:** 2026-03-17T06:15:00Z
**Status:** gaps_found
**Confidence:** MEDIUM -- all internal metrics independently confirmed, but the anchor comparison (accuracy) shows unresolved tension.

**STATIC ANALYSIS NOTE:** Full code execution available via project venv. All computational checks executed with actual output.

---

## Contract Coverage

| Contract ID | Kind | Status | Confidence | Evidence |
|---|---|---|---|---|
| claim-rare-improvement | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Rare-class macro-F1=0.3028 established as baseline; per-class F1 table complete; gap=50.5pp documented |
| deliv-cnn-baseline-metrics | deliverable | VERIFIED | INDEPENDENTLY CONFIRMED | metrics.json exists with all required fields; macro-F1 recomputed from per-class data matches exactly (diff=0) |
| deliv-cnn-per-class-table | deliverable | VERIFIED | INDEPENDENTLY CONFIRMED | 23-row CSV with all 8 required columns; total test=48845 matches manifest; CSV-JSON F1 values match exactly |
| deliv-cnn-confusion-matrix | deliverable | VERIFIED | STRUCTURALLY PRESENT | File exists at figures/cnn_confusion_matrix.png (image, not computationally verifiable) |
| deliv-gap-analysis | deliverable | VERIFIED | INDEPENDENTLY CONFIRMED | Gap = (80.75% - 30.28%) = 50.47pp, recomputed from raw per-class data matches reported value exactly |
| test-accuracy-reproduction | acceptance_test | PARTIAL | INDEPENDENTLY CONFIRMED (value), FAILED (threshold) | Accuracy 91.81% confirmed but below [95%, 99%] range |
| test-rare-class-gap | acceptance_test | VERIFIED | INDEPENDENTLY CONFIRMED | Gap=50.5pp >> 5pp threshold |
| test-focal-loss-implementation | acceptance_test | VERIFIED | INDEPENDENTLY CONFIRMED | All 3 tests pass with rel errors < 1e-6 |
| test-metric-consistency | acceptance_test | VERIFIED | INDEPENDENTLY CONFIRMED | sklearn vs torchmetrics diff = 7.34e-08 |
| ref-gravity-spy | reference | COMPLETED | INDEPENDENTLY CONFIRMED | Compare+cite actions documented; tension noted |
| fp-overall-accuracy | forbidden_proxy | REJECTED | INDEPENDENTLY CONFIRMED | metrics.json primary_metric="macro_f1"; accuracy note contains "SANITY CHECK ONLY" and "fp-overall-accuracy" |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| results/02-cnn-baseline/metrics.json | Macro-F1, per-class F1, CIs, accuracy | EXISTS, SUBSTANTIVE | 342 lines, all fields present |
| results/02-cnn-baseline/per_class_f1.csv | 23 rows, 8+ columns | EXISTS, SUBSTANTIVE | 24 lines (header+23), all required columns |
| results/02-cnn-baseline/rare_class_gap_analysis.md | Gap quantification | EXISTS, SUBSTANTIVE | 79 lines with definition, summary, per-class table, anchor comparison |
| results/02-cnn-baseline/training_log.json | Per-epoch metrics | EXISTS, SUBSTANTIVE | 24 epochs, config, timing |
| figures/cnn_confusion_matrix.png | 23x23 matrix | EXISTS | Image file, cannot verify content computationally |
| figures/cnn_per_class_f1.png | Per-class bar chart | EXISTS | Image file |
| checkpoints/02-cnn-baseline/best_model.pt | Model weights | NOT VERIFIED | Would require loading torch checkpoint; skipped |

---

## Computational Verification Details

### Spot-Check Results (INDEPENDENTLY CONFIRMED)

| Expression | Test Point | Computed | Expected | Match |
|---|---|---|---|---|
| Focal loss FL(p=0.9, gamma=2, alpha=0.25) | p=0.9 | 0.00026340 | 0.00026340 | PASS (rel err 8.42e-7) |
| Focal loss FL(p=0.1, gamma=2, alpha=0.25) | p=0.1 | 0.46627346 | 0.46627348 | PASS (rel err 5.26e-8) |
| Focal loss gamma=0 vs weighted CE | p=0.7 | 0.08916874 | 0.08916874 | PASS (rel err 7.08e-8) |
| Macro-F1 = mean(per-class F1) | 23 classes | 0.6786466698931387 | 0.6786466698931387 | EXACT MATCH |
| Rare-class macro-F1 = mean(4 rare F1) | 4 classes | 0.3028322440087146 | 0.3028322440087146 | EXACT MATCH |
| Common-class avg F1 | 15 classes | 0.8074931179796714 | 0.8074931179796714 | EXACT MATCH |
| Gap (pp) = 100*(common - rare) | -- | 50.466087 | 50.466087 | EXACT MATCH |
| Overall accuracy from recall*support | 23 classes | 0.91810830 | 0.91810830 | EXACT MATCH |
| sklearn vs torchmetrics macro-F1 | -- | diff=7.34e-08 | < 1e-6 | PASS |

### Training Schedule Verification (INDEPENDENTLY CONFIRMED)

| Check | Expected | Actual | Match |
|---|---|---|---|
| Warmup LR epoch 1 | 0.0002 | 0.000200 | EXACT |
| Warmup LR epoch 2 | 0.0004 | 0.000400 | EXACT |
| Warmup LR epoch 3 | 0.0006 | 0.000600 | EXACT |
| Warmup LR epoch 4 | 0.0008 | 0.000800 | EXACT |
| Warmup LR epoch 5 | 0.001 | 0.001000 | EXACT |
| Cosine LR epoch 6 | 0.0009997266 | 0.0009997266 | EXACT |
| Cosine LR epoch 7 | 0.0009989068 | 0.0009989068 | EXACT |
| Cosine LR epoch 8 | 0.0009975415 | 0.0009975415 | EXACT |
| Cosine LR epoch 9 | 0.0009956320 | 0.0009956320 | EXACT |
| Best epoch | 14 | 14 (max val_macro_f1) | MATCH |
| Early stopping patience | 10 | stopped_at=24, best=14, gap=10 | MATCH |

### Cross-Checks Performed

| Result | Primary Method | Cross-Check | Agreement |
|---|---|---|---|
| Macro-F1 | sklearn f1_score(average='macro') | Manual mean of per-class F1 from JSON | EXACT |
| Macro-F1 | sklearn | torchmetrics MulticlassF1Score | diff=7.34e-08 |
| Overall accuracy | np.mean(preds==labels) | sum(recall_c * n_test_c) / total | EXACT |
| CSV per-class F1 | CSV file | JSON per-class_f1 dict | 0 mismatches across 23 classes |
| Focal loss | torch implementation | Python math.log analytic formula | rel errors < 1e-6 |

---

## Physics Consistency Summary

| Check | Status | Confidence | Notes |
|---|---|---|---|
| 5.1 Dimensional analysis | N/A | -- | ML classification; no physical units in equations |
| 5.2 Numerical spot-check | PASS | INDEPENDENTLY CONFIRMED | Macro-F1, rare-F1, gap, accuracy all recomputed from raw per-class data; exact match |
| 5.3 Limiting cases | PASS | INDEPENDENTLY CONFIRMED | Focal loss gamma=0 reduces to weighted CE (rel err 7.1e-8) |
| 5.4 Cross-check | PASS | INDEPENDENTLY CONFIRMED | sklearn vs torchmetrics, CSV vs JSON, recall*support vs accuracy |
| 5.5 Intermediate spot-check | PASS | INDEPENDENTLY CONFIRMED | Warmup schedule (5 epochs), cosine decay (4 epochs) match analytic formula exactly |
| 5.6 Symmetry | N/A | -- | Classification task, no symmetry constraints |
| 5.7 Conservation | PASS | INDEPENDENTLY CONFIRMED | Total test samples = 48845 across CSV, consistent with manifest |
| 5.8 Math consistency | PASS | INDEPENDENTLY CONFIRMED | All CI satisfy lower <= point <= upper, all F1/recall in [0,1] |
| 5.9 Convergence | WARNING | STRUCTURALLY PRESENT | Val macro-F1 oscillates heavily (std=0.044, 16 sign changes). Best epoch 14 is a local peak, not a stable plateau. Train loss mostly decreasing (2 non-monotonic steps in 23). |
| 5.10 Literature agreement | TENSION | INDEPENDENTLY CONFIRMED | 91.81% vs ~97% (Zevin et al. 2017). Plausible explanation but no ablation to confirm. |
| 5.11 Plausibility | PASS | INDEPENDENTLY CONFIRMED | All F1 in [0,1], all CIs valid, bootstrap resamples=10K |
| 5.12 Statistics | PASS with WARNING | INDEPENDENTLY CONFIRMED | Bootstrap CIs computed correctly. WARNING: Chirp CI width=0.602 (n_test=7), 1080Lines CI=[1,1] (n_test=6) -- small-sample CIs unreliable |
| 5.13 Thermodynamic consistency | N/A | -- | Not applicable |
| 5.14 Spectral/analytic | N/A | -- | Not applicable |
| Gate A: Cancellation | N/A | -- | Classification metrics, no floating-point cancellation risk |
| Gate B: Analytical-numerical cross | PASS | INDEPENDENTLY CONFIRMED | Focal loss analytic formula matches torch implementation |
| Gate C: Integration measure | N/A | -- | No coordinate transforms |
| Gate D: Approximation validity | PASS | STRUCTURALLY PRESENT | Focal loss gamma=2.0 is standard (Lin et al. 2017); no hyperparameter sweep but acknowledged in uncertainty markers |
| Config-protocol compliance | PASS | INDEPENDENTLY CONFIRMED | All 17 config parameters match locked protocol exactly |
| Forbidden proxy enforcement | PASS | INDEPENDENTLY CONFIRMED | primary_metric="macro_f1", accuracy note contains "SANITY CHECK ONLY" and "fp-overall-accuracy" |

---

## Forbidden Proxy Audit

| Proxy ID | Status | Evidence | Why It Matters |
|---|---|---|---|
| fp-overall-accuracy | REJECTED | metrics.json: primary_metric="macro_f1"; accuracy.note contains "SANITY CHECK ONLY -- not the primary metric (fp-overall-accuracy)"; gap_analysis.md leads with macro-F1 | Overall accuracy hides rare-class failure; 91.81% sounds good but rare-class F1 is only 30.3% |

---

## Comparison Verdict Ledger

| Subject ID | Comparison Kind | Verdict | Threshold | Notes |
|---|---|---|---|---|
| test-accuracy-reproduction | benchmark (ref-gravity-spy) | tension | [0.95, 0.99] | 91.81% < 95%. O3 temporal split plausible but unvalidated. |
| test-rare-class-gap | benchmark | pass | > 5pp | 50.5pp >> 5pp |
| test-focal-loss-implementation | consistency | pass | rel err < 1e-6 | All 3 tests pass |
| test-metric-consistency | consistency | pass | diff < 1e-6 | 7.34e-08 |

---

## Discrepancies Found

| Severity | Location | Computation Evidence | Root Cause | Suggested Fix |
|---|---|---|---|---|
| SIGNIFICANT | Overall accuracy 91.81% vs [95%,99%] anchor | Accuracy recomputed exactly as 0.91810830 | O3 temporal split harder than O1/O2 random split; additionally Air_Compressor(F1=0.15), Power_Line(F1=0.24), Paired_Doves(F1=0.10) are anomalously poor | Run ablation with random split on same O3 data to isolate cause; OR cite O3-specific published results showing similar degradation |
| MINOR | Rare threshold: protocol says 25, plan uses 200 | With threshold=25: rare=Chirp only (F1=0.47); threshold=200: 4 classes (F1=0.30) | Plan explicitly chose 200 for broader gap analysis | Document the override; ensure Phase 3 uses same threshold=200 for fair comparison |
| WARNING | Val macro-F1 oscillation (std=0.044, 16 sign changes in 23 epochs) | Epoch-by-epoch F1: range [0.486, 0.662] | LR=1e-3 likely too high after warmup; extreme val class imbalance (1080Lines=1, Chirp=1 val samples) amplifies noise | Consider LR=5e-4 or 3e-4 for future runs; or use stratified val sampling |
| WARNING | Chirp bootstrap CI width=0.602, 1080Lines CI=[1,1] with n_test=6 | Bootstrap produces degenerate CIs for n_test < 10 | Fundamental small-sample limitation | Already documented in gap_analysis.md. No fix possible without more data. |
| INFO | Paired_Doves: recall=0.92 but precision=0.05 (F1=0.10) | n_train=216, n_test=64 | Model predicts many non-Paired_Doves samples as Paired_Doves | Confusion matrix analysis would reveal which classes are confused; may benefit from class-specific threshold tuning |

---

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| MODL-01: CNN baseline with modern recipe | SATISFIED | ResNet-50v2 BiT trained with AdamW, cosine LR, fp16, focal loss, class-balanced sampling; all protocol parameters match |

---

## Anti-Patterns Found

| Pattern | Location | Severity | Physics Impact |
|---|---|---|---|
| No convergence plateau | training_log.json | WARNING | Val macro-F1 never stabilized; best epoch 14 is a local peak in oscillatory trajectory. Model selection is correct (best val F1) but the oscillation suggests training instability. |
| No hyperparameter sweep | configs/cnn_baseline.yaml | INFO | gamma=2.0 and lr=1e-3 used without tuning. Acknowledged in uncertainty markers. Standard practice for establishing baseline. |

---

## Expert Verification Required

None. All checks are computationally verifiable for this phase.

---

## Confidence Assessment

**Overall: MEDIUM**

Strengths:
- All internal metric computations independently confirmed with exact numerical matches
- Focal loss implementation verified against analytic formula (3 tests, all < 1e-6 rel error)
- sklearn/torchmetrics macro-F1 consistency confirmed (diff=7.34e-08)
- Training schedule (warmup + cosine decay) verified against analytic formula (all exact)
- Early stopping logic verified (best epoch 14, stopped at 24, patience=10)
- Protocol compliance verified (all 17 config parameters match)
- Forbidden proxy enforcement verified (macro-F1 is primary throughout)
- Gap computation verified from raw per-class data (exact match)
- CSV-JSON cross-consistency verified (0 mismatches across 23 classes)

Weaknesses:
- **Accuracy tension unresolved:** 91.81% vs [95%, 99%] anchor range. The O3 temporal split explanation is plausible but no ablation confirms it. This is not a blocker for the rare-class baseline claim (which is the decisive output) but it is a gap in the anchor comparison.
- **Val macro-F1 oscillation:** 16 sign changes in 23 epochs (std=0.044) suggests training instability. The best-epoch selection is correct but the oscillation raises questions about whether a better model exists at different LR.
- **Small-sample CIs:** Chirp (n_test=7, CI width=0.602) and 1080Lines (n_test=6, CI=[1,1]) have unreliable per-class CIs. The rare-class aggregate CI [0.2085, 0.3751] is more meaningful.

**Why not HIGH:** The anchor accuracy comparison failed its threshold, and no ablation confirms the cause. While the decisive metric (rare-class F1 baseline) is solid, the unresolved tension prevents HIGH confidence.

**Why not LOW:** Every internal computation was independently confirmed with exact numerical matches. The focal loss, metrics, training schedule, and gap computation are all verified. The accuracy tension has a plausible explanation and does not invalidate the decisive output.

---

## Gaps Summary

**Gap 1 (SIGNIFICANT): Accuracy anchor tension**

Overall accuracy 91.81% falls below the [95%, 99%] contract threshold for test-accuracy-reproduction. The executor attributes this to the O3 temporal split being harder than the O1/O2 random split used in Zevin et al. 2017, plus 23 vs ~20 classes. This explanation is plausible -- temporal splits are known to be harder because they prevent data leakage from temporally correlated glitches. However, no ablation was performed (e.g., random split on same O3 data) to quantify how much of the gap is split-method vs dataset-era.

Notably, excluding the 3 most anomalous common classes (Air_Compressor, Power_Line, Paired_Doves) only raises accuracy to 91.96%, showing these classes alone do not explain the gap. The deficit appears broadly distributed across many classes.

**Impact on downstream phases:** LOW. The decisive Phase 2 output is the rare-class macro-F1 baseline (0.3028), not overall accuracy. The forbidden proxy (fp-overall-accuracy) explicitly demotes accuracy to a sanity check. The gap quantification (50.5pp) and per-class F1 table are the artifacts consumed by Phase 3, and these are independently verified.

**Recommended action:** Document the tension as a known limitation. Optionally run a random-split ablation before Phase 5 (paper writing) to provide a quantitative explanation for the accuracy gap.

**Gap 2 (MINOR): Rare threshold discrepancy**

The locked protocol (experimental_protocol.md Section 6) defines N_rare=25 (making only Chirp rare), but the plan uses rare_threshold_train=200 (making 4 classes rare). The plan's choice is more informative (4 classes give a more robust rare-class average than 1 class), but it creates a discrepancy with the locked protocol. Phase 3 must use the same threshold for a fair comparison.

**Recommended action:** Add a note to the protocol or state.json documenting that the gap analysis uses threshold=200, distinct from the protocol's N_rare=25 which defines the "at-risk" classes for Chirp-specific tracking.
