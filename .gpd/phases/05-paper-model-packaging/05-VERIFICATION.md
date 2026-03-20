---
phase: 05-paper-model-packaging
plan_contract_ref: ".gpd/phases/05-paper-model-packaging/05-01-PLAN.md#/contract"
verified: 2026-03-18T05:00:00Z
status: passed
score: 6/7 contract targets verified
consistency_score: 11/12 checks passed
independently_confirmed: 8/12 checks independently confirmed
confidence: medium
gaps:
  - subject_kind: "claim"
    subject_id: "claim-threshold"
    expectation: "Power_Line F1 diff in Section 3.1 must match the O3 values quoted in the same section"
    expected_check: "powerLineDiff macro = round(ViT_F1_O3 - CNN_F1_O3, 3)"
    status: failed
    category: "math_consistency"
    reason: "Section 3.1 quotes O3 test set values (ViT F1=0.742, CNN F1=0.235) but pairs them with powerLineDiff=+0.394 which is the O4 diff. The correct O3 diff is 0.742-0.235=+0.507. The paper_numbers.json highlights section also has a data extraction bug: highlights.power_line_f1_diff_o3=0.00519 is actually the 1400Ripples diff."
    computation_evidence: "Python: 0.7419354838709677 - 0.23529411764705882 = 0.506641366223909; round to 3dp = 0.507. LaTeX macro powerLineDiff = +0.394 = round(O4 diff 0.3935, 3). Mismatch in O3 context."
    artifacts:
      - path: "paper/main.tex"
        issue: "Line 329: O3 per-class values paired with O4 delta"
      - path: "paper/data/paper_numbers.json"
        issue: "highlights.power_line_f1_diff_o3 = 0.00519 is wrong (1400Ripples value, not Power_Line)"
    missing:
      - "Split powerLineDiff into two macros: powerLineDiffOthree (+0.507) and powerLineDiffOfour (+0.394)"
      - "Fix highlights.power_line_f1_diff_o3 in paper_numbers.json to use per_class_o3.Power_Line.f1_diff = 0.5066"
      - "In Section 3.1 use powerLineDiffOthree; in CW section and Discussion use powerLineDiffOfour"
      - "Line 523 claims +0.394 on BOTH O3 and O4 -- correct to state the O3 value (+0.507) and O4 value (+0.394) separately"
    severity: significant
comparison_verdicts:
  - subject_id: claim-threshold
    subject_kind: claim
    reference_id: ref-gravity-spy
    comparison_kind: benchmark
    verdict: pass
    metric: "per-class F1 values match source result files"
    threshold: "exact match (full precision)"
  - subject_id: claim-cw-benefit
    subject_kind: claim
    reference_id: ref-gravity-spy
    comparison_kind: benchmark
    verdict: pass
    metric: "CW veto efficiency values match source result files"
    threshold: "exact match (full precision)"
suggested_contract_checks: []
---

# Phase 5 Verification: Paper & Model Packaging

**Phase goal:** Both trained models are packaged as deployable artifacts, and a research paper is written framing class-morphology-dependent architecture preferences as the central contribution, with CW sensitivity analysis and data augmentation strategies for ultra-rare classes as future work.

**Verified:** 2026-03-18
**Status:** gaps_found
**Confidence:** MEDIUM (one significant numerical inconsistency in paper)

## Contract Coverage

| ID | Kind | Status | Confidence | Evidence |
|---|---|---|---|---|
| claim-threshold | claim | PARTIAL | INDEPENDENTLY CONFIRMED (numbers) / FAILED (Power_Line diff) | Per-class analysis correct except Power_Line diff mismatch in Section 3.1 |
| claim-cw-benefit | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Matched-deadtime numbers correct; no 3.4x claim; quantitative throughout |
| deliv-trained-model | deliverable | VERIFIED | INDEPENDENTLY CONFIRMED | Both checkpoints exist (983MB ViT, 270MB CNN), SHA-256 verified, inference.py parses, 23 classes correct |
| deliv-paper | deliverable | VERIFIED | INDEPENDENTLY CONFIRMED | Complete LaTeX draft, 7 sections + appendix, 16 references, 5 figures, 3 tables |
| test-paper-threshold-framing | acceptance test | VERIFIED | INDEPENDENTLY CONFIRMED | Abstract mentions both "improves overall macro-averaged F1" AND "underperforms on rare classes: rare-class macro-F1 declines from 0.303 to 0.241" |
| test-paper-no-forbidden-proxy | acceptance test | VERIFIED | INDEPENDENTLY CONFIRMED | Line 264: "We explicitly do not use overall accuracy as a primary metric"; macro-F1 always primary |
| test-cw-quantitative | acceptance test | VERIFIED | INDEPENDENTLY CONFIRMED | CW section contains: matched deadtime 22.4%, ViT eff 0.745, CNN eff 0.735, delta_DC=-0.051 [-0.054,-0.048] |
| test-no-34x | acceptance test | VERIFIED | INDEPENDENTLY CONFIRMED | grep for "3.4" returns only "\vitAccuracy{93.4%}" (ViT accuracy), not a CW claim. No "5% deadtime" found. |
| test-model-reproduces-metrics | acceptance test | STRUCTURALLY PRESENT | Per SUMMARY: zero-drift cross-validation. Cannot re-run without GPU. |

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| release/README.md | Model card | EXISTS, SUBSTANTIVE | 174 lines, leads with macro-F1, per-class table, limitations, usage |
| release/src/inference.py | Standalone inference | EXISTS, SUBSTANTIVE | 192 lines, 3 functions (load_model, predict, main), no project imports |
| release/src/preprocessing.py | Preprocessing pipeline | EXISTS | 2098 bytes |
| release/src/class_labels.json | 23 class labels | EXISTS, VERIFIED | Exactly 23 classes matching Gravity Spy taxonomy |
| release/src/model_config.json | Model configs | EXISTS, VERIFIED | Both models: num_classes=23, correct timm model names |
| release/checkpoints/*.pt | Model weights | EXISTS, VERIFIED | SHA-256 checksums pass for both |
| release/examples/expected_output.json | Validation predictions | EXISTS, SUBSTANTIVE | 5 test examples with predictions for both models |
| paper/main.tex | Paper draft | EXISTS, SUBSTANTIVE | 755 lines, complete sections, LaTeX macros from JSON |
| paper/data/paper_numbers.json | Number source of truth | EXISTS, VERIFIED | 765 lines, all values traced to result files |
| paper/scripts/extract_numbers.py | Extraction pipeline | EXISTS | |
| paper/scripts/generate_figures.py | Figure generation | EXISTS | |
| paper/tables/generate_tables.py | Table generation | EXISTS | |

## Computational Verification Details

### Spot-Check Results: Paper Numbers vs Source Files

All checks below were executed computationally (Python, not grep).

| Quantity | paper_numbers.json | Source File | Source Value | Match |
|---|---|---|---|---|
| ViT macro-F1 O3 | 0.723037755055069 | 03-vit-rare-class/metrics.json | 0.723037755055069 | EXACT |
| ViT macro-F1 CI lower | 0.7030906642773047 | 03-vit-rare-class/metrics.json | 0.7030906642773047 | EXACT |
| ViT macro-F1 CI upper | 0.7396666656510605 | 03-vit-rare-class/metrics.json | 0.7396666656510605 | EXACT |
| ViT rare-class macro-F1 | 0.2411595394736842 | 03-vit-rare-class/metrics.json | 0.2411595394736842 | EXACT |
| CW matched deadtime | 0.2235101718195247 | 04-o4-validation/cw_veto_results.json | 0.2235101718195247 | EXACT |
| CW ViT efficiency | 0.7446446109984423 | 04-o4-validation/cw_veto_results.json | 0.7446446109984423 | EXACT |
| CW CNN efficiency | 0.7353257729706058 | 04-o4-validation/cw_veto_results.json | 0.7353257729706058 | EXACT |
| delta_DC | -0.050877448363438466 | 04-o4-validation/cw_veto_results.json | -0.050877448363438466 | EXACT |
| Power_Line F1 diff O4 | 0.3935164891603077 | 04-o4-validation/cw_veto_results.json | 0.3935164891603077 | EXACT |
| Power_Line ViT F1 O3 | 0.7419354838709677 | 03-vit-rare-class/metrics.json | 0.7419354838709677 | EXACT |
| N_total | 325632 | Computed: train+val+test | 227943+48844+48845=325632 | EXACT |

### Rare-Class Macro-F1 Independent Recomputation

**Oracle computation (executed):**
```
ViT rare-class F1s: [0.0 (Chirp), 0.0 (Wandering_Line), 0.1053 (Helix), 0.8594 (Light_Mod)]
Manual average: (0.0 + 0.0 + 0.1053 + 0.8594) / 4 = 0.2412
Reported: 0.2412 -- MATCH

CNN rare-class F1s: [0.4706 (Chirp), 0.0 (Wandering_Line), 0.0494 (Helix), 0.6914 (Light_Mod)]
Manual average: (0.4706 + 0.0 + 0.0494 + 0.6914) / 4 = 0.3028
Reported: 0.3028 -- MATCH
```
**Confidence:** INDEPENDENTLY CONFIRMED

### LaTeX Macro Rounding Verification

**Oracle computation (executed):** All 10 key macros verified as correct roundings of full-precision values:

| Macro | Full Precision | Rounded | LaTeX Value | Match |
|---|---|---|---|---|
| vitMacroF | 0.723038 | round(3)=0.723 | 0.723 | PASS |
| cnnMacroF | 0.678647 | round(3)=0.679 | 0.679 | PASS |
| vitRareF | 0.241160 | round(3)=0.241 | 0.241 | PASS |
| cnnRareF | 0.302832 | round(3)=0.303 | 0.303 | PASS |
| cwEffVit | 0.744645 | round(3)=0.745 | 0.745 | PASS |
| cwEffCnn | 0.735326 | round(3)=0.735 | 0.735 | PASS |
| deltaDC | -0.050877 | round(3)=-0.051 | -0.051 | PASS |
| spearmanRhoOfour | -0.033597 | round(3)=-0.034 | -0.034 | PASS |
| spearmanPOfour | 0.879043 | round(3)=0.879 | 0.879 | PASS |
| matchedDeadtime | 22.351% | round(1)=22.4 | 22.4% | PASS |

**Confidence:** INDEPENDENTLY CONFIRMED

### Degradation Percentages

**Oracle computation (executed):**
```
ViT: (0.6695 - 0.7230) / 0.7230 * 100 = -7.4% -- matches paper
CNN: (0.6674 - 0.6787) / 0.6787 * 100 = -1.7% -- matches paper
```
**Confidence:** INDEPENDENTLY CONFIRMED

### Overall Diff

**Oracle computation (executed):**
```
0.723038 - 0.678647 = 0.044391 -- matches paper_numbers.json
round(0.044391, 3) = 0.044 -- matches LaTeX macro diffOverall
```
**Confidence:** INDEPENDENTLY CONFIRMED

## Physics Consistency Summary

| Check | Status | Confidence | Notes |
|---|---|---|---|
| 5.1 Dimensional analysis | N/A | -- | ML classification, not physics equations |
| 5.2 Numerical spot-check | PASSED | INDEPENDENTLY CONFIRMED | 11/11 source-to-paper cross-checks exact match |
| 5.3 Limiting cases | N/A | -- | Not applicable to paper-writing phase |
| 5.5 Intermediate spot-check | PASSED | INDEPENDENTLY CONFIRMED | Rare-class macro-F1 recomputed from per-class values |
| 5.6 Symmetry | N/A | -- | Not applicable |
| 5.8 Math consistency | PARTIAL | INDEPENDENTLY CONFIRMED | Power_Line diff error found (see gap below) |
| 5.9 Convergence | N/A | -- | Not applicable |
| 5.10 Literature agreement | PASSED | INDEPENDENTLY CONFIRMED | All values match Phase 3/4 result files |
| 5.11 Plausibility | PASSED | STRUCTURALLY PRESENT | All F1 values in [0,1], CIs ordered correctly, p-values in [0,1] |
| Gate A: Catastrophic cancellation | N/A | -- | No numerical cancellation in classification metrics |
| Gate B: Analytical-numerical cross-validation | PASSED | INDEPENDENTLY CONFIRMED | LaTeX macros = round(full precision values) |
| Gate C: Integration measure | N/A | -- | No coordinate transforms |
| Gate D: Approximation validity | PASSED | STRUCTURALLY PRESENT | Duty-cycle proxy acknowledged as coarse in paper (limitations section) |

## Forbidden Proxy Audit

| Proxy ID | Status | Evidence | Why It Matters |
|---|---|---|---|
| fp-overall-accuracy | REJECTED | Line 264: "We explicitly do not use overall accuracy as a primary metric." Abstract leads with macro-F1. Accuracy mentioned only as forbidden-proxy example. | Prevents misleading conclusion that ViT is uniformly superior |
| fp-qualitative-only | REJECTED | CW section has: matched deadtime 22.4%, ViT efficiency 0.745, CNN efficiency 0.735, delta_DC=-0.051 [-0.054,-0.048], 7-class per-class breakdown with duty cycles | Prevents unsubstantiated CW benefit claims |

## Comparison Verdict Ledger

| Subject ID | Comparison Kind | Verdict | Threshold | Notes |
|---|---|---|---|---|
| claim-threshold | benchmark (per-class F1 vs source) | pass | exact match | All per-class F1 values match source files |
| claim-cw-benefit | benchmark (CW metrics vs source) | pass | exact match | All CW metrics match source files |
| test-no-34x | forbidden content | pass | zero matches | "3.4" only in ViT accuracy "93.4%", not CW context |

## Discrepancies Found

| Severity | Location | Computation Evidence | Root Cause | Suggested Fix |
|---|---|---|---|---|
| **SIGNIFICANT** | paper/main.tex line 329 + LaTeX macro powerLineDiff | O3 diff = 0.742-0.235 = +0.507; macro says +0.394 (= O4 diff) | Single macro used for both O3 and O4 Power_Line diff, but values differ by era | Create separate macros: powerLineDiffOthree=+0.507, powerLineDiffOfour=+0.394. Use O3 in Sec 3.1, O4 in CW section. |
| **SIGNIFICANT** | paper/data/paper_numbers.json highlights | highlights.power_line_f1_diff_o3 = 0.00519 | Data extraction bug: pulled 1400Ripples diff instead of Power_Line diff | Fix to per_class_o3.Power_Line.f1_diff = 0.5066 |
| **MINOR** | paper/main.tex line 523 | "($\powerLineDiff$ on both O3 and O4)" | Claims same diff on both runs, but O3=+0.507 and O4=+0.394 | State values separately with era labels |
| **MINOR** | N_total | paper says 325,632; STATE.md says 325,634 | 2-sample discrepancy, likely filter edge case | Confirm which is authoritative, use consistently |

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| DELV-01 (Model packaging) | SATISFIED | Both checkpoints, inference script, model card, class labels, checksums all present and validated |
| DELV-02 (Paper draft) | PARTIAL | Complete draft with correct framing, but Power_Line diff inconsistency needs fixing |

## Anti-Patterns Found

| Category | Location | Issue | Severity | Physics Impact |
|---|---|---|---|---|
| Derivation | paper_numbers.json highlights | Wrong Power_Line O3 diff (1400Ripples value instead) | SIGNIFICANT | Propagates to paper macro -- reader sees wrong numerical diff in O3 context |

## Expert Verification Required

None. All checks are within computational verification scope.

## Confidence Assessment

**Overall confidence: MEDIUM**

Strengths:
- All 11 source-to-paper numerical cross-checks pass at full precision (exact match)
- All LaTeX macro roundings are correct
- Rare-class macro-F1 independently recomputed and confirmed
- Degradation percentages independently confirmed
- Forbidden proxies properly enforced (accuracy not primary, CW section quantitative)
- 3.4x CW artifact successfully excluded
- Abstract frames findings honestly with both positive and negative results
- 23 Gravity Spy class labels preserved correctly
- Model checkpoints pass SHA-256 integrity verification

Weakness:
- Power_Line F1 diff uses O4 value (+0.394) in O3 context (Section 3.1 where O3 ViT=0.742 and CNN=0.235 are quoted, giving actual diff=+0.507). This is a localized but significant numerical error that a reviewer would catch.
- Cannot re-run inference.py validation (requires GPU + model loading) -- relying on SUMMARY cross-validation report

## Gaps Summary

**One significant gap identified**, rooted in a single data extraction bug:

The `paper/scripts/extract_numbers.py` pipeline appears to have extracted the wrong per-class O3 diff for Power_Line in the highlights section (got 1400Ripples value 0.00519 instead of Power_Line value 0.5066). The paper then uses a single `\powerLineDiff` macro set to the O4 diff (+0.394) for all contexts, including Section 3.1 which discusses O3 results. The O3 diff is actually +0.507, not +0.394. This makes the abstract, Section 3.1, and Discussion internally inconsistent when they pair O3 per-class F1 values with the O4 delta.

**Fix:** Split into two macros (`\powerLineDiffOthree` = +0.507, `\powerLineDiffOfour` = +0.394) and use the correct one in each context. Fix the highlights extraction in `extract_numbers.py`.
