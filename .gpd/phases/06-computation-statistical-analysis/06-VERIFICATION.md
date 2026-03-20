---
phase: 06-computation-statistical-analysis
verified: 2026-03-18T13:45:00Z
status: passed
score: 2/2 contract claims verified
consistency_score: 10/10 physics checks passed
independently_confirmed: 5/10 checks independently confirmed
confidence: high
comparison_verdicts:
  - subject_id: claim-split-explains-gap
    subject_kind: claim
    reference_id: ref-gravity-spy
    comparison_kind: benchmark
    verdict: pass
    metric: "overall_accuracy"
    threshold: ">= 0.948 AND within [0.95, 0.99]"
  - subject_id: claim-rare-underpowered
    subject_kind: claim
    reference_id: null
    comparison_kind: consistency
    verdict: pass
    metric: "power_at_observed_effect"
    threshold: "< 0.80 for >= 2 of 4 smallest classes"
suggested_contract_checks: []
---

# Phase 6 Verification: Computation & Statistical Analysis

**Phase Goal:** Produce the two new quantitative results the paper revision requires: (1) a random-split ablation confirming the temporal split explains the 5pp accuracy gap from published benchmarks, and (2) a power analysis quantifying the minimum detectable effect size for the rare-class comparison.

**Verified:** 2026-03-18
**Status:** PASSED
**Confidence:** HIGH
**Profile:** review (maximum rigor)
**Research Mode:** balanced
**Autonomy:** balanced

## Contract Coverage

| ID | Kind | Status | Confidence | Evidence |
|---|---|---|---|---|
| claim-split-explains-gap | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Accuracy 0.9544, gap +3.63pp, within [0.95, 0.99] |
| claim-rare-underpowered | claim | VERIFIED | INDEPENDENTLY CONFIRMED | All 4/4 smallest classes power < 0.80; analytical cross-check confirms |
| deliv-random-split-accuracy | deliverable | VERIFIED | INDEPENDENTLY CONFIRMED | JSON exists, all required fields present, values internally consistent |
| deliv-power-analysis | deliverable | VERIFIED | INDEPENDENTLY CONFIRMED | JSON exists, all required fields present, 8 classes + aggregate |
| test-accuracy-range | acceptance_test | PASSED | INDEPENDENTLY CONFIRMED | 0.9544 >= 0.948 AND in [0.95, 0.99] |
| test-power-insufficient | acceptance_test | PASSED | INDEPENDENTLY CONFIRMED | 4/4 smallest classes underpowered (need >= 2) |
| ref-gravity-spy | reference | COMPLETED | STRUCTURALLY PRESENT | 95.44% compared to published 95-99%; at lower end, consistent with stricter filtering |

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `results/.../random_split_ablation.json` | accuracy, macro-F1 with CIs | EXISTS, SUBSTANTIVE, INTEGRATED | 23 per-class F1s, all CIs, gap computation |
| `results/.../power_analysis.json` | MDE per class, power curves | EXISTS, SUBSTANTIVE, INTEGRATED | 8 classes + aggregate + sanity check |
| `results/.../ablation_summary.txt` | Human-readable summary | EXISTS, SUBSTANTIVE, INTEGRATED | 3 sections, values match JSON |
| `scripts/17_random_split_ablation.py` | Ablation script | EXISTS, SUBSTANTIVE | 515 lines, no stubs, no TODOs |
| `scripts/18_power_analysis.py` | Power analysis script | EXISTS, SUBSTANTIVE | 510 lines, no stubs, no TODOs |

## Computational Verification Details

### Spot-Check Results (Ablation JSON)

| Expression | Test | Computed | Expected | Match |
|---|---|---|---|---|
| accuracy_gap_pp | (0.9544 - 0.9181) * 100 | 3.630773 | 3.630773 (stored) | PASS |
| macro_f1 from per-class | mean(23 per-class F1 values) | 0.750701 | 0.750701 (stored) | PASS |
| CI ordering (accuracy) | CI_lo < point < CI_hi | 0.9525 < 0.9544 < 0.9563 | True | PASS |
| CI ordering (macro-F1) | CI_lo < point < CI_hi | 0.7273 < 0.7507 < 0.7716 | True | PASS |
| Split proportions | train/val/test | 0.7000/0.1500/0.1500 | 70/15/15 within 1% | PASS |
| Total samples | train + val + test | 325,634 | 325,634 | PASS |

### Spot-Check Results (Power Analysis JSON)

| Expression | Test | Computed | Expected | Match |
|---|---|---|---|---|
| 1080Lines critical_value | Analytical: recall=1.0 => F1=1.0 always => null diff=0 | 0.0 | 0.0 (stored) | PASS |
| 1080Lines power@0.25 | Analytical: 1 - 0.75^6 = 0.822 | 0.822 | 0.8188 (stored) | PASS (within 0.5%) |
| 1080Lines power@observed | Analytical: 1 - 0.938^6 = 0.319 | 0.319 | 0.310 (stored) | PASS (within 3%) |
| All power values | Bounded [0,1] | 99 values checked | All in [0,1] | PASS |
| n_test values | Match plan | 8/8 match | As specified | PASS |

### Independent Analytical Cross-Check

For 1080Lines (n=6, recall=1.0, FP=0), F1 is analytically deterministic:
- When all TP=6: F1=1.0. Any missed prediction: F1 < 1.0.
- Critical value: Under H0 (both models recall=1.0), |F1_a - F1_b| = 0 always. Critical value = 0.0. Stored: 0.0. **MATCH.**
- Power at delta=0.25: ViT recall=0.75. Power = P(at least one miss) = 1 - 0.75^6 = 0.8220. Stored: 0.8188. **MATCH within simulation noise (~0.4%).**
- Power at observed effect (delta=0.062): ViT recall=0.938. Power = 1 - 0.938^6 = 0.3189. Stored: 0.3100. **MATCH within simulation noise (~3%).**

This cross-check INDEPENDENTLY CONFIRMS the simulation methodology is correct.

### Power Curve Monotonicity and Saturation Analysis

Power curves show three distinct behaviors:
1. **Well-behaved monotonic** (1080Lines, 1400Ripples, Light_Modulation, Power_Line): Power increases monotonically with delta, reaching 0.80 at finite MDE.
2. **Saturating** (Helix at ~0.60, Air_Compressor at ~0.72): Power plateaus below 0.80 because when delta exceeds the baseline recall, ViT recall floors at 0. The null distribution variance at low recall is high enough that the critical value cannot be reliably exceeded.
3. **Perpetually low** (Wandering_Line, Chirp): n is so small that the critical value is extremely high (e.g., 0.667 for Wandering_Line), making detection nearly impossible at any effect size.

All three behaviors are physically correct consequences of the Binomial(n, recall) model and do NOT indicate bugs. MDE=Infinity is correctly assigned when the maximum power never reaches 0.80.

### Dimensional/Units Analysis

| Quantity | Units/Range | Consistent |
|---|---|---|
| Accuracy | Dimensionless [0,1] | YES |
| Macro-F1 | Dimensionless [0,1] | YES |
| accuracy_gap_pp | Percentage points | YES (correctly multiplied by 100) |
| Power | Dimensionless [0,1] | YES |
| MDE (delta_F1) | Dimensionless [0,1] or Inf | YES |
| n_test | Count (integer) | YES |
| bootstrap_resamples | Count (integer) | YES (10000) |
| Training time | Seconds | YES (3227s ~ 53.8 min) |

### Convention Compliance

| Convention | Lock Value | Artifact Status | Compliant |
|---|---|---|---|
| primary_metric | macro-F1 | Early stopping on macro-F1; macro-F1 reported with CI; per-class F1 tabulated | YES |
| forbidden_primary | overall accuracy | Accuracy used ONLY for benchmark comparison (appropriate: published benchmarks report accuracy) | YES |
| bootstrap | 10K resamples, seed=42 | Stored: resamples=10000, seed=42, confidence=0.95 | YES |
| ASSERT_CONVENTION (script 17) | primary_metric=macro_f1 | Line 15 | YES |
| ASSERT_CONVENTION (script 18) | primary_metric=macro_f1, alpha=0.05 | Line 13 | YES |

## Forbidden Proxy Audit

| Proxy ID | Status | Evidence |
|---|---|---|
| fp-no-ablation | REJECTED | CNN retrained on random split with identical hyperparameters; JSON contains actual computed metrics, not narrative |
| fp-no-power | REJECTED | 10K-iteration simulation computed for all 8 classes; MDE and power values are quantitative |
| fp-overall-accuracy-primary | REJECTED | Early stopping on macro-F1 (line 353 of script 17); accuracy used only for ref-gravity-spy comparison |

## Comparison Verdict Ledger

| Subject ID | Comparison Kind | Verdict | Threshold | Notes |
|---|---|---|---|---|
| claim-split-explains-gap | benchmark (ref-gravity-spy) | PASS | >= 0.948 AND in [0.95, 0.99] | 0.9544 at lower end of range; consistent with stricter data filtering |
| claim-rare-underpowered | consistency (power analysis) | PASS | power < 0.80 for >= 2/4 | 4/4 underpowered; all 8/8 rare classes underpowered |

## Backtracking Trigger Checks

| Trigger | Condition | Result | Fired? |
|---|---|---|---|
| Pipeline bug | Random-split accuracy < temporal (0.9181) | 0.9544 > 0.9181 | NO |
| Rare-class regression real | Power > 0.80 for most rare classes | 0/8 have power > 0.80 | NO |

## Anti-Patterns Found

| Category | Severity | File | Finding |
|---|---|---|---|
| None | - | - | No TODOs, FIXMEs, placeholders, suppressed warnings, or hardcoded magic numbers found |

## Discrepancies Found

None. All computed values match stored values within simulation noise (< 3% relative error for simulation-based quantities).

## Methodology Assessment

### Random-Split Ablation
- **Design:** Single-variable ablation (only split method changed, all hyperparameters frozen) -- sound experimental design
- **Limitation (documented):** Single seed (42); a seed sweep would strengthen the claim but is out of scope
- **Statistical validity:** Bootstrap CIs (10K resamples) properly quantify uncertainty on the point estimate

### Power Analysis
- **Method:** Simulation-based with permutation test critical values -- appropriate for F1 on tiny samples where parametric assumptions are violated
- **FP=0 assumption:** Conservative (overestimates F1, underestimates power). The underpowering conclusion is strengthened, not weakened, by this assumption.
- **Limitation (documented):** Assumes independent per-class F1 scores; bootstrap correlation structure not modeled
- **Sanity check:** n=100, delta=0.40 gives power=1.0 as expected

## Confidence Assessment

**Overall: HIGH**

Justification:
1. All numerical values cross-checked against JSON sources -- exact matches
2. Independent analytical derivation for 1080Lines confirms simulation methodology (critical value, power at 2 effect sizes all match within < 3%)
3. Split proportions, sample counts, and CI ordering all verified computationally
4. Power curve behaviors (monotonic, saturating, perpetually low) explained by the underlying Binomial model
5. No anti-patterns, no convention violations, no forbidden proxy usage
6. Both acceptance tests pass clearly (not borderline)
7. Backtracking triggers correctly not fired

The only factor preventing maximum confidence is the single-seed limitation for the ablation, which is a documented weakness and out of scope for this revision cycle.

## Requirements Coverage

| Requirement | Description | Status |
|---|---|---|
| COMP-01 | Random-split ablation for accuracy gap | SATISFIED |
| STAT-01 | Power analysis for rare-class comparison | SATISFIED |

---

_Verified by: GPD Phase Verifier_
_Phase: 06-computation-statistical-analysis_
_Date: 2026-03-18_
