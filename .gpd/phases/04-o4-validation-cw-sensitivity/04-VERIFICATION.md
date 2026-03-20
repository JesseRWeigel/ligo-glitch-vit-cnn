---
phase: 04-o4-validation-cw-sensitivity
plan_contract_ref: ".gpd/phases/04-o4-validation-cw-sensitivity/04-01-PLAN.md#/contract"
verified: 2026-03-17T18:30:00Z
status: gaps_found
score: 4/6 contract targets verified
consistency_score: 10/11 physics checks passed
independently_confirmed: 8/11 checks independently confirmed
confidence: medium
gaps:
  - subject_kind: acceptance_test
    subject_id: test-cw-improvement
    expectation: "Veto efficiency at 5% deadtime comparison must be at matched operating points"
    expected_check: "Both models evaluated at the SAME deadtime for fair comparison"
    status: failed
    category: plausibility
    reason: >
      The claimed "3.4x veto efficiency at 5% deadtime" (ViT=0.741 vs CNN=0.220) is
      comparing models at DIFFERENT operating points. ViT cannot achieve 5% deadtime
      in the threshold sweep 0.50-0.95; its minimum deadtime is 22.3%. The code uses
      np.interp(0.05, sorted_deadtimes, sorted_efficiencies), which clamps to the
      boundary value when 0.05 is below the ViT's minimum deadtime. The ViT value
      0.741 is actually its efficiency at 22.3% deadtime, not 5%.
      At matched deadtime (~22%), CNN interpolated efficiency is ~0.732, making the
      ratio ~1.01x, not 3.4x. The headline CW comparison number is misleading.
    computation_evidence: >
      ViT ROC last point: eff=0.7415, deadtime=0.2225 (threshold=0.95).
      CNN ROC last point: eff=0.2200, deadtime=0.0652 (threshold=0.95).
      np.interp(0.05, [0.2225,...], [0.7415,...]) returns 0.7415 (clamped).
      At matched deadtime=0.2225, CNN interpolated eff~0.732 (from CNN ROC data).
      Ratio at matched deadtime: 0.7415/0.732 = 1.01x.
    artifacts:
      - path: results/04-o4-validation/cw_veto_results.json
        issue: "efficiency_at_5pct_deadtime_vit is ViT efficiency at 22.3% deadtime, not 5%"
      - path: scripts/16_cw_veto_analysis.py
        issue: "Line 302: np.interp clamps out-of-range targets to boundary values silently"
    missing:
      - "Fix efficiency_at_deadtime() to return NaN or flag when target_deadtime is outside the ROC range"
      - "Report matched-deadtime comparison instead of or alongside the 5% figure"
      - "If keeping the 5% comparison, extend ViT ROC to higher thresholds (0.95-0.99) to reach 5% deadtime"
    severity: significant
comparison_verdicts:
  - subject_id: test-threshold-o4
    subject_kind: acceptance_test
    reference_id: ref-o3-per-class
    comparison_kind: cross_method
    verdict: fail
    metric: "Spearman rho"
    threshold: "p < 0.05"
    notes: "rho=-0.034, p=0.879. No correlation between training set size and ViT advantage on O4."
  - subject_id: test-o4-generalization
    subject_kind: acceptance_test
    reference_id: ref-o3-per-class
    comparison_kind: benchmark
    verdict: pass
    metric: "relative degradation"
    threshold: "<= 0.20"
    notes: "CNN: 1.7%, ViT: 7.4%. Both well within 20% threshold."
  - subject_id: claim-cw-benefit
    subject_kind: claim
    reference_id: ref-gravity-spy-cw
    comparison_kind: baseline
    verdict: tension
    metric: "delta_DC and matched-deadtime efficiency"
    threshold: "measurable improvement"
    notes: >
      Overall delta_DC=-0.051 favors CNN. The claimed 3.4x efficiency advantage
      at 5% deadtime is an artifact of np.interp boundary clamping. At matched
      deadtime, the models are nearly equivalent (~1.01x ratio).
      The genuine finding is that ViT can operate at a wider range of efficiency/deadtime
      tradeoffs (its ROC curve has higher floor efficiency), but this is not a simple
      "ViT is 3.4x better" claim.
suggested_contract_checks:
  - check: "Matched-deadtime veto efficiency comparison"
    reason: "The current comparison at '5% deadtime' uses different operating points for each model due to np.interp clamping"
    suggested_subject_kind: acceptance_test
    suggested_subject_id: test-cw-matched-deadtime
    evidence_path: "results/04-o4-validation/cw_veto_results.json"
---

# Phase 04 Verification: O4 Validation & CW Sensitivity

**Phase goal:** Both O3-trained models (ViT and CNN) are evaluated on O4 data to test whether the sample-efficiency threshold holds under distribution shift, and CW search data quality improvement from ViT-based glitch vetoes is quantified for the classes where ViT outperforms CNN.

**Verified:** 2026-03-17
**Status:** GAPS FOUND
**Confidence:** MEDIUM
**Profile:** review | **Autonomy:** balanced | **Mode:** balanced

## Contract Coverage

| ID | Kind | Status | Confidence | Evidence |
|----|------|--------|------------|----------|
| claim-threshold-o4 | claim | FAILED (honest) | INDEPENDENTLY CONFIRMED | Spearman rho=-0.034, p=0.879; recomputed and verified |
| claim-cw-benefit | claim | PARTIAL | MEDIUM | delta_DC=-0.051 favors CNN; "3.4x at 5% deadtime" is misleading (see gap) |
| test-threshold-o4 | acceptance_test | FAIL (correct) | INDEPENDENTLY CONFIRMED | rho/p recomputed from CSV; sign test 9/20 verified |
| test-o4-generalization | acceptance_test | PASS | INDEPENDENTLY CONFIRMED | CNN -1.7%, ViT -7.4% recomputed from per-class F1 |
| test-cw-improvement | acceptance_test | PARTIAL | LOW | Headline "3.4x" number is flawed; overall delta_DC is real but favors CNN |
| test-cw-quantitative | acceptance_test | PASS | INDEPENDENTLY CONFIRMED | Every CW claim backed by numbers (42 numeric values across 7 classes) |
| deliv-o4-comparison | deliverable | PASS | INDEPENDENTLY CONFIRMED | CSV exists, 23 classes, all F1/deg values verified |
| deliv-cw-analysis | deliverable | PASS | STRUCTURALLY PRESENT | Multi-panel PNG exists (256KB), non-trivial |
| deliv-cw-roc | deliverable | PASS | STRUCTURALLY PRESENT | ROC PNG exists (82KB), non-trivial |
| deliv-cw-data | deliverable | PASS | INDEPENDENTLY CONFIRMED | JSON with all metrics, verified computationally |
| fp-overall-accuracy | forbidden_proxy | REJECTED | INDEPENDENTLY CONFIRMED | Overall accuracy labeled SANITY_CHECK in o4_metrics.json |
| fp-qualitative-only | forbidden_proxy | REJECTED | INDEPENDENTLY CONFIRMED | 42 numeric values in CW class findings; no qualitative-only claims |

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| results/04-o4-validation/o4_comparison_table.csv | Per-class F1 table | EXISTS, SUBSTANTIVE | 23 classes + 3 MACRO rows, all columns populated |
| results/04-o4-validation/o4_metrics.json | Overall metrics | EXISTS, SUBSTANTIVE | 1.1KB, macro-F1, CIs, degradation data |
| results/04-o4-validation/o4_threshold_test.json | Spearman + sign test | EXISTS, SUBSTANTIVE | 4.4KB, rho, p, sensitivity analysis |
| results/04-o4-validation/cw_veto_results.json | CW veto metrics | EXISTS, SUBSTANTIVE | 11.9KB, per-class DC, ROC, combined metrics |
| figures/o4_threshold_scatter.png | Threshold scatter | EXISTS | 88KB |
| figures/o4_degradation_per_class.png | Degradation bars | EXISTS | 136KB |
| figures/cw_sensitivity_summary.png | CW summary | EXISTS | 257KB |
| figures/cw_veto_roc.png | Veto ROC | EXISTS | 82KB |
| scripts/14_acquire_o4_data.py | O4 data acquisition | EXISTS | Not verified (execution artifact) |
| scripts/15_evaluate_o4.py | O4 evaluation | EXISTS | Not verified (execution artifact) |
| scripts/16_cw_veto_analysis.py | CW analysis | EXISTS, VERIFIED | Bug found in efficiency_at_deadtime() |

## Computational Verification Details

### Spot-Check Results (Executed)

```
=== SPOT-CHECK 1: Macro-level degradation ===
CNN abs degradation: -0.01129 (matches CSV to machine precision) -- PASS
ViT abs degradation: -0.05357 (matches CSV to machine precision) -- PASS
CNN relative degradation: 1.663% (matches claimed 1.7%) -- PASS
ViT relative degradation: 7.409% (matches claimed 7.4%) -- PASS
Both < 20%: True -- PASS

=== SPOT-CHECK 2: Power_Line per-class ===
F1 diff O4 (ViT-CNN): +0.3935 (matches CSV) -- PASS
CNN degradation: +0.0958 (improved on O4, matches CSV) -- PASS

=== SPOT-CHECK 3: Chirp per-class ===
Chirp F1 diff O4: -0.3732 (matches CSV) -- PASS

=== SPOT-CHECK 4: Duty cycle arithmetic ===
DC_ViT = 1 - 0.2624 = 0.7376 (matches stored) -- PASS
DC_CNN = 1 - 0.2115 = 0.7885 (matches stored) -- PASS
delta_DC = DC_ViT - DC_CNN = -0.0509 (matches stored within 2e-5) -- PASS (minor rounding)
Negative sign correct (ViT removes MORE time) -- PASS

=== SPOT-CHECK 5: Deadtime from counts ===
ViT: 10127/38587 = 0.2624 (matches stored) -- PASS
CNN: 8163/38587 = 0.2115 (matches stored) -- PASS

=== SPOT-CHECK 6: Macro-F1 recomputation ===
CNN: sum(per-class F1)/23 = 0.6674 (matches stored) -- PASS
ViT: sum(per-class F1)/23 = 0.6695 (matches stored) -- PASS

=== SPOT-CHECK 7: Rare/Common macro-F1 ===
Rare (4 classes): CNN=0.4298, ViT=0.3464 (match MACRO_RARE) -- PASS
Common (19 classes): CNN=0.7174, ViT=0.7375 (match MACRO_COMMON) -- PASS
```

### Spearman Correlation Independent Recomputation (Executed)

Spearman rho recomputed from CSV (23 classes, n_train vs f1_diff_o4):
- Computed rho: -0.033597
- Claimed rho: -0.033597
- Match within 0.001: **PASS**

### Sign Test Independent Verification (Executed)

Classes with n_train >= 100: 20 (3 excluded: Chirp=11, Wandering_Line=30, Helix=33)
ViT wins among 100+ classes: 9/20 (matches claimed) -- **PASS**

### Sensitivity Analysis Verification (Executed)

Identical rho/p for min_o4_samples=5,10,15,20 is CORRECT: the minimum O4 count across all 23 classes is Chirp at 21 samples, so no class is excluded below cutoff=20. At cutoff=30, Chirp is excluded (n_o4=21 < 30), giving 22 classes and a different rho=-0.181. -- **PASS**

### Critical Finding: "Efficiency at 5% Deadtime" Bug (Executed)

**This is the primary verification finding.**

The function `efficiency_at_deadtime(roc_points, target_deadtime=0.05)` at line 296 of `scripts/16_cw_veto_analysis.py` uses `np.interp`, which clamps to boundary values when the target is outside the data range.

- ViT ROC deadtime range: [0.2225, 0.2763] (all points above 22%)
- CNN ROC deadtime range: [0.0652, 0.2525] (lowest point at 6.5%)
- Target: 0.05 (5% deadtime)

For ViT, `np.interp(0.05, [0.2225, 0.2267, ...], [0.7415, 0.7896, ...])` returns 0.7415 (the value at the boundary). This is the ViT efficiency at 22.3% deadtime, **not** 5% deadtime.

For CNN, `np.interp(0.05, [0.0652, ...], [0.2200, ...])` returns 0.2200 (also clamped, since 0.05 < 0.0652).

So the "3.4x" ratio (0.741/0.220) compares:
- ViT at 22.3% deadtime
- CNN at 6.5% deadtime

At matched deadtime (~22.3%), CNN interpolated efficiency is ~0.732, giving a ratio of 0.741/0.732 = **1.01x**, not 3.4x.

**Severity:** SIGNIFICANT. The headline CW comparison number in the SUMMARY and STATE is misleading. The delta_DC=-0.051 finding (CNN advantage overall) is correct and honestly reported, but the "3.4x at matched 5% deadtime" claim that appears in key results needs correction.

## Physics Consistency

| Check | Status | Confidence | Notes |
|-------|--------|------------|-------|
| 5.1 Dimensional analysis | CONSISTENT | INDEPENDENTLY CONFIRMED | All F1 in [0,1], DC in [0,1], efficiency in [0,1], degradation in [-1,+1], rho in [-1,+1], p in [0,1] |
| 5.2 Numerical spot-check | PASS | INDEPENDENTLY CONFIRMED | 7 spot-checks on degradation, F1 diff, DC arithmetic all match to machine precision |
| 5.3 Limiting cases | PASS | INDEPENDENTLY CONFIRMED | Sensitivity analysis at varying cutoffs shows correct behavior; at cutoff=0 all classes included |
| 5.6 Symmetry | PASS | INDEPENDENTLY CONFIRMED | delta_DC sign convention consistent (negative = ViT removes more) throughout |
| 5.8 Math consistency | PASS (mostly) | INDEPENDENTLY CONFIRMED | Macro-F1 = mean of per-class F1 verified; deadtime = n_vetoed/n_total verified. Minor: delta_DC point estimate vs bootstrap mean differ by 2e-5 (acceptable) |
| 5.9 Convergence | N/A | -- | No iterative computation; bootstrap uses 10K resamples (adequate) |
| 5.10 Literature agreement | PASS | STRUCTURALLY PRESENT | Both models degrade on O4 as expected for distribution shift; Violin_Mode/Whistle catastrophic degradation plausible given detector changes |
| 5.11 Physical plausibility | PARTIAL | INDEPENDENTLY CONFIRMED | Degradation patterns plausible. BUT: the "5% deadtime" comparison is at mismatched operating points (see gap) |
| 5.12 Statistical rigor | PASS | INDEPENDENTLY CONFIRMED | Bootstrap 10K resamples, CIs non-degenerate and properly ordered, Spearman p-value correctly interpreted |
| Gate A: Cancellation | PASS | INDEPENDENTLY CONFIRMED | No catastrophic cancellation; quantities are ratios and differences of O(0.1-1) values |
| Gate B: Analytical-numerical | PASS | INDEPENDENTLY CONFIRMED | Macro-F1 recomputed from per-class values matches stored values exactly |

## Forbidden Proxy Audit

| Proxy ID | Status | Evidence |
|----------|--------|----------|
| fp-overall-accuracy | REJECTED | Overall accuracy key in o4_metrics.json explicitly labeled `_SANITY_CHECK`; macro-F1 is the primary metric throughout |
| fp-qualitative-only | REJECTED | 42 numeric values across 7 CW class findings; every claim backed by duty cycle, delta_DC, CI, and efficiency numbers |

## Comparison Verdict Ledger

| Subject ID | Comparison Kind | Verdict | Threshold | Notes |
|------------|----------------|---------|-----------|-------|
| test-threshold-o4 | cross_method | FAIL | p < 0.05 | rho=-0.034, p=0.879; independently verified |
| test-o4-generalization | benchmark | PASS | degradation < 20% | CNN 1.7%, ViT 7.4%; independently verified |
| claim-cw-benefit | baseline | TENSION | measurable improvement | delta_DC=-0.051 (CNN advantage); "3.4x" claim is flawed; see gap |

## Discrepancies Found

| Severity | Location | Evidence | Root Cause | Fix |
|----------|----------|----------|------------|-----|
| SIGNIFICANT | cw_veto_results.json: efficiency_at_5pct_deadtime_vit | ViT value 0.741 is at 22.3% deadtime, not 5% | `np.interp` clamps to boundary value when target is out of range | Fix interpolation function; report matched-deadtime comparison |
| MINOR | delta_DC point estimate vs bootstrap mean | 2e-5 discrepancy | Bootstrap mean of delta_DC differs slightly from point estimate (DC_ViT - DC_CNN) | Acceptable floating-point precision; document which is reported |
| INFO | Rare class threshold | Phase 1 defined N_rare=25 (Chirp only); Phase 4 uses <200 (4 classes) | Convention shift in plan 04-01 | Not a physics error; just note the threshold changed |

## Anti-Patterns Found

| Pattern | Location | Impact |
|---------|----------|--------|
| Silent boundary clamping | scripts/16_cw_veto_analysis.py:302 | `np.interp` silently returns boundary value for out-of-range targets, producing misleading "5% deadtime" comparison | SIGNIFICANT |

No TODO/FIXME/placeholder patterns found in result files. No suppressed warnings. No hardcoded magic numbers in analysis scripts.

## Expert Verification Required

None. All verification checks were performed computationally. The "5% deadtime" issue is a clear software bug, not a physics ambiguity.

## Confidence Assessment

**Overall: MEDIUM**

Most contract targets are independently confirmed with matching numerical values. The threshold test failure (test-threshold-o4) is honestly reported and correctly evaluated. The O4 generalization test passes cleanly.

The confidence is MEDIUM rather than HIGH because:

1. The headline CW comparison number ("3.4x at 5% deadtime") is based on a flawed interpolation that compares models at different operating points. This is the primary gap.
2. The overall CW finding (delta_DC=-0.051, CNN advantage) is correct and honestly reported, but the SUMMARY and STATE propagate the misleading "3.4x" number as a key result.
3. At matched deadtime, the models are nearly equivalent in veto efficiency (~1.01x), which significantly weakens the claim-cw-benefit.

Strengths:
- All per-class F1 values, macro-F1 values, degradation values, and Spearman correlations are independently confirmed to machine precision
- Sign test count verified exactly
- Sensitivity analysis behavior correct
- Bootstrap CIs properly constructed (non-degenerate, ordered, 10K resamples)
- Forbidden proxies properly enforced
- Honest reporting of threshold failure with appropriate backtracking

## Gaps Summary

**One gap found, one root cause:**

The `efficiency_at_deadtime()` function in `scripts/16_cw_veto_analysis.py` uses `np.interp` which silently clamps when the target deadtime (5%) falls outside the ROC curve's deadtime range. For ViT, the minimum achievable deadtime is 22.3%, so "efficiency at 5% deadtime" actually returns the efficiency at 22.3% deadtime. This makes the "3.4x veto efficiency" claim compare ViT at 22.3% deadtime against CNN at 6.5% deadtime -- fundamentally different operating points.

**To close this gap:**
1. Fix `efficiency_at_deadtime()` to return NaN or raise when target is out of range, or extend the threshold sweep to higher values (0.95-0.99+) so ViT can reach 5% deadtime
2. Report matched-deadtime comparison as the primary CW metric
3. Update STATE.md and SUMMARY to remove or qualify the "3.4x at 5% deadtime" claim
4. The genuine CW finding -- that ViT achieves higher veto efficiency at its operating range but at the cost of higher deadtime -- is valid and should be the framing

---

_Verified by GPD phase verifier_
_Phase: 04-o4-validation-cw-sensitivity_
_Date: 2026-03-17_
