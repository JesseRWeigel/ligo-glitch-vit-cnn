---
phase: 03-vit-training-rare-class-optimization
plan: 02
depth: full
one-liner: "ViT-B/16 test evaluation reveals forbidden proxy scenario: overall macro-F1 improves (0.7230 vs 0.6786, p<0.001) but rare-class macro-F1 regresses (0.2412 vs 0.3028, p=0.88) -- backtracking trigger ACTIVATED"
subsystem: [analysis, computation]
tags: [vit-evaluation, paired-bootstrap, rare-class-f1, forbidden-proxy, backtracking-trigger]

requires:
  - phase: 03-vit-training-rare-class-optimization
    plan: 01
    provides: ["Trained ViT-B/16 checkpoint (checkpoints/03-vit-rare-class/best_model.pt)"]
  - phase: 02-cnn-baseline-reproduction
    provides: ["CNN checkpoint (checkpoints/02-cnn-baseline/best_model.pt)", "CNN test metrics (macro-F1=0.6786, rare-class F1=0.3028)"]
provides:
  - "ViT test macro-F1 = 0.7230 [0.7031, 0.7397] -- significantly better than CNN overall"
  - "ViT rare-class macro-F1 = 0.2412 [0.2019, 0.2957] -- WORSE than CNN (0.3028)"
  - "Paired bootstrap rare-class p=0.8842 -- NO significant rare improvement"
  - "Paired bootstrap overall macro-F1 p=0.0002 -- significant overall improvement"
  - "Backtracking trigger: ACTIVATED -- claim-rare-improvement FAILS"
  - "Per-class comparison table, confusion matrices, statistical summary with contract assessment"
affects: [phase-04-decision, project-direction]

methods:
  added: [paired-percentile-bootstrap, per-sample-prediction-comparison]
  patterns: [forbidden-proxy-detection, backtracking-trigger-assessment]

key-files:
  created:
    - src/evaluation/paired_bootstrap.py
    - scripts/12_evaluate_vit.py
    - scripts/13_comparison_deliverables.py
    - results/03-vit-rare-class/metrics.json
    - results/03-vit-rare-class/per_class_f1.csv
    - results/03-vit-rare-class/predictions.npz
    - results/03-vit-rare-class/paired_bootstrap_results.json
    - results/03-vit-rare-class/comparison_table.csv
    - results/03-vit-rare-class/statistical_summary.md
    - figures/vit_confusion_matrix.png
    - figures/comparison_confusion_matrices.png
    - figures/comparison_per_class_f1.png
  modified: []

key-decisions:
  - "Backtracking trigger ACTIVATED: ViT rare-class macro-F1 (0.2412) is below CNN baseline (0.3028)"
  - "Forbidden proxy scenario confirmed: ViT improves overall accuracy and macro-F1 but NOT rare classes"
  - "Result is honest and documented per contract -- not a pipeline bug, but a genuine negative finding"

patterns-established:
  - "Paired bootstrap with identical indices for fair model comparison"
  - "CNN re-evaluation before comparison to verify metric reproducibility"
  - "Forbidden proxy enforcement: accuracy labeled SANITY CHECK in all outputs"

conventions:
  - "Macro-F1 as primary metric; overall accuracy is SANITY CHECK only"
  - "Rare threshold = 200 train samples (4 classes: Chirp, Wandering_Line, Helix, Light_Modulation)"
  - "Bootstrap >= 10K resamples, seed=42, 95% CIs"
  - "ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]"

plan_contract_ref: ".gpd/phases/03-vit-training-rare-class-optimization/03-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-rare-improvement:
      status: failed
      summary: "ViT rare-class macro-F1 (0.2412) is BELOW CNN baseline (0.3028). Paired bootstrap p=0.8842 (H0: ViT<=CNN NOT rejected). The ViT improves overall macro-F1 significantly (p<0.001) but this comes from common classes -- the forbidden proxy scenario."
      linked_ids: [deliv-comparison-table, deliv-confusion-matrix, deliv-vit-metrics, deliv-paired-bootstrap, deliv-statistical-summary, test-rare-f1-improvement, test-metric-consistency, test-paired-bootstrap-valid, test-same-test-set]
  deliverables:
    deliv-comparison-table:
      status: passed
      path: "results/03-vit-rare-class/comparison_table.csv"
      summary: "23 data rows + 3 summary rows (MACRO_ALL, MACRO_RARE, MACRO_COMMON). All required columns present: per-class F1, recall, precision for both models, bootstrap CIs, f1_diff, is_rare flag, sample counts."
      linked_ids: [claim-rare-improvement, test-rare-f1-improvement]
    deliv-confusion-matrix:
      status: passed
      path: "figures/comparison_confusion_matrices.png"
      summary: "Side-by-side 23x23 confusion matrices (ViT left, CNN right), row-normalized with raw count annotations, rare classes highlighted with red border, sorted by n_train ascending."
      linked_ids: [claim-rare-improvement]
    deliv-vit-metrics:
      status: passed
      path: "results/03-vit-rare-class/metrics.json"
      summary: "ViT test metrics mirroring Phase 2 structure: macro_f1=0.7230 [0.7031, 0.7397], rare_class_macro_f1=0.2412 [0.2019, 0.2957], overall_accuracy=0.9343 (SANITY CHECK). Test manifest hash recorded."
      linked_ids: [claim-rare-improvement, test-metric-consistency, test-same-test-set]
    deliv-paired-bootstrap:
      status: passed
      path: "results/03-vit-rare-class/paired_bootstrap_results.json"
      summary: "Rare-class macro-F1 diff=-0.0617 [-0.1463, 0.0408], p=0.8842 (NOT significant). Overall macro-F1 diff=+0.0444 [0.0210, 0.0675], p=0.0002 (significant). 10K resamples, paired indices, test hash verified."
      linked_ids: [claim-rare-improvement, test-rare-f1-improvement, test-paired-bootstrap-valid]
    deliv-statistical-summary:
      status: passed
      path: "results/03-vit-rare-class/statistical_summary.md"
      summary: "Structured summary with verdict (ViT WORSE on rare classes), per-class breakdown, contract assessment (FAIL), backtracking trigger (ACTIVATED), interpretation of forbidden proxy scenario."
      linked_ids: [claim-rare-improvement]
  acceptance_tests:
    test-rare-f1-improvement:
      status: failed
      summary: "ViT rare-class macro-F1 (0.2412) <= CNN (0.3028). Paired bootstrap p=0.8842 >> 0.05. Both conditions of the pass criterion fail."
      linked_ids: [claim-rare-improvement, deliv-comparison-table, deliv-paired-bootstrap]
    test-metric-consistency:
      status: passed
      summary: "ViT: sklearn=0.72303776, torchmetrics=0.72303772, diff=3.53e-08 < 1e-6. CNN: diff=7.34e-08 < 1e-6."
      linked_ids: [claim-rare-improvement, deliv-vit-metrics]
    test-paired-bootstrap-valid:
      status: passed
      summary: "(1) Same indices for both models in each resample -- verified by single rng per resample. (2) Test manifest hash c8f3865d... matches Phase 2. (3) n_resamples=10,000 >= 10,000. (4) Rare classes: Chirp, Wandering_Line, Helix, Light_Modulation (threshold=200)."
      linked_ids: [claim-rare-improvement, deliv-paired-bootstrap]
    test-same-test-set:
      status: passed
      summary: "SHA-256 hash c8f3865d1bf36e1c435b05329bcddb32d5994b65314a824409758aa7492f5124 matches Phase 2. Both models evaluated on identical 48,845-sample test set."
      linked_ids: [deliv-vit-metrics]
  references:
    ref-cnn-baseline:
      status: completed
      completed_actions: [read, compare, cite]
      missing_actions: []
      summary: "CNN checkpoint re-evaluated: macro-F1=0.6786 and accuracy=0.9181 reproduced exactly (diff=0.00e+00). Per-sample predictions obtained for paired bootstrap."
    ref-gravity-spy:
      status: completed
      completed_actions: [read, cite]
      missing_actions: []
      summary: "23-class taxonomy preserved throughout. Class labels match in both models."
  forbidden_proxies:
    fp-overall-accuracy:
      status: rejected
      notes: "FORBIDDEN PROXY SCENARIO DETECTED AND DOCUMENTED: ViT improves overall accuracy (0.9343 vs 0.9181) and overall macro-F1 (0.7230 vs 0.6786, p=0.0002) but rare-class macro-F1 DECREASES (0.2412 vs 0.3028). All outputs label accuracy as SANITY CHECK. The project claim correctly uses rare-class F1 as primary, preventing false-positive conclusions from overall improvement."
  uncertainty_markers:
    weakest_anchors:
      - "Chirp has only 7 test samples: F1 drops from 0.471 to 0.000, but a single correct/incorrect prediction moves F1 by ~0.14"
      - "Wandering_Line has only 6 test samples: both models score F1=0.000 -- no statistical power"
      - "Rare-class macro-F1 pools only 93 test samples across 4 classes; bootstrap CIs are wide"
    unvalidated_assumptions: []
    competing_explanations:
      - "Chirp F1 regression could be due to ViT's attention mechanism focusing on dominant spectral features that happen to not capture Chirp morphology"
      - "With only 11 training examples for Chirp, any model is effectively guessing; the CNN's 0.471 may itself be a lucky outcome"
    disconfirming_observations:
      - "ViT rare-class macro-F1 (0.2412) < CNN (0.3028) -- primary claim is disconfirmed"
      - "Light_Modulation (the largest rare class, 142 train) improved substantially (0.691->0.859), suggesting the ViT CAN help rare classes with sufficient samples"
      - "Overall macro-F1 improvement is significant but irrelevant to the primary claim per contract"

comparison_verdicts:
  - subject_id: test-rare-f1-improvement
    subject_kind: acceptance_test
    subject_role: decisive
    reference_id: ref-cnn-baseline
    comparison_kind: benchmark
    metric: rare_class_macro_f1
    threshold: "> 0.3028 AND p < 0.05"
    verdict: fail
    recommended_action: "Backtracking trigger activated. Consider: (1) targeted rare-class augmentation, (2) few-shot learning, (3) synthetic data, (4) contrastive pretraining, (5) accepting partial improvement (Light_Modulation) while acknowledging Chirp/Wandering_Line/Helix limitations."
    notes: "ViT rare-class F1=0.2412 < CNN 0.3028. The ViT architecture alone does not solve rare-class classification."
  - subject_id: fp-overall-accuracy
    subject_kind: forbidden_proxy
    subject_role: guard
    reference_id: ref-cnn-baseline
    comparison_kind: sanity
    metric: overall_macro_f1
    threshold: "N/A (forbidden as primary)"
    verdict: confirmed_violation
    recommended_action: "Document as cautionary example: architecture swap improves common classes but not rare ones."
    notes: "Overall macro-F1 improved 0.6786->0.7230 (p=0.0002) but this is EXACTLY the forbidden proxy -- improvement on common classes masking failure on rare classes."

duration: 15min
completed: 2026-03-17
---

# Plan 03-02: Evaluate ViT-B/16 vs CNN with Paired Bootstrap

**ViT-B/16 test evaluation reveals forbidden proxy scenario: overall macro-F1 improves significantly (0.7230 vs 0.6786, p<0.001) but rare-class macro-F1 regresses (0.2412 vs 0.3028, p=0.88) -- backtracking trigger ACTIVATED**

## Performance

- **Duration:** ~15 min (13 min inference + bootstrap, 2 min deliverables)
- **Started:** 2026-03-17T14:10:00Z
- **Completed:** 2026-03-17T14:31:00Z
- **Tasks:** 2/2
- **Files modified:** 12

## Key Results

- **ViT rare-class macro-F1 (PRIMARY): 0.2412 [0.2019, 0.2957]** -- BELOW CNN baseline of 0.3028 [CONFIDENCE: HIGH -- 3 independent checks: paired bootstrap, per-class breakdown, CNN re-eval match]
- **Paired bootstrap (rare-class F1): diff = -0.0617 [-0.1463, 0.0408], p = 0.8842** -- NOT significant [CONFIDENCE: HIGH]
- **ViT overall macro-F1: 0.7230 [0.7031, 0.7397]** -- above CNN 0.6786, p=0.0002 (significant but FORBIDDEN PROXY)
- **ViT overall accuracy (SANITY CHECK): 0.9343** -- above CNN 0.9181
- **CNN re-evaluation: EXACT match** -- macro-F1 diff=0.00e+00, accuracy diff=0.00e+00
- **Metric consistency: PASS** -- sklearn vs torchmetrics diff=3.53e-08 (ViT), 7.34e-08 (CNN)
- **claim-rare-improvement: FAIL** -- backtracking trigger ACTIVATED

### Per-Class Rare Results

| Class | n_train | n_test | CNN F1 | ViT F1 | Diff | Interpretation |
|---|---|---|---|---|---|---|
| Chirp | 11 | 7 | 0.471 | 0.000 | -0.471 | Complete regression -- ViT fails on all Chirp samples |
| Wandering_Line | 30 | 6 | 0.000 | 0.000 | 0.000 | Both models fail -- too few samples |
| Helix | 33 | 14 | 0.049 | 0.105 | +0.056 | Marginal improvement |
| Light_Modulation | 142 | 66 | 0.691 | 0.859 | +0.168 | Substantial improvement (largest rare class) |

## Key Quantities and Uncertainties

| Quantity | Value | Uncertainty | Source | Valid Range |
|---|---|---|---|---|
| ViT rare-class macro-F1 | 0.2412 | [0.2019, 0.2957] | Bootstrap 10K, 95% CI | 4 rare classes, 93 test samples |
| CNN rare-class macro-F1 | 0.3028 | [0.2085, 0.3751] | Bootstrap 10K, 95% CI | 4 rare classes, 93 test samples |
| Rare-class F1 difference | -0.0617 | [-0.1463, 0.0408] | Paired bootstrap 10K | Same test indices |
| Paired bootstrap p-value | 0.8842 | N/A | One-sided (H0: ViT<=CNN) | 10K resamples |
| ViT overall macro-F1 | 0.7230 | [0.7031, 0.7397] | Bootstrap 10K, 95% CI | N=48,845 |
| Overall macro-F1 difference | +0.0444 | [0.0210, 0.0675] | Paired bootstrap 10K | Same test indices |
| Overall macro-F1 p-value | 0.0002 | N/A | One-sided | FORBIDDEN PROXY |

## Task Commits

1. **Task 1: Evaluate ViT+CNN with paired bootstrap** -- `9f0aa62` (compute)
2. **Task 2: Comparison deliverables and statistical summary** -- `92c0619` (analyze)

## Files Created/Modified

- `src/evaluation/paired_bootstrap.py` -- Paired bootstrap module with unit test
- `scripts/12_evaluate_vit.py` -- Combined ViT+CNN evaluation with CNN re-verification
- `scripts/13_comparison_deliverables.py` -- Comparison table, figures, statistical summary
- `results/03-vit-rare-class/metrics.json` -- ViT test metrics with bootstrap CIs
- `results/03-vit-rare-class/per_class_f1.csv` -- ViT per-class metrics
- `results/03-vit-rare-class/predictions.npz` -- y_true, vit_preds, cnn_preds (48,845 each)
- `results/03-vit-rare-class/paired_bootstrap_results.json` -- Paired test results
- `results/03-vit-rare-class/comparison_table.csv` -- 26-row comparison (23 classes + 3 summaries)
- `results/03-vit-rare-class/statistical_summary.md` -- Verdict and contract assessment
- `figures/vit_confusion_matrix.png` -- Standalone ViT confusion matrix
- `figures/comparison_confusion_matrices.png` -- Side-by-side ViT vs CNN
- `figures/comparison_per_class_f1.png` -- Grouped bar chart with CIs

## Contract Coverage

- **Claim IDs:** claim-rare-improvement -> FAILED (ViT rare F1 < CNN)
- **Deliverable IDs:** all 5 produced and passed (comparison-table, confusion-matrix, vit-metrics, paired-bootstrap, statistical-summary)
- **Acceptance test IDs:** test-rare-f1-improvement -> FAILED, test-metric-consistency -> PASSED, test-paired-bootstrap-valid -> PASSED, test-same-test-set -> PASSED
- **Reference IDs:** ref-cnn-baseline -> read+compare+cite (re-evaluated with exact match), ref-gravity-spy -> read+cite (23 classes preserved)
- **Forbidden proxies:** fp-overall-accuracy -> REJECTED and DETECTED (forbidden proxy scenario confirmed)
- **Comparison verdicts:** test-rare-f1-improvement -> fail (decisive), fp-overall-accuracy -> confirmed_violation

## Validations Completed

- CNN re-evaluation reproduces Phase 2 exactly: macro-F1 diff=0.00e+00, accuracy diff=0.00e+00
- ViT and CNN evaluated on identical test set (hash c8f3865d...)
- Paired bootstrap uses same RNG indices for both models
- Metric consistency: sklearn vs torchmetrics < 1e-6 for both models
- Comparison table: 23 data rows + 3 summary, no NaN, CNN column matches Phase 2
- Confusion matrices: 23x23, row-normalized, rare classes highlighted
- Statistical summary: all required fields present
- Forbidden proxy enforced: accuracy labeled SANITY CHECK in all outputs

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|---|---|---|---|
| Paired percentile bootstrap CI | N_test large enough for stable resampling | Stable for overall (N=48,845); wide for rare classes (N=93 pooled) | Individual rare classes with N_test < 10 (Chirp=7, Wandering_Line=6) |
| Macro-F1 over 4 rare classes | Each class contributes equally | Dominated by best/worst class | When classes have very different test set sizes (7 vs 66) |

## Figures Produced

| Figure | File | Description | Key Feature |
|---|---|---|---|
| Fig. 03.2 | figures/comparison_confusion_matrices.png | Side-by-side ViT vs CNN confusion matrices | Rare classes highlighted red; ViT shows better common-class patterns |
| Fig. 03.3 | figures/comparison_per_class_f1.png | Grouped bar chart comparing all 23 classes | Rare classes marked; error bars from bootstrap CIs |
| Fig. 03.4 | figures/vit_confusion_matrix.png | Standalone ViT confusion matrix | Row-normalized with rare-class highlighting |

## Decisions Made

- Backtracking trigger ACTIVATED per contract: ViT rare-class macro-F1 (0.2412) < CNN baseline (0.3028)
- Result documented honestly as a genuine negative finding, not a pipeline bug
- Forbidden proxy scenario documented as a cautionary example for the project narrative

## Deviations from Plan

None. Both tasks executed as planned. The negative result is a genuine finding, not a deviation.

## Issues Encountered

- Forbidden proxy check flagged two false-positive warnings (JSON key name `overall_accuracy` and a sentence explaining the forbidden proxy scenario). Both are properly qualified in context.

## Open Questions

- Would targeted rare-class augmentation (mixup, SMOTE-like for images) help the ViT on Chirp/Helix?
- Light_Modulation improved substantially (0.691->0.859) -- what is different about this rare class?
- Is the Chirp regression (0.471->0.000) due to the ViT's attention mechanism or statistical noise with N=7?
- Would a two-stage approach (ViT for common classes + specialized rare-class model) be viable?
- Should Phase 4 pivot from "optimize ViT for rare classes" to "targeted rare-class interventions"?

## Self-Check: PASSED

- [x] All 12 output files exist and are non-empty
- [x] Both task commits verified (9f0aa62, 92c0619)
- [x] CNN re-evaluation matches Phase 2 exactly
- [x] Paired bootstrap resamples >= 10,000
- [x] Test manifest hash matches across both models
- [x] Metric consistency < 1e-6 for both models
- [x] Comparison table has all required columns and no NaN
- [x] Confusion matrices are 23x23 with rare-class highlighting
- [x] Statistical summary contains verdict and contract assessment
- [x] Forbidden proxy enforced in all outputs
- [x] Backtracking trigger correctly assessed as ACTIVATED

---

_Phase: 03-vit-training-rare-class-optimization_
_Plan: 02_
_Completed: 2026-03-17_
