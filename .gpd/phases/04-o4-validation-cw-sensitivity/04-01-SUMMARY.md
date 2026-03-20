---
phase: 04-o4-validation-cw-sensitivity
plan: 01
status: complete
one_liner: "O4 evaluation complete — sample-efficiency threshold NOT confirmed (Spearman ρ=−0.034, p=0.88); both models pass <20% degradation; Power_Line still shows strong ViT advantage (+0.394)"
started: 2026-03-17
completed: 2026-03-17
tasks_completed: 3
tasks_total: 3
deviations:
  - description: "Sample-efficiency threshold does not persist on O4 as a monotonic correlation"
    severity: "high"
    impact: "Paper claim must be re-scoped from systematic threshold to class-specific architecture preferences"
    resolution: "Report honestly; Power_Line and Air_Compressor show strong ViT advantage independent of threshold narrative"
  - description: "O3 cross-check also shows non-significant correlation (ρ=−0.119, p=0.59)"
    severity: "medium"
    impact: "The original Phase 3 'threshold' was driven by specific classes (Power_Line, Chirp), not a monotonic trend"
    resolution: "Reframe as class-morphology-dependent rather than sample-size-dependent"
key_results:
  - "CNN macro-F1 O4: 0.6674 [0.6567, 0.6765] — degradation: −1.7% relative (PASS)"
  - "ViT macro-F1 O4: 0.6695 [0.6555, 0.6816] — degradation: −7.4% relative (PASS)"
  - "Spearman ρ(n_train, ViT−CNN F1 on O4) = −0.034, p = 0.879 — threshold NOT confirmed"
  - "Sign test: 9/20 ViT wins for 100+ classes (p=0.82) — chance level"
  - "Power_Line: ViT F1 0.725 vs CNN 0.331, diff = +0.394 (strongest ViT advantage)"
  - "Air_Compressor: ViT F1 0.545 vs CNN 0.309, diff = +0.237"
  - "Chirp: ViT F1 0.308 vs CNN 0.681, diff = −0.373 (CNN better for rarest class on O4)"
key_files:
  created:
    - scripts/14_acquire_o4_data.py
    - scripts/15_evaluate_o4.py
    - scripts/16_threshold_test.py
    - results/04-o4-validation/o4_comparison_table.csv
    - results/04-o4-validation/o4_metrics.json
    - results/04-o4-validation/o4_degradation.json
    - results/04-o4-validation/o4_threshold_test.json
    - results/04-o4-validation/o4_predictions.npz
    - figures/o4_threshold_scatter.png
    - figures/o4_degradation_per_class.png
    - data/o4/metadata/o4_evaluation_manifest.csv
    - data/o4/metadata/o4_class_distribution.json
  modified: []
  consumed:
    - results/03-vit-rare-class/comparison_table.csv
    - checkpoints/02-cnn-baseline/best_model.pt
    - checkpoints/03-vit-rare-class/best_model.pt
    - data/metadata/test_manifest.csv
gpd_return:
  state_updates:
    phase: "04"
    plan: "01"
    status: "complete"
  decisions:
    - "test-threshold-o4: FAIL — Spearman ρ=−0.034, p=0.879; threshold is not a monotonic sample-size effect"
    - "test-o4-generalization: PASS — both models degrade <20% (CNN: −1.7%, ViT: −7.4%)"
    - "test-o3-reproduction: PASS — per-class F1 matches Phase 3 to <1e-6"
    - "BACKTRACKING: claim-threshold must be re-scoped from 'systematic threshold' to 'class-specific architecture preferences'"
  metrics:
    cnn_macro_f1_o4: 0.6674
    vit_macro_f1_o4: 0.6695
    cnn_degradation_relative: 0.017
    vit_degradation_relative: 0.074
    spearman_rho_o4: -0.034
    spearman_p_o4: 0.879
    sign_test_vit_wins: 9
    sign_test_total: 20
    n_o4_samples: 38587
    n_classes_reliable: 23
---

# Plan 04-01 Summary: O4 Validation & Threshold Persistence Test

## What was done

Evaluated both O3-trained models (ViT-B/16 and ResNet-50 CNN) on 38,587 O4a Gravity Spy glitches across all 23 classes. Tested whether the sample-efficiency threshold discovered in Phase 3 persists under O3→O4 distribution shift.

## Key findings

### 1. O3 metrics reproduced exactly (test-o3-reproduction: PASS)
Both model checkpoints load correctly and reproduce Phase 3 per-class F1 to <1e-6.

### 2. Both models generalize to O4 (test-o4-generalization: PASS)
- CNN: macro-F1 0.6674 (−1.7% relative degradation from O3)
- ViT: macro-F1 0.6695 (−7.4% relative degradation from O3)
- Both well within the <20% contract threshold

### 3. Sample-efficiency threshold NOT confirmed (test-threshold-o4: FAIL)
- **Spearman ρ = −0.034, p = 0.879** — no correlation between O3 training set size and ViT advantage on O4
- **Sign test: 9/20 ViT wins** for 100+ sample classes — indistinguishable from chance
- **O3 cross-check: ρ = −0.119, p = 0.59** — the "threshold" was never a robust monotonic effect
- **Sensitivity analysis:** correlation is near-zero regardless of minimum O4 sample cutoff (5–30)

### 4. Class-specific patterns persist
Despite no systematic threshold, specific classes show consistent ViT advantage:
- **Power_Line** (n=1582): ViT F1 0.725 vs CNN 0.331 → +0.394 (strongest)
- **Air_Compressor** (n=1361): ViT F1 0.545 vs CNN 0.309 → +0.237
- **Paired_Doves** (n=216): ViT F1 0.728 vs CNN 0.620 → +0.108

And specific classes show consistent CNN advantage:
- **Chirp** (n=11): CNN F1 0.681 vs ViT 0.308 → CNN wins by 0.373
- **1400Ripples** (n=2428): CNN F1 0.911 vs ViT 0.731 → CNN wins by 0.180

## Backtracking assessment

The Phase 3 "sample-efficiency threshold at ~100 samples" was an overfit narrative. The actual pattern is **class-morphology-dependent architecture preference**: ViT consistently excels at classes with distinctive spectral features (Power_Line's 60 Hz harmonics, Air_Compressor's broadband pattern) regardless of sample count, while CNN excels at classes with variable/complex morphologies (Chirp, 1400Ripples). The paper claim should be re-scoped accordingly.

## Deliverables produced
- ✓ deliv-o4-comparison: `results/04-o4-validation/o4_comparison_table.csv`
- ✓ deliv-threshold-scatter: `figures/o4_threshold_scatter.png`
- ✓ deliv-degradation-plot: `figures/o4_degradation_per_class.png`
- ✓ deliv-o4-metrics: `results/04-o4-validation/o4_metrics.json`

## Self-Check: PASS
- [x] O3 metrics reproduced exactly
- [x] Both models evaluated on O4 with bootstrap CIs
- [x] Overall degradation < 20% for both models
- [x] Threshold test completed with Spearman + sign test
- [x] Sensitivity analysis across sample size cutoffs
- [x] All figures generated with required annotations
- [x] Forbidden proxy (overall accuracy) marked SANITY CHECK only
- [x] Honest reporting: threshold NOT confirmed, backtracking noted
