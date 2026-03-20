# Research Digest: v1.0 Transformer-Based Rare Glitch Classification

Generated: 2026-03-18
Milestone: v1.0
Phases: 1-5

## Narrative Arc

This project began with a clear hypothesis: Vision Transformers, with their global self-attention over spectrograms, should outperform CNNs at classifying rare LIGO glitch morphologies. We built a clean O3 pipeline (325K glitches, 23 Gravity Spy classes), established a fair CNN baseline (ResNet-50v2 BiT, macro-F1=0.6786), then trained ViT-B/16 with an identical recipe. The result was a **forbidden proxy scenario**: ViT improved overall macro-F1 significantly (0.7230, p=0.0002) but *regressed* on rare classes (0.2412 vs 0.3028). Per-class analysis revealed this was class-morphology-dependent — Light_Modulation (142 train) improved +17pp while Chirp (11 train) collapsed entirely. We attempted to frame this as a "sample-efficiency threshold" but O4 validation showed no monotonic correlation (Spearman ρ=−0.034, p=0.879). The final framing is **class-morphology-dependent architecture preferences**: ViTs excel on classes with distinctive spectral features (Power_Line: +0.507pp O3, +0.394pp O4) regardless of sample size, while CNNs are more robust for morphologically ambiguous rare classes. CW veto analysis showed matched-deadtime efficiency is approximately equal (~1.01x ratio), with the genuine finding being class-specific: Power_Line veto is the strongest ViT CW advantage. The paper honestly reports both the positive per-class findings and the negative rare-class and O4 threshold results.

## Key Results

| Phase | Result | Equation / Value | Validity Range | Confidence |
|-------|--------|-----------------|----------------|------------|
| 1 | O3 dataset size | 325,634 glitches (23 classes) | ml_confidence > 0.9 | HIGH |
| 1 | Temporal split | 227,943 / 48,844 / 48,845 (70/15/15%) | 60s gap enforced | HIGH |
| 1 | Rare class count | 1 (Chirp: 19 total, 11 train) at N_rare=25 | Protocol definition | HIGH |
| 2 | CNN macro-F1 | 0.6786 [0.6598, 0.6944] | O3 temporal test set | HIGH |
| 2 | CNN rare-class F1 | 0.3028 [0.2085, 0.3751] | 4 classes at threshold=200 | MEDIUM (wide CI) |
| 2 | Rare-class gap | 50.5pp (common 80.8% vs rare 30.3%) | Threshold=200 | HIGH |
| 3 | ViT macro-F1 | 0.7230 [0.7031, 0.7397] | O3 temporal test set | HIGH |
| 3 | ViT rare-class F1 | 0.2412 [0.2019, 0.2957] | 4 classes | MEDIUM (wide CI) |
| 3 | Rare-class bootstrap p | 0.8842 (NOT significant) | H0: ViT <= CNN | HIGH |
| 3 | Overall bootstrap p | 0.0002 (significant — FORBIDDEN PROXY) | H0: ViT <= CNN | HIGH |
| 4 | CNN O4 macro-F1 | 0.6674 [0.6567, 0.6765] | O4a, 38,587 glitches | HIGH |
| 4 | ViT O4 macro-F1 | 0.6695 [0.6555, 0.6816] | O4a | HIGH |
| 4 | CNN O4 degradation | −1.7% relative | PASS (<20%) | HIGH |
| 4 | ViT O4 degradation | −7.4% relative | PASS (<20%) | HIGH |
| 4 | Threshold test | ρ = −0.034, p = 0.879 | NOT confirmed | HIGH |
| 4 | CW matched deadtime | 22.4%: ViT eff=0.745, CNN eff=0.735 | ~equal | HIGH |
| 4 | CW delta_DC | −0.051 [−0.054, −0.048] | CNN preserves more time | HIGH |
| 4 | Power_Line ViT advantage | +0.394 F1 diff (O4), +0.507 (O3) | Both observing runs | HIGH |

## Methods Employed

- **Phase 1:** temporal-split-with-gap — GPS-time-ordered split with 60s minimum gap to prevent temporal leakage
- **Phase 1:** async-http-download — Concurrent download of 1.3M pre-made spectrograms from Zooniverse CDN
- **Phase 2:** focal-loss-gamma2 — FL(p_t) = −α_t(1−p_t)^γ log(p_t), γ=2.0 for class imbalance
- **Phase 2:** cosine-warmup-scheduler — 5-epoch linear warmup then cosine decay
- **Phase 2:** sqrt-inverse-class-weights — Class-balanced batch sampling
- **Phase 3:** layer-wise-lr-decay — ViT fine-tuning with decay=0.75 across 12 transformer blocks
- **Phase 3:** paired-percentile-bootstrap — 10K resamples with identical indices for fair model comparison
- **Phase 4:** softmax-confidence-threshold-sweep — Sweep 0.50–0.95 for veto ROC curves
- **Phase 4:** veto-ROC-analysis — Efficiency vs deadtime at each confidence threshold
- **Phase 5:** number-extraction-pipeline — All paper numbers from JSON source of truth
- **Phase 5:** programmatic-figure-generation — 5 vector PDF figures from result data

## Convention Evolution

| Phase | Convention | Description | Status |
|-------|-----------|-------------|--------|
| 1 | SI units | Strain dimensionless, frequency Hz, time s | Active |
| 1 | Spectrogram format | 224×224 px, RGB PNG, min-max [0,1] after SNR clip [0,25.5] | Active |
| 1 | Macro-F1 primary | Overall accuracy is forbidden proxy | Active (ENFORCED in Phase 3) |
| 1 | Bootstrap protocol | ≥10K resamples, seed=42, p<0.05 | Active |
| 2 | Rare threshold | N_rare=200 for evaluation (protocol says 25 for planning) | Active |
| 2 | ImageNet normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Active |
| 4 | delta_DC sign | Positive = ViT removes LESS time (higher duty cycle = better) | Active |
| 4 | CW-critical taxonomy | HIGH: Scattered_Light, Violin_Mode, Low_Freq_Lines; MEDIUM: 1080Lines, Whistle, Power_Line | Active |

## Figures and Data Registry

| File | Phase | Description | Paper-ready? |
|------|-------|-------------|--------------|
| figures/class_distribution_o3.png | 1 | Class distribution (23 classes, log scale) | No (exploratory) |
| figures/split_class_distribution.png | 1 | Per-class split distribution | No (exploratory) |
| figures/cnn_confusion_matrix.png | 2 | CNN 23×23 confusion matrix | No (superseded) |
| figures/cnn_per_class_f1.png | 2 | CNN per-class F1 bar chart | No (superseded) |
| figures/comparison_confusion_matrices.png | 3 | Side-by-side ViT vs CNN confusion matrices | Yes |
| figures/comparison_per_class_f1.png | 3 | Grouped bar chart comparing all 23 classes | Yes |
| figures/o4_threshold_scatter.png | 4 | Training size vs ViT advantage scatter | Yes |
| figures/o4_degradation_per_class.png | 4 | Per-class O3→O4 degradation | Yes |
| figures/cw_sensitivity_summary.png | 4 | CW veto summary (multi-panel) | Yes |
| figures/cw_veto_roc.png | 4 | Veto efficiency vs deadtime ROC | Yes |
| paper/figures/fig_per_class_f1.pdf | 5 | Journal-quality per-class comparison | Yes (vector) |
| paper/figures/fig_threshold_scatter.pdf | 5 | Journal-quality threshold scatter | Yes (vector) |
| paper/figures/fig_confusion_matrices.pdf | 5 | Journal-quality confusion matrices | Yes (vector) |
| paper/figures/fig_o4_degradation.pdf | 5 | Journal-quality O4 degradation | Yes (vector) |
| paper/figures/fig_cw_veto.pdf | 5 | Journal-quality CW veto analysis | Yes (vector) |

## Open Questions

1. Would rare-class-specific interventions (cDVGAN augmentation, contrastive pretraining, few-shot learning) resolve the rare-class gap?
2. Is frequency-resolved PSD analysis needed to quantify Power_Line CW benefit more precisely?
3. How should the paper frame the CW finding: operating-point-dependent or class-specific?
4. Which specific rare glitch classes in Gravity Spy taxonomy have the most impact on CW searches?
5. Which literature anchors beyond Gravity Spy exist for transformer-based GW glitch classification?

## Dependency Graph

    Phase 1 "Data Pipeline & Experimental Design"
      provides: filtered metadata (325K), temporal split manifests, experimental protocol
      requires: nothing (entry phase)
    → Phase 2 "CNN Baseline Reproduction"
      provides: CNN macro-F1=0.6786, rare-class F1=0.3028, training infrastructure
      requires: split manifests, protocol
    → Phase 3 "ViT Training & Per-Class Comparison"
      provides: ViT macro-F1=0.7230, rare-class F1=0.2412, comparison table, bootstrap results
      requires: CNN baseline, training infrastructure, data pipeline
    → Phase 4 "O4 Validation & CW Sensitivity"
      provides: O4 metrics, threshold test (FAIL), CW veto results
      requires: both model checkpoints, comparison table
    → Phase 5 "Paper & Model Packaging"
      provides: paper draft, packaged models, number extraction pipeline
      requires: all Phase 2-4 results and artifacts

## Mapping to Original Objectives

| Requirement | Status | Fulfilled by | Key Result |
|-------------|--------|-------------|------------|
| DATA-01: Download O3 Gravity Spy | Complete | Phase 1 | 325,634 samples |
| DATA-02: Generate spectrograms | Complete (modified) | Phase 1 | Pre-made images (Option C) |
| DATA-03: Temporal split | Complete | Phase 1 | 70/15/15, 60s gap |
| DATA-04: Rare-class audit | Complete | Phase 1 | Chirp flagged (19 total) |
| MODL-01: CNN baseline | Complete | Phase 2 | macro-F1=0.6786 |
| MODL-02: Train ViT-B/16 | Complete | Phase 3 | macro-F1=0.7230 |
| MODL-03: Focal loss + balanced | Complete | Phase 3 | Identical recipe verified |
| EVAL-01: Per-class F1 | Complete | Phase 3 | Full table with CIs |
| EVAL-02: O4 validation | Complete | Phase 4 | Both <20% degradation |
| EVAL-03: Statistical significance | Partial (honest FAIL) | Phase 3 | Rare p=0.88, overall p=0.0002 |
| CWSS-01: CW quantification | Partial (class-specific) | Phase 4 | Matched-deadtime ~equal |
| DELV-01: Trained models | Complete | Phase 5 | Both packaged with SHA-256 |
| DELV-02: Research paper | Complete | Phase 5 | CQG draft, 755 lines |
