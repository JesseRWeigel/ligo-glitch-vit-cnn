# Conventions: Transformer-Based Rare Glitch Classification for LIGO CW Searches

**Established:** 2026-03-16
**Project type:** ML / signal processing for GW detector characterization

## Unit System

| Quantity | Convention | Notes |
| -------- | ---------- | ----- |
| Strain | Dimensionless h(t); spectral density in 1/√Hz | LIGO standard |
| Frequency | Hz | SI |
| Time | Seconds (s); GPS seconds for event timestamps | LIGO standard |
| ML metrics | Dimensionless, range [0, 1] | F1, recall, accuracy |

## Spectrogram Conventions (Q-transform)

| Parameter | Value | Rationale |
| --------- | ----- | --------- |
| Transform | Q-transform via `gwpy.timeseries.TimeSeries.q_transform` | Matches Gravity Spy pipeline |
| Q range | (4, 150) | Gravity Spy default |
| Frequency range | 10–2048 Hz, logarithmic axis | Gravity Spy convention |
| Time windows | 4 durations: 0.5, 1.0, 2.0, 4.0 s centered on trigger | Gravity Spy multi-view |
| Image size | 224 × 224 pixels | ViT-B/16 standard input |
| Normalization | Min-max to [0, 1] after SNR clipping to [0, 25.5] | Prevents outlier saturation |
| Color channels | Single-channel (grayscale) replicated to 3 channels | For ImageNet-pretrained backbone |

## ML Evaluation Metric Conventions

| Metric | Definition | Role |
| ------ | ---------- | ---- |
| **Primary: Macro-F1** | (1/N_cls) × Σ F1_c, unweighted mean of per-class F1 | Core evaluation metric; weights all classes equally |
| **Per-class F1** | F1_c = 2 × precision_c × recall_c / (precision_c + recall_c) | Standard harmonic mean |
| **Rare-class macro-F1** | Same formula restricted to classes with < N_rare training examples | N_rare threshold set in Phase 1 |
| **Secondary: Per-class recall** | TP_c / (TP_c + FN_c) for each class c | Measures completeness of detection |
| **Sanity check: Overall accuracy** | Reported but never optimized for | **Forbidden proxy** per project contract |
| **Statistical testing** | Bootstrap ≥ 10,000 resamples; p < 0.05 significance | EVAL-03 requirement |
| **Confidence intervals** | 95% bootstrap CI on all reported metrics | Standard for ML papers |
| **Reporting precision** | 3 significant figures for F1 and recall | EVAL-01 requirement |

## Data Conventions

| Convention | Value | Rationale |
| ---------- | ----- | --------- |
| Taxonomy | Gravity Spy O3: ~24 classes | Zenodo record 5649212 |
| Class naming | Exact Gravity Spy label strings (e.g., "Scattered_Light", "Blip") | Prevents mapping errors |
| Label quality filter | ml_confidence > 0.9 for training set | Filters noisy labels |
| Temporal split | Train/val/test by GPS time, ≥ 60 s gap, no overlap | Prevents temporal leakage |
| Detectors | H1 (Hanford) and L1 (Livingston), labeled per-event | Both LIGO O3 detectors |
| Rare class threshold | N_rare = TBD (locked in Phase 1 after class audit) | Depends on actual distribution |

## Model Conventions

| Convention | Value | Rationale |
| ---------- | ----- | --------- |
| ViT variant | ViT-B/16 (patch 16, 86M params), ImageNet-21k pretrained, via `timm` | Standard; patch-16 > patch-32 on small datasets |
| CNN baseline | ResNet-50 with identical training recipe | Fair comparison per contract |
| Loss | Focal loss: FL(p_t) = −α_t(1 − p_t)^γ log(p_t), γ = 2.0 | Standard for class imbalance |
| Label smoothing | ε = 0.1 | Mitigates overconfident predictions |
| Optimizer | AdamW | Standard for ViT fine-tuning |
| Precision | Mixed precision (fp16) | RTX 5090 memory budget |
| Random seeds | Recorded per run | Reproducibility |

## CW Search Conventions (Phase 4)

| Convention | Value |
| ---------- | ----- |
| CW frequency band | 20–2000 Hz |
| CW signal model | f(t) = f₀ + ḟt + ½f̈t² + Doppler |
| CW-critical glitch classes | TBD; candidates: Scattered_Light, Violin_Mode, Low_Frequency_Lines, 1080Lines |
| Data quality metric | Duty cycle or strain upper limit (exact choice in Phase 4) |

## Standard Physics Conventions (Not Applicable)

The following are null for this project (ML/signal-processing, not theoretical physics):

Metric signature, Fourier convention, gauge choice, regularization scheme, renormalization scheme, covariant derivative sign, gamma matrix convention, Levi-Civita sign, generator normalization, creation/annihilation ordering, state normalization, commutation convention, time ordering, spin basis, index positioning, coupling convention.

---

_Established: 2026-03-16 during project initialization_
