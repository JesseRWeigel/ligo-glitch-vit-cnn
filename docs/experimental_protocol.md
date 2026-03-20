# Experimental Protocol: ViT vs CNN for LIGO Glitch Classification

**Status:** LOCKED -- all downstream training and evaluation must follow this protocol.
**Locked by:** Plan 01-02 (Phase 01 - Data Pipeline & Experimental Design)
**Date:** 2026-03-16

---

## 1. Dataset

- **Source:** Gravity Spy O3 (Zenodo record 5649212 + 1486046)
- **Filtering:** `ml_confidence > 0.9`, `None_of_the_Above` (NOTA) class excluded
- **Total samples:** 325,634 across 23 classes (22 glitch morphologies + No_Glitch)
- **Detectors:** H1 (121,627 samples) and L1 (204,007 samples)
- **Spectrograms:** Pre-made Gravity Spy Q-transform images from Zooniverse CDN, resized to 224x224 RGB PNG
- **Duration views:** 4 per glitch (0.5s, 1.0s, 2.0s, 4.0s)

### Temporal Split

- **Method:** Temporal split by GPS event_time with >= 60s gap between consecutive splits
- **Split boundaries:**
  - Train: GPS <= 1261013846.52 (all events up to boundary)
  - Gap 1: 125.3s excluded (2 samples)
  - Val: GPS in [1261013971.81, 1263297523.59]
  - Gap 2: 267.8s excluded (0 samples)
  - Test: GPS >= 1263297791.34
- **Split sizes:**
  - Train: 227,943 (70.0%)
  - Val: 48,844 (15.0%)
  - Test: 48,845 (15.0%)
  - Gap-excluded: 2 samples
- **Verification:** All 6 programmatic checks pass (temporal gap, ID uniqueness, time range separation, class coverage, split ratios) -- see `data/metadata/split_statistics.json`

### Rare Classes

- **Rare-class threshold (N_rare):** 25 training samples
- **Rare classes:** Chirp (11 training samples)
- **Near-rare classes (25-50 training samples):** Wandering_Line (30), Helix (33)

---

## 2. Primary Evaluation Metric

- **Decisive metric: Macro-averaged F1 score**

  $$\text{Macro-F1} = \frac{1}{N_{\text{cls}}} \sum_{c=1}^{N_{\text{cls}}} F1_c$$

  where $F1_c = 2 \cdot \text{precision}_c \cdot \text{recall}_c / (\text{precision}_c + \text{recall}_c)$ for class $c$.

- **Per-class F1 and recall** reported for all 23 classes
- **Rare-class macro-F1** computed separately for classes with < N_rare = 25 training samples
- **FORBIDDEN: Overall accuracy must NOT be used as the primary comparison metric**
  - Reason: Overall accuracy is dominated by common classes (Scattered_Light alone is 32% of data). A model can achieve ~97% accuracy by excelling on the top 5 classes while completely failing on rare classes. Macro-F1 weights all classes equally, making rare-class performance visible.
  - Overall accuracy is reported as a secondary sanity-check metric only.
  - This prohibition is per project contract `fp-overall-accuracy`.
- **Secondary metrics (reported but not decisive):**
  - Overall accuracy (sanity check)
  - Per-class precision and recall
  - Confusion matrix
  - Macro-averaged recall

---

## 3. Statistical Testing

- **Bootstrap resampling:** >= 10,000 resamples of the test set predictions
- **Significance threshold:** p < 0.05 for declaring ViT improvement over CNN
- **95% bootstrap confidence intervals** reported for all metrics (macro-F1, per-class F1, rare-class macro-F1)
- **Test statistic:** Difference in macro-F1 between ViT and CNN on bootstrap samples

  $$\Delta_b = \text{Macro-F1}^{\text{ViT}}_b - \text{Macro-F1}^{\text{CNN}}_b, \quad b = 1, \ldots, 10000$$

- **p-value:** Fraction of bootstrap samples where $\Delta_b \leq 0$
- **Effect size:** Report the median and 95% CI of $\Delta_b$ as the effect size
- **Multiple comparisons:** If testing multiple ViT variants, apply Bonferroni correction

---

## 4. Training Recipe (Identical for CNN and ViT)

All hyperparameters below apply identically to both CNN and ViT models to ensure a fair comparison. Any deviation must be documented and justified.

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | weight_decay=0.01 |
| Learning rate | 1e-4 (ViT), 1e-3 (CNN) | Standard pretrained transfer rates; tuned per architecture |
| LR schedule | Cosine decay with linear warmup | 5 epochs warmup |
| Loss function | Focal loss, gamma=2.0 | Down-weights well-classified examples |
| Class balancing | Class-balanced batch sampling | Effective class-uniform sampling via oversampling rare classes |
| Label smoothing | epsilon=0.1 | Prevents overconfident predictions |
| Data augmentation | See below | Identical for both architectures |
| Mixed precision | fp16 | RTX 5090 memory efficiency |
| Batch size | 64 | Adjustable for memory; document if changed |
| Max epochs | 100 | With early stopping |
| Early stopping | Patience=10 on val macro-F1 | Monitors macro-F1, not accuracy |
| Random seed | Recorded per run | For reproducibility |

### Data Augmentation (identical for both)

- Random horizontal flip (p=0.5)
- Random rotation (+/- 10 degrees)
- Random resized crop (224 from 256)
- Color jitter (brightness=0.2, contrast=0.2)
- No vertical flip (spectrograms have physical up/down orientation)

### Learning Rate Exception

The base learning rate differs between CNN (1e-3) and ViT (1e-4) because pretrained transformers require lower learning rates to avoid catastrophic forgetting. This is a standard practice in transfer learning and does not violate the "identical recipe" requirement -- the LR is architecture-dependent by design, not a free parameter being tuned to favor one model.

---

## 5. Model Specifications

| Property | CNN Baseline | ViT |
|---|---|---|
| Architecture | ResNet-50 | ViT-B/16 |
| Pretrained weights | ImageNet-1K (torchvision) | ImageNet-21K (timm) |
| Classification head | Replace final FC (2048 -> 23) | Replace classification head (768 -> 23) |
| Input size | 224x224x3 | 224x224x3 |
| Parameters (approx) | ~25.6M | ~86.6M |

### Input Configuration

- **Initial experiments:** 1.0s duration view only (single spectrogram per glitch)
- **Extension (if time permits):** Multi-view fusion (all 4 duration views concatenated or averaged)
- **Normalization:** ImageNet mean/std normalization applied after loading PNG

---

## 6. Rare-Class Evaluation Strategy

- **N_rare threshold:** 25 training samples
- **Rare classes:** Chirp (11 training samples, 7 test samples)
- **Near-rare classes (for monitoring):** Wandering_Line (30 train, 6 test), Helix (33 train, 14 test)

### Rare-Class Metrics

- **Rare-class macro-F1:** Average of F1 scores across rare classes only (currently: Chirp)
- **Per-class F1 and recall:** Reported for every class, with rare classes highlighted
- **Minimum test sample rule:** If any rare class has < 5 test samples, report per-class metrics but exclude from macro-F1 with documentation (sample too small for reliable F1 estimation)
  - Current status: Chirp has 7 test samples -- included in evaluation

### Rare-Class Mitigation Strategies

1. **Class-balanced sampling:** Oversample rare classes during training (already in training recipe)
2. **Focal loss:** Down-weight easy examples, focus learning on hard/rare cases (already in training recipe)
3. **Label smoothing:** Prevent overconfident predictions on rare classes (already in training recipe)
4. **Confusion matrix analysis:** Highlight rare-class rows/columns to identify systematic misclassification patterns
5. **Per-class learning curves:** Monitor rare-class val F1 during training to detect overfitting

---

## 7. Reproducibility

- **Code:** All training, evaluation, and analysis scripts version-controlled in this repository
- **Data splits:** `data/metadata/{train,val,test}_manifest.csv` (locked, not to be regenerated)
- **Split verification:** `scripts/07_verify_split.py` (re-runnable; must produce all PASS)
- **Split statistics:** `data/metadata/split_statistics.json` (GPS boundaries, per-class counts, verification results)
- **Random seeds:** Recorded per run in training logs; must be reported in results
- **Results:** Reported with 95% bootstrap confidence intervals, not point estimates
- **Hardware:** RTX 5090 (local); training script records GPU model, CUDA version, PyTorch version
- **Library versions:** Locked in `requirements.txt` or `pyproject.toml` before training begins

### Artifact Checklist (per trained model)

- [ ] Model weights saved (`.pt` or `.safetensors`)
- [ ] Training log with per-epoch metrics (train loss, val macro-F1, val accuracy)
- [ ] Evaluation results on test set (per-class F1, confusion matrix, bootstrap CIs)
- [ ] Random seed recorded
- [ ] Hyperparameters recorded (LR, batch size, epochs trained, early stop epoch)
- [ ] Wall-clock training time recorded

---

## 8. Evaluation Procedure

### Step-by-step evaluation (applied identically to CNN and ViT)

1. Load trained model checkpoint (best val macro-F1 epoch)
2. Run inference on test set (48,845 samples)
3. Compute per-class F1, precision, recall
4. Compute macro-F1 (primary metric)
5. Compute rare-class macro-F1 (Chirp only, currently)
6. Compute overall accuracy (secondary sanity check)
7. Generate confusion matrix
8. Run bootstrap resampling (10,000 samples) for confidence intervals
9. Compute p-value for ViT vs CNN comparison
10. Report all metrics with 95% CIs

### Decision Criteria

- **ViT improves over CNN:** Macro-F1 difference is positive AND p < 0.05
- **ViT matches CNN:** Macro-F1 difference is within CI of zero (p >= 0.05)
- **ViT worse than CNN:** Macro-F1 difference is negative AND p < 0.05

---

_Protocol locked by Plan 01-02. Any modifications require documented justification and re-approval._
