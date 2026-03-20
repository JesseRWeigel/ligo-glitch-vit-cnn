# Methods Research

**Domain:** Gravitational-wave detector characterization -- glitch classification via Vision Transformers
**Researched:** 2026-03-16
**Confidence:** MEDIUM-HIGH

---

## 1. Spectrogram Generation Methods

### Recommended: Q-transform via GWpy

| Method | Purpose | Why Recommended |
|--------|---------|-----------------|
| Q-transform (constant-Q) | Time-frequency representation of transient glitches | Standard in LIGO/Virgo detector characterization; logarithmic frequency spacing matches glitch morphology; directly comparable to Gravity Spy baseline |

**Use GWpy's `TimeSeries.q_transform()`** because it is the same pipeline Gravity Spy uses, ensuring label-spectrogram consistency. The Q-transform tiles the time-frequency plane with constant quality factor Q, giving better frequency resolution at low frequencies and better time resolution at high frequencies -- exactly what glitch morphologies require.

**Key parameters (match Gravity Spy conventions):**

| Parameter | Gravity Spy Default | Notes |
|-----------|-------------------|-------|
| `qrange` | (4, 150) | Range of Q values searched; higher Q = better frequency resolution |
| `frange` | (10, 2048) Hz | Detector sensitive band; below 10 Hz seismic noise dominates |
| `mismatch` | 0.35 | Controls tiling density; lower = finer but slower |
| Time windows | 0.5s, 1.0s, 2.0s, 4.0s | Four durations per event; multi-scale morphology capture |
| Output image | 170x140 px (per window) | Gravity Spy standard; consider upscaling to 224x224 for ViT input |

**Multi-view strategy:** Gravity Spy generates four spectrograms per glitch event at different time durations (0.5s, 1.0s, 2.0s, 4.0s). The recent O4 classifier (Wu et al. 2024, arXiv:2401.12913) showed that attention-based multi-view fusion across these windows significantly outperforms single-view or naive concatenation. **Recommendation:** Generate all four time windows and use multi-view fusion, not single-window classification.

### Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|------------------------|
| Q-transform via GWpy | Omega scan (gwdetchar) | Only if you need the full pipeline with significance testing and multi-channel analysis; heavier infrastructure |
| Q-transform | CWT (continuous wavelet transform) | If you need phase information; Q-transform discards phase but is standard for Gravity Spy compatibility |
| Q-transform | STFT/mel-spectrogram | Do not use for GW glitches -- fixed time-frequency resolution is poorly matched to glitch morphologies spanning decades in frequency |

### What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Mel-scale spectrograms | Designed for human auditory perception, not GW signal morphology; compresses high-frequency information where many glitch classes have distinguishing features | Q-transform with logarithmic frequency axis |
| Raw time-series input | Loses the 2D morphological structure that defines glitch classes; time-series CNNs underperform spectrogram-based classifiers for this task | Q-transform spectrograms |
| Fixed-duration single window | Misses multi-scale temporal structure; scattered light glitches need 4s windows while blips need 0.5s | All four Gravity Spy time windows |

---

## 2. ML Architectures for GW Glitch Classification

### Baseline: Gravity Spy CNN

The Gravity Spy classifier is a custom CNN (~1.9M parameters) trained on multi-view Q-transform spectrograms across 22-24 glitch classes (22 original + Blip_Low_Frequency and Fast_Scattering added for O3). It processes four time-window spectrograms and fuses them. Reported accuracy: ~97% on well-represented classes, but degrades substantially on rare classes (Chirp, Wandering_Line).

**Limitations motivating ViT exploration:**
- Poor generalization to unseen glitch morphologies (O4 revealed this)
- CNN local receptive fields miss long-range correlations in spectrograms
- Naive fusion of multi-view inputs (concatenation) fails to capture cross-window dependencies
- Label noise from citizen-science annotations biases confident predictions

### Recommended: Fine-tuned ViT-B/16 with Multi-View Fusion

| Architecture | Purpose | Why Recommended |
|--------------|---------|-----------------|
| ViT-B/16 (ImageNet-21k pretrained) | Primary glitch classifier | Proven on Gravity Spy data (Srivastava 2025, arXiv:2510.06273); 86M params fits in 32GB VRAM with mixed precision; patch-16 outperforms patch-32 by 5-7pp on small datasets |
| Attention-based multi-view fusion | Cross-window correlation | Wu et al. (2024) demonstrated this outperforms concatenation and averaging for multi-duration spectrograms |

**Why ViT-B/16 over alternatives:**

| Variant | Parameters | Strengths | Weaknesses | Verdict |
|---------|-----------|-----------|------------|---------|
| ViT-B/16 | 86M | Best accuracy on small fine-tuning datasets; rich ImageNet-21k pretrained features; 224x224 input standard | Higher VRAM than patch-32; quadratic attention cost | **Use this** -- 32GB VRAM is sufficient with mixed precision and batch size 32-64 |
| ViT-B/32 | 88M | Faster inference; lower memory | 5-7pp accuracy drop vs patch-16 on small datasets; coarser spatial resolution loses fine glitch features | Only as a rapid prototyping baseline |
| DeiT-B | 86M | Knowledge distillation improves data efficiency; no ImageNet-21k needed | Requires teacher model (adds training complexity); marginal gains over ViT-B/16 when pretrained weights available | Consider if training from scratch |
| Swin-T | 28M | Linear attention scaling O(N); hierarchical features; strong fine-tuning performance (96% reported) | Shifted-window attention adds implementation complexity; fewer pretrained checkpoints for 224x224 | **Strong fallback** if ViT-B/16 overfits or VRAM is tight |
| Audio Spectrogram Transformer (AST) | 87M | Pre-trained on AudioSet; spectrograms are native input modality; recent success on LIGO O3/O4 (arXiv:2601.20034) | Less community tooling; AudioSet pretraining may not transfer well to Q-transform morphology | **Investigate as comparison** -- conceptually compelling but unproven advantage over ImageNet ViT |

### Architectural Decisions

**Input resolution:** 224x224 pixels. Upscale Gravity Spy's 170x140 spectrograms using bilinear interpolation. ViT-B/16 divides this into 14x14 = 196 patches of 16x16 pixels each, which is manageable for self-attention.

**Multi-view fusion approach:** Use a shared ViT encoder for all four time windows, then fuse CLS tokens via cross-attention (not concatenation). This follows the proven approach from Wu et al. (2024). The attention weights across windows also provide interpretability -- which time scale is the model using for each class?

**Classification head:** Linear layer on fused CLS token. For 24 classes, this is trivial.

### Recent and Emerging Approaches

| Approach | Reference | Status | Relevance |
|----------|-----------|--------|-----------|
| ViT-B/32 on Gravity Spy + O3a | Srivastava 2025, arXiv:2510.06273 | Published, 92.26% accuracy on 24 classes | Direct precedent; we aim to exceed this with ViT-B/16 and better class balancing |
| Multi-view attention fusion | Wu et al. 2024, arXiv:2401.12913 | Published, deployed for O4 | Provides the multi-view fusion architecture to adopt |
| CTSAE (CNN-ViT autoencoder) | arXiv:2404.15552 | Published | Unsupervised clustering; useful for discovering new glitch classes, not direct classification |
| Pre-trained Audio Transformer | arXiv:2601.20034 | Preprint, Jan 2026 | AST on O3/O4 strain data; alternative to ImageNet pretraining |
| cDVGAN synthetic glitches | arXiv:2401.16356 | Published in PRD | GAN-generated training data; up to 4.2% AUC improvement |
| GAN spectrogram augmentation | arXiv:2207.04001 | Published in MNRAS | High-resolution spectrogram generation for minority classes |

---

## 3. Class Imbalance Strategies

The Gravity Spy dataset has severe class imbalance. Common classes (Blip, Koi_Fish, Scattered_Light) have tens of thousands of examples; rare classes (Chirp, Wandering_Line, Paired_Doves) may have fewer than 100. This is the central challenge.

### Recommended: Layered Strategy

Use all three layers simultaneously -- they address different aspects of the problem.

**Layer 1 -- Loss Function: Focal Loss with Label Smoothing**

| Method | Purpose | Why Recommended |
|--------|---------|-----------------|
| Focal loss (gamma=2.0, class-weighted alpha) | Down-weight easy/common examples, focus on hard/rare ones | Standard for imbalanced classification; gamma=2.0 is the well-tested default; class-weighted alpha provides additional per-class rebalancing |
| Label smoothing (epsilon=0.1) | Mitigate overconfident predictions from noisy citizen-science labels | Wu et al. (2024) showed this improves generalization for Gravity Spy specifically |

**Do not use** plain cross-entropy loss. It will be dominated by the majority classes and the model will achieve high overall accuracy by ignoring rare classes entirely.

**Layer 2 -- Sampling: Class-Balanced Batch Sampling**

| Method | Purpose | Why Recommended |
|--------|---------|-----------------|
| Class-balanced sampling (square-root rebalancing) | Each batch contains roughly equal representation from all classes | More stable than pure uniform resampling (which causes severe overfitting on minority classes); square-root schedule is a compromise between instance-balanced and class-balanced |

**Do not use** SMOTE or similar feature-space interpolation. SMOTE was designed for tabular data and produces blurred, unphysical spectrograms when applied to image pixels. It creates artifacts the model can learn to recognize as "synthetic" rather than learning genuine glitch features.

**Layer 3 -- Data Augmentation for Minority Classes**

| Method | Purpose | Why Recommended |
|--------|---------|-----------------|
| GAN-generated synthetic spectrograms (cDVGAN) | Expand minority classes with realistic synthetic examples | Demonstrated 4.2% AUC improvement (arXiv:2401.16356); generates time-domain waveforms that produce physically plausible spectrograms |
| SpecAugment (time/frequency masking) | Regularization; forces model to use distributed features | Cheap, effective, well-tested on spectrogram inputs; apply to all classes |
| Geometric augmentation (flips, small rotations) | Basic data expansion | Horizontal flip is valid (spectrograms have no inherent left-right asymmetry in time for short windows); vertical flip is NOT valid (frequency axis has physical meaning) |

### Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|------------|---------|
| Focal loss + class-weighted alpha | Plain class-weighted CE | Focal loss additionally down-weights easy examples; CE only reweights by class frequency |
| Class-balanced batch sampling | Full oversampling (replicate minority) | Causes overfitting on minority class examples; the model memorizes them |
| cDVGAN synthetic data | Standard image-space augmentation only | Augmentation alone cannot compensate for 100x class ratio; need genuinely new examples |
| SpecAugment masking | Mixup/CutMix across classes | Mixing spectrograms from different glitch classes creates unphysical composites; SpecMix within-class is acceptable but between-class mixing confuses the model |

### Contrastive Learning (Advanced, Phase 2+)

If focal loss + balanced sampling + augmentation still yields poor rare-class recall:

**Supervised contrastive pre-training** (Khosla et al. 2020) learns an embedding space where same-class spectrograms cluster together regardless of class size. This is particularly effective for rare classes because the contrastive loss treats each positive pair equally, independent of class frequency.

**Protocol:** (1) Pre-train ViT encoder with supervised contrastive loss on all classes, (2) freeze encoder, (3) train linear classifier on top. This two-stage approach is more robust to class imbalance than end-to-end training with cross-entropy.

**Equilibrium Contrastive Learning (ECL)** is a recent framework specifically designed for imbalanced datasets that promotes geometric equilibrium in representation space. Consider this if standard SupCon still shows majority-class bias.

---

## 4. Data Augmentation Techniques for Spectrograms

### Physically Valid Augmentations

| Augmentation | Valid? | Rationale | Implementation |
|-------------|--------|-----------|----------------|
| Time-axis shift (small) | YES | Glitch trigger time has ~10ms uncertainty | Shift spectrogram left/right by up to 5% of width |
| Horizontal flip | YES | No inherent time-arrow asymmetry for short-duration glitches | `transforms.RandomHorizontalFlip(p=0.5)` |
| Vertical flip | NO | Frequency axis has physical meaning; flipping inverts the morphology | Do not apply |
| Brightness/contrast jitter | YES (mild) | Accounts for varying SNR and background noise level | ColorJitter(brightness=0.2, contrast=0.2) |
| SpecAugment time masking | YES | Forces distributed feature learning | Mask up to 20% of time bins |
| SpecAugment frequency masking | YES | Forces distributed feature learning | Mask up to 15% of frequency bins |
| Additive Gaussian noise | YES (mild) | Simulates varying noise floors | sigma = 0.01-0.05 of image std |
| Large rotation (>15 deg) | NO | Distorts time-frequency relationship; creates unphysical artifacts | Do not apply |
| Mixup within same class | YES | Interpolation between same-class examples is plausible | alpha=0.2, same-class pairs only |
| Mixup between classes | NO | Creates chimeric spectrograms with no physical meaning | Do not apply |
| CutMix between classes | CAUTION | May work if cut region is small; large cuts create confusion | Test carefully; prefer SpecAugment |

### GAN-Based Augmentation

**Recommended: cDVGAN** (arXiv:2401.16356) for minority class expansion.

cDVGAN is a conditional Derivative GAN that generates realistic time-domain glitch waveforms conditioned on class label. Advantages over image-space GANs:
- Operates in time domain, so generated spectrograms preserve physical time-frequency structure
- Conditional generation targets specific minority classes
- Published validation: FID scores and downstream classification improvement

**Workflow:**
1. Train cDVGAN on O3 time-domain glitch data for underrepresented classes
2. Generate synthetic waveforms (aim for ~1000 per minority class)
3. Apply Q-transform to synthetic waveforms to produce spectrograms
4. Add to training set with a "synthetic" flag for ablation studies

**Alternative:** Standard spectrogram-space GANs (arXiv:2207.04001) are simpler to train but produce less physically faithful outputs. Use only if cDVGAN training proves unstable.

---

## 5. Transfer Learning Approaches

### Recommended: ImageNet-21k Pretrained ViT-B/16, Fine-Tuned

| Approach | Performance | When to Use |
|----------|-------------|-------------|
| ImageNet-21k pretrained, full fine-tune | Best accuracy; features transfer well to spectrograms | Default approach; 32GB VRAM is sufficient |
| ImageNet-21k pretrained, frozen encoder + linear probe | Fast training; good baseline | Initial rapid experimentation; establish lower bound |
| ImageNet-21k pretrained, LoRA fine-tune | Memory-efficient; prevents catastrophic forgetting | If full fine-tuning overfits on small Gravity Spy dataset |
| AudioSet pretrained AST | Spectrogram-native pretraining | Worth comparing; may capture time-frequency features better |
| Training from scratch | Poor unless dataset is very large | Do not use; Gravity Spy dataset (~10K labeled) is far too small for ViT from scratch |

**Why ImageNet-21k works for spectrograms:** ViT learns general visual features (edges, textures, spatial relationships) that transfer to Q-transform spectrograms. The low-level feature representations (patch embeddings, early attention layers) capture the same kind of oriented gradients and texture patterns present in glitch morphologies. This has been empirically validated by Srivastava (2025).

**Fine-tuning protocol:**
1. Load `google/vit-base-patch16-224-in21k` from HuggingFace
2. Replace classification head with 24-class linear layer
3. Train with small learning rate for encoder (1e-5) and larger for head (1e-3)
4. Use cosine learning rate schedule with linear warmup (5% of total steps)
5. Mixed precision (torch.amp) to fit batch size 32-64 in 32GB VRAM
6. Train for 30-50 epochs with early stopping on validation macro-F1

### What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Training ViT from scratch | Requires millions of images; 10K Gravity Spy samples causes severe overfitting | ImageNet-21k pretrained fine-tuning |
| ImageNet-1k pretrained (vs 21k) | 21k has richer feature diversity; 1k is biased toward natural image categories | ImageNet-21k checkpoint |
| Freezing entire encoder | Spectrogram features differ enough from natural images that adaptation is needed | At minimum, fine-tune last 2-3 transformer blocks |

---

## 6. Evaluation Metrics for Imbalanced Classification

### Primary Metric: Macro-Averaged F1

**Do not use overall accuracy as the primary metric.** A model achieving 95% accuracy can completely ignore the 5 rarest classes. Macro-F1 weights all classes equally regardless of prevalence.

| Metric | Purpose | When to Use |
|--------|---------|-------------|
| Macro-F1 | Primary evaluation; equal weight to all classes | Model selection, hyperparameter tuning, final reporting |
| Per-class recall | Identify which rare classes the model fails on | Diagnostic; drives targeted augmentation |
| Per-class precision | Identify false positive patterns | Diagnostic; confusion between similar morphologies |
| Macro-averaged AUC-ROC | Threshold-independent performance across classes | Model comparison; complements F1 |
| Confusion matrix | Full error structure | Essential for understanding misclassification patterns (e.g., Blip vs Blip_Low_Frequency confusion) |
| Overall accuracy | Sanity check only | Report but do not optimize for |

### Validation Strategy

| Check | Expected Result | Tolerance | Reference |
|-------|----------------|-----------|-----------|
| Reproduce Gravity Spy CNN baseline | ~97% accuracy, ~90% macro-F1 (estimated) | Within 2pp of published values | Zevin et al. 2017, Bahaadini et al. 2018 |
| ViT-B/32 baseline reproduction | 92.26% accuracy on 24 classes | Within 1pp | Srivastava 2025, arXiv:2510.06273 |
| Per-class recall on rare classes | Higher than CNN baseline | Statistically significant improvement (McNemar's test) | This project's hypothesis |
| Cross-detector generalization | Similar performance on H1 and L1 data | Within 5pp per-class F1 | Test on held-out detector |
| O4 temporal generalization | Reasonable performance on O4 data not seen in training | Degradation <10pp macro-F1 from O3 | O4 validation set |

### Statistical Rigor

- **Use stratified k-fold cross-validation** (k=5) to ensure all classes appear in every fold
- **Report confidence intervals** on macro-F1 using bootstrap resampling (1000 iterations)
- **Use McNemar's test** for pairwise model comparison (ViT vs CNN), not just point estimates
- **Calibration curve** for each class: are predicted probabilities reliable? Important for downstream use in detector characterization pipelines

---

## 7. Software Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| PyTorch | >=2.2 | Deep learning framework | Native ViT support via torchvision; best ecosystem for research; CUDA 12.x for RTX 5090 |
| HuggingFace transformers | >=4.40 | Pretrained ViT checkpoints and tokenizers | One-line loading of `google/vit-base-patch16-224-in21k`; ViTForImageClassification |
| GWpy | >=3.0 | Q-transform spectrogram generation from LIGO strain data | Standard LIGO data analysis library; same pipeline as Gravity Spy |
| timm (PyTorch Image Models) | >=1.0 | Additional ViT variants (DeiT, Swin, BEiT) | Ross Wightman's library; most comprehensive collection of pretrained vision transformers |

### Supporting Libraries

| Library | Purpose | When to Use |
|---------|---------|-------------|
| gwdatafind | Locate LIGO frame files on GWOSC | Data download and discovery |
| scikit-learn | Metrics, stratified splits, confusion matrices | Evaluation pipeline |
| torchmetrics | GPU-accelerated metrics during training | Per-epoch macro-F1 without CPU transfer |
| albumentations | Fast image augmentation pipeline | SpecAugment-style masking, geometric transforms |
| wandb or tensorboard | Experiment tracking | Log per-class metrics across training runs |
| matplotlib + seaborn | Confusion matrices, per-class performance plots | Results visualization |

### Installation

```bash
# Core computational environment (assumes CUDA 12.x for RTX 5090)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers timm torchmetrics albumentations

# GW data tools
pip install gwpy gwdatafind

# Evaluation and tracking
pip install scikit-learn wandb matplotlib seaborn

# Optional: GAN augmentation
pip install pytorch-fid  # for evaluating synthetic spectrogram quality
```

---

## 8. Method Selection by Problem Type

**If class imbalance is the primary bottleneck (likely):**
- Use focal loss + class-balanced sampling + GAN augmentation for minority classes
- Evaluate with macro-F1, not accuracy
- Consider supervised contrastive pre-training if focal loss plateau is reached

**If overfitting on small dataset is the primary bottleneck:**
- Use LoRA fine-tuning instead of full fine-tuning
- Increase augmentation aggressiveness (SpecAugment + noise + geometric)
- Consider Swin-T (28M params) instead of ViT-B/16 (86M params)

**If new/unknown glitch classes appear in O4 validation:**
- Use CTSAE-style unsupervised clustering (arXiv:2404.15552) to discover new morphologies
- Train with an "other/unknown" class using out-of-distribution detection
- The AST approach (arXiv:2601.20034) showed t-SNE clustering of embeddings naturally separates unknown classes

**If multi-view fusion is not improving over single-view:**
- Verify that all four time windows are correctly aligned to the same event
- Check attention weights: if one window dominates, the fusion is not helping
- Fall back to late fusion (average logits) as a simpler baseline

---

## Sources

### Direct Precedents for This Project
- Srivastava 2025, "Vision Transformer for Transient Noise Classification," arXiv:2510.06273 -- ViT-B/32 on Gravity Spy, 92.26% accuracy
- Wu et al. 2024, "Advancing Glitch Classification in Gravity Spy: Multi-view Fusion with Attention-based Machine Learning for Advanced LIGO's Fourth Observing Run," arXiv:2401.12913 -- O4 classifier with attention fusion
- arXiv:2601.20034, "The Sound of Noise: Leveraging the Inductive Bias of Pre-trained Audio Transformers for Glitch Identification in LIGO" -- AST approach on O3/O4

### Gravity Spy Baseline
- Zevin et al. 2017, "Gravity Spy: Integrating Advanced LIGO Detector Characterization, Machine Learning, and Citizen Science," CQG 34 064003
- Bahaadini et al. 2018, "Machine Learning for Gravity Spy: Glitch Classification and Dataset," Information Sciences 444, 172-186
- Gravity Spy O1-O3b labels: https://zenodo.org/records/5649212

### GAN Augmentation
- arXiv:2401.16356, "cDVGAN: One Flexible Model for Multi-class Gravitational Wave Signal and Glitch Generation," PRD 110 022004
- arXiv:2207.04001, "On Improving the Performance of Glitch Classification for Gravitational Wave Detection by using Generative Adversarial Networks," MNRAS 515 4606

### Unsupervised/Hybrid Approaches
- arXiv:2404.15552, "Cross-Temporal Spectrogram Autoencoder (CTSAE): Unsupervised Dimensionality Reduction for Clustering Gravitational Wave Glitches"

### ViT Architecture and Training
- Dosovitskiy et al. 2021, "An Image is Worth 16x16 Words," ICLR 2021
- Touvron et al. 2021, "Training data-efficient image transformers & distillation through attention," ICML 2021 (DeiT)
- Liu et al. 2021, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV 2021
- Steiner et al. 2022, "How to Train Your ViT?," TMLR 2022

### Data Augmentation
- Park et al. 2019, "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," Interspeech 2019
- Khosla et al. 2020, "Supervised Contrastive Learning," NeurIPS 2020

### GW Data Tools
- GWpy documentation: https://gwpy.github.io/docs/stable/
- GWDetChar omega scans: https://gwdetchar.readthedocs.io/en/stable/omega/

---

_Methods research for: ViT-based LIGO glitch classification_
_Researched: 2026-03-16_
