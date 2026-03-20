# Research Summary

**Project:** Transformer-Based Rare Glitch Classification for LIGO Continuous Wave Searches
**Domain:** Gravitational-wave detector characterization / machine learning
**Researched:** 2026-03-16
**Confidence:** MEDIUM-HIGH

## Unified Notation

| Symbol | Quantity | Units/Dimensions | Convention Notes |
|--------|---------|------------------|------------------|
| h(t) | Strain time series | 1/sqrt(Hz) | LIGO convention; dimensionless strain spectral density |
| SNR | Signal-to-noise ratio | dimensionless | Matched-filter SNR for glitch triggers; Gravity Spy uses Omicron SNR > 7.5 |
| Q | Quality factor of Q-transform | dimensionless | Range (4, 150) in Gravity Spy; controls time-frequency resolution tradeoff |
| f | Frequency | Hz | CW search band: 20--2000 Hz; glitch spectrograms: 10--2048 Hz |
| tau | Glitch duration | s | 0.01--10 s; captured by multi-duration windows (0.5, 1.0, 2.0, 4.0 s) |
| F1_macro | Macro-averaged F1 score | dimensionless | Primary evaluation metric; weights all classes equally regardless of prevalence |
| alpha | Focal loss class weight | dimensionless | Per-class rebalancing weight; inverse frequency or tuned |
| gamma | Focal loss focusing parameter | dimensionless | Default 2.0; down-weights easy/common examples |
| epsilon | Label smoothing parameter | dimensionless | Default 0.1; mitigates overconfident predictions from noisy labels |
| N_cls | Number of glitch classes | integer | 22 (O1/O2), 24 (O3: +Blip_Low_Frequency, +Fast_Scattering) |
| ml_confidence | Gravity Spy ML confidence | [0, 1] | Metadata field; filter training set at > 0.9 for label quality |

**Unit system:** SI for physical quantities (strain, frequency, time). Dimensionless for ML metrics and hyperparameters. Natural units are not used in this project.

**Spectrogram convention:** Q-transform with logarithmic frequency axis, GWpy implementation, normalized to [0, 1] via min-max after SNR clipping to [0, 25.5]. Image dimensions: 224x224 pixels (upscaled from Gravity Spy native 170x140 via bilinear interpolation).

**Metric convention:** "F1" without qualifier means macro-averaged F1 throughout this project. "Accuracy" refers to overall accuracy and is reported only as a secondary/sanity-check metric.

## Executive Summary

The literature strongly supports applying Vision Transformers to LIGO glitch classification as a timely and well-motivated research direction. The Gravity Spy CNN baseline (~97% overall accuracy, George et al. 2018 reaching >98.8% with deep transfer learning) is acknowledged by its own team as insufficient for O4 and beyond (Zevin et al. 2024). The single published ViT attempt (Srivastava and Niedzielski 2025, ViT-B/32, 92.26% accuracy on 24 classes) used a basic fine-tuning recipe without modern ViT training techniques, class balancing, or multi-view fusion -- leaving substantial room for improvement. The gap between CNN performance and ViT potential is a training recipe problem, not a fundamental architectural limitation. ViT-B/16 with ImageNet-21k pretraining, focal loss, class-balanced sampling, and multi-view attention fusion is the recommended approach.

The central technical challenge is class imbalance, not model architecture. The Gravity Spy dataset has 100x class ratio between common classes (Blip: ~7500 samples) and rare targets (Paired_Doves: ~100, Wandering_Line: ~25 training samples). No published work has optimized for rare-class F1 as the primary metric. A layered strategy -- focal loss with label smoothing, class-balanced batch sampling, GAN-based augmentation (cDVGAN, demonstrated 4.2% AUC improvement in PRD), and potentially supervised contrastive pretraining -- addresses this from multiple angles. The evaluation must use macro-F1 and per-class recall, never overall accuracy, as the primary metrics.

The principal risks are: (1) temporal data leakage inflating test metrics if train/test splits are not time-based, (2) unfair CNN baseline comparison if the CNN does not receive the same modern training recipe as the ViT, (3) label noise in rare classes from citizen-science annotation, and (4) O3-to-O4 distribution shift invalidating trained models. All are preventable with proper experimental design locked in before any model training begins. The CW search connection is strongest for glitch classes with spectral-line-like characteristics (Scattered_Light, Violin_Mode, Low_Frequency_Lines), not for all 23 classes equally.

## Key Findings

### Computational Approaches

The full pipeline is computationally tractable on a single RTX 5090 (32 GB VRAM). The main bottleneck is spectrogram generation (5--10 hours on 12 CPU cores for 400K spectrograms), not model training (4--8 hours for ViT-B/16 fine-tuning). Total storage: 400--560 GB with custom spectrograms from strain data, or 15--25 GB using pre-made Gravity Spy images. A prototyping shortcut exists: use pre-made Gravity Spy images from Zenodo to reach model training within hours. [CONFIDENCE: HIGH]

**Core approach:**

- **Q-transform spectrograms via GWpy**: Standard LIGO representation; 4 time durations per glitch; directly comparable to Gravity Spy baseline -- ensures label-spectrogram consistency
- **ViT-B/16 (ImageNet-21k pretrained) via timm**: 86M parameters; fits in 32 GB VRAM with mixed precision at batch size 64-128; patch-16 outperforms patch-32 by 5-7pp on small datasets
- **AdamW + cosine annealing + mixed precision**: Standard ViT training recipe; differential learning rates (1e-5 encoder, 1e-3 head); early stopping on validation macro-F1

### Prior Work Landscape

The field has a clear trajectory: CNN baselines established (2017-2018), acknowledged as insufficient (2024), and ViT exploration just beginning (2025). [CONFIDENCE: HIGH]

**Must reproduce (benchmarks):**

- Gravity Spy CNN ~97% overall accuracy on common classes (Zevin et al. 2017) -- community baseline
- Deep transfer learning >98.8% overall accuracy (George et al. 2018) -- accuracy ceiling for CNN approaches
- ViT-B/32 92.26% on 24 classes (Srivastava 2025) -- ViT baseline to exceed

**Novel predictions (contributions):**

- ViT-B/16 with modern training recipe exceeding CNN macro-F1, particularly on rare classes
- Quantified CW search sensitivity improvement from better rare-class classification
- Self-supervised pretraining on unlabeled LIGO glitches (unexplored in literature, high potential given ~600K+ classified glitches)

**Defer (future work):**

- O4 deployment and real-time inference pipeline
- Novel glitch class discovery (open-set recognition) -- distinct from rare-class classification
- Cross-detector generalization to Virgo/KAGRA

### Methods and Tools

The software stack is mature and well-documented: PyTorch >= 2.2, timm >= 1.0, GWpy >= 3.0, scikit-learn for evaluation. All tools are open-source and widely used. The recommended architecture is ViT-B/16 with attention-based multi-view fusion of 4-duration spectrograms (following Wu et al. 2024). Swin-T (28M params) is the fallback if ViT-B/16 overfits. Audio Spectrogram Transformer (AST) is worth investigating as a comparison but is unproven for Q-transform spectrograms. [CONFIDENCE: HIGH]

**Major components:**

1. **GWpy Q-transform pipeline** -- spectrogram generation matching Gravity Spy conventions (4 durations, 10-2048 Hz, logarithmic frequency)
2. **timm ViT-B/16** -- pretrained backbone with linear classification head for 24 classes
3. **Multi-view attention fusion** -- shared encoder for 4 time windows, cross-attention on CLS tokens (Wu et al. 2024 architecture)
4. **Focal loss + class-balanced sampling + cDVGAN augmentation** -- layered class imbalance strategy

### Critical Pitfalls

1. **Accuracy paradox from class imbalance** -- Use macro-F1 and per-class recall from day one; never optimize for overall accuracy. A 95% accurate model can ignore all rare classes. [Phase: experimental design]
2. **Temporal data leakage** -- Use time-based train/test splits with >= 60s gap, not random stratified splits. LIGO glitches cluster temporally; adjacent glitches share instrumental state. Discovery after training invalidates ALL results. [Phase: data preparation]
3. **Unfair CNN baseline** -- Train CNN baseline (ResNet-50) with identical modern recipe (AdamW, cosine schedule, same augmentation, same epochs) as ViT. Match model capacity. Otherwise, any ViT advantage may reflect training recipe, not architecture. [Phase: experimental design]
4. **ViT data hunger** -- Never train from scratch; always fine-tune from ImageNet-21k. Gravity Spy (~10K high-confidence labeled) is 200x smaller than ImageNet. Consider LoRA if full fine-tuning overfits. [Phase: model architecture]
5. **Label noise in rare classes** -- Filter by ml_confidence > 0.9; apply label smoothing (epsilon=0.1); manually audit 50 samples per rare class before training. [Phase: data preparation]
6. **O3-to-O4 distribution shift** -- Evaluate on temporally disjoint data; report per-detector results; include novel class detection mechanism. [Phase: evaluation design]

## Approximation Landscape

| Method | Valid Regime | Breaks Down When | Controlled? | Complements |
|--------|-------------|-----------------|-------------|-------------|
| ViT-B/16 fine-tuning (ImageNet-21k) | Labeled dataset >= 1K samples per class; spectrograms as 224x224 images | Rare classes with < 50 samples; genuinely novel morphologies not in training set | No formal expansion parameter; empirical convergence via validation macro-F1 | GAN augmentation for rare classes; contrastive pretraining |
| CNN (ResNet-50, transfer learning) | Same as ViT; stronger inductive bias helps with smaller datasets | Long-range spatial dependencies in spectrograms; multi-scale fusion | Same as ViT | ViT for global attention; serves as controlled comparison |
| cDVGAN augmentation | Minority classes with identifiable morphology; time-domain generation | Classes with highly heterogeneous morphology (None_of_the_Above); if GAN training is unstable | Validated via FID scores and downstream AUC | Class-balanced sampling; focal loss |
| Focal loss (gamma=2.0) | Any class imbalance; down-weights easy examples | Extreme imbalance (>1000x) with very noisy labels | Gamma is tunable; well-studied | Label smoothing; contrastive pretraining |
| Supervised contrastive pretraining | Small labeled datasets; class-size-independent embedding learning | Requires meaningful positive pairs; may not help if morphological variation within-class exceeds between-class | SupCon loss is well-characterized | Focal loss (for the classification head stage) |

**Coverage gap:** No reliable method exists for classifying genuinely novel glitch morphologies (open-set recognition) with high confidence. This is explicitly out of scope but relevant for O4 deployment. Within-scope, the layered imbalance strategy has no coverage gap for the 24 known classes, though classes with < 25 training samples (Wandering_Line) remain high-risk.

## Theoretical Connections

### Cross-Domain Transfer Learning (Established)

ImageNet-pretrained features transfer to Q-transform spectrograms because low-level visual features (edges, textures, oriented gradients) are shared between natural images and time-frequency representations. This has been empirically validated by multiple groups (George et al. 2018, Srivastava 2025). The transfer is imperfect -- spectrograms have physical semantics (frequency axis, energy scale) that natural images lack -- but sufficient for fine-tuning.

### Audio-Visual Domain Bridge (Conjectured)

The Audio Spectrogram Transformer (AST, arXiv:2601.20034) suggests that AudioSet-pretrained transformers may transfer better to Q-transform spectrograms than ImageNet-pretrained ones, since both domains involve time-frequency representations. This is plausible but unvalidated for LIGO data. Worth investigating as a comparison but not as the primary approach.

### GAN-Classifier Symbiosis (Established)

cDVGAN generates time-domain waveforms conditioned on class label, which are then Q-transformed into spectrograms. This preserves physical time-frequency structure that image-space GANs cannot guarantee. The downstream classifier benefits from physically plausible synthetic examples. This pipeline (GAN augmentation -> classifier training) is established in the GW literature (PRD publication).

### Self-Supervised Pretraining Transfer from GW Parameter Estimation (Conjectured)

GraviBERT (arXiv:2512.21390) demonstrated that transformer + self-supervised pretraining dramatically improves GW parameter estimation. The analogous experiment for glitch classification -- masked spectrogram prediction or contrastive learning on ~600K unlabeled Gravity Spy spectrograms -- has not been published. Given the massive unlabeled data pool, this is a high-potential unexplored direction. [CONFIDENCE: MEDIUM -- the analogy is strong but untested for classification]

### Glitch-CW Connection (Established but Nuanced)

The link between glitch classification and CW search sensitivity is indirect and class-dependent. Glitches primarily affect transient GW searches. For CW searches, persistent spectral lines are the dominant artifact, but certain glitch classes with quasi-periodic or spectral-line-like character (Scattered_Light, Violin_Mode, 1080Lines, Low_Frequency_Lines) contaminate CW frequency bands. Better classification of these boundary cases improves CW data quality products. This connection is strongest for line-like glitches, not for all classes equally.

## Critical Claim Verification

| # | Claim | Source | Verification | Result |
|---|-------|--------|--------------|--------|
| 1 | ViT-B/32 achieves 92.26% on 24 LIGO glitch classes | PRIOR-WORK.md | web_search: arXiv:2510.06273 | CONFIRMED -- published in Acta Astronomica Vol. 74 (2024) |
| 2 | cDVGAN yields 4.2% AUC improvement for glitch classification | METHODS.md | web_search: arXiv:2401.16356 | CONFIRMED -- published in PRD 110 022004 (2024) |
| 3 | Gravity Spy team acknowledges CNN insufficient for O4 | PRIOR-WORK.md | web_search: arXiv:2308.15530 | CONFIRMED -- CNN confidently mislabeled novel O4 glitches; published in EPJ Plus 139 100 (2024) |
| 4 | George et al. achieve >98.8% with deep transfer learning | PRIOR-WORK.md | arXiv:1706.07446, PRD 97 101501 | CONFIRMED -- peer-reviewed PRD publication (2018) |
| 5 | Gravity Spy dataset has ~614K ML-classified glitches O1-O3 | PRIOR-WORK.md | Zenodo 5649212 | CONFIRMED -- public dataset release |

## Cross-Validation Matrix

|                    | CNN (ResNet-50) | ViT-B/16 | Swin-T | AST | Gravity Spy CNN |
|--------------------|:---:|:---:|:---:|:---:|:---:|
| CNN (ResNet-50)    | -- | All 24 classes, same test set | All 24 classes | All 24 classes | Common classes only (different taxonomy possible) |
| ViT-B/16           | All 24 classes | -- | All 24 classes | All 24 classes | Common classes |
| Exact/Analytical   | None (no analytical solution for glitch classification) | None | None | None | None |
| Experiment (human) | Citizen-science labels on test set | Same | Same | Same | Gravity Spy volunteer consensus |

**Key observation:** There is no analytical ground truth for glitch classification. All cross-validation is empirical, against human labels. The quality of those labels is itself uncertain for rare classes. This means benchmark values (accuracy, F1) carry systematic label-noise uncertainty that cannot be reduced without expert re-annotation.

## Implications for Roadmap

### Suggested Phase Structure

### Phase 1: Data Pipeline and Experimental Design

**Rationale:** All downstream results depend on correct data preparation. Temporal leakage and label quality issues are irrecoverable if discovered late. The experimental design (metrics, splitting strategy, baseline protocol) must be locked before any model training.
**Delivers:** Clean spectrogram dataset with time-based train/val/test splits; class frequency analysis; label quality audit for rare classes; documented experimental design protocol specifying identical training recipe for CNN and ViT.
**Validates:** Spectrogram generation matches Gravity Spy website images; no GPS time overlap within 60s between splits; class distribution documented.
**Avoids:** Pitfalls 1 (accuracy paradox -- metrics locked), 2 (temporal leakage -- splits defined), 5 (label noise -- audit completed), 6 (normalization -- convention fixed).

### Phase 2: Baseline Reproduction and Fair CNN Comparison

**Rationale:** Must establish the performance floor before claiming improvement. The CNN baseline must use the same modern training recipe as the ViT to ensure a fair comparison. Reproducing published results validates the data pipeline.
**Delivers:** ResNet-50 baseline trained with modern recipe; reproduced Gravity Spy CNN ~97% accuracy and ViT-B/32 ~92% accuracy; per-class F1 breakdown establishing the rare-class performance gap.
**Uses:** PyTorch, timm, same augmentation/optimizer/schedule for both CNN and ViT-B/32.
**Builds on:** Phase 1 dataset and splitting protocol.

### Phase 3: ViT-B/16 with Class Imbalance Strategy

**Rationale:** This is the core contribution. ViT-B/16 with focal loss, class-balanced sampling, and GAN augmentation targets the rare-class performance gap identified in Phase 2. Multi-view attention fusion captures cross-duration morphological information.
**Delivers:** ViT-B/16 classifier with rare-class F1 improvement over CNN baseline; ablation study isolating architecture vs. training recipe vs. imbalance strategy contributions; attention map analysis.
**Uses:** ViT-B/16 (timm, ImageNet-21k), focal loss, cDVGAN, multi-view attention fusion.
**Builds on:** Phase 2 baselines and per-class performance analysis.

### Phase 4: CW Search Sensitivity Analysis

**Rationale:** The CW connection is the project's scientific motivation. Must quantify the impact of improved rare-class classification on CW search data quality. This requires identifying which glitch classes affect CW frequency bands and measuring the duty cycle or sensitivity improvement.
**Delivers:** Quantified CW sensitivity improvement from better glitch classification; identification of CW-critical glitch classes; data quality veto comparison.
**Uses:** CW search sensitivity estimation; glitch-frequency-band overlap analysis.
**Builds on:** Phase 3 trained classifier and per-class performance.

### Phase 5: Paper Writing and O4 Validation

**Rationale:** O4 validation tests generalization beyond the training distribution. Paper writing is the final deliverable. O4 data provides the distribution-shift evaluation that reviewers will expect.
**Delivers:** Research paper; O4 generalization results; per-detector analysis; calibration assessment.
**Builds on:** All prior phases.

### Phase Ordering Rationale

- **Phase 1 before all else:** Data pipeline correctness is a prerequisite; leakage or normalization errors invalidate everything downstream.
- **Phase 2 before Phase 3:** Fair baselines must exist before claiming improvement; reproducing published results validates the pipeline.
- **Phase 3 is the core:** Architecture + imbalance strategy is the novel contribution; depends on validated baselines.
- **Phase 4 after Phase 3:** CW analysis requires a trained classifier; cannot be parallelized with Phase 3.
- **Phase 5 last:** Writing and O4 validation integrate all results.

### Phases Requiring Deep Investigation

- **Phase 3:** Novel combination of ViT architecture with multi-view fusion and layered imbalance strategy for LIGO data. No direct precedent for this specific combination. The interaction between focal loss, class-balanced sampling, and GAN augmentation needs ablation. Self-supervised pretraining is an optional advanced direction with high potential but no literature precedent for glitch classification.
- **Phase 4:** The glitch-to-CW-sensitivity pipeline is asserted in the literature but not quantified. Defining the right CW sensitivity metric and connecting it to per-class glitch classification improvement requires domain expertise.

Phases with established methodology (straightforward execution):

- **Phase 1:** Standard data engineering with well-documented tools (GWpy, Gravity Spy Zenodo data). The key decisions (time-based splits, normalization) are documented in the research files.
- **Phase 2:** Transfer learning fine-tuning with timm/PyTorch is a well-trodden path. Reproducing published results is standard.
- **Phase 5:** Paper writing follows standard publication conventions.

## Input Quality -> Roadmap Impact

| Input File | Quality | Affected Recommendations | Impact if Wrong |
|------------|---------|------------------------|-----------------|
| METHODS.md | Good | Architecture selection (ViT-B/16), training recipe, augmentation strategy | Would need to revisit model choice and hyperparameters |
| PRIOR-WORK.md | Good | Benchmark values (97%, 98.8%, 92.26%), gap analysis, CW connection nuance | Phases 2-3 success criteria may need revision |
| COMPUTATIONAL.md | Good | Resource estimates, data pipeline, GPU memory budget | Tool substitution or timeline adjustment |
| PITFALLS.md | Good | Risk mitigation across all phases, experimental design constraints | Blind spots in Phases 1-3 |

## Confidence Assessment

| Area | Confidence | Notes |
|------|-----------|-------|
| Computational Approaches | HIGH | Mature software stack (PyTorch, timm, GWpy); RTX 5090 has ample capacity; resource estimates well-characterized |
| Prior Work | HIGH | Multiple peer-reviewed publications; Gravity Spy dataset publicly available on Zenodo; key claims independently verified |
| Methods | MEDIUM-HIGH | ViT fine-tuning is well-established for vision tasks; application to LIGO spectrograms has one precedent (Srivastava 2025); multi-view fusion demonstrated for CNNs but not ViTs on this data |
| Pitfalls | HIGH | Comprehensive failure mode analysis; all critical pitfalls have documented prevention strategies; temporal leakage and accuracy paradox are well-known in the GW-ML community |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Self-supervised pretraining effectiveness:** GraviBERT showed promise for parameter estimation but the transfer to classification is conjectured, not demonstrated. Investigate in Phase 3 as an optional advanced direction.
- **CW sensitivity quantification:** The operational link between glitch classification improvement and CW search sensitivity gain is qualitatively understood but not quantified in the literature. Phase 4 must develop this methodology.
- **Rare-class label quality:** Manual audit of rare-class labels has not been performed. Phase 1 must include this (budget 2-4 hours).
- **Optimal multi-view fusion for ViT:** Wu et al. (2024) demonstrated multi-view attention fusion for CNNs. The ViT-native version (cross-attention on CLS tokens) is untested and may require architecture experimentation.
- **Evaluation metric for "CW benefit":** The specific metric connecting better glitch classification to CW search improvement (duty cycle gain? upper limit improvement? frequency band recovery?) needs definition in Phase 4.

## Open Questions

1. **Can ViT outperform CNN on rare glitch classes with proper training recipe?** -- The only ViT result (92.26% overall, no per-class breakdown) used ViT-B/32 with basic fine-tuning. ViT-B/16 with focal loss, class-balanced sampling, and augmentation has not been tested. [Priority: HIGH, blocks Phase 3 success criterion]

2. **What is the optimal multi-view fusion strategy for ViT?** -- Cross-attention on CLS tokens from 4 duration-specific encoders vs. 4-channel input vs. late fusion. No published comparison for ViT on LIGO data. [Priority: HIGH, blocks Phase 3 architecture]

3. **Does self-supervised pretraining on ~600K unlabeled Gravity Spy spectrograms improve rare-class F1?** -- High potential based on GraviBERT analogy but untested. [Priority: MEDIUM, optional Phase 3 extension]

4. **How does rare-class F1 improvement map to CW search sensitivity?** -- The connection is qualitatively motivated but never quantified. [Priority: HIGH, blocks Phase 4]

5. **What is the right evaluation metric for CW benefit?** -- Duty cycle, upper limit, frequency band recovery are all candidates. [Priority: HIGH, blocks Phase 4]

6. **How severe is O3-to-O4 distribution shift for the ViT classifier?** -- LIGO detector characterization team noted new glitch types in O4. [Priority: MEDIUM, Phase 5]

## Sources

### Primary (HIGH)

- Zevin et al. (2017), arXiv:1611.04596, CQG 34(6) 064003 -- Gravity Spy foundational paper, CNN baseline, glitch taxonomy
- George, Shen & Huerta (2018), arXiv:1706.07446, PRD 97 101501 -- Deep transfer learning, >98.8% accuracy benchmark, unsupervised clustering
- Zevin et al. (2024), arXiv:2308.15530, EPJ Plus 139 100 -- Lessons learned, O3 taxonomy, CNN limitations acknowledged for O4
- Bahaadini et al. (2018), Information Sciences 444 -- Gravity Spy dataset formalization, class distribution statistics
- Glanzer et al. (2022), Zenodo 5649212 -- Gravity Spy ML classifications O1-O3b dataset release
- Dosovitskiy et al. (2021), arXiv:2010.11929, ICLR 2021 -- ViT architecture paper

### Secondary (MEDIUM)

- Srivastava & Niedzielski (2025), arXiv:2510.06273, Acta Astronomica 74(3) -- First ViT application to LIGO glitches, 92.26% accuracy
- Wu/Raza et al. (2024), arXiv:2401.12913 -- Multi-view attention fusion for O4 Gravity Spy
- cDVGAN (2024), arXiv:2401.16356, PRD 110 022004 -- GAN augmentation for glitch classification, 4.2% AUC improvement
- Colgan et al. (2022), arXiv:2207.04001, MNRAS 515 4606 -- GAN spectrogram augmentation
- LIGO DetChar O4 (2024), arXiv:2409.02831 -- O4 detector characterization, data quality products
- Touvron et al. (2021), arXiv:2012.12877, ICML 2021 -- DeiT training recipe for data-efficient ViTs
- Khosla et al. (2020), NeurIPS 2020 -- Supervised contrastive learning

### Tertiary (LOW)

- GraviBERT (2025), arXiv:2512.21390 -- Transformer + self-supervised pretraining for GW inference (not classification)
- AST for LIGO (2026), arXiv:2601.20034 -- Audio Spectrogram Transformer on O3/O4 (preprint, Jan 2026)
- CTSAE (2024), arXiv:2404.15552 -- Unsupervised clustering for glitch grouping
- Lee et al. (2022), arXiv:2112.13492 -- ViT for small-size datasets (SPT + LSA modifications)
- Bai et al. (2021) -- "Are Transformers More Robust Than CNNs?" (training recipe dominance)

---

_Research synthesis completed: 2026-03-16_
_Ready for research plan: yes_

```yaml
# --- ROADMAP INPUT (machine-readable, consumed by gpd-roadmapper) ---
synthesis_meta:
  project_title: "Transformer-Based Rare Glitch Classification for LIGO Continuous Wave Searches"
  synthesis_date: "2026-03-16"
  input_files: [METHODS.md, PRIOR-WORK.md, COMPUTATIONAL.md, PITFALLS.md]
  input_quality: {METHODS: good, PRIOR-WORK: good, COMPUTATIONAL: good, PITFALLS: good}

conventions:
  unit_system: "SI"
  metric_signature: "N/A"
  fourier_convention: "N/A"
  coupling_convention: "N/A"
  renormalization_scheme: "N/A"
  spectrogram_convention: "Q-transform, GWpy, log-frequency, min-max normalized to [0,1], 224x224 px"
  primary_metric: "macro-averaged F1"

methods_ranked:
  - name: "ViT-B/16 fine-tuning (ImageNet-21k)"
    regime: "Labeled spectrograms >= 1K/class; 224x224 images; single GPU with 32GB VRAM"
    confidence: MEDIUM
    cost: "4-8 hours training on RTX 5090; 86M parameters"
    complements: "CNN baseline for fair comparison; cDVGAN for rare-class augmentation"
  - name: "ResNet-50 (modern training recipe)"
    regime: "Same as ViT; stronger inductive bias on small datasets"
    confidence: HIGH
    cost: "2-4 hours training; 25M parameters"
    complements: "ViT-B/16 as controlled comparison target"
  - name: "Focal loss + class-balanced sampling"
    regime: "Any class imbalance ratio; combined with any architecture"
    confidence: HIGH
    cost: "Negligible additional cost over standard training"
    complements: "GAN augmentation for extreme imbalance (>100x)"
  - name: "cDVGAN augmentation"
    regime: "Minority classes with identifiable morphology; generates time-domain waveforms"
    confidence: MEDIUM
    cost: "Additional GAN training time (~hours); spectrogram generation from synthetic waveforms"
    complements: "Focal loss and class-balanced sampling"
  - name: "Multi-view attention fusion"
    regime: "4-duration Q-transform spectrograms; improves morphologically ambiguous classes"
    confidence: MEDIUM
    cost: "~2x inference cost vs single-view; shared encoder mitigates training cost"
    complements: "Single-view ViT as baseline"
  - name: "Swin-T"
    regime: "Fallback if ViT-B/16 overfits; 28M params; linear attention scaling"
    confidence: MEDIUM
    cost: "Similar to ViT-B/16"
    complements: "ViT-B/16 as primary; useful if dataset proves too small"

phase_suggestions:
  - name: "Data Pipeline and Experimental Design"
    goal: "Build clean spectrogram dataset with time-based splits and locked experimental protocol"
    methods: ["Q-transform via GWpy"]
    depends_on: []
    needs_research: false
    risk: LOW
    pitfalls: ["temporal-leakage", "label-noise", "normalization-mismatch"]
  - name: "Baseline Reproduction and Fair CNN Comparison"
    goal: "Establish validated performance floor with fair CNN baseline using modern training recipe"
    methods: ["ResNet-50 (modern training recipe)", "Focal loss + class-balanced sampling"]
    depends_on: ["Data Pipeline and Experimental Design"]
    needs_research: false
    risk: LOW
    pitfalls: ["unfair-baseline", "accuracy-paradox"]
  - name: "ViT-B/16 with Class Imbalance Strategy"
    goal: "Demonstrate rare-class F1 improvement over CNN baseline via ViT architecture + layered imbalance strategy"
    methods: ["ViT-B/16 fine-tuning (ImageNet-21k)", "Focal loss + class-balanced sampling", "cDVGAN augmentation", "Multi-view attention fusion"]
    depends_on: ["Baseline Reproduction and Fair CNN Comparison"]
    needs_research: true
    risk: MEDIUM
    pitfalls: ["vit-data-hunger", "accuracy-paradox", "attention-overinterpretation"]
  - name: "CW Search Sensitivity Analysis"
    goal: "Quantify impact of improved rare-class classification on CW search data quality"
    methods: []
    depends_on: ["ViT-B/16 with Class Imbalance Strategy"]
    needs_research: true
    risk: HIGH
    pitfalls: ["cw-benefit-not-quantified"]
  - name: "Paper Writing and O4 Validation"
    goal: "Generalization test on O4 data and research paper deliverable"
    methods: ["ViT-B/16 fine-tuning (ImageNet-21k)"]
    depends_on: ["CW Search Sensitivity Analysis"]
    needs_research: false
    risk: MEDIUM
    pitfalls: ["distribution-shift", "cherry-picked-attention-maps"]

critical_benchmarks:
  - quantity: "Gravity Spy CNN overall accuracy (common classes)"
    value: "~97%"
    source: "Zevin et al. 2017, CQG 34(6) 064003"
    confidence: HIGH
  - quantity: "Deep transfer learning overall accuracy"
    value: ">98.8%"
    source: "George et al. 2018, PRD 97 101501"
    confidence: HIGH
  - quantity: "ViT-B/32 overall accuracy (24 classes)"
    value: "92.26%"
    source: "Srivastava & Niedzielski 2025, Acta Astronomica 74(3)"
    confidence: HIGH
  - quantity: "cDVGAN AUC improvement"
    value: "4.2%"
    source: "cDVGAN 2024, PRD 110 022004"
    confidence: HIGH
  - quantity: "Gravity Spy O3 total ML-classified glitches"
    value: "~614K (234K H1 + 380K L1)"
    source: "Glanzer et al. 2022, Zenodo 5649212"
    confidence: HIGH

open_questions:
  - question: "Can ViT-B/16 with modern training recipe outperform CNN on rare glitch class F1?"
    priority: HIGH
    blocks_phase: "ViT-B/16 with Class Imbalance Strategy"
  - question: "What is the optimal multi-view fusion strategy for ViT on multi-duration spectrograms?"
    priority: HIGH
    blocks_phase: "ViT-B/16 with Class Imbalance Strategy"
  - question: "How does rare-class F1 improvement quantitatively map to CW search sensitivity gain?"
    priority: HIGH
    blocks_phase: "CW Search Sensitivity Analysis"
  - question: "Does self-supervised pretraining on unlabeled Gravity Spy spectrograms improve rare-class F1?"
    priority: MEDIUM
    blocks_phase: "none"
  - question: "How severe is O3-to-O4 distribution shift for the trained classifier?"
    priority: MEDIUM
    blocks_phase: "Paper Writing and O4 Validation"

contradictions_unresolved: []
```
