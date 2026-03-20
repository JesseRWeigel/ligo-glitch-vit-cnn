# Phase 2: CNN Baseline Reproduction - Research

**Researched:** 2026-03-16
**Domain:** Transfer learning / image classification / gravitational-wave glitch taxonomy
**Confidence:** HIGH

## Summary

Phase 2 establishes the fair CNN baseline that all subsequent ViT comparisons are measured against. The goal is to train a ResNet-50 with a modern training recipe (AdamW, cosine LR, mixed precision, focal loss, class-balanced sampling) on the Gravity Spy spectrogram dataset prepared in Phase 1, reproduce published Gravity Spy CNN accuracy within 2%, and -- critically -- produce the per-class F1 breakdown that quantifies the rare-class performance gap motivating the ViT work.

The key insight driving this phase is that a fair baseline requires the CNN to receive the same modern training recipe the ViT will use in Phase 3. The "ResNet strikes back" result (Wightman et al. 2021, arXiv:2110.00476) demonstrated that a vanilla ResNet-50 jumps from 75.3% to 80.4% top-1 on ImageNet simply by upgrading the training recipe. If the CNN baseline uses an outdated recipe (SGD with step decay, no augmentation), any ViT advantage in Phase 3 may reflect training recipe differences, not architectural superiority. This is the single most important methodological pitfall for the project's central claim.

The original Gravity Spy CNN is a custom ~1.9M parameter network with multi-view fusion across 4 duration windows (Bahaadini et al. 2018). We do NOT reproduce the original architecture -- instead we use ResNet-50 (~25M params) as a stronger, well-understood CNN baseline. This is deliberate: the goal is not to replicate Gravity Spy exactly, but to establish the best CNN performance achievable with modern methods, so that any ViT improvement is attributable to architecture, not training.

**Primary recommendation:** Train `timm/resnetv2_50x1_bit.goog_in21k_ft_in1k` (ResNet-50v2, ImageNet-21k pretrained) with AdamW (lr=1e-4, wd=0.05), cosine schedule with 5-epoch warmup, focal loss (gamma=2.0, class-weighted alpha), class-balanced sampling via `WeightedRandomSampler`, mixed precision (bf16), for 50 epochs with early stopping on validation macro-F1. Process single best-duration view (1.0s) first as a sanity check, then train multi-view (4-duration) model for the production baseline.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| ref-gravity-spy: Zevin et al. 2017 | benchmark | CNN overall accuracy ~97% is the reproduction target; per-class breakdown establishes the gap | Reproduce overall accuracy within 2%; compare per-class patterns | plan, execution, verification, Phase 3 comparison |
| fp-overall-accuracy | forbidden proxy | Overall accuracy must NOT be the primary output; per-class F1 on rare classes is the decisive deliverable | Report accuracy only as sanity check; optimize and select on macro-F1 | plan, execution, all downstream |
| Phase 1 dataset artifacts | prior artifact | Temporal train/val/test splits, 224x224 spectrograms, class distributions, label quality audit results | Load directly; do NOT re-split or re-process | execution |

**Missing or weak anchors:** Gravity Spy papers do not publish per-class F1 scores -- only overall accuracy and confusion matrices. The rare-class F1 baseline value this phase establishes is genuinely new and cannot be directly compared to literature. The "within 2%" reproduction target applies only to overall accuracy as a pipeline validation check.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| CNN architecture | ResNet-50v2 (BiT variant, ImageNet-21k pretrained) | Original Gravity Spy CNN; ResNet-18; EfficientNetV2 | Modern standard; capacity-matched to ViT-B/16 comparison |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.05) | LAMB; SGD+momentum | AdamW is standard for transfer learning fine-tuning |
| LR schedule | Cosine annealing with linear warmup (5 epochs) | Step decay; exponential | Cosine is robust, no extra hyperparameters beyond warmup |
| Loss function | Focal loss (gamma=2.0, class-weighted alpha) | Cross-entropy; weighted CE | Focal loss is strictly better for imbalanced classification |
| Sampling | Class-balanced via WeightedRandomSampler (sqrt weights) | Uniform; pure inverse-frequency | Sqrt rebalancing avoids minority-class overfitting |
| Mixed precision | bf16 (preferred over fp16 on RTX 5090) | fp16; fp32 | bf16 avoids attention underflow; RTX 5090 natively supports bf16 |
| Input | 224x224 px, 3-channel (RGB from colormap), [0,1] normalized | Grayscale; 384x384 | Matches ImageNet pretraining and Phase 1 convention |
| Primary metric | Macro-averaged F1 | Overall accuracy (FORBIDDEN as primary) | Project contract |
| Epochs | 50 max with early stopping (patience=10 on val macro-F1) | Fixed 100/300 epochs | Sufficient for transfer learning; prevents overfitting |
| Batch size | 64 (fits in 32 GB VRAM with bf16 and ResNet-50) | 32; 128 | Good gradient estimate without excessive memory |
| Label smoothing | epsilon=0.1 | 0.0; 0.05 | Mitigates label noise; Wu et al. 2024 validated for Gravity Spy |

**CRITICAL: The CNN and ViT (Phase 3) MUST share the same optimizer, LR schedule, augmentation pipeline, loss function, sampling strategy, number of epochs, and early stopping criterion. Only the model architecture differs.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t) | Focal loss | Lin et al. (2017), arXiv:1708.02002 | Training loss function |
| alpha_c = 1 / sqrt(N_c) / sum_k(1/sqrt(N_k)) | Sqrt-inverse class weights | Cui et al. (2019), CVPR | Per-class alpha in focal loss and sampler weights |
| F1_c = 2 * P_c * R_c / (P_c + R_c) | Per-class F1 score | Standard | Primary per-class evaluation metric |
| F1_macro = (1/N_cls) * sum_c F1_c | Macro-averaged F1 | Standard | Model selection criterion |
| CI = [F1_boot(alpha/2), F1_boot(1-alpha/2)] | Bootstrap confidence interval | Efron (1979) | Statistical significance of per-class F1 |
| p-value via McNemar's test | Pairwise model comparison | McNemar (1947) | CNN vs ViT statistical comparison in Phase 3 |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Transfer learning (fine-tuning) | Adapts ImageNet-21k features to spectrogram domain | ResNet-50 backbone initialization | Kolesnikov et al. (2020), BiT paper |
| Differential learning rates | Higher LR for classification head, lower for backbone | Prevents catastrophic forgetting of pretrained features | Standard transfer learning practice |
| Cosine annealing with warmup | Smooth LR decay; warmup prevents early divergence | LR schedule throughout training | Loshchilov & Hutter (2017) |
| Gradient clipping (max_norm=1.0) | Prevents gradient explosion from extreme class weights | Training loop | Standard with focal loss |
| Early stopping on macro-F1 | Prevents overfitting; selects model at best rare-class performance | Training loop; patience=10 epochs | Standard |
| Bootstrap resampling (N=1000) | Confidence intervals for per-class metrics | Post-training evaluation | scipy.stats.bootstrap or confidenceinterval package |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| Single-view (1.0s) for sanity check | Information loss from ignoring 3 other durations | Common morphologically distinct classes | ~3-5pp accuracy loss vs multi-view; adequate for pipeline validation | Multi-view fusion (production baseline) |
| Focal loss with gamma=2.0 (default) | Deviation from optimal gamma | Moderate imbalance (10x-100x class ratio) | Sub-optimal by ~0.5-1pp F1 vs tuned gamma | Grid search gamma in {1.0, 1.5, 2.0, 3.0, 5.0} |
| Sqrt class weights | Deviation from optimal rebalancing | Imbalance ratio < 500x | Empirically robust; avoids extremes of uniform (ignores imbalance) and inverse-frequency (overfits rare) | Effective number of samples (Cui et al. 2019) |

## Standard Approaches

### Approach 1: ResNet-50v2 (BiT) with Modern Training Recipe (RECOMMENDED)

**What:** Fine-tune a ResNet-50v2 pretrained on ImageNet-21k using the same training recipe that Phase 3 will apply to the ViT. This establishes the strongest possible CNN baseline, ensuring any ViT improvement is architectural.

**Why standard:** "ResNet strikes back" (Wightman et al. 2021) demonstrated that modern training recipes close most of the gap between CNNs and ViTs. BiT (Big Transfer, Kolesnikov et al. 2020) showed that ImageNet-21k pretraining gives ResNets a substantial boost on downstream tasks, especially small datasets. Using the same pretraining source (ImageNet-21k) for both CNN and ViT eliminates a confound.

**Track record:** ResNet-50 with modern training reaches 80.4% top-1 on ImageNet (from 75.3% with legacy recipe). George et al. (2018) achieved >98.8% on Gravity Spy using deep transfer learning with a similar approach (though with older training recipe).

**Key steps:**

1. Load `resnetv2_50x1_bit.goog_in21k_ft_in1k` from timm (or `resnetv2_50x1_bit.goog_in21k` for 21k-only)
2. Replace classification head with `nn.Linear(2048, N_cls)` for 23-24 classes
3. Set up differential learning rates: backbone 1e-5, head 1e-3
4. Configure focal loss with gamma=2.0 and sqrt-inverse class weights
5. Configure `WeightedRandomSampler` with sqrt-inverse weights for the DataLoader
6. Train with AdamW (weight_decay=0.05), cosine schedule (warmup=5 epochs), bf16 mixed precision
7. Apply augmentations: horizontal flip, brightness/contrast jitter, SpecAugment masking
8. Early stop on validation macro-F1 (patience=10)
9. Evaluate on temporal test set: per-class F1, recall, precision, confusion matrix, overall accuracy
10. Compute bootstrap 95% CIs for all per-class metrics (N=1000 resamples)

**Known difficulties at each step:**

- Step 1: The BiT model expects 3-channel input; spectrograms must be converted from grayscale to 3-channel (either replicate channel or apply colormap in Phase 1)
- Step 4: Extreme class weights (>50x for rarest classes) can cause gradient instability; cap at 10x or use gradient clipping
- Step 5: WeightedRandomSampler replacement=True causes some rare-class samples to repeat many times per epoch; monitor for memorization
- Step 8: Early stopping on macro-F1 can oscillate for rare classes with few test samples; use smoothed macro-F1 (exponential moving average over 3 epochs)
- Step 9: Per-class F1 for classes with <50 test samples has high variance; report bootstrap CIs

### Approach 2: Multi-View Fusion CNN (PRODUCTION BASELINE)

**What:** After validating the pipeline with single-view, extend to process all 4 duration windows (0.5s, 1.0s, 2.0s, 4.0s) through a shared ResNet-50 backbone, then fuse features via concatenation of global-average-pooled features + linear head.

**When to use:** After single-view baseline validates the pipeline (step 1 sanity check passes).

**Why:** Multi-view is the Gravity Spy standard. Comparing single-view ViT vs single-view CNN AND multi-view ViT vs multi-view CNN provides a complete ablation matrix.

**Key difference from single-view:** 4x forward passes per sample through shared backbone; concatenate 4 x 2048-dim features -> 8192-dim -> linear head. Memory cost ~4x inference but manageable at batch=16 per view.

### Anti-Patterns to Avoid

- **Reproducing the original Gravity Spy CNN architecture:** The original ~1.9M param custom CNN uses an outdated architecture and training recipe. Reproducing it exactly would give a weak baseline that inflates ViT advantage. Use ResNet-50 with modern recipe instead.
  - _Example:_ If CNN baseline gets 90% accuracy with old architecture, and ViT gets 95% with modern recipe, the "5pp improvement" is meaningless -- it measures recipe, not architecture.

- **Using SGD with step-decay for CNN but AdamW with cosine for ViT:** This is the most common unfair comparison in the ViT literature. Both models must use identical optimizer and schedule.
  - _Example:_ Bai et al. (2021) showed that CNNs match ViT robustness when given the same training recipe.

- **Optimizing hyperparameters on the test set:** Hyperparameter search (gamma, LR, etc.) must use the validation set only. The temporal test set is touched exactly once for final evaluation.
  - _Example:_ "Tuning" gamma on the test set and reporting test-set macro-F1 gives overly optimistic results.

- **Reporting only overall accuracy:** This is the project's forbidden proxy (fp-overall-accuracy). Overall accuracy is a pipeline sanity check, not a result.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Gravity Spy CNN ~97% overall accuracy | 97.1% on 20 classes (O1/O2 data) | Zevin et al. (2017), CQG 34 064003 | Reproduction target: our ResNet-50 on same-era data should be within 2pp |
| George et al. >98.8% with deep transfer | 98.84% on similarity search task; CNN classification accuracy similar range | George et al. (2018), PRD 97 101501 | Upper bound for CNN accuracy with transfer learning |
| Focal loss formulation | FL(p_t) = -alpha_t (1-p_t)^gamma log(p_t) | Lin et al. (2017), arXiv:1708.02002 | Use directly; do not re-derive |
| ResNet-50 modern recipe reaches 80.4% on ImageNet | A1 procedure: LAMB + cosine + augmentation, 600 epochs | Wightman et al. (2021), arXiv:2110.00476 | Validates that modern recipe substantially improves ResNet; our recipe is adapted from A2/A3 |
| Sqrt-inverse class weighting | alpha_c = 1/sqrt(N_c) normalized | Cui et al. (2019), CVPR | Use directly for sampler weights and focal loss alpha |
| Bootstrap BCa confidence intervals | Bias-corrected accelerated percentile method | Efron (1987) | Use via scipy.stats.bootstrap for per-class F1 CIs |

**Key insight:** The per-class F1 breakdown for Gravity Spy is NOT published anywhere. This phase generates a genuinely new result. Do not waste effort trying to find it -- it does not exist.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| Phase 1 class distribution | Per-class sample counts for train/val/test after temporal split | Phase 1 artifacts | Directly determines alpha weights and sampling weights |
| Phase 1 label audit | Which rare classes have noisy/ambiguous labels | Phase 1 artifacts | May need to exclude or downweight certain classes |
| Gravity Spy confusion patterns | Blip vs Blip_Low_Frequency, Scattered_Light variants | Zevin et al. (2024), arXiv:2308.15530 | Known confusions to check in our confusion matrix |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Gravity Spy CNN | Zevin et al. | 2017 | Defines the baseline we must match or exceed on overall accuracy | Overall accuracy ~97%; confusion patterns; class taxonomy |
| Deep Transfer Learning for LIGO | George, Shen, Huerta | 2018 | CNN ceiling with transfer learning | >98.8% accuracy achievable; validates transfer learning for spectrograms |
| ResNet strikes back | Wightman et al. | 2021 | Modern training recipe for ResNet-50 | Training recipe components: LAMB/AdamW, cosine, augmentation, longer training |
| BiT: Big Transfer | Kolesnikov et al. | 2020 | ImageNet-21k pretrained ResNets for downstream transfer | BiT-M ResNet-50x1 is the recommended checkpoint; transfer protocol |
| Focal Loss | Lin et al. | 2017 | Loss function for class imbalance | gamma=2.0 default; alpha weighting; implementation details |
| Class-Balanced Loss | Cui et al. | 2019 | Effective number of samples for weighting | Provides principled alternative to sqrt-inverse weighting |
| Gravity Spy Lessons Learned | Zevin et al. | 2024 | Known CNN failure modes; O3 taxonomy update | CNN mislabels novel morphologies; confidence overfitting documented |
| O4 Multi-view Classifier | Wu/Raza et al. | 2024 | Multi-view attention fusion; label smoothing benefit | Label smoothing epsilon=0.1 validated for Gravity Spy |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| PyTorch | >= 2.2 | Training framework | Native AMP, DataLoader, distributed support; CUDA 12.x for RTX 5090 |
| timm | >= 1.0 | ResNet-50v2 BiT pretrained model | `timm.create_model('resnetv2_50x1_bit.goog_in21k', pretrained=True, num_classes=N)` |
| scikit-learn | >= 1.4 | Metrics: `classification_report`, `confusion_matrix`, `f1_score(average='macro')` | Standard; comprehensive per-class metrics |
| scipy | >= 1.12 | `scipy.stats.bootstrap` for confidence intervals | BCa method for per-class F1 CIs |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| torchmetrics | GPU-side macro-F1 computation during training | Per-epoch validation without CPU transfer overhead |
| albumentations | Augmentation pipeline (horizontal flip, color jitter, SpecAugment) | Training data augmentation |
| wandb or tensorboard | Experiment tracking: per-class metrics across epochs | Monitor rare-class F1 evolution during training |
| matplotlib + seaborn | Confusion matrices, per-class F1 bar charts, training curves | Results visualization |
| confidenceinterval | Bootstrap CIs for macro/per-class F1 with scikit-learn API | Post-training evaluation; alternative to manual scipy bootstrap |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
| --- | --- | --- |
| timm ResNet-50v2 BiT | torchvision ResNet-50 | torchvision lacks ImageNet-21k pretrained weights; timm has superior checkpoint ecosystem |
| AdamW | LAMB (from ResNet strikes back A1) | LAMB better for large batch (2048+); AdamW simpler and standard for fine-tuning at batch 64 |
| Manual focal loss | kornia.losses.FocalLoss | kornia adds dependency; focal loss is ~15 lines in PyTorch -- implement directly |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Single-view ResNet-50 training (50 epochs) | ~1-2 hours on RTX 5090 | Forward/backward pass (negligible) | bf16 mixed precision |
| Multi-view ResNet-50 training (50 epochs) | ~4-6 hours on RTX 5090 | 4x forward passes per sample | Shared backbone; batch=16 per view |
| Bootstrap CI computation (1000 resamples) | ~5 minutes CPU | Resampling loop | Vectorize with numpy |
| Hyperparameter grid search (5 gamma values x 3 LRs) | ~15-30 hours total | Multiple training runs | Use single-view for search; apply best to multi-view |
| VRAM per batch (single-view, batch=64, bf16) | ~8 GB | Backward pass activations | Gradient checkpointing if needed (unlikely) |
| VRAM per batch (multi-view, batch=16x4, bf16) | ~12 GB | 4 forward passes cached | Accumulate gradients if needed |

**Installation / Setup:**
```bash
# Core (assumes CUDA 12.x environment from Phase 1)
pip install timm>=1.0 torchmetrics albumentations
# Evaluation
pip install confidenceinterval
# Experiment tracking (optional)
pip install wandb
```

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| Overall accuracy vs Gravity Spy | Pipeline correctness; data is processed correctly | Compare our ResNet-50 overall accuracy to published ~97% | Within 2pp: 95-99% |
| Per-class F1 ordering | Common classes should outperform rare classes | Sort classes by F1; correlate with class size | Positive correlation (r > 0.5) between log(N_c) and F1_c |
| Known confusion pairs | Confusion matrix shows expected patterns | Check Blip vs Blip_Low_Frequency; Scattered_Light variants | Non-zero off-diagonal entries at known confusion locations |
| Train vs val loss gap | Overfitting detection | Plot training and validation loss curves | Gap < 0.3 nats; no divergence after epoch 15 |
| Rare-class recall > random | Model is learning rare classes, not ignoring them | Check recall for each rare class | Recall > 1/N_cls for all classes (better than random) |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| CNN overall accuracy ceiling | Common classes with >1000 training samples | ~97-99% | Zevin et al. 2017; George et al. 2018 |
| Rare-class F1 floor | Classes with <100 training samples | Expected F1 < 0.7 (this is the gap we quantify) | Novel measurement; no literature value |
| Single-view vs multi-view accuracy gap | Removing 3 of 4 duration windows | Expected ~3-5pp accuracy drop | Wu et al. 2024 |
| Transfer learning vs from-scratch | ImageNet-21k pretrained vs random init | 15-25pp accuracy gap on small datasets | Kolesnikov et al. 2020 |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
| --- | --- | --- | --- |
| Focal loss implementation | Compare FL(p=0.9, gamma=2, alpha=0.25) to analytic value | < 1e-6 relative error | FL = -0.25 * (0.1)^2 * log(0.9) = 0.000264 |
| Macro-F1 computation | Compare torchmetrics vs scikit-learn on same predictions | Exact match (< 1e-10) | Must agree |
| Class weight normalization | Verify sum of sampler weights equals dataset size | Exact (up to float precision) | sum(weights) = len(dataset) |
| Bootstrap CI coverage | On synthetic data with known F1, check 95% CI contains true value | ~95% empirical coverage over 100 trials | 95% +/- 3% |

### Red Flags During Computation

- If overall accuracy is below 90%: data pipeline is likely broken (wrong normalization, corrupted images, incorrect labels). Stop and debug before proceeding.
- If all rare-class F1 scores are 0.0: focal loss alpha weights or class-balanced sampling is misconfigured. Check class weight computation.
- If training loss is NaN after first few epochs: class weights are too extreme or learning rate is too high. Reduce max class weight or LR.
- If validation macro-F1 peaks in epoch 1-3 and then monotonically decreases: the model is overfitting immediately. Reduce learning rate, increase weight decay, or freeze more backbone layers.
- If common-class F1 is much lower than expected (< 0.90): the focal loss gamma is too high, causing the model to underfit common classes. Reduce gamma.

## Common Pitfalls

### Pitfall 1: Unfair CNN Baseline (Training Recipe Mismatch)

**What goes wrong:** CNN is trained with legacy recipe (SGD, step decay, no augmentation), ViT with modern recipe. ViT appears to win, but the improvement is from training, not architecture.
**Why it happens:** Legacy Gravity Spy work used 2017-era training practices. It is tempting to "reproduce" those settings for the CNN baseline.
**How to avoid:** Use IDENTICAL training recipe for CNN and ViT. The only variable is the model architecture. Document the recipe explicitly. This is the #1 priority.
**Warning signs:** CNN baseline accuracy is below published Gravity Spy values; ViT improves by >5pp on all metrics.
**Recovery:** Retrain CNN with modern recipe. Prior ViT results may still be valid.

### Pitfall 2: Focal Loss Hyperparameter Sensitivity

**What goes wrong:** gamma too high (>3) causes underfitting on common classes; gamma too low (=0, reduces to CE) fails to help rare classes. Interaction between gamma and alpha is non-trivial.
**Why it happens:** Focal loss has two interacting hyperparameters. The optimal gamma depends on the class imbalance ratio AND the dataset difficulty.
**How to avoid:** Grid search gamma in {1.0, 2.0, 3.0} with fixed alpha=sqrt-inverse. Evaluate on validation macro-F1, not accuracy. Start with gamma=2.0 (default from Lin et al.).
**Warning signs:** Common-class F1 drops >5pp relative to CE baseline when gamma is introduced.
**Recovery:** Reduce gamma; if still problematic, fall back to weighted CE with label smoothing.

### Pitfall 3: Gradient Instability from Extreme Class Weights

**What goes wrong:** For classes with <50 samples, inverse-frequency weights can exceed 100x. Combined with focal loss, gradients from rare-class samples can be 10,000x larger than common-class gradients, causing NaN losses.
**Why it happens:** Weight = 1/N_c for rare class with N_c=25 and common class with N_c=7500 gives a 300x ratio. Focal loss multiplier (1-p_t)^gamma can add another 10-100x factor.
**How to avoid:** Use sqrt-inverse weights (reduces 300x to ~17x). Apply gradient clipping (max_norm=1.0). Cap class weights at 10x the minimum weight.
**Warning signs:** NaN loss; loss spikes; rare-class predictions oscillate wildly between epochs.
**Recovery:** Reduce class weight cap; increase gradient clipping; switch to weighted CE temporarily to diagnose.

### Pitfall 4: Overfitting on Repeated Rare-Class Samples

**What goes wrong:** Class-balanced sampling with replacement causes rare-class samples to appear 50-100x per epoch. The model memorizes these specific images rather than learning the class morphology.
**Why it happens:** With N_rare=50 and effective oversampling to balance with N_common=5000, each rare sample appears ~100 times per epoch.
**How to avoid:** Apply strong augmentation (random augmentation is ESSENTIAL for oversampled classes). Monitor per-class train-vs-val F1 gap -- large gap for rare classes indicates memorization. Consider limiting max oversampling factor (e.g., no class appears >20x per epoch).
**Warning signs:** Rare-class train F1 = 1.0 but val F1 < 0.5; attention maps for rare classes focus on image-specific artifacts rather than morphological features.
**Recovery:** Increase augmentation aggressiveness for rare classes; reduce sampling weight; add dropout.

### Pitfall 5: Misinterpreting Per-Class F1 with Small Test Sets

**What goes wrong:** A rare class with 10 test samples can swing from F1=0.4 to F1=0.8 based on getting 2 more samples correct. Point estimates without CIs are misleading.
**Why it happens:** F1 is a ratio metric with high variance when the denominator (TP + FP + FN) is small.
**How to avoid:** ALWAYS report bootstrap 95% CIs for per-class F1. For classes with <30 test samples, flag the CI width explicitly. Use the CI overlap (not point estimates) to determine whether CNN vs ViT differences are significant.
**Warning signs:** Rare-class F1 varies by >0.2 across random seeds.
**Recovery:** Increase test set size for rare classes (but cannot violate temporal split!). Report exact counts alongside F1.

## Level of Rigor

**Required for this phase:** Empirical machine learning standard with statistical significance testing.

**Justification:** This is an experimental ML phase, not a theoretical derivation. The deliverable is a trained model and its evaluation metrics. Rigor comes from proper experimental design (fair comparison, temporal split, bootstrap CIs) rather than formal proof.

**What this means concretely:**

- All reported metrics must include 95% bootstrap confidence intervals
- CNN vs ViT comparisons (Phase 3) must use McNemar's test for statistical significance
- Hyperparameter search must be on validation set only; test set touched exactly once
- The exact training recipe (optimizer, LR, schedule, augmentation, loss, sampling) must be documented with full reproducibility (config file or script)
- Per-class F1 must be reported for ALL classes, not just aggregates
- Random seed must be reported; ideally, results averaged over 3-5 seeds

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Custom small CNN (~1.9M params) with CE loss | ResNet-50 with focal loss and modern recipe | 2021+ (ResNet strikes back; BiT) | 5-10pp accuracy improvement from recipe alone |
| SGD + step decay LR | AdamW + cosine annealing | 2019+ | More stable training; better convergence for transfer learning |
| Random stratified split | Temporal split with gap | Increasingly standard in GW-ML since 2022 | Prevents inflated metrics; required for publication |
| Overall accuracy as metric | Macro-F1 and per-class recall | Recognized as important since Gravity Spy O4 work (2024) | Exposes rare-class failure modes hidden by accuracy |
| No data augmentation | SpecAugment + geometric + color jitter | Standard since 2019+ | 2-5pp improvement; essential for oversampled rare classes |

**Superseded approaches to avoid:**

- **Original Gravity Spy CNN with CE loss:** Outdated architecture and loss function. Do not reproduce this as the baseline -- it is too weak and makes the ViT comparison unfair in the opposite direction (too easy to beat).
- **SMOTE for class rebalancing:** Creates unphysical interpolated spectrograms. Use focal loss + balanced sampling instead.

## Open Questions

1. **What is the optimal focal loss gamma for this specific dataset?**
   - What we know: gamma=2.0 is the standard default; literature suggests 2-3 for moderate imbalance
   - What's unclear: Gravity Spy's specific imbalance profile may favor higher/lower gamma
   - Impact on this phase: Affects rare-class F1 by up to 2-3pp
   - Recommendation: Grid search {1.0, 2.0, 3.0} on validation set; use 2.0 as starting point

2. **Should we use ImageNet-21k or ImageNet-1k pretrained weights for ResNet-50?**
   - What we know: 21k gives richer features; BiT paper shows benefit on small downstream datasets
   - What's unclear: Whether the benefit persists for Q-transform spectrograms (visual domain very different from natural images)
   - Impact on this phase: Could affect baseline accuracy by 1-3pp
   - Recommendation: Use ImageNet-21k (matches Phase 3 ViT pretraining source); compare against 1k as ablation if time permits

3. **How many rare classes are there, and what is the exact class distribution?**
   - What we know: Gravity Spy has extreme imbalance; rare classes include Chirp, Wandering_Line, Paired_Doves with <100 samples
   - What's unclear: Exact counts depend on Phase 1 temporal split and confidence filtering
   - Impact on this phase: Determines class weights, sampling strategy, and which classes to focus rare-class analysis on
   - Recommendation: Define "rare" as classes with <200 training samples after filtering; this threshold to be finalized when Phase 1 class distribution is available

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| ResNet-50v2 BiT does not reach 95% accuracy | Domain gap between ImageNet and spectrograms is too large | EfficientNetV2-S (21M params, strong on small datasets) | ~2 hours to retrain; same recipe applies |
| Focal loss causes training instability | Class weights + focal interaction causes NaN | Weighted cross-entropy + label smoothing (epsilon=0.1) | Minimal; just change loss function |
| Single-view 1.0s baseline is too far below 97% | 1.0s window misses morphologies only visible at other durations | Jump directly to multi-view (4 durations) | ~4x training time; may be necessary anyway |
| Overall accuracy >99% but rare-class F1 <0.3 | Model ignoring rare classes despite focal loss | Increase gamma to 5.0; switch to 2-stage training (pretrain with CE, fine-tune with focal) | 2x training time for 2-stage approach |
| Entire modern recipe fails to reproduce ~97% | Data pipeline issue from Phase 1 | Debug data: check spectrograms visually, verify labels match images, check normalization | Full Phase 1 re-validation; HIGH cost |

**Decision criteria:** Abandon the single-view sanity check after 2 failed attempts. If multi-view also fails to reach 93% overall accuracy after hyperparameter search, the Phase 1 data pipeline must be audited.

## Sources

### Primary (HIGH confidence)

- Zevin et al. (2017), "Gravity Spy: Integrating Advanced LIGO Detector Characterization, Machine Learning, and Citizen Science," CQG 34 064003 -- baseline CNN accuracy benchmark (~97%)
- Lin et al. (2017), "Focal Loss for Dense Object Detection," arXiv:1708.02002 -- focal loss formulation, gamma=2.0 default
- Wightman et al. (2021), "ResNet strikes back: An improved training procedure in timm," arXiv:2110.00476 -- modern CNN training recipe; A1/A2/A3 procedures
- Kolesnikov et al. (2020), "Big Transfer (BiT): General Visual Representation Learning," arXiv:1912.11370 -- ImageNet-21k pretrained ResNet-50; transfer learning protocol
- Cui et al. (2019), "Class-Balanced Loss Based on Effective Number of Samples," CVPR -- class rebalancing strategies; effective number of samples

### Secondary (MEDIUM confidence)

- George, Shen & Huerta (2018), "Deep Transfer Learning: A new deep learning glitch classification method for advanced LIGO," PRD 97 101501 -- CNN ceiling with transfer learning (>98.8%)
- Zevin et al. (2024), "Gravity Spy: Lessons Learned and a Path Forward," EPJ Plus 139 100, arXiv:2308.15530 -- CNN limitations; O3 taxonomy; label quality issues
- Wu/Raza et al. (2024), arXiv:2401.12913 -- multi-view attention fusion; label smoothing epsilon=0.1 for Gravity Spy
- Bahaadini et al. (2018), "Machine Learning for Gravity Spy: Glitch Classification and Dataset," Information Sciences 444 -- original Gravity Spy CNN architecture details
- Bai et al. (2021), "Are Transformers More Robust Than CNNs?" -- training recipe dominates architecture for robustness

### Tertiary (LOW confidence)

- [khornlund/pytorch-balanced-sampler](https://github.com/khornlund/pytorch-balanced-sampler) -- alpha-parameterized balanced BatchSampler implementation
- [jacobgil/confidenceinterval](https://github.com/jacobgil/confidenceinterval) -- bootstrap CI package for scikit-learn metrics
- [ufoym/imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler) -- alternative imbalanced dataset sampler

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH -- focal loss, macro-F1, bootstrap CIs are well-established
- Standard approaches: HIGH -- ResNet-50 fine-tuning with modern recipe is a well-trodden path; "ResNet strikes back" provides the exact recipe
- Computational tools: HIGH -- timm, PyTorch, scikit-learn are mature, well-documented, widely used
- Validation strategies: HIGH -- overall accuracy reproduction target is well-defined; bootstrap CIs are standard

**Research date:** 2026-03-16
**Valid until:** Indefinitely for methodology; tool versions may need updating after 2027

## Caveats and Self-Critique

1. **Assumption: ImageNet-21k pretraining transfers well to Q-transform spectrograms.** This is validated by Srivastava (2025) and George et al. (2018) for CNNs, but the domain gap is real. If transfer fails, the BiT model may need more aggressive fine-tuning (unfreeze all layers from epoch 1) or we may need a spectrogram-specific pretraining step.

2. **Assumption: ResNet-50 is the right capacity for a fair comparison.** ResNet-50 has 25M params vs ViT-B/16's 86M. This is a ~3.4x capacity gap. We justify this by noting that ResNet-50 has stronger inductive biases (locality, translation equivariance) which partially compensate for fewer parameters. However, a reviewer might argue for ResNet-152 (~60M) or EfficientNetV2-L (~120M) as a more capacity-matched baseline. If ViT improvement is marginal (<2pp F1), this becomes a critical concern.

3. **Assumption: Sqrt-inverse weighting is near-optimal.** We chose this over the more principled "effective number of samples" (Cui et al. 2019) for simplicity. The difference is typically <1pp, but on extremely imbalanced classes it could matter. Worth ablating if time permits.

4. **Simpler alternative considered but not dismissed quickly enough:** A linear probe on frozen ResNet-50 features (no fine-tuning) could serve as an even simpler baseline. We skip this because it would underperform full fine-tuning and is less relevant to the fair-comparison goal. However, it could serve as a useful lower-bound data point that takes minutes to train.

5. **What would a GW-ML specialist disagree with?** They might argue that reproducing the *exact* Gravity Spy CNN architecture matters for direct literature comparison, even if it produces a weaker baseline. Our counter: the project goal is to demonstrate ViT improvement over the *best possible* CNN, not over a specific historical model. But we should report both if feasible (original architecture reproduction as a secondary result).
