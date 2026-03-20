# Phase 3: ViT Training & Rare-Class Optimization - Research

**Researched:** 2026-03-17
**Domain:** Deep learning for class-imbalanced image classification (LIGO glitch spectrograms)
**Confidence:** HIGH

## Summary

Phase 3 is the decisive phase for the project's primary claim (claim-rare-improvement): a ViT-B/16 classifier must demonstrate statistically significant improvement in rare-class macro-F1 over the CNN baseline established in Phase 2 (rare-class macro-F1 = 0.3028 [0.2085, 0.3751]). The CNN baseline already achieves 91.8% overall accuracy but only 30.3% rare-class macro-F1, with a 50.5 pp gap between common and rare classes. The four rare classes are Chirp (11 train, F1=0.47), Wandering_Line (30 train, F1=0.00), Helix (33 train, F1=0.05), and Light_Modulation (142 train, F1=0.69).

The recommended approach is straightforward: fine-tune ViT-B/16 from ImageNet-21k pretrained weights using the same training infrastructure from Phase 2 (focal loss, sqrt-inverse class weights, class-balanced sampling, AdamW + cosine schedule), with three ViT-specific enhancements: (1) layer-wise learning rate decay (decay factor 0.75) to preserve pretrained low-level features, (2) lower base learning rate (1e-4 vs CNN's 1e-3), and (3) optional addition of SpecAugment-style frequency/time masking. The existing Phase 2 code (dataset, transforms, focal loss, training loop, evaluation, bootstrap CI) is almost entirely reusable -- only the model builder and LR scheduling need modification.

The statistical comparison requires a paired bootstrap test on the difference in rare-class macro-F1 between ViT and CNN predictions on the same test set, with >= 10,000 resamples and p < 0.05. The existing bootstrap_ci.py can be extended for this paired comparison. The bar to clear is modest in absolute terms (CNN rare-class F1 is only 0.30) but must be statistically significant given the very small rare-class test sets (Wandering_Line: 6 test samples, Chirp: 7, Helix: 14).

**Primary recommendation:** Fine-tune `timm` ViT-B/16 (ImageNet-21k+1k, `vit_base_patch16_224.augreg_in21k_ft_in1k`) with layer-wise LR decay 0.75, base LR 1e-4, and the identical loss/sampling/augmentation recipe from Phase 2. Add SpecAugment (time + frequency masking) as the only new augmentation. Use paired bootstrap on rare-class macro-F1 difference for statistical testing.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| CNN baseline metrics (results/02-cnn-baseline/metrics.json) | prior artifact | Defines the performance floor the ViT must beat; rare-class macro-F1 = 0.3028 | compare against | plan, execution, verification |
| CNN predictions on test set | prior artifact | Needed for paired statistical test (same test samples) | load predictions | execution, verification |
| ref-gravity-spy (Zevin et al. 2017) | benchmark | Gravity Spy taxonomy defines the 23 class labels | preserve class labels | plan, execution, deliverables |
| Srivastava 2025 (arXiv:2510.06273) | benchmark | Only published ViT on Gravity Spy (92.26% accuracy, ViT-B/32) | exceed with ViT-B/16 | discussion/paper |
| fp-overall-accuracy | forbidden proxy | Overall accuracy must NOT be used as success metric | enforce in evaluation | plan, execution, verification |

**Missing or weak anchors:** CNN test-set predictions (logits or predicted labels per sample) must be saved or recoverable from Phase 2 checkpoint to enable paired bootstrap. If only aggregate metrics were saved, the CNN checkpoint must be re-evaluated to produce per-sample predictions.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Input normalization | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Dataset-specific stats | Locked in Phase 1 protocol; src/data/transforms.py |
| Image size | 224x224 RGB | 384x384 (higher res ViT) | Locked; matches pretrained ViT-B/16 resolution |
| Primary metric | Macro-averaged F1 | Weighted F1, accuracy | Locked; contract requirement |
| Loss function | Focal loss gamma=2.0, sqrt-inverse alpha, label_smoothing=0.1 | Class-balanced CE, Balanced BCE | Locked from Phase 2 config |
| Class-balanced sampling | WeightedRandomSampler, sqrt-inverse weights | Uniform oversampling | Locked from Phase 2 |
| Mixed precision | fp16 via torch.amp | bf16, fp32 | Locked from Phase 2 |
| Rare-class threshold | < 200 train samples | < 100, < 50 | Locked from Phase 2 metrics.json |
| Statistical test | Bootstrap >= 10,000 resamples, p < 0.05 | Permutation test, McNemar | Locked from contract |

**CRITICAL: All equations and results below use these conventions. The ViT training recipe MUST use identical augmentation, loss, and sampling to the CNN -- only architecture and base LR differ (per cnn_baseline.yaml comment).**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) | Focal loss | Lin et al. 2017; src/losses/focal_loss.py | Already implemented; reuse directly |
| w_c = 1/sqrt(N_c), normalized so sum(w_c * N_c) = len(dataset) | Sqrt-inverse class weights | Phase 2 dataset.py | Already implemented; reuse directly |
| lr_l = lr_base * decay^(L - l) | Layer-wise LR decay | Steiner et al. 2022 (How to train your ViT) | NEW: apply decay=0.75, L=12 layers for ViT-B |
| delta = F1_vit - F1_cnn (on same bootstrap sample) | Paired bootstrap difference | Standard paired bootstrap | NEW: statistical significance test |
| p-value = fraction of bootstrap samples where delta <= 0 | One-sided bootstrap p-value | Standard paired bootstrap | NEW: test H0: ViT <= CNN |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Transfer learning (fine-tuning) | Adapts ImageNet features to spectrograms | ViT-B/16 initialization | Dosovitskiy et al. 2021 |
| Layer-wise LR decay | Lower LR for early layers preserves pretrained features | Optimizer param groups | Steiner et al. 2022; Clark et al. (BEiT) |
| Gradient clipping | Prevents training instability from large class weights | Already in train_cnn.py | Standard practice |
| Paired bootstrap | Tests significance of metric difference on same test set | Post-training evaluation | Efron & Tibshirani 1993 |
| SpecAugment masking | Time/frequency masking regularizes spectrogram models | Training augmentation | Park et al. 2019 |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| Single-view (1.0s duration) | Morphological info in other durations | Most glitches distinguishable at 1.0s; rare classes may need multi-scale | Unknown; ablate by testing 0.5s, 2.0s, 4.0s | Multi-view fusion (Phase 3 extension, not primary) |
| ImageNet features transfer to spectrograms | Domain gap (natural images vs Q-transform) | Validated empirically (George 2018, Srivastava 2025) | ~5-7pp accuracy gap vs domain-specific pretraining | AudioSet pretraining (AST), self-supervised on GW data |
| Bootstrap percentile CI | Test set size large enough for stable percentiles | Works well for overall macro-F1 (N=48,845) | May be unstable for per-class metrics with N<10 test samples | BCa bootstrap (bias-corrected accelerated); exact binomial for tiny classes |

## Standard Approaches

### Approach 1: ViT-B/16 Fine-Tuning with Layer-Wise LR Decay (RECOMMENDED)

**What:** Load `vit_base_patch16_224.augreg_in21k_ft_in1k` from timm, replace classification head with 23-class linear layer, train with layer-wise LR decay and the existing Phase 2 recipe.

**Why standard:** Steiner et al. 2022 ("How to train your ViT?") established that layer-wise LR decay with factor 0.65-0.75 is the single most important hyperparameter for fine-tuning pretrained ViTs. The augreg (augmentation + regularization) checkpoints from timm are specifically designed for downstream fine-tuning.

**Track record:** ViT-B/16 with ImageNet-21k pretraining consistently outperforms ViT-B/32 by 5-7pp on small datasets. Srivastava 2025 achieved 92.26% on Gravity Spy with ViT-B/32 and a basic recipe; ViT-B/16 with proper fine-tuning should exceed this.

**Key steps:**

1. Build ViT-B/16 model via `timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=23)`
2. Construct optimizer param groups with layer-wise LR decay (decay=0.75, 12 transformer blocks)
3. Train with identical recipe: AdamW, cosine+warmup, focal loss gamma=2.0, sqrt-inverse alpha, label_smoothing=0.1, batch_size=64, gradient_clip=1.0, fp16
4. Early stop on val macro-F1 with patience=10
5. Evaluate on test set: per-class F1/recall, rare-class macro-F1 with bootstrap CI
6. Run paired bootstrap comparing ViT vs CNN rare-class macro-F1 on same test predictions
7. Produce comparison table and confusion matrices

**Known difficulties at each step:**

- Step 1: timm model name must be exact; the `augreg_in21k_ft_in1k` variant is preferred over `orig_in21k_ft_in1k` (better fine-tuning performance)
- Step 2: Layer-wise LR decay requires iterating over named parameters and assigning decay based on layer depth; ViT-B has 12 blocks, each block gets lr * 0.75^(12-block_idx)
- Step 3: Base LR should be 1e-4 (10x lower than CNN's 1e-3); ViT is more sensitive to LR
- Step 5: Rare-class bootstrap CIs will be wide due to small test sets (Wandering_Line N=6); this is inherent, not a bug
- Step 6: Must save per-sample CNN predictions from Phase 2 (or re-evaluate CNN checkpoint) for paired test

### Approach 2: ViT-B/16 with Enhanced Augmentation (FALLBACK)

**What:** Same as Approach 1 but add SpecAugment (time masking + frequency masking) and stronger color jitter to the augmentation pipeline.

**When to switch:** If Approach 1 shows overfitting (val macro-F1 peaks early and drops, or rare-class val F1 is volatile).

**Tradeoffs:** More regularization may help rare classes (fewer examples to memorize) but could hurt common classes. Apply SpecAugment uniformly to all classes to maintain fair comparison.

### Approach 3: Swin-T as Architecture Alternative (EMERGENCY FALLBACK)

**What:** Replace ViT-B/16 (86M params) with Swin-T (28M params) if ViT-B/16 overfits severely.

**When to switch:** If ViT-B/16 val loss diverges within 5 epochs despite LR tuning and augmentation.

**Tradeoffs:** Swin-T has stronger inductive biases (local attention windows) that help on small datasets but may limit long-range attention that could help distinguish rare classes. 3x fewer parameters means less risk of overfitting but lower capacity.

### Anti-Patterns to Avoid

- **Training ViT from scratch:** With 228K training samples total but only 11-142 for rare classes, a from-scratch ViT will learn nothing about rare classes. Always fine-tune from pretrained weights.
  - _Example:_ Dosovitskiy et al. showed ViTs trained from scratch on ImageNet-1K (1.2M images) underperform ResNets; our dataset is 5x smaller.

- **Using overall accuracy to select hyperparameters:** The CNN already gets 91.8% accuracy. A ViT that gets 92% accuracy but worse rare-class F1 is a FAILURE for this project, not an improvement.
  - _Example:_ The model could learn to reclassify Scattered_Light more accurately (4158 test samples) while completely ignoring Wandering_Line (6 test samples) and still improve accuracy.

- **Tuning focal loss gamma separately for ViT vs CNN:** The experimental protocol locks gamma=2.0 for both. Changing gamma for ViT would make the comparison unfair (training recipe difference, not architecture difference).

- **Oversampling rare classes to exact balance:** Extreme oversampling (replicating 11 Chirp samples 2000x) leads to memorization. The sqrt-inverse weighting in WeightedRandomSampler is the correct compromise -- it increases rare-class frequency without forcing exact balance.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form / Value | Source | How to Use |
| --- | --- | --- | --- |
| CNN rare-class macro-F1 | 0.3028 [0.2085, 0.3751] | results/02-cnn-baseline/metrics.json | Beat this; use for paired bootstrap |
| CNN overall macro-F1 | 0.6786 [0.6598, 0.6944] | results/02-cnn-baseline/metrics.json | Compare but NOT optimize for |
| CNN per-class F1 (all 23 classes) | See per_class_f1.csv | results/02-cnn-baseline/ | Direct comparison in deliv-comparison-table |
| Focal loss implementation | Validated with unit tests | src/losses/focal_loss.py | Reuse directly; do not rewrite |
| Sqrt-inverse class weights | w_c = 1/sqrt(N_c) normalized | src/data/dataset.py | Reuse directly from GravitySpyDataset |
| Class-balanced sampler | WeightedRandomSampler | src/data/dataset.py | Reuse get_sampler() method |
| Training loop (epoch, validation) | Mixed precision, grad clip, cosine warmup | src/training/train_cnn.py | Reuse with minor modifications for ViT |
| Bootstrap CI computation | Paired percentile bootstrap | src/evaluation/bootstrap_ci.py | Extend for paired difference test |
| Augmentation pipeline | HFlip, rotation, color jitter, ImageNet norm | src/data/transforms.py | Reuse; optionally add SpecAugment |

**Key insight:** Phase 2 built a complete, tested training infrastructure. Phase 3 should modify as little as possible -- swap the model, adjust the LR, add layer-wise decay. Every unnecessary change introduces confounds in the ViT-vs-CNN comparison.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| CNN best epoch = 14 | Training time estimate (ViT likely needs ~20-30 epochs) | metrics.json | Similar convergence expected |
| CNN training took ~82 min | ViT-B/16 will take ~2-3x longer (86M vs 25M params) | Phase 2 execution | Same hardware, same batch size |
| Wandering_Line F1 = 0.00 | Hardest rare class; 30 train / 6 test samples | per_class_f1.csv | Likely remains difficult for ViT |
| Paired_Doves F1 = 0.10 | Not a "rare" class (216 train) but very low F1 | per_class_f1.csv | Monitor as potential ViT improvement target |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Vision Transformer for Transient Noise Classification | Srivastava & Niedzielski | 2025 | Only ViT on Gravity Spy; ViT-B/32, 92.26% accuracy | Baseline to exceed; no per-class or rare-class metrics reported |
| How to train your ViT? | Steiner et al. | 2022 | Definitive guide to ViT fine-tuning hyperparameters | Layer-wise LR decay = 0.75; augmentation strength; batch size guidelines |
| Learning Imbalanced Data with Vision Transformers (LiVT) | Xu et al. | 2023 | CVPR 2023; ViT-specific imbalanced learning | Balanced BCE loss as alternative if focal loss underperforms; MGP pretraining |
| Focal Loss for Dense Object Detection | Lin et al. | 2017 | Foundational focal loss paper | gamma=2.0 default validated across many domains |
| Class-Balanced Loss (Effective Number) | Cui et al. | 2019 | Effective number of samples framework | Alternative alpha computation: alpha_t = (1-beta)/(1-beta^n_t) |
| Advancing Glitch Classification (O4) | Wu/Raza et al. | 2024 | Multi-view attention fusion for Gravity Spy | Architecture for future multi-view extension |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| timm | >= 1.0 | ViT-B/16 pretrained model loading | Ross Wightman's library; best ViT checkpoint collection; already used for CNN baseline |
| PyTorch | >= 2.2 | Training framework, mixed precision | Already installed and validated in Phase 2 |
| torchmetrics | MulticlassF1Score | Per-epoch macro-F1 tracking | Already used in train_cnn.py |
| scikit-learn | f1_score, confusion_matrix | Final evaluation metrics | Already used in evaluate.py |
| albumentations | Compose, transforms | Augmentation pipeline | Already used in transforms.py |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| matplotlib + seaborn | Confusion matrix visualization (deliv-confusion-matrix) | After evaluation |
| numpy | Bootstrap resampling | Statistical testing |
| pandas | Per-class metrics table (deliv-comparison-table) | After evaluation |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
| --- | --- | --- |
| timm ViT-B/16 | HuggingFace ViTForImageClassification | HF has fewer checkpoint variants; timm already in use |
| Manual layer-wise LR | timm's `param_groups_layer_decay()` utility | timm utility is cleaner; check if available in installed version |
| albumentations SpecAugment | torchaudio.transforms.FrequencyMasking/TimeMasking | albumentations is already the pipeline; adding torchaudio for one transform is unnecessary |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| ViT-B/16 forward pass (batch=64, 224x224, fp16) | ~4 GB VRAM per batch | Attention computation (196 tokens, 12 heads) | fp16 keeps it well within 32GB |
| ViT-B/16 full training (86M params, ~30 epochs) | ~150-250 min on RTX 5090 | Backprop through 12 transformer blocks | ~2-3x CNN training time (82 min); acceptable |
| ViT-B/16 total VRAM (model + optimizer + activations) | ~12-16 GB at batch=64, fp16 | Adam state (2x model size) | Plenty of headroom in 32 GB |
| Bootstrap 10K resamples on 48,845 test samples | ~2-5 min | CPU-bound numpy resampling | Already fast enough from Phase 2 |
| Hyperparameter sweep (3 LRs x 2 augmentation configs) | ~6x base training time (~15-25 hours total) | Multiple training runs | Run sequentially; each ~2.5-4 hours |

**Installation / Setup:**
```bash
# No additional packages needed beyond Phase 2 environment
# Verify timm has the required checkpoint:
python -c "import timm; m = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=False); print(f'Params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M')"
```

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| Identical test set | ViT and CNN evaluated on same samples | Assert test_manifest.csv hash matches Phase 2 | Exact match |
| Identical augmentation at eval | No augmentation leakage in evaluation | Use eval_transforms() (resize + normalize only) | Same as Phase 2 |
| Class weight consistency | Same focal loss alpha for ViT and CNN | Assert class_weights tensor matches Phase 2 | Exact match |
| Metric cross-check | sklearn vs torchmetrics macro-F1 agreement | Compute both; compare | Difference < 1e-6 (as in Phase 2) |
| Overall accuracy sanity | Model is learning something | Compute overall accuracy | Should be > 85% (CNN was 91.8%) |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| CNN baseline (beat this) | Same dataset, same recipe | Rare-class macro-F1 = 0.3028 | Phase 2 |
| CNN per-class best-case | Chirp F1 = 0.47 | Upper end of rare-class CNN performance | Phase 2 |
| CNN per-class worst-case | Wandering_Line F1 = 0.00 | Complete failure on rarest class | Phase 2 |
| ViT-B/32 overall accuracy | 24 classes, basic recipe | 92.26% | Srivastava 2025 |
| Random baseline (23 classes) | Uniform random predictions | Macro-F1 ~ 0.04, accuracy ~ 4.3% | 1/23 per class |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
| --- | --- | --- | --- |
| Training loss convergence | Loss decreasing over epochs | Should be < initial loss after 3 epochs | N/A |
| Val macro-F1 improvement | Monitor per epoch | Should exceed 0.60 (below CNN indicates problem) | CNN val macro-F1 as reference |
| Per-class F1 sanity | All 23 classes have F1 > 0 | At least 20/23 classes should have F1 > 0.3 | CNN per-class F1 as reference |
| Paired bootstrap validity | CI width for rare-class difference | Should be < 0.5 (if wider, test has no power) | N/A |

### Red Flags During Computation

- **Val loss increases within first 3 epochs:** LR too high. Reduce base LR from 1e-4 to 5e-5 or 3e-5.
- **Overall accuracy > 92% but rare-class F1 < 0.30:** Model is optimizing for common classes at expense of rare. Check that class-balanced sampling is active (not accidentally using sequential sampler).
- **Wandering_Line F1 = 0.0 for ViT also:** With only 6 test samples and 30 training samples, this may be fundamentally intractable. Not necessarily a failure of the approach, but document it.
- **Macro-F1 lower than CNN (0.6786):** If the ViT is worse overall, something is wrong with the training setup (likely LR, missing warmup, or broken augmentation pipeline).
- **Training time > 8 hours:** Likely a batch size or mixed precision issue. Check that torch.amp.autocast is active.
- **NaN in loss:** Likely gradient explosion from extreme class weights interacting with attention. Reduce gradient clip norm or cap class weights.

## Common Pitfalls

### Pitfall 1: Wrong timm Checkpoint Variant

**What goes wrong:** timm has multiple ViT-B/16 checkpoints with very different fine-tuning performance. Using `vit_base_patch16_224.orig_in21k` (no augmentation regularization) instead of `vit_base_patch16_224.augreg_in21k_ft_in1k` gives 2-5pp worse downstream performance.

**Why it happens:** Model names look similar; easy to pick the wrong one.

**How to avoid:** Use exactly `vit_base_patch16_224.augreg_in21k_ft_in1k`. Verify with `timm.list_models("vit_base_patch16_224*")`.

**Warning signs:** Unexpectedly low val macro-F1 in early epochs compared to CNN.

**Recovery:** Swap checkpoint and retrain. Low cost (just restart).

### Pitfall 2: Missing Layer-Wise LR Decay

**What goes wrong:** Using a flat learning rate for all layers destroys pretrained features in early layers. Fine-tuning performance drops by 2-5pp.

**Why it happens:** Default optimizer applies the same LR to all parameters. Must explicitly construct param groups.

**How to avoid:** Implement `get_layer_wise_lr_groups(model, base_lr, decay=0.75)` that assigns lower LR to earlier layers. ViT-B has 12 blocks; block 0 gets lr * 0.75^12 ~ 0.032 * base_lr.

**Warning signs:** Model "forgets" basic visual features; attention maps in early layers become random.

**Recovery:** Add layer-wise decay and retrain.

### Pitfall 3: Paired Bootstrap Done Wrong

**What goes wrong:** Running independent (unpaired) bootstraps on ViT and CNN separately, then comparing CIs. This ignores the correlation between models' predictions on the same test samples and produces overly conservative p-values.

**Why it happens:** The existing bootstrap_ci.py computes CIs for single models. Must be extended for paired comparison.

**How to avoid:** In each bootstrap resample, draw the SAME indices for both ViT and CNN predictions. Compute the DIFFERENCE in macro-F1 within each resample. The p-value is the fraction of resamples where the difference is <= 0.

**Warning signs:** p-value is 0.06-0.10 (borderline) when the point estimate difference is large. This suggests unpaired testing is losing power.

**Recovery:** Rewrite the bootstrap to be properly paired.

### Pitfall 4: Small Test Set Instability for Rare Classes

**What goes wrong:** Wandering_Line has only 6 test samples. A single additional correct prediction changes F1 dramatically. Bootstrap CIs for individual rare classes will be very wide.

**Why it happens:** Inherent dataset limitation. Cannot be fixed without more data.

**How to avoid:** Report rare-class MACRO-F1 (averaged across all 4 rare classes) as the primary comparison metric, not individual class F1. This pools information across rare classes and is more stable. Also report individual class metrics but with explicit CI widths.

**Warning signs:** Per-class F1 bootstrap CIs span 0.0 to 1.0 for the smallest classes.

**Recovery:** This is expected, not a bug. Document the instability. The macro-F1 over 4 rare classes is the contract metric.

### Pitfall 5: Confounding Training Recipe with Architecture

**What goes wrong:** If ANY training hyperparameter differs between ViT and CNN (beyond base LR), the comparison becomes unfair.

**Why it happens:** Temptation to "optimize" the ViT recipe (different augmentation, different gamma, different epochs).

**How to avoid:** The Phase 2 config explicitly states: "This recipe is IDENTICAL to the ViT recipe in Phase 3, except for: model architecture and learning rate." Enforce this by loading the same config and only overriding model and LR fields. Layer-wise LR decay is the ONE allowed addition (it's ViT-specific and has no CNN equivalent).

**Warning signs:** Reviewer asks "is the improvement from the architecture or the training recipe?"

**Recovery:** Ablation study: train CNN with ViT's base LR (1e-4). If CNN improves, the LR was confounded.

## Level of Rigor

**Required for this phase:** Controlled experiment with statistical significance testing.

**Justification:** This is the decisive phase for the project's primary claim. The comparison must be fair, the statistical test must be valid, and all confounds must be controlled or documented.

**What this means concretely:**

- All hyperparameters except model architecture and base LR must be identical between ViT and CNN
- Statistical significance via paired bootstrap with p < 0.05 on rare-class macro-F1 difference
- Bootstrap CIs reported for all metrics (point estimates alone are insufficient)
- Per-class F1 and recall reported for all 23 classes (not just aggregates)
- If ViT rare-class macro-F1 does not exceed CNN, this is a STOP condition (backtracking trigger)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Flat LR for ViT fine-tuning | Layer-wise LR decay (0.65-0.75) | Steiner et al. 2022 | 2-5pp improvement on downstream tasks |
| Standard CE loss for imbalanced data | Focal loss + class-balanced sampling | Lin 2017, Cui 2019 | Essential for rare-class performance |
| ViT-B/32 for classification | ViT-B/16 for small-dataset fine-tuning | Steiner et al. 2022 | Patch-16 consistently outperforms patch-32 by 5-7pp on small datasets |
| ImageNet-1k pretraining | ImageNet-21k pretraining | Dosovitskiy et al. 2021 | 21k has much richer feature diversity for transfer |
| Independent bootstrap comparison | Paired bootstrap on same test samples | Standard statistics | Much higher statistical power for detecting differences |

**Superseded approaches to avoid:**

- **ViT-B/32:** Coarser patches lose fine spatial detail in spectrograms. Use ViT-B/16.
- **Training from scratch on small datasets:** Always fine-tune from pretrained. Lee et al. 2022 propose SPT+LSA modifications for scratch training, but these are unnecessary with 21k pretraining.
- **SMOTE for spectrograms:** Generates unphysical pixel-space interpolations. Use class-balanced sampling instead.

## Open Questions

1. **Will layer-wise LR decay be sufficient, or is partial freezing needed?**
   - What we know: Steiner et al. recommend decay=0.75 as default. Full fine-tuning is standard for datasets of this size (~228K samples).
   - What's unclear: Whether the domain gap (natural images to spectrograms) requires freezing early layers entirely.
   - Impact on this phase: If overfitting occurs, may need to freeze first 6/12 blocks.
   - Recommendation: Start with full fine-tuning + layer-wise decay. If val loss diverges, freeze first 6 blocks as fallback.

2. **Can ViT improve Wandering_Line (F1=0.00, N_train=30, N_test=6)?**
   - What we know: CNN completely fails on this class. Only 6 test samples.
   - What's unclear: Whether ViT's global attention helps distinguish this rare morphology, or whether 30 training samples is simply insufficient.
   - Impact on this phase: Even a single correct Wandering_Line prediction would be meaningful (F1 jumps from 0 to ~0.29 with 1 TP out of 6).
   - Recommendation: Expect this to remain difficult. Document results honestly. The rare-class MACRO-F1 can still improve significantly from Helix and Light_Modulation gains.

3. **Should SpecAugment be added to the augmentation pipeline?**
   - What we know: SpecAugment (time + frequency masking) is standard for spectrogram-based models. EnViTSA (2023) showed it helps ViTs on acoustic event classification.
   - What's unclear: Whether it helps for Q-transform spectrograms specifically, and whether adding it violates the "identical recipe" constraint.
   - Impact on this phase: If added, must also add to CNN for fairness (or run ablation showing it helps ViT but not CNN).
   - Recommendation: Add SpecAugment as a clearly documented augmentation extension. Run one ViT training with and one without for ablation. If it helps, apply to CNN also and recompute baseline.

4. **Is the `augreg_in21k_ft_in1k` checkpoint the best starting point, or should `augreg_in21k` (without ImageNet-1k fine-tuning) be used?**
   - What we know: The `in21k_ft_in1k` variant has already been fine-tuned on ImageNet-1k, which adds task-specific features. The `in21k` variant has more general features.
   - What's unclear: Whether the extra ImageNet-1k fine-tuning helps or hurts for spectrogram transfer.
   - Impact on this phase: Minor (likely < 1pp difference).
   - Recommendation: Use `augreg_in21k_ft_in1k` as primary. If results are borderline, try `augreg_in21k` as a quick ablation.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| ViT-B/16 fine-tuning | Overfitting (val loss diverges) | Swin-T (28M params, stronger inductive bias) | ~4 hours (retrain from scratch with Swin-T) |
| ViT-B/16 fine-tuning | Rare-class F1 not improving over CNN | Add SpecAugment + increase augmentation strength | ~4 hours (retrain) |
| ViT-B/16 with standard recipe | Rare-class F1 borderline (p > 0.05) | Sweep LR (5e-5, 1e-4, 3e-4) and layer_decay (0.65, 0.75, 0.85) | ~20 hours (grid search) |
| All ViT variants | Fundamental ViT limitation on this dataset | Contract STOP condition; document honestly | N/A -- this is the backtracking trigger |

**Decision criteria:** If ViT rare-class macro-F1 does not exceed CNN's 0.3028 after (1) LR sweep, (2) augmentation sweep, and (3) Swin-T alternative, invoke the backtracking trigger per contract.

## Caveats and Self-Critique

1. **Assumption that may be wrong:** I assume that single-view (1.0s) is sufficient for rare classes. Some rare morphologies (Wandering_Line, Helix) may be more distinguishable at longer durations. Multi-view fusion could be the key difference, but it's architecturally complex and not in the minimal viable approach.

2. **Alternative dismissed too quickly:** Balanced BCE loss (LiVT, CVPR 2023) was shown to outperform focal loss specifically for ViTs on imbalanced data. However, changing the loss function from focal to BCE would break the "identical recipe" constraint. If focal loss fails, Balanced BCE should be investigated as a loss-function ablation.

3. **Limitation I may be understating:** The very small rare-class test sets (6, 7, 14, 66 samples) mean that statistical significance may be genuinely unachievable even if the ViT is meaningfully better. A 1-sample improvement on Wandering_Line (6 test) is not statistically significant by any test. The paired bootstrap on rare-class MACRO-F1 (pooling all 4 rare classes) is the best we can do, but power is inherently limited.

4. **Simpler method overlooked?** A simple logistic regression or k-NN on pretrained ViT features (no fine-tuning) could serve as a useful diagnostic. If ViT features already separate rare classes better than CNN features, the improvement is in the representation, not the fine-tuning. This is a 30-minute experiment worth running.

5. **Would a specialist disagree?** A Gravity Spy specialist might argue that multi-view fusion is essential (not optional) because rare classes like Wandering_Line are specifically defined by their behavior across time scales. They would be right in principle, but multi-view fusion adds substantial architectural complexity. The single-view approach should be tried first as the minimal experiment.

## Sources

### Primary (HIGH confidence)

- Steiner et al. 2022, "How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers," TMLR (arXiv:2106.10270) -- Layer-wise LR decay, augmentation/regularization guidelines for ViT fine-tuning
- Lin et al. 2017, "Focal Loss for Dense Object Detection" -- Focal loss formulation (gamma=2.0 default)
- Dosovitskiy et al. 2021, "An Image is Worth 16x16 Words," ICLR 2021 (arXiv:2010.11929) -- ViT architecture, ImageNet-21k pretraining importance
- Phase 2 artifacts: results/02-cnn-baseline/metrics.json, per_class_f1.csv -- CNN baseline to beat
- timm documentation and model hub (huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) -- Pretrained checkpoint details

### Secondary (MEDIUM confidence)

- Srivastava & Niedzielski 2025, arXiv:2510.06273 -- ViT-B/32 on Gravity Spy (92.26% accuracy, no per-class breakdown)
- Xu et al. 2023, "Learning Imbalanced Data with Vision Transformers," CVPR 2023 -- Balanced BCE loss for ViTs on imbalanced data
- Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of Samples" (arXiv:1901.05555) -- Alternative class weighting scheme
- Park et al. 2019, "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" -- Time/frequency masking for spectrogram models
- Wu/Raza et al. 2024, arXiv:2401.12913 -- Multi-view fusion for Gravity Spy (future extension reference)

### Tertiary (LOW confidence)

- EnViTSA (Sensors 2023) -- Ensemble ViT + SpecAugment for acoustic classification; confirms SpecAugment helps ViTs on spectrograms
- timm `param_groups_layer_decay()` utility -- Implementation reference for layer-wise LR; verify availability in installed version

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH -- Focal loss, transfer learning, and bootstrap testing are well-established
- Standard approaches: HIGH -- ViT-B/16 fine-tuning with layer-wise LR decay is the standard recipe (Steiner et al. 2022)
- Computational tools: HIGH -- timm, PyTorch, all already validated in Phase 2
- Validation strategies: MEDIUM -- Paired bootstrap is sound but statistical power is limited by tiny rare-class test sets

**Research date:** 2026-03-17
**Valid until:** 2027-03 (timm checkpoint names may change; core methodology is stable)
