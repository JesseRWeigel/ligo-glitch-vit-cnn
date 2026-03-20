# Known Pitfalls Research

**Domain:** ML-based gravitational wave glitch classification (ViT vs CNN, rare-class imbalance, LIGO spectrograms)
**Researched:** 2026-03-16
**Confidence:** MEDIUM-HIGH

## Critical Pitfalls

### Pitfall 1: Accuracy Paradox from Class Imbalance

**What goes wrong:**
The Gravity Spy dataset has extreme class imbalance -- Blip glitches have ~7500 samples while rare classes like Paired_doves have ~100. A model that ignores rare classes entirely can still report >95% overall accuracy. Papers that report only accuracy are uninformative about the very classes this project targets.

**Why it happens:**
Cross-entropy loss on imbalanced data produces gradients dominated by majority classes. The model learns to minimize total loss by becoming a majority-class predictor. This is well-documented in the Gravity Spy literature: confidence overfitting from cross-entropy loss combined with class imbalance and noisy labels produces classifiers that appear accurate but fail on rare morphologies.

**How to avoid:**
- Use macro-averaged F1 and per-class recall as primary metrics, never overall accuracy
- Report confusion matrices for all classes, not just aggregate numbers
- Use class-weighted or focal loss during training
- Set the success criterion on rare-class F1/recall from day one
- Monitor per-class metrics during training, not just aggregate loss

**Warning signs:**
- Validation accuracy is high (>95%) but rare-class recall is below 50%
- Confusion matrix shows rare classes systematically predicted as the nearest majority class
- Training loss decreases smoothly but rare-class F1 oscillates or stagnates

**Phase to address:** Data preparation and model training phases. Metric selection must be locked in the experimental design phase before any model is trained.

**References:**
- Gravity Spy O4 classifier work, arXiv:2401.12913 (Bahaadini et al. discuss confidence overfitting from CE loss + imbalance)
- Soni et al. 2021, arXiv:2208.12849 (Gravity Spy O3 data quality)

---

### Pitfall 2: Data Leakage via Temporal Correlation in Glitches

**What goes wrong:**
LIGO glitches are not independent events. Many glitch types occur in clusters or bursts (e.g., scattering glitches correlate with ground motion, scattered light glitches appear in temporal sequences). Naive random train/test splitting puts temporally adjacent glitches -- which share instrumental state, PSD shape, and often morphology -- into both train and test sets. The model memorizes instrumental conditions rather than learning glitch morphology. Test performance is inflated.

**Why it happens:**
Standard ML practice uses random stratified splits. In LIGO data, glitches separated by seconds share the same noise background, and glitches of the same type may cluster within minutes or hours due to a common instrumental/environmental cause. Livingston's most common O4a glitches were seasonal ground-motion related; Hanford's were instrumental-condition related. These temporal correlations mean that adjacent glitches are not independent samples.

**How to avoid:**
- Use time-based splitting: train on earlier data, validate/test on later data (e.g., train on O3a, validate on O3b, or use temporal blocks within a run)
- At minimum, enforce a temporal gap (e.g., 60 seconds) between any train sample and any test sample
- For cross-validation, use GroupKFold with groups defined by GPS time blocks, not random folds
- Report the splitting strategy explicitly in all results

**Warning signs:**
- Test performance significantly exceeds what is reported in the literature for the same task
- Performance drops sharply when evaluating on a temporally disjoint set (e.g., a later month)
- Model attention maps focus on background noise texture rather than glitch morphology

**Phase to address:** Data preparation phase. The splitting strategy must be defined before any model training begins. This is not recoverable after the fact -- if leakage is discovered late, all results must be discarded.

**References:**
- George et al. 2018, arXiv:1706.07446 (deep transfer learning for LIGO glitches)
- Soni et al. 2021, arXiv:2208.12849 (temporal evolution of glitch populations across O1-O3)

---

### Pitfall 3: Unfair CNN Baseline in ViT Comparison

**What goes wrong:**
The central claim of this project -- "ViT improves rare-class classification over CNN" -- is invalidated if the CNN baseline is not properly optimized. ViTs are typically trained with modern recipes (AdamW, cosine schedule, strong augmentation, longer training). If the CNN baseline uses SGD with step-decay and weak augmentation (as is common in older Gravity Spy work), any ViT advantage may simply reflect better training, not better architecture.

**Why it happens:**
The ViT literature (Dosovitskiy et al. 2021, Touvron et al. DeiT 2021) demonstrated that training recipe matters enormously. Bai et al. (2021) showed CNNs can match ViT robustness when given the same training recipe. Many papers claiming ViT superiority compare against legacy CNN baselines trained with outdated hyperparameters. The Gravity Spy CNN was originally trained with transfer learning from ImageNet using standard recipes circa 2017.

**How to avoid:**
- Train the CNN baseline (ResNet-50 or equivalent) with the SAME modern training recipe as the ViT: AdamW optimizer, cosine learning rate schedule, identical augmentation pipeline, identical number of epochs
- Match model capacity: compare ViT-Small (~22M params) against ResNet-50 (~25M params), not ViT-Base (~86M params) against ResNet-18 (~11M params)
- Report computational cost (FLOPs, wall time) alongside performance
- Include at least one CNN with modern architecture (ConvNeXt, EfficientNetV2) as a second baseline
- If using transfer learning, use the same pretrained backbone source for both (e.g., both ImageNet-21k pretrained)

**Warning signs:**
- ViT outperforms CNN by >5% on aggregate metrics but uses a qualitatively different training setup
- CNN baseline performance is below what the original Gravity Spy papers reported
- No ablation separating architecture effect from training recipe effect

**Phase to address:** Experimental design phase. Baseline specification must be locked before model development begins. A fair-comparison protocol should be a deliverable of the experimental design phase.

**References:**
- Bai et al. 2021, "Are Transformers More Robust Than CNNs?" (JHU, shows training recipe dominates)
- Touvron et al. 2021, "Training data-efficient image transformers" (DeiT, arXiv:2012.12877)

---

### Pitfall 4: ViT Data Hunger on Small/Imbalanced Datasets

**What goes wrong:**
ViTs lack the inductive biases (locality, translation equivariance) that make CNNs effective on small datasets. Standard ViT trained from scratch on a dataset the size of Gravity Spy (~25k labeled spectrograms, with rare classes having <200 samples) will underperform CNNs. The self-attention layers cannot learn to aggregate local features from insufficient data, and lower attention layers fail to focus on neighboring patches.

**Why it happens:**
CNNs have built-in locality via convolution kernels. ViTs must learn spatial structure entirely from data. The original ViT paper (Dosovitskiy et al.) showed ViTs only beat CNNs when pretrained on JFT-300M (300 million images). On ImageNet alone (~1.2M images), ViTs underperform. Gravity Spy is 200x smaller than ImageNet.

**How to avoid:**
- Use pretrained ViT (ImageNet-21k or similar), fine-tune on Gravity Spy spectrograms -- do NOT train from scratch
- Consider hybrid architectures: convolutional stem + transformer body (e.g., early convolution layers feeding into transformer blocks)
- Use Shifted Patch Tokenization (SPT) and Locality Self-Attention (LSA) from Lee et al. (arXiv:2112.13492) if training from scratch is required
- Apply strong data augmentation (mixup, cutmix, random erasing) to compensate
- For rare classes specifically, consider few-shot learning heads or prototypical networks on top of ViT features

**Warning signs:**
- ViT training loss decreases but validation loss plateaus or increases early (overfitting with <5 epochs)
- Attention maps in early layers are diffuse/random rather than focusing on local structure
- Performance is worse than a simple ResNet-18 baseline

**Phase to address:** Model architecture phase. The decision to fine-tune vs. train from scratch must be made early and justified by dataset size analysis.

**References:**
- Dosovitskiy et al. 2021, "An Image is Worth 16x16 Words" (arXiv:2010.11929)
- Lee et al. 2022, "Vision Transformer for Small-Size Datasets" (arXiv:2112.13492, BMVC 2022)
- arXiv:2510.06273, "Vision Transformer for Transient Noise Classification" (achieves >92% accuracy on GW glitch classification with ViT)

---

### Pitfall 5: O3-to-O4 Distribution Shift Invalidating Trained Models

**What goes wrong:**
Glitch morphologies change between observing runs due to detector upgrades, commissioning changes, and environmental shifts. New glitch classes appear, old ones disappear, and existing classes change in frequency or morphology. A model trained on O3 data may fail on O4 data, and even within O3, the glitch population evolves over time. Livingston and Hanford have different glitch distributions.

**Why it happens:**
Detector hardware changes between runs (e.g., squeezing upgrades, suspension changes). Environmental conditions are seasonal (ground motion at Livingston varies with logging activity and weather). The O4 run has seen increased rates of Low Frequency Burst and Low Frequency Lines at both detectors from unidentified sources. The Gravity Spy team explicitly noted that O4 may require site-specific training sets and CNN models.

**How to avoid:**
- Evaluate on temporally held-out data from a different period or run segment, not just random test split
- Include domain adaptation or continual learning as part of the methodology
- Track per-class performance over time windows within the evaluation set
- If claiming O4 applicability, validate on actual O4 data (available via GWOSC after embargo)
- Report per-detector results separately for Hanford (H1) and Livingston (L1)
- Build in a "novel class detection" mechanism (out-of-distribution detection) rather than forcing all glitches into known categories

**Warning signs:**
- High performance on O3a test set but degraded performance on O3b
- Different performance between H1 and L1 that cannot be explained by class frequency differences alone
- Certain classes show bimodal confidence distributions (model is uncertain on the shifted subpopulation)

**Phase to address:** Evaluation design phase. Distribution shift evaluation must be a planned experiment, not an afterthought.

**References:**
- Soni et al. 2021, arXiv:2208.12849 (O3 glitch evolution)
- LIGO Detector Characterization O4, arXiv:2409.02831 (O4a new glitch types)
- Bahaadini et al. 2024, arXiv:2401.12913 (O4 classifier requiring updates)

---

### Pitfall 6: Gravity Spy Label Quality and the "None of the Above" Problem

**What goes wrong:**
Gravity Spy labels come from a combination of citizen science volunteers and machine learning bootstrapping. Label quality is uneven: common classes have reliable consensus labels, but rare classes have fewer volunteer classifications and higher label noise. The "None of the Above" (NOTA) class was removed from training because it confused the model, but this means glitches with genuinely novel morphology get forced into known categories. Labels for ambiguous or borderline glitches are effectively random.

**Why it happens:**
The citizen science workflow uses majority voting with tiered volunteer expertise. For rare classes, fewer volunteers have seen enough examples to classify confidently. The ML-human feedback loop can create confirmation bias: the ML model suggests a label, volunteers are primed by it, and the label gets confirmed even if marginal. Retired glitches (no longer shown to volunteers) accumulate uncertain classifications.

**How to avoid:**
- Filter the training set by Gravity Spy confidence score (use only high-confidence labels, e.g., >0.9)
- Use label smoothing during training (as done in the O4 classifier, arXiv:2401.12913)
- Manually audit a random sample of rare-class labels before training -- budget 2-4 hours for this
- Implement a noise-robust loss function (e.g., symmetric cross-entropy, generalized cross-entropy)
- Consider the NOTA samples as a separate validation set for OOD detection
- Do NOT assume Gravity Spy labels are ground truth -- treat them as noisy annotations

**Warning signs:**
- Inter-annotator agreement (where available) is low for certain classes
- Model confidence is bimodal for a class (some samples are easy, others are near-random)
- Rare-class performance varies wildly between random seeds (sign of noisy labels)

**Phase to address:** Data preparation phase. Label quality audit must happen before any model training.

**References:**
- Zevin et al. 2024, "Gravity Spy: Lessons Learned and a Path Forward" (arXiv:2308.15530, EPJ Plus)
- Soni et al. 2021, arXiv:2208.12849 (NOTA removal, label quality discussion)

---

## Approximation Shortcuts

Shortcuts that seem reasonable but introduce systematic errors.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
| --- | --- | --- | --- |
| Using overall accuracy as the metric | Simple to report, always high | Completely hides rare-class failures; masks the accuracy paradox | Never for this project |
| Training on single-duration spectrograms (e.g., 1.0s only) | Simpler pipeline, 4x less data to handle | Loses multi-scale morphological information; some glitches are only distinguishable at specific durations | Only for initial prototyping; multi-duration fusion required for final results |
| Random train/test split without temporal awareness | Standard ML practice, easy to implement | Inflated metrics from temporal leakage; results not reproducible on future data | Never -- use time-based splits |
| Using SMOTE/random oversampling for rare classes | Quick fix for imbalance | SMOTE generates synthetic samples via linear interpolation in pixel space, which creates unphysical spectrogram blends that do not correspond to real glitch morphologies; causes overfitting to artifacts | Never for spectrograms. Use augmentation (time-shift, frequency-shift, noise injection) or class-weighted loss instead |
| Evaluating only on the Gravity Spy test split | Convenient, standardized | Does not test generalization to new time periods, new detector states, or new glitch morphologies | Only as one of multiple evaluation protocols |

## Convention Traps

Common mistakes when converting between different conventions or comparing with literature.

| Convention Issue | Common Mistake | Correct Approach |
| --- | --- | --- |
| Q-transform normalization | Different papers use different normalization (absolute energy vs. row-normalized vs. min-max). Comparing models trained with different normalizations is meaningless. | Fix one normalization scheme and document it. Use min-max normalization to [0,1] as in the standard Gravity Spy pipeline. |
| Spectrogram time/frequency axes | Axes flipped between different Q-transform implementations; some use linear frequency, others log frequency | Use the GWpy Q-transform output with logarithmic frequency axis as in Gravity Spy. Verify axis orientation matches model input expectations. |
| Multi-duration image ordering | The four duration views (0.5s, 1.0s, 2.0s, 4.0s) can be concatenated in different orders or as a 4-channel input vs. 2x2 grid | Match the Gravity Spy convention (2x2 grid) for baseline comparisons. Document the layout. |
| SNR threshold for glitch inclusion | Gravity Spy uses SNR > 7.5 with peak frequency 10-2048 Hz. Other pipelines use different thresholds. | Use the same Omicron trigger threshold as Gravity Spy when building the dataset. State the threshold explicitly. |
| Class taxonomy versions | Gravity Spy has evolved from 22 classes (O1/O2) to 24+ classes (O3). Papers use different taxonomies. | State which taxonomy version is used. Map to the specific Gravity Spy data release (e.g., Zenodo release for O1-O3b). |
| Macro vs. micro vs. weighted F1 | "F1 score" without qualification is ambiguous. Macro-F1 weights all classes equally; micro-F1 is dominated by majority classes. | Always specify: "macro-averaged F1" for rare-class evaluation. Report per-class F1 in supplementary material. |

## Numerical Traps

Patterns that work for simple cases but fail for realistic calculations.

| Trap | Symptoms | Prevention | When It Breaks |
| --- | --- | --- | --- |
| ViT patch size mismatch | Model trains but attention maps are meaningless; performance is poor | Patch size must be compatible with spectrogram resolution. A 140x170 px spectrogram with 16x16 patches leaves fractional patches. Resize to patch-compatible dimensions (e.g., 224x224). | When input image dimensions are not divisible by patch size |
| Positional embedding interpolation | Fine-tuning a pretrained ViT on different resolution spectrograms produces degraded performance | If input resolution differs from pretraining resolution, positional embeddings must be interpolated (bicubic). Validate that interpolated positional embeddings still capture spatial structure. | When fine-tuning resolution differs from pretraining resolution by >2x |
| Gradient explosion with class weights | Training diverges or produces NaN losses when using large class weights for rare classes | Clip class weights (e.g., cap at 10x) or use focal loss instead of weighted CE. Use gradient clipping (max_norm=1.0). | When the rarest class is >100x less frequent than the most common class |
| Batch composition starvation | Rare classes appear in <1% of batches; model "forgets" rare classes between encounters | Use class-balanced sampling: each batch contains at least one sample from every class (or use oversampling in the dataloader). | When rare-class frequency < 1/batch_size |
| Mixed-precision underflow in attention | Attention scores for rare-class features underflow to zero in fp16 | Use bf16 instead of fp16 if available; keep attention computation in fp32. | With very long attention sequences or extreme value ranges in spectrograms |

## Interpretation Mistakes

Domain-specific errors in interpreting results beyond computational bugs.

| Mistake | Risk | Prevention |
| --- | --- | --- |
| Claiming "ViT learns physics" from attention maps | Attention maps show where the model looks, not what it understands. High attention on a glitch region does not mean the model has learned the physical mechanism. | Use attention maps for debugging and qualitative analysis only. Do not claim physical interpretation without ablation studies. |
| Interpreting high confidence as correct classification | Gravity Spy CNN is known to be overconfident due to cross-entropy training. A 99% confidence prediction can still be wrong, especially for rare classes. | Calibrate model outputs (temperature scaling, Platt scaling). Report calibration curves alongside accuracy. |
| Assuming improved rare-class F1 means the model generalizes | High rare-class F1 on the test set may reflect memorization of the few rare-class examples, especially if temporal leakage is present. | Validate on temporally disjoint data. Check that rare-class performance does not collapse on held-out time periods. |
| Confusing "novel glitch detection" with "rare-class classification" | This project classifies known rare classes. Detecting genuinely novel (unseen) glitch types is a different, harder problem (open-set recognition). Do not conflate the two. | State clearly that the scope is closed-set classification of known rare classes. If OOD detection is included, evaluate it separately with proper metrics (AUROC on OOD vs. ID). |
| Attributing performance differences to architecture when training differs | If ViT uses augmentation/regularization that CNN does not, the comparison measures training recipe, not architecture. | Ablation study isolating architecture effect: same data, same augmentation, same optimizer, same schedule. |

## Publication Pitfalls

Common mistakes specific to writing up and presenting physics results.

| Pitfall | Impact | Better Approach |
| --- | --- | --- |
| Reporting only best-of-N runs | Hides variance; rare-class metrics are high-variance with small test sets | Report mean +/- std over at least 3-5 random seeds. Use bootstrap confidence intervals for per-class metrics. |
| Comparing against Gravity Spy CNN without retraining it | The published Gravity Spy CNN may use a different dataset version, different preprocessing, or different class taxonomy | Retrain the CNN baseline on the exact same dataset split with the same preprocessing. |
| Not reporting per-class results | Aggregate metrics hide that the model excels on common classes but fails on the rare classes that are the stated contribution | Include a full per-class precision/recall/F1 table for all classes, especially the rare targets. |
| Claiming O4 applicability without O4 evaluation | O3-trained models may not generalize to O4 due to distribution shift | Either evaluate on O4 data or explicitly state the limitation. |
| Cherry-picking attention map visualizations | Showing only examples where attention aligns with glitch features creates misleading impression | Show attention maps for correct AND incorrect predictions, common AND rare classes. Quantify attention-morphology alignment. |

## "Looks Correct But Is Not" Checklist

Things that appear right but are missing critical pieces.

- [ ] **Test set metrics:** High overall F1 -- but check macro-F1 vs. weighted-F1. They can differ by 10+ points when imbalance is extreme.
- [ ] **Data split:** Stratified split preserves class proportions -- but check for temporal leakage. Adjacent GPS times in train and test share noise realization.
- [ ] **Augmentation:** Standard image augmentation applied -- but check that augmentations are physically meaningful for spectrograms. Horizontal flip reverses time (may change morphology). Vertical flip reverses frequency (unphysical). Color jitter on Q-transform spectrograms changes the energy scale.
- [ ] **Transfer learning:** ImageNet-pretrained backbone used -- but check that the pretrained model expects 3-channel RGB input while Q-transform spectrograms may be grayscale or have different channel semantics.
- [ ] **Class-weighted loss:** Weights set inversely proportional to class frequency -- but check that extreme weights (>50x) do not cause gradient instability.
- [ ] **Multi-duration input:** All four Q-transform durations used -- but check that the model is actually using information from all four (ablation: does removing one duration change results?).
- [ ] **Confidence calibration:** Model outputs probabilities -- but check Expected Calibration Error (ECE). Uncalibrated models from cross-entropy training are systematically overconfident.
- [ ] **Reproducibility:** Results reported -- but check that random seed, exact data split, and preprocessing pipeline are fully specified and code is available.

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
| --- | --- | --- |
| Temporal data leakage discovered | HIGH | All results must be recomputed with proper splits. No shortcut. |
| Unfair CNN baseline | MEDIUM | Retrain CNN with modern recipe. Re-run comparison. Prior ViT results may still be valid. |
| Label noise corrupting rare classes | MEDIUM | Filter to high-confidence labels, retrain. Consider noise-robust loss. May need manual audit of rare-class samples. |
| ViT overfitting on small dataset | LOW-MEDIUM | Switch to pretrained ViT with fine-tuning. Add regularization (dropout, weight decay, augmentation). Reduce model size. |
| SMOTE artifacts in training data | MEDIUM | Remove synthetic samples, retrain with class-weighted loss or class-balanced sampling instead. |
| Distribution shift not evaluated | LOW | Run existing model on held-out temporal block. Report results. No retraining needed, just additional evaluation. |
| Wrong spectrogram normalization | HIGH | Reprocess all spectrograms with correct normalization. Retrain all models. |

## Pitfall-to-Phase Mapping

How research phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
| --- | --- | --- |
| Accuracy paradox | Experimental design | Confirm macro-F1 and per-class metrics are computed and logged from first training run |
| Temporal data leakage | Data preparation | Verify no GPS time overlap within 60s between train/val/test; check with explicit temporal gap analysis |
| Unfair CNN baseline | Experimental design | Baseline protocol document specifying identical training recipe for CNN and ViT; reviewer sign-off |
| ViT data hunger | Model architecture | Preliminary experiment: ViT-from-scratch vs. ViT-pretrained vs. CNN on 10% of data. If ViT-scratch fails, abandon that path early. |
| O3-O4 distribution shift | Evaluation design | Temporal evaluation protocol specifying held-out time blocks; per-detector reporting |
| Label quality | Data preparation | Audit 50 random samples from each rare class; compute inter-annotator agreement where available; filter by confidence threshold |
| Spectrogram normalization | Data preparation | Verify normalization matches Gravity Spy standard; document Q-transform parameters (Q range, frequency range, SNR threshold) |
| Class weight instability | Model training | Monitor per-class gradient norms during first epoch; cap class weights; use gradient clipping |
| Attention map overinterpretation | Results analysis | Include negative examples (wrong predictions) in attention analysis; do not claim physical interpretation without ablation |

## Sources

- Bahaadini et al. 2024, "Advancing Glitch Classification in Gravity Spy: Multi-view Fusion with Attention-based Machine Learning for Advanced LIGO's Fourth Observing Run" ([arXiv:2401.12913](https://arxiv.org/abs/2401.12913))
- Soni et al. 2022, "Data quality up to the third observing run of Advanced LIGO: Gravity Spy glitch classifications" ([arXiv:2208.12849](https://arxiv.org/abs/2208.12849))
- Zevin et al. 2024, "Gravity Spy: Lessons Learned and a Path Forward" ([arXiv:2308.15530](https://arxiv.org/abs/2308.15530), EPJ Plus)
- George et al. 2018, "Deep Transfer Learning: A new deep learning glitch classification method for advanced LIGO" ([arXiv:1706.07446](https://arxiv.org/abs/1706.07446))
- LIGO Detector Characterization in O4, ([arXiv:2409.02831](https://arxiv.org/abs/2409.02831))
- "Vision Transformer for Transient Noise Classification" ([arXiv:2510.06273](https://arxiv.org/abs/2510.06273))
- Lee et al. 2022, "Vision Transformer for Small-Size Datasets" ([arXiv:2112.13492](https://arxiv.org/abs/2112.13492), BMVC 2022)
- Bai et al. 2021, "Are Transformers More Robust Than CNNs?" (JHU)
- Touvron et al. 2021, "Training data-efficient image transformers" (DeiT, [arXiv:2012.12877](https://arxiv.org/abs/2012.12877))
- Dosovitskiy et al. 2021, "An Image is Worth 16x16 Words" ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929))
- Gravity Spy data releases on Zenodo ([zenodo.org/records/5649212](https://zenodo.org/records/5649212))
- Colgan et al. 2020, "On Improving the Performance of Glitch Classification for Gravitational Wave Detection by using GANs" ([arXiv:2207.04001](https://arxiv.org/abs/2207.04001))
- Living Reviews in Relativity: "Applications of machine learning in gravitational-wave research" ([Springer](https://link.springer.com/article/10.1007/s41114-024-00055-8))

---

_Known pitfalls research for: ML-based LIGO glitch classification (ViT vs CNN, rare-class imbalance)_
_Researched: 2026-03-16_
