# Phase 4: O4 Validation & CW Sensitivity - Research

**Researched:** 2026-03-17
**Domain:** Distribution shift evaluation (ML on LIGO data) + continuous gravitational wave data quality
**Confidence:** MEDIUM

## Summary

Phase 4 has two distinct subgoals that share the O4 evaluation dataset but differ in methodology. **Subgoal A** (O4 validation) evaluates both the O3-trained ViT and CNN on O4 data to test whether the sample-efficiency threshold discovered in Phase 3 persists under distribution shift. This is methodologically straightforward -- inference with saved checkpoints on new data, per-class F1 comparison, degradation analysis. **Subgoal B** (CW sensitivity) quantifies how ViT-based glitch vetoes improve continuous wave search data quality compared to CNN-based vetoes. This is methodologically novel for this project and requires constructing a veto-generation pipeline, identifying CW-critical glitch classes, and computing a duty cycle or upper-limit metric on real O4 data segments.

The principal challenge is **O4 data availability and format**. GWOSC released O4a bulk strain data (May 2023 -- Jan 2024) in August 2025, including data quality flags and auxiliary channels. However, there is no public O4 Gravity Spy labeled dataset on Zenodo as of March 2026. The Gravity Spy classifier is running on O4 data via Zooniverse, and volunteer classifications through July 2024 were released (Zenodo 13904421), but pre-made spectrogram images with O4 labels may need to be obtained from the Zooniverse API or generated from O4 strain data using the existing GWpy pipeline. This data acquisition step is the primary risk for the phase.

The CW sensitivity quantification requires a concrete metric. The recommended approach is **duty cycle improvement**: for a selected O4 data segment, compute the fraction of observation time surviving after vetoing segments flagged by each classifier for CW-critical glitch classes. A secondary metric is **veto efficiency vs. deadtime**: how much glitch contamination is removed per unit of observation time lost. These are standard LIGO data quality metrics (used in e.g., arXiv:2409.02831). The key insight from Phase 3 is that ViT outperforms CNN specifically on classes with 100+ training samples -- the CW-relevant question is whether any CW-critical glitch classes fall in this regime.

**Primary recommendation:** Obtain O4a Gravity Spy classifications (Zenodo 13904421 volunteer data or Zooniverse API for ML classifications + spectrogram URLs), download/generate O4a spectrograms for evaluation, run both O3-trained models on O4a data, compute per-class F1 with bootstrap CIs, test threshold persistence via correlation analysis, then build a veto pipeline for CW-critical classes comparing ViT vs CNN duty cycle on the same O4a data segment.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| ref-gravity-spy (Zevin et al. 2017) | benchmark | CNN-based vetoes serve as CW baseline comparison | CNN predictions on O4 data are the comparison target | plan, execution, verification |
| O3 per-class comparison (results/03-vit-rare-class/comparison_table.csv) | prior artifact | Defines the sample-efficiency threshold pattern to test on O4 | Load O3 per-class F1 values; correlate with O4 per-class F1 | plan, execution, verification |
| CNN checkpoint (checkpoints/02-cnn-baseline/best_model.pt) | prior artifact | Must evaluate on O4 data for fair comparison | Load and run inference | execution |
| ViT checkpoint (checkpoints/03-vit-rare-class/best_model.pt) | prior artifact | Must evaluate on O4 data | Load and run inference | execution |
| Phase 3 statistical summary | prior artifact | Documents threshold: ViT advantage for 100+ samples, disadvantage for <50 | Reference for O4 comparison | execution, verification |
| fp-qualitative-only | forbidden proxy | CW benefit MUST be a number, not a qualitative argument | Compute duty cycle or equivalent metric | plan, execution, verification |
| fp-overall-accuracy | forbidden proxy | Per-class F1, not overall accuracy, remains the decisive metric | Enforce in all O4 evaluations | plan, execution, verification |

**Missing or weak anchors:** No public O4 Gravity Spy labeled dataset with spectrogram images exists on Zenodo as of March 2026. The volunteer classification dataset (Zenodo 13904421) provides labels but accessing spectrograms requires the Zooniverse CDN URLs or regeneration from strain data. This is the primary data acquisition risk. If O4 spectrogram images cannot be obtained, the phase must generate Q-transform spectrograms from GWOSC O4a strain data using the GWpy pipeline -- which is feasible but significantly more expensive.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Input normalization | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Dataset-specific stats | Locked from Phase 1; must match training convention exactly |
| Image size | 224x224 RGB | 384x384 | Locked from Phase 1 |
| Primary metric | Macro-averaged F1 (per-class and rare-class) | Overall accuracy | Locked; contract requirement |
| Statistical test | Bootstrap >= 10,000 resamples, p < 0.05 | Permutation test | Locked from contract |
| Rare-class threshold | < 200 O3 training samples | Other cutoffs | Locked from Phase 2/3 |
| Sample-efficiency threshold | ~100 O3 training samples | Other cutoffs | Discovered in Phase 3; test on O4 |
| CW metric | Duty cycle (fraction of time surviving vetoes) | Strain upper limit, h0 sensitivity depth | Recommended here; standard DQ metric |
| O4a GPS range | 1369180818 -- 1389657618 (May 24 2023 -- Jan 16 2024) | O4b (not yet public) | GWOSC O4a release |
| Unit system | SI for physical quantities; dimensionless for ML metrics | Natural units | Locked from project conventions |

**CRITICAL: O4 evaluation must use the identical preprocessing pipeline (transforms, normalization, image size) as O3 training. Any deviation invalidates the distribution shift analysis.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| F1_c = 2 * P_c * R_c / (P_c + R_c) | Per-class F1 score | Standard | Computed per class on O4 test set |
| delta_c = F1_c(O4) - F1_c(O3) | Per-class degradation | This phase | Measures distribution shift per class |
| rho = corr(N_train, F1_vit - F1_cnn) on O4 | Threshold persistence test | This phase | Tests whether ViT advantage still correlates with training set size on O4 |
| DC = T_clean / T_total | Duty cycle | Standard LIGO DQ | Fraction of observation time surviving glitch vetoes |
| eta_veto = N_vetoed_glitches / N_total_glitches | Veto efficiency | Standard LIGO DQ | Fraction of glitches correctly vetoed |
| tau_dead = T_vetoed / T_total | Deadtime fraction | Standard LIGO DQ | Fraction of observation time removed by vetoes |
| ROC: eta_veto vs tau_dead at varying confidence thresholds | Veto ROC curve | Standard LIGO DQ | Compares ViT vs CNN veto quality |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Transfer evaluation (no retraining) | Tests O3-trained models on O4 data without fine-tuning | Both model evaluations | Standard domain shift protocol |
| Spearman rank correlation | Tests whether ViT-CNN F1 difference correlates with training set size on O4 | Threshold persistence test | Non-parametric; robust to outliers |
| Bootstrap confidence intervals | Provides uncertainty on per-class O4 metrics | All per-class F1 values | Efron & Tibshirani 1993 |
| Segment-based veto generation | Converts per-glitch classifier predictions into time-domain veto segments | CW sensitivity analysis | LIGO DQ standard practice |
| Duty cycle computation | Computes fraction of clean observation time | CW metric | arXiv:2409.02831 |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| O3 training generalizes to O4 | Detector configuration change (O3->O4) | Morphological classes that are physically stable across runs (e.g., Blip, Koi_Fish) | Unknown a priori; measured by degradation analysis | Fine-tune on O4 labeled data (out of scope for this paper) |
| Single-view (1.0s) evaluation on O4 | Morphological info in other durations | Most classes distinguishable at 1.0s | Consistent with O3 evaluation protocol | Multi-view if single-view O4 performance is anomalously low |
| Duty cycle as CW sensitivity proxy | CW search sensitivity depends linearly on observation time for same noise floor | Valid when noise floor is comparable across vetoed/unvetoed segments | Approximation breaks if vetoes preferentially remove high-noise segments (which improves sensitivity beyond just duty cycle) | Compute actual strain upper limits using CW search code (much more expensive) |

## Standard Approaches

### Approach 1: O4 Evaluation via GWOSC + Gravity Spy Classifications (RECOMMENDED)

**What:** Obtain O4a Gravity Spy classifications (volunteer or ML labels) with spectrogram image URLs from Zenodo/Zooniverse, download O4a spectrograms using the same CDN pipeline as Phase 1, evaluate both O3-trained models on these images, compute per-class F1 with bootstrap CIs.

**Why standard:** This mirrors the Phase 1 data acquisition approach (download pre-made spectrograms from Zooniverse CDN) and ensures consistency between O3 and O4 evaluation. The Gravity Spy project provides labels for O4 glitches classified by both ML and volunteers.

**Track record:** The existing `scripts/03_download_spectrograms.py` downloads spectrograms from Zooniverse CDN URLs. The same infrastructure can be reused for O4 data if the O4 metadata includes CDN URLs.

**Key steps:**

1. Acquire O4a Gravity Spy metadata with spectrogram URLs (Zenodo 13904421 volunteer classifications, or query Zooniverse API for ML classifications with image URLs)
2. Filter to high-confidence labels (ml_confidence > 0.9 or volunteer consensus >= 3/5)
3. Download O4a spectrograms using existing CDN download pipeline (adapted for O4 metadata format)
4. Verify O4a spectrograms match expected format (224x224 RGB, same normalization)
5. Run both O3-trained models (CNN checkpoint, ViT checkpoint) on O4a spectrograms -- inference only, no training
6. Compute per-class F1, recall, precision with bootstrap CIs for both models on O4
7. Compute per-class degradation: delta_c = F1_c(O4) - F1_c(O3)
8. Test threshold persistence: Spearman correlation between N_train(O3) and (F1_vit - F1_cnn) on O4
9. Generate comparison table, confusion matrices, degradation plots

**Known difficulties at each step:**

- Step 1: O4 Gravity Spy metadata format may differ from O3. The volunteer classification dataset (Zenodo 13904421) may not include spectrogram URLs -- may need Zooniverse API queries. **FALLBACK:** Generate spectrograms from GWOSC O4a strain data using GWpy Q-transform.
- Step 2: O4 may have new glitch classes not in O3 taxonomy (24 O3 classes). The O4 classifier (Wu et al. 2024) updated the taxonomy. Models can only classify into the 23 O3 classes they were trained on.
- Step 3: CDN URLs may have changed format between O3 and O4. Test a small batch first.
- Step 5: O4 glitches from new/modified classes will be forced into O3 categories, producing misclassifications. This is expected and informative -- it measures how well the O3 taxonomy covers O4 glitches.
- Step 7: Some O3 classes may have zero or very few O4 examples. Per-class metrics will have wide CIs or be undefined. Report class-by-class sample sizes.
- Step 8: Correlation test requires enough classes with sufficient O4 test samples. Classes with < 10 O4 test samples should be excluded from the correlation analysis.

### Approach 2: Generate O4a Spectrograms from GWOSC Strain Data (FALLBACK)

**What:** Download O4a strain data from GWOSC, obtain glitch trigger times from Omicron or Gravity Spy metadata, generate Q-transform spectrograms using GWpy, then evaluate as in Approach 1.

**When to switch:** If O4 spectrogram images cannot be obtained from Zooniverse CDN (URLs unavailable, format mismatch, or CDN access blocked).

**Tradeoffs:** Much more compute-intensive (hours of Q-transform computation vs. minutes of CDN download). But produces spectrograms with guaranteed consistency to the O3 pipeline if the same GWpy Q-transform parameters are used. May actually be preferable for ensuring identical preprocessing.

**Key steps:**

1. Download O4a strain data segments from GWOSC using `gwpy.timeseries.TimeSeries.fetch_open_data()`
2. Obtain O4a glitch trigger times and labels from Gravity Spy metadata (Zenodo 13904421 or Zooniverse API)
3. For each trigger, generate 4-duration Q-transform spectrograms using the same GWpy parameters as Phase 1
4. Continue from Step 4 of Approach 1

### Anti-Patterns to Avoid

- **Fine-tuning on O4 data:** Do NOT retrain or fine-tune models on O4. The entire point is to test O3-trained models under distribution shift. Any O4 training invalidates the distribution shift analysis.
- **Cherry-picking O4 time segments:** Evaluate on a contiguous O4a segment, not hand-picked "clean" periods. Distribution shift includes the bad data.
- **Using O4 labels as ground truth without scrutiny:** O4 Gravity Spy labels may be noisier than O3 (new classifier, fewer volunteer classifications). Be transparent about label quality.
- **Reporting only overall degradation:** Must report per-class degradation. A model can maintain overall F1 while catastrophically failing on specific classes.
- **Qualitative CW claims:** The fp-qualitative-only forbidden proxy requires a computed number, not "ViT-based vetoes should improve CW searches because..."

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| O3 per-class F1 for both models | See comparison_table.csv (23 classes) | Phase 3: results/03-vit-rare-class/comparison_table.csv | Reference values for degradation computation |
| Sample-efficiency threshold | ViT > CNN for N_train >= ~100; ViT < CNN for N_train < ~50 | Phase 3 statistical summary | The hypothesis to test on O4 |
| O3 rare-class macro-F1 | CNN: 0.3028 [0.2085, 0.3751]; ViT: 0.2412 [0.2019, 0.2957] | Phase 3 paired bootstrap | Baseline for O4 degradation |
| O3 overall macro-F1 | CNN: 0.6786; ViT: 0.7230 | Phase 3 metrics.json | Baseline for O4 degradation |
| Gravity Spy O3 class distribution | 23 classes, N_train per class from 11 (Chirp) to 34,555 (Fast_Scattering) | Phase 1 split_statistics.json | N_train values for threshold correlation test on O4 |

**Key insight:** The O3 results are fully computed and saved. Phase 4 only needs to run inference on O4 data and compare.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| CNN and ViT checkpoints | Trained model weights for O4 inference | checkpoints/02-cnn-baseline/, checkpoints/03-vit-rare-class/ | Must use identical transforms.py preprocessing |
| Existing evaluation scripts | Bootstrap CI, per-class metrics, confusion matrix code | src/evaluation/evaluate.py, bootstrap_ci.py, paired_bootstrap.py | Reusable with minimal modification |
| Spectrogram download pipeline | Async CDN download with retry logic | scripts/03_download_spectrograms.py | Adapt for O4 metadata format |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| LIGO DetChar O4a (arXiv:2409.02831) | Soni et al. | 2024 | Documents O4a glitch landscape, new glitch types, DQ products | Which O3 glitch classes changed in O4; data quality flag definitions; veto methodology |
| Gravity Spy O4 classifier (arXiv:2401.12913) | Wu/Raza et al. | 2024 | O4 Gravity Spy classifier with updated taxonomy and multi-view fusion | O4 class taxonomy changes; known O4 distribution shifts; confidence calibration issues |
| Gravity Spy lessons learned (arXiv:2308.15530) | Zevin et al. | 2024 | CNN limitations acknowledged; O4 novel glitches misclassified | Specific failure modes of O3-trained classifiers on O4 data |
| O4a open data paper (arXiv:2508.18079) | LVK | 2025 | O4a data release documentation; GPS times, data quality flags, available channels | O4a GPS range; data quality flag names; science mode segments |
| CW search methods review (arXiv:2111.12575) | Tenorio et al. | 2021 | Comprehensive review of CW search methods and noise handling | How spectral lines and glitches affect CW sensitivity; veto methodology |
| Adapting to noise distribution shifts (PRD 107 084046) | Wildberger et al. | 2023 | Domain adaptation for GW inference under PSD changes | Methodology for quantifying distribution shift impact on ML models |
| Effects of DQ vetoes on CBC searches | Davis et al. | 2021 | DQ veto impact on search sensitivity (up to 20% improvement) | Veto efficiency and deadtime metrics; ROC analysis for DQ flags |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| PyTorch | >= 2.2 | Model inference on O4 data | Same as training; ensures checkpoint compatibility |
| timm | >= 1.0 | ViT-B/16 model loading | Same library used for training |
| GWpy | >= 3.0 | O4a strain data access, Q-transform, DQ flags | Standard LIGO data access library; `fetch_open_data()` for GWOSC |
| scikit-learn | >= 1.3 | Per-class metrics, confusion matrix | Same as Phase 2/3 evaluation |
| scipy.stats | spearmanr | Threshold correlation test | Standard non-parametric correlation |
| numpy | >= 1.24 | Bootstrap resampling, array operations | Same as Phase 2/3 |
| matplotlib | >= 3.7 | Degradation plots, veto ROC curves, CW comparison figures | Same as Phase 2/3 |
| pandas | >= 2.0 | Metadata handling, comparison tables | Same as Phase 1 |
| aiohttp | >= 3.9 | Async spectrogram download from CDN | Same as Phase 1 pipeline |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| gwpy.segments.DataQualityFlag | O4a science mode segments | Identify valid observation time for duty cycle computation |
| gwpy.timeseries.StateVector | O4a data quality bit vector | Parse DQ flags from GWOSC |
| gwosc (Python package) | Programmatic GWOSC data discovery | Find O4a segment lists and data URLs |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
| --- | --- | --- |
| Duty cycle as CW metric | Actual CW upper limits via PyFstat/LALSuite | Much more compute-intensive and requires CW search expertise; duty cycle is a well-accepted proxy |
| Zooniverse CDN spectrograms | Generate from GWOSC strain via GWpy Q-transform | More compute but guaranteed preprocessing consistency |
| Spearman correlation for threshold test | Logistic regression on (N_train, ViT_advantage) | Spearman is simpler and sufficient for 23 data points |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| O4a spectrogram download (CDN) | 30-60 min (network I/O) | CDN rate limiting | Use existing async downloader with retry |
| O4a spectrogram generation (fallback) | 2-6 hours (CPU) | Q-transform computation | Parallelize across CPU cores |
| Model inference (both models on O4a) | 10-30 min (GPU) | Minimal | Batch inference on RTX 5090 |
| Bootstrap CIs (per-class, 10K resamples) | 5-10 min (CPU) | Minimal | Existing bootstrap_ci.py code |
| Duty cycle computation | < 5 min (CPU) | Minimal | Simple segment arithmetic |

**Installation / Setup:**
```bash
# gwosc package for programmatic O4a data discovery
pip install gwosc
# All other packages (gwpy, torch, timm, scikit-learn) already installed from prior phases
```

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| Re-evaluate O3 test set with loaded checkpoints | Checkpoint loading and preprocessing correctness | Run inference on O3 test set; compare with saved Phase 3 metrics | Exact match (< 1e-6 difference) with Phase 3 per-class F1 |
| O4 class distribution vs O3 | Whether O4 taxonomy maps to O3 | Count per-class O4 samples; compare with O3 distribution | Similar relative ordering; some classes may be absent or have new prevalence |
| Per-class sample size in O4 | Whether per-class bootstrap CIs are meaningful | Report N_test per class on O4 | Flag classes with < 10 O4 test samples as unreliable |
| Degradation symmetry | Both models should degrade, not just one | Compare CNN degradation vs ViT degradation | Both should show some degradation; if only one degrades, suspect preprocessing bug |
| Veto deadtime sanity | Vetoes should not remove > 50% of observation time | Compute tau_dead for both models | If > 50%, the confidence threshold is too aggressive |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| O3-trained CNN on O4 | All classes | O3 CNN confidently mislabels novel O4 glitches | Zevin et al. 2024, arXiv:2308.15530 |
| O4 classifier performance | O4-trained model | Wu et al. 2024 report improved O4 accuracy with updated classifier | arXiv:2401.12913 |
| DQ veto improvement for CBC | O3 DQ flags | Up to 20% improvement in detectable signals | Davis et al. 2021 |
| Maximum expected overall degradation | O3->O4 | < 20% overall F1 drop (contract criterion) | Based on O1->O2->O3 evolution in literature |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
| --- | --- | --- | --- |
| O3 metric reproduction | Re-inference on O3 test set | < 1e-6 | Phase 3 metrics.json values |
| Bootstrap CI stability | Run bootstrap twice with different seeds | CI width difference < 5% | Internal consistency |
| Duty cycle bounds | DC must be in [0, 1] | Exact | Physical constraint |
| Veto efficiency bounds | eta must be in [0, 1] | Exact | Physical constraint |

### Red Flags During Computation

- **O4 per-class F1 is higher than O3 for most classes:** Likely indicates O4 label set is biased toward easy/confident examples. Check label quality filter.
- **Both models show identical O4 performance:** Likely a preprocessing bug or both models collapsed to same predictions. Check distinct confusion matrices.
- **ViT advantage reverses for ALL classes on O4 (not just rare):** May indicate O4 images have systematic visual differences (different spectrogram generation pipeline). Verify O4 spectrograms visually.
- **Duty cycle is identical for ViT and CNN vetoes:** Models agree on all veto decisions. Check that per-class veto construction uses different predictions, not just different confidences.
- **Correlation p-value is very high (> 0.5) with wrong sign:** Threshold pattern does not hold on O4. This is the backtracking trigger -- report with caveat.

## Common Pitfalls

### Pitfall 1: O4 Spectrogram Preprocessing Mismatch

**What goes wrong:** O4 spectrograms downloaded from a different source or generated with different Q-transform parameters than O3 spectrograms will have different pixel distributions, making model predictions unreliable. The model sees "domain shift" that is actually a preprocessing artifact.

**Why it happens:** Gravity Spy periodically updates its spectrogram generation pipeline. O4 spectrograms may use different normalization, resolution, or Q-transform parameters than O3. If using GWOSC strain data to generate spectrograms, even small parameter differences (e.g., different `qrange` or `frange`) change the output images.

**How to avoid:** (1) If downloading from CDN: visually compare O3 and O4 spectrograms of the same glitch type to confirm consistency. (2) If generating from strain: use identical GWpy Q-transform parameters documented in Phase 1. (3) Before full evaluation, run O3 models on a small O4 batch and verify predictions are reasonable (not all same class, not all confidence ~0.5).

**Warning signs:** Model confidence distribution on O4 is bimodal or shifted relative to O3. All predictions cluster in 1-2 classes. Per-class O4 accuracy is near random (1/23) for all classes.

**Recovery:** Generate O4 spectrograms from scratch using the exact Phase 1 GWpy pipeline applied to GWOSC O4a strain data.

### Pitfall 2: O4 Taxonomy Mismatch

**What goes wrong:** O4 Gravity Spy taxonomy may include new classes or rename/merge existing classes. The O3-trained models can only predict the 23 O3 classes. If O4 glitches belong to classes not in the O3 taxonomy, they will be misclassified, and using O4 labels as ground truth for non-O3 classes inflates error rates misleadingly.

**Why it happens:** The LIGO detector undergoes upgrades between runs. O4 introduced new glitch morphologies (arXiv:2409.02831). The O4 Gravity Spy classifier (Wu et al. 2024) updated the taxonomy.

**How to avoid:** (1) Filter O4 evaluation set to only include glitches labeled with O3 class names. (2) Separately document any O4-only classes encountered and their prevalence. (3) Report the fraction of O4 glitches excluded due to taxonomy mismatch -- this is informative about distribution shift severity.

**Warning signs:** O4 metadata contains class labels not in the O3 class list. A large fraction (> 20%) of O4 glitches have unknown labels.

**Recovery:** If taxonomy mismatch is severe, map O4 classes to nearest O3 classes where physically motivated, and document the mapping. Any unmappable classes are excluded.

### Pitfall 3: CW Benefit Conflation with Overall Model Quality

**What goes wrong:** Claiming "ViT-based vetoes improve CW sensitivity" when the improvement comes from common classes (e.g., Blip, Tomte) that are not CW-critical. This would be a misleading claim -- the CW benefit must come specifically from CW-critical classes.

**Why it happens:** ViT outperforms CNN on overall macro-F1 due to common-class improvement. If veto construction uses all-class predictions, the overall veto quality will favor ViT regardless of CW-critical class performance.

**How to avoid:** (1) Identify CW-critical classes first (Scattered_Light, Violin_Mode, Low_Frequency_Lines, 1080Lines, Whistle -- classes with spectral-line-like or narrowband character in the 20-2000 Hz CW band). (2) Compute CW metrics using ONLY vetoes from CW-critical classes. (3) Report results for CW-critical classes and non-CW-critical classes separately.

**Warning signs:** CW duty cycle improvement disappears when restricted to CW-critical classes. Improvement is driven entirely by one very common class.

**Recovery:** Narrow the claim to specific class subsets. The backtracking trigger acknowledges this: "If CW improvement is not measurable, narrow to specific line-like glitch classes."

### Pitfall 4: Insufficient O4 Test Samples for Per-Class Statistics

**What goes wrong:** Some O3 classes may have very few O4 examples (especially rare classes that were already rare in O3). Per-class F1 with N < 10 has enormous variance and bootstrap CIs are uninformative.

**Why it happens:** Class prevalence varies between observing runs. Some glitch types are tied to specific instrumental conditions that may not exist in O4.

**How to avoid:** (1) Report per-class O4 sample size alongside all metrics. (2) Exclude classes with < 10 O4 test samples from the threshold correlation analysis. (3) For the threshold test, use only classes where both O3 and O4 have meaningful sample sizes. (4) Use BCa (bias-corrected accelerated) bootstrap for small samples instead of percentile bootstrap.

**Warning signs:** Bootstrap CIs span [0, 1] for multiple classes. Spearman correlation is driven by 1-2 extreme points.

**Recovery:** Report threshold correlation with and without small-sample classes. Use a sensitivity analysis showing how the correlation changes as the minimum sample size threshold varies.

### Pitfall 5: Confusing Distribution Shift with Label Noise

**What goes wrong:** O4 per-class F1 drops are attributed to distribution shift (morphological change) when they actually reflect worse O4 label quality (different volunteers, less mature ML classifier, fewer classifications per glitch).

**Why it happens:** O4 Gravity Spy is newer, with potentially lower label consensus than the well-curated O3 dataset. The O4 ML classifier was retrained and may have different biases.

**How to avoid:** (1) Filter O4 labels by the same quality thresholds used for O3 (ml_confidence > 0.9 or volunteer consensus). (2) Report label quality statistics for O4 vs O3 (average confidence, number of volunteer classifications per glitch). (3) If possible, examine a random subsample of O4 labels manually.

**Warning signs:** O4 F1 drops are uniform across all classes (suggests label noise, not class-specific morphological shift). O4 label confidence distribution is significantly lower than O3.

**Recovery:** Tighten the label quality filter for O4 (e.g., require 4/5 volunteer agreement instead of 3/5). Report results at multiple quality thresholds.

## CW-Critical Glitch Classes

This section identifies which Gravity Spy glitch classes are most relevant to continuous wave searches, based on their spectral characteristics and frequency band overlap with the CW search band (20-2000 Hz).

### Classes with Direct CW Impact

| Class | Frequency Band | CW Relevance | O3 N_train | O3 ViT-CNN F1 Diff | Notes |
| --- | --- | --- | --- | --- | --- |
| Scattered_Light | 10-120 Hz | HIGH: arch-like artefacts in low-frequency CW band | 68,160 | -0.092 (CNN better) | Most common class; ViT underperforms CNN |
| Violin_Mode | ~500 Hz + harmonics (1000, 1500, 2000) | HIGH: directly overlaps CW search frequencies; persistent narrowband | 274 | -0.139 (CNN better) | Moderate sample size; ViT underperforms CNN |
| Low_Frequency_Lines | 10-100 Hz | HIGH: persistent narrowband in low-frequency CW band | 2,853 | -0.086 (CNN better) | ViT underperforms CNN |
| 1080Lines | ~1080 Hz | MEDIUM: specific narrowband at known CW-relevant frequency | 341 | -0.294 (CNN better) | Small O4 sample likely; CNN better |
| Whistle | 200-4000 Hz | MEDIUM: brief narrowband wandering through CW band | 6,299 | +0.001 (essentially tied) | Very common; no model advantage |
| Power_Line | 60 Hz + harmonics | MEDIUM: fixed-frequency narrowband | 1,582 | +0.507 (ViT much better) | ViT dramatically better; CW-relevant |
| Low_Frequency_Burst | 10-50 Hz | LOW-MEDIUM: broadband transient in low-frequency CW region | 19,834 | -0.008 (tied) | Common; minimal CW impact per glitch |

### Key Observation for CW-Threshold Connection

**The ViT advantage and CW-critical classes partially overlap but not perfectly.** The most CW-critical classes (Scattered_Light, Violin_Mode, Low_Frequency_Lines) are ones where CNN actually outperforms ViT on O3. However, Power_Line (N_train = 1,582, ViT advantage = +0.507) is both CW-relevant and strongly ViT-advantaged. This means the CW benefit story must be class-specific, not blanket.

**Recommended framing:** "For the subset of CW-critical glitch classes where ViT outperforms CNN (specifically Power_Line and potentially others with N_train > 100), ViT-based vetoes provide measurably better CW data quality. For CW-critical classes where CNN outperforms ViT (Scattered_Light, Violin_Mode), the CW benefit comes from the existing CNN baseline."

### Glitch-to-CW Impact Mechanism

Glitches affect CW searches through two distinct mechanisms:

1. **Broadband transients** (Blip, Extremely_Loud, Low_Frequency_Burst): Increase the noise floor transiently. CW searches average over long durations, so transients are typically averaged out. Impact is minimal unless glitch rate is very high.

2. **Narrowband/line-like glitches** (Scattered_Light arches, Violin_Mode, Power_Line, Low_Frequency_Lines, 1080Lines, Whistle): These create excess power in narrow frequency bands that can mimic or contaminate CW signals. CW searches use spectral line lists to veto contaminated frequency bins, but imperfect glitch classification means some contaminated segments are not vetoed.

**The CW benefit from better classification comes from mechanism (2):** correctly identifying narrowband/line-like glitches enables cleaner frequency-domain vetoes, reducing false alarms in CW searches and potentially recovering observation time that would otherwise be discarded conservatively.

## Veto Generation Methodology

### Step 1: Define CW-Critical Glitch Set

Select glitch classes with known CW band contamination: Scattered_Light, Violin_Mode, Low_Frequency_Lines, 1080Lines, Power_Line, Whistle. For each class, document the frequency band affected.

### Step 2: Generate Per-Class Veto Segments

For each classifier (ViT, CNN) independently:
1. Run inference on all O4a glitches in the evaluation set
2. For each CW-critical class, identify glitches predicted to belong to that class with confidence above threshold
3. Create veto segments: [GPS_trigger - duration/2, GPS_trigger + duration/2] for each identified glitch
4. Merge overlapping segments per class

### Step 3: Compute Duty Cycle

For a selected O4a data segment (e.g., 1 week of continuous science mode):
1. Obtain total science-mode observation time T_total from GWOSC DQ flags
2. Subtract time covered by CW-critical veto segments to get T_clean
3. DC = T_clean / T_total
4. Compare DC_vit vs DC_cnn

### Step 4: Compute Veto Efficiency vs Deadtime (ROC)

Vary the classifier confidence threshold from 0.5 to 0.99:
1. At each threshold, compute veto efficiency (fraction of known CW-critical glitches vetoed)
2. Compute deadtime (fraction of observation time removed)
3. Plot ROC curve for ViT and CNN
4. Compare area under curve or efficiency at fixed deadtime

## Level of Rigor

**Required for this phase:** Controlled empirical evaluation with statistical significance testing.

**Justification:** This is an ML evaluation phase, not a theoretical derivation. The claims are empirical (does the threshold persist on O4? does duty cycle improve?). The rigor standard is clear statistical methodology with uncertainty quantification.

**What this means concretely:**

- All per-class F1 values must have bootstrap confidence intervals (>= 10,000 resamples)
- The threshold persistence test must have a p-value from Spearman correlation
- CW duty cycle improvement must be a computed number with uncertainty, not a qualitative argument
- Classes with insufficient O4 test samples must be flagged, not silently included
- The degradation analysis must compare both models on the same O4 test set (paired evaluation)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Single-run evaluation only | Cross-run distribution shift analysis | ~2023 (Zevin et al. acknowledged O4 shift) | Models must be tested across observing runs to claim robustness |
| Overall accuracy for DQ evaluation | Per-class metrics with rare-class focus | ~2024 (Wu et al. O4 classifier) | Rare-class performance matters for DQ applications |
| Manual DQ flag generation | ML-assisted DQ classification | ~2017 (Gravity Spy) | ML classifiers integrated into LIGO DQ pipeline |
| Category-based DQ vetoes only | ML classification + DQ vetoes combined | ~2021 (Davis et al.) | Up to 20% sensitivity improvement from combined approach |

**Superseded approaches to avoid:**

- **Evaluating only overall accuracy on O4:** The Phase 3 lesson (forbidden proxy scenario) applies equally to O4 evaluation. Overall accuracy hides per-class failures.
- **Claiming CW benefit from overall model improvement:** CW benefit must be traced to specific CW-critical classes, not overall model superiority.

## Open Questions

1. **Is there a public O4 Gravity Spy labeled dataset with spectrogram URLs?**
   - What we know: Zenodo 13904421 has volunteer classifications through July 2024. Gravity Spy is running on O4 via Zooniverse. Pre-made spectrogram images exist on the Zooniverse CDN.
   - What's unclear: Whether the O4 metadata includes CDN URLs in the same format as the O3 dataset. Whether ML classifications (as opposed to volunteer-only) are publicly available for O4.
   - Impact on this phase: Determines whether Approach 1 (CDN download) or Approach 2 (strain-based generation) is used.
   - Recommendation: Attempt Approach 1 first by querying the Zooniverse API or downloading the O4 volunteer classification dataset. If URLs are unavailable, fall back to Approach 2.

2. **Which O3 classes have changed morphology in O4?**
   - What we know: arXiv:2409.02831 documents O4a detector characterization. O4 classifier (Wu et al. 2024) shows lower confidence for Whistle, Wandering_Line, Violin_Mode, Scratchy.
   - What's unclear: Exact per-class morphological changes. Whether any O3 classes are absent in O4.
   - Impact on this phase: Classes with severe morphological change will show degradation for both models -- this is distribution shift, not model failure.
   - Recommendation: Document per-class O4 sample sizes and model confidence distributions. Classes with dramatic change are informative for the paper.

3. **Is duty cycle the right CW metric, or should we compute actual upper limits?**
   - What we know: Duty cycle is standard and widely understood. Actual CW upper limits require running a full CW search pipeline (e.g., PyFstat, LALSuite).
   - What's unclear: Whether the duty cycle improvement is large enough to be meaningful for CW searches. A 1% duty cycle improvement may be negligible.
   - Impact on this phase: If duty cycle improvement is too small to be meaningful, the CW claim is weak.
   - Recommendation: Start with duty cycle. If the improvement is > 1%, it is reportable. If < 1%, consider narrowing to specific frequency sub-bands where CW-critical classes dominate.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| O4 spectrogram download (CDN) | URLs unavailable or format mismatch | Generate from GWOSC O4a strain data via GWpy | +2-6 hours compute; need GWOSC O4a trigger times |
| O4 threshold persistence | Distribution shift destroys correlation | Report as O3-specific finding with caveat; still publishable as a negative result | Minimal extra work; framing change in paper |
| CW duty cycle improvement | Too small to be measurable (< 0.1%) | Narrow to specific frequency sub-bands; or focus on veto efficiency at fixed deadtime | Reframe the CW analysis; may need to report null result |
| CW improvement for ViT-advantaged classes | ViT-advantaged classes are not CW-critical | Report the disconnect honestly; CW benefit comes from CNN baseline instead | Framing change; the threshold finding and CW finding become independent results |

**Decision criteria:**
- O4 evaluation: Proceed unless O4 data is completely inaccessible (unlikely given GWOSC O4a release). If threshold correlation p > 0.1 on O4, report as non-significant with honest interpretation.
- CW analysis: If duty cycle difference between ViT and CNN vetoes is < 0.1% (one part per thousand), the CW claim is null. Narrow to class-specific or frequency-band-specific analysis before giving up entirely.

## Caveats and Self-Critique

1. **Assumption that may be wrong:** I assume O4 Gravity Spy spectrogram images are accessible via the same CDN infrastructure as O3. If the Zooniverse platform has changed its image hosting, this could require significant pipeline adaptation.

2. **Alternative approach possibly dismissed too quickly:** Computing actual CW strain upper limits (rather than the duty cycle proxy) would be much more convincing to CW search experts. I dismissed it due to computational cost and pipeline complexity, but if the duty cycle improvement is marginal, the proxy may be too weak to support the claim. Consider PyFstat as a more compelling (but harder) alternative.

3. **Limitation I may be understating:** The CW-critical class analysis shows CNN outperforms ViT on most CW-relevant classes (Scattered_Light, Violin_Mode, Low_Frequency_Lines). The CW benefit may genuinely be null for ViT -- Power_Line is the main hope, and it is a single class. The paper framing should be prepared for this outcome.

4. **Simpler method overlooked?** For the threshold persistence test, instead of Spearman correlation across all classes, a simpler approach would be a paired sign test: for classes with N_train >= 100, does ViT beat CNN on O4 more often than not? This is more robust to outliers and easier to interpret.

5. **Would a CW expert disagree?** Yes, possibly. A CW expert would likely argue that duty cycle is too coarse a metric -- what matters is the noise floor in specific frequency bins, not just the total observation time. They might push for a frequency-resolved analysis. I recommend the planner consider a "stretch goal" of computing PSD improvement in CW-critical frequency bands after vetoing.

## Sources

### Primary (HIGH confidence)

- Soni et al. 2024, arXiv:2409.02831 -- LIGO O4a detector characterization; glitch landscape; data quality products
- LVK 2025, arXiv:2508.18079 -- O4a open data release documentation; GPS ranges; data quality flags
- GWOSC Data Sets page (https://gwosc.org/data/) -- O4a strain data access
- Phase 3 results: results/03-vit-rare-class/comparison_table.csv, metrics.json, statistical_summary.md

### Secondary (MEDIUM confidence)

- Wu/Raza et al. 2024, arXiv:2401.12913 -- O4 Gravity Spy classifier; taxonomy updates; distribution shift observations
- Zevin et al. 2024, arXiv:2308.15530 -- Gravity Spy lessons learned; O3 CNN limitations on O4 data
- Tenorio et al. 2021, arXiv:2111.12575 -- CW search methods review; spectral line handling
- Davis et al. 2021 -- DQ veto effects on CBC searches; veto efficiency methodology
- Wildberger et al. 2023, PRD 107 084046 -- Noise distribution shift adaptation for GW inference
- Gravity Spy volunteer classifications O4, Zenodo 13904421

### Tertiary (LOW confidence)

- Gravity Spy Zooniverse project page (https://www.zooniverse.org/projects/zooniverse/gravity-spy) -- O4 classification status
- GWOSC O3 spectral lines (https://gwosc.org/O3/o3speclines/) -- Reference for CW-critical frequencies (O4 lines page not yet public)
- LIGO DCC T1300876 -- Violin mode frequency summary

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH -- standard ML evaluation metrics and LIGO DQ metrics; no novel math needed
- Standard approaches: MEDIUM-HIGH -- O4 evaluation is straightforward; CW veto methodology is well-documented but novel for this project
- Computational tools: HIGH -- all tools already used in prior phases; GWpy O4a access documented
- Validation strategies: HIGH -- clear validation checks with concrete expected values
- O4 data availability: MEDIUM -- O4a strain data is public; Gravity Spy O4 labeled spectrograms with CDN URLs unconfirmed
- CW sensitivity quantification: MEDIUM -- duty cycle is a well-accepted proxy but may understate/overstate actual CW impact

**Research date:** 2026-03-17
**Valid until:** Stable for ML evaluation methodology. O4 data availability may improve (O4b/O4c releases expected 2026). CW methodology stable.
