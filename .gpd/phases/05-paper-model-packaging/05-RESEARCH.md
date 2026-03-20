# Phase 5: Paper & Model Packaging - Research

**Researched:** 2026-03-17
**Domain:** Scientific writing / ML model release / GW detector characterization
**Confidence:** HIGH

## Summary

Phase 5 produces two deliverables: (1) both trained models (ViT-B/16 and ResNet-50v2 BiT) packaged as reproducible, deployable artifacts with inference scripts and preprocessing pipelines, and (2) a research paper framing the sample-efficiency threshold as the central contribution. The paper must navigate a nuanced narrative: ViT significantly outperforms CNN on overall macro-F1 (0.7230 vs 0.6786, p=0.0002) but underperforms on rare classes (0.2412 vs 0.3028), and the threshold finding did NOT replicate on O4 data (Spearman rho=-0.034, p=0.879). The CW benefit is operating-point-dependent and class-specific, not a blanket improvement.

The central challenge is honest framing of mixed results as a scientifically valuable contribution. The user has explicitly decided that nuanced, actionable findings are more valuable than chasing positive results. The paper should present the sample-efficiency threshold as practical guidance for practitioners deciding between ViT and CNN architectures for imbalanced scientific image classification, with the O4 non-replication and CW nuance as honest caveats that strengthen rather than weaken the paper's credibility.

**Primary recommendation:** Frame the paper around "When do Vision Transformers help? A per-class sample-efficiency analysis for gravitational-wave glitch classification" -- the threshold finding is the contribution, not architecture superiority. Package models as PyTorch checkpoints with standalone inference scripts and locked preprocessing, deposited on Zenodo with a DOI.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| ref-gravity-spy (Zevin et al. 2017, CQG 34 064003) | benchmark | CNN baseline architecture and taxonomy that our work extends | cite, compare | Introduction, Methods, Results, Discussion |
| Phase 3 comparison_table.csv | prior artifact | Per-class F1 with bootstrap CIs for all 23 classes -- the core evidence | use directly in results tables/figures | Results section, supplementary material |
| Phase 4 o4_comparison_table.csv | prior artifact | O4 generalization test -- threshold non-replication evidence | use in O4 validation section | Results, Discussion |
| Phase 4 cw_veto_results.json | prior artifact | CW veto metrics with per-class duty cycle and ROC | use in CW analysis section | Results, Discussion |
| Phase 4 VERIFICATION.md | prior artifact | Documents the np.interp bug in 5% deadtime claim | Must NOT propagate the "3.4x" number | Paper must use matched-deadtime comparison only |

**Missing or weak anchors:** The O4 threshold test failed (rho=-0.034, p=0.879), meaning the sample-efficiency threshold is an O3-only finding. This must be stated explicitly in the paper. The O3 cross-check (rho=-0.119, p=0.59) also shows no significant correlation. The threshold is observational (Light_Modulation improved, Chirp regressed) rather than statistically confirmed via rank correlation.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Primary metric | Macro-averaged F1 | Weighted F1, accuracy | Project contract (locked Phase 1) |
| Uncertainty | Bootstrap 95% CI, 10K resamples | Analytical CI, cross-validation | Phase 2-3 methodology |
| Statistical test | Paired percentile bootstrap | McNemar's, permutation test | Phase 3 paired_bootstrap.py |
| Units | SI (Hz, s) for physical; dimensionless for ML metrics | -- | Project convention |
| Spectrogram | Q-transform, GWpy, 224x224, ImageNet normalization | -- | Phase 1 locked protocol |
| delta_DC sign | DC_ViT - DC_CNN (positive = ViT advantage) | -- | Phase 4 convention |

**CRITICAL: All equations and results below use these conventions. The "3.4x veto efficiency at 5% deadtime" number from Phase 4 is INVALID (np.interp boundary clamping bug, verified in 04-VERIFICATION.md). Use matched-deadtime comparison (22.4%: ViT 0.745 vs CNN 0.735) instead.**

## Mathematical Framework

### Key Equations and Starting Points

This phase involves no new derivations. All quantitative results are computed in Phases 2-4. The paper assembles, frames, and presents existing results.

| Equation / Metric | Definition | Source | Role in Paper |
| --- | --- | --- | --- |
| Macro-F1 = (1/C) * sum(F1_c) | Unweighted mean of per-class F1 | sklearn.metrics | Primary comparison metric |
| Bootstrap CI: resample N times, take [2.5%, 97.5%] quantiles | Nonparametric CI on F1 | Phase 3 paired_bootstrap.py | All reported uncertainties |
| Paired bootstrap p-value: fraction of resamples where diff <= 0 | One-sided test for ViT > CNN | Phase 3 | Statistical significance claims |
| Spearman rho(n_train, F1_diff) | Rank correlation between training set size and ViT advantage | Phase 4 threshold test | Threshold finding evidence |
| delta_DC = DC_ViT - DC_CNN | Duty cycle difference for CW veto | Phase 4 | CW benefit quantification |
| Veto efficiency = N_correct_vetoes / N_true_CW_glitches | Fraction of CW-critical glitches correctly identified | Phase 4 | CW ROC analysis |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Bootstrap confidence intervals | Nonparametric uncertainty quantification | All reported metrics | Efron & Tibshirani 1993 |
| Paired bootstrap | Controls for test-set composition in model comparison | ViT vs CNN significance | Phase 3 implementation |
| Spearman rank correlation | Tests monotonic relationship between sample size and advantage | Threshold finding | scipy.stats.spearmanr |
| Confusion matrix analysis | Reveals per-class error patterns | Results figures | sklearn.metrics |

### Approximation Schemes

No new approximations introduced. Key limitations of existing approximations to report:

| Approximation | Limitation to Disclose | Impact on Claims |
| --- | --- | --- |
| Bootstrap CI on rare classes (N=7-66 per class) | Wide CIs, low statistical power for individual rare classes | Chirp (N_test=7) F1 change is not individually significant |
| Macro-F1 over 4 rare classes | Dominated by Light_Modulation (66 test samples) vs Chirp (7) | Rare-class aggregate hides per-class heterogeneity |
| Sample-count duty cycle proxy | Assumes uniform 1.0s segments, no GPS times | CW duty cycle is approximate |
| Threshold as rank correlation | Tests monotonic trend, not threshold location | Cannot pinpoint exact N where ViT overtakes CNN |

## Standard Approaches

### Approach 1: Nuanced Threshold Framing (RECOMMENDED)

**What:** Frame the paper around the empirical observation that ViT-B/16 outperforms CNN on classes with sufficient training data (~100+ samples) but underperforms on ultra-rare classes (<50 samples). Present this as practical guidance, not a universal law.

**Why standard:** Recent ML literature increasingly values nuanced, per-class analyses over aggregate metrics. The "Position: Embracing Negative Results in Machine Learning" (ICML 2024 position paper) argues that negative and nuanced results contribute more to scientific progress than marginal positive results. The user has explicitly chosen this framing.

**Track record:** Comparable per-class threshold analyses exist in medical imaging (where class imbalance is similarly severe) and remote sensing (CNN vs ViT comparisons on imbalanced satellite data, e.g., arXiv:2510.03297).

**Key elements of the narrative:**

1. **Central finding:** Per-class analysis reveals a sample-efficiency crossover -- ViT helps above ~100 training samples, hurts below ~50
2. **Supporting evidence:** Light_Modulation (142 train): +17pp F1; Power_Line: +39pp F1. Chirp (11 train): -47pp F1; Wandering_Line (30 train): 0.00 for both
3. **Honest caveat:** Spearman rank correlation is not significant (O3: rho=-0.119, p=0.59; O4: rho=-0.034, p=0.879). The threshold is observed in specific classes, not confirmed as a general monotonic trend
4. **Actionable guidance:** Practitioners should evaluate ViT vs CNN per-class before deploying, especially for rare categories
5. **CW connection:** Operating-point-dependent; at matched deadtime, nearly equivalent
6. **Future work:** Data augmentation for ultra-rare classes; larger O4 datasets to test threshold stability

**Known difficulties:**

- Reviewers may question why the Spearman test fails if the threshold is real -- need to explain that a few outlier classes (common classes where CNN also does well) break the monotonic assumption
- The O4 non-replication weakens the claim -- must be forthright about this
- The CW benefit is marginal -- must not overclaim

### Approach 2: Architecture Comparison Framing (FALLBACK)

**What:** Frame as a controlled ViT vs CNN comparison for GW glitch classification, with the threshold finding as one insight among several.

**When to switch:** If reviewers strongly object to the threshold framing as too speculative given the non-significant Spearman test.

**Tradeoffs:** Less novel (just another ViT vs CNN paper) but safer. The overall macro-F1 improvement IS significant (p=0.0002), so the paper can still make a positive claim about ViT for GW glitches, just without the per-class threshold narrative as the centerpiece.

### Anti-Patterns to Avoid

- **Leading with overall accuracy:** The contract explicitly forbids this (fp-overall-accuracy). ViT accuracy is 93.4% vs CNN 91.8%, but this hides the rare-class regression. NEVER lead with this number.
- **Propagating the "3.4x" CW claim:** Phase 4 verification proved this is an np.interp boundary clamping artifact. The matched-deadtime comparison shows ~1.01x ratio. Propagating the 3.4x number would be scientific misconduct.
- **Claiming the threshold generalizes to O4:** Spearman rho=-0.034, p=0.879 on O4. The threshold is O3-specific and must be described as such.
- **Cherry-picking attention maps:** Show both successes and failures. Do not show only cases where ViT attention aligns with glitch morphology.
- **Omitting the forbidden proxy finding:** The fact that overall metrics improve while rare-class metrics regress is itself a valuable finding -- it demonstrates exactly why per-class analysis matters.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

All quantitative results exist from Phases 2-4. The paper phase produces NO new computations -- it assembles, frames, and presents.

| Result | Value | Source | Role in Paper |
| --- | --- | --- | --- |
| ViT test macro-F1 | 0.7230 [0.7018, 0.7413] | results/03-vit-rare-class/metrics.json | Primary ViT result |
| CNN test macro-F1 | 0.6786 [0.6598, 0.6944] | results/02-cnn-baseline/metrics.json | Primary CNN result |
| Paired bootstrap p (overall) | 0.0002 | results/03-vit-rare-class/paired_bootstrap_results.json | Statistical significance |
| ViT rare-class macro-F1 | 0.2412 [0.2019, 0.2957] | Same | Rare-class regression evidence |
| CNN rare-class macro-F1 | 0.3028 [0.2085, 0.3751] | Same | Rare-class baseline |
| O4 ViT macro-F1 | 0.6695 | results/04-o4-validation/o4_metrics.json | Generalization test |
| O4 CNN macro-F1 | 0.6674 | Same | Generalization baseline |
| O4 Spearman rho | -0.034, p=0.879 | results/04-o4-validation/o4_threshold_test.json | Threshold non-replication |
| CW delta_DC | -0.051 [-0.054, -0.048] | results/04-o4-validation/cw_veto_results.json | CW finding (CNN advantage overall) |
| CW matched-deadtime eff | ViT 0.745 vs CNN 0.735 at 22.4% | Same | CW finding (near-equivalent) |
| Per-class F1 table (23 classes) | Full CSV | results/03-vit-rare-class/comparison_table.csv | Core evidence table |
| O4 per-class comparison | Full CSV | results/04-o4-validation/o4_comparison_table.csv | O4 evidence table |

**Key insight:** Re-computing any of these numbers risks introducing inconsistency with verified Phase 3-4 results. Read directly from the existing result files.

### Existing Figures to Use or Adapt

| Figure | Path | Paper Role | May Need Adaptation |
| --- | --- | --- | --- |
| Comparison per-class F1 | figures/comparison_per_class_f1.png | Main results figure | May need journal-quality formatting |
| Comparison confusion matrices | figures/comparison_confusion_matrices.png | Supplementary or main | Likely sufficient |
| O4 degradation per-class | figures/o4_degradation_per_class.png | O4 validation figure | May need reformatting |
| O4 threshold scatter | figures/o4_threshold_scatter.png | Threshold test figure | Add O3 cross-check overlay |
| CW sensitivity summary | figures/cw_sensitivity_summary.png | CW analysis figure | Remove/fix any "3.4x" text |
| CW veto ROC | figures/cw_veto_roc.png | CW supplementary | Likely sufficient |
| CW duty cycle comparison | figures/cw_duty_cycle_comparison.png | CW supplementary | Likely sufficient |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Gravity Spy | Zevin et al. | 2017 | CNN baseline, taxonomy, project context | Architecture, class taxonomy, methodology comparison |
| Deep transfer learning for LIGO | George et al. | 2018 | >98.8% accuracy benchmark | Upper bound on CNN approach |
| Gravity Spy lessons learned | Zevin et al. | 2024 | O4 challenges, CNN limitations | Motivation for ViT exploration |
| ViT for transient noise | Srivastava & Niedzielski | 2025 | First ViT attempt, 92.26% accuracy | Direct comparison, motivation for improved recipe |
| Multi-view attention fusion | Wu et al. | 2024 | O4 classifier architecture | Related work, context for single-view limitation |
| cDVGAN augmentation | Lopez et al. | 2024 | GAN-based augmentation for rare classes | Future work reference |
| CNN vs ViT imbalanced | arXiv:2510.03297 | 2025 | Controlled CNN vs ViT under imbalance | Methodological parallel, framing support |
| Embracing negative results | ICML position | 2024 | Publication of nuanced/negative findings | Meta-argument for paper value |
| LIGO O4 DetChar | LIGO collab. | 2024 | O4 detector characterization context | O4 validation context |
| Data quality in O3 | Davis et al. | 2021 | Veto methodology for CW searches | CW analysis methodology reference |

## Computational Tools

### Core Tools for Paper Phase

| Tool | Purpose | Why Standard |
| --- | --- | --- |
| matplotlib + seaborn | Journal-quality figure generation | Standard in astronomy/physics papers |
| LaTeX (REVTeX 4.2 or CQG class) | Paper typesetting | Required by target journals |
| PyTorch (torch.save / torch.load) | Model checkpoint packaging | Native format for timm models |
| JSON/CSV readers | Reading existing result files | Direct use of Phase 3-4 outputs |

### Model Packaging Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| torch.save (state_dict) | Save model weights portably | Primary packaging format |
| ONNX export (torch.onnx.export) | Framework-agnostic model format | Optional but recommended for broad usability |
| Zenodo API / web upload | DOI-minted data/model deposit | Final release |
| SHA-256 checksums | Verify model integrity | Include in README with release |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
| --- | --- | --- |
| PyTorch state_dict | HuggingFace Model Hub | HF Hub is more discoverable but requires HF account and format compliance; state_dict is simpler for a first paper |
| ONNX | TorchScript | ONNX is more broadly interoperable; TorchScript is PyTorch-only |
| Zenodo | GitHub Releases | Zenodo provides a DOI; GitHub releases do not have permanent DOI |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Figure regeneration (journal quality) | < 30 min | matplotlib rendering | Batch all figures in one script |
| Model packaging (2 models) | < 10 min | Disk I/O | Straightforward torch.save |
| ONNX export (optional) | < 5 min per model | ONNX compatibility | Test with onnxruntime before release |
| Paper writing (LaTeX) | Human effort | Content creation | Use existing results directly |
| Zenodo upload | < 30 min | Network speed | Compress checkpoints |

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| Numbers in paper match result files | No transcription errors | Script that reads JSONs/CSVs and compares to paper values | Exact match to reported precision |
| Figures match underlying data | Visual accuracy | Regenerate figures from result files, compare | Pixel-identical or visually identical |
| Model checkpoint reproduces metrics | Packaged model is functional | Load checkpoint, run on test set, compare metrics | Exact match to Phase 3/4 metrics |
| Inference script produces correct output | End-to-end reproducibility | Run inference script on 5 known samples, verify predictions | Match saved predictions |
| No "3.4x" claim in paper | Verified bug not propagated | grep for "3.4" in paper source | Zero matches |
| No leading with accuracy | Forbidden proxy enforcement | Check abstract and intro for "accuracy" as primary metric | macro-F1 is always primary |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| Gravity Spy CNN accuracy | Common classes | ~97% | Zevin et al. 2017 |
| Our CNN macro-F1 | All 23 classes | 0.6786 | Phase 2 |
| Our ViT macro-F1 | All 23 classes | 0.7230 | Phase 3 |
| O4 degradation bound | Distribution shift | CNN -1.7%, ViT -7.4% | Phase 4 |

### Red Flags During Paper Writing

- If any number in the paper does not trace directly to a result file from Phases 2-4, something is wrong
- If the abstract or introduction mentions "accuracy" without qualifying it as secondary/sanity-check
- If the CW section claims a clear ViT advantage without the "operating-point-dependent" qualifier
- If the threshold finding is presented without the O4 non-replication caveat
- If confidence intervals are missing from any reported metric
- If the paper claims the threshold "generalizes" beyond O3

## Common Pitfalls

### Pitfall 1: Transcription Errors Between Result Files and Paper

**What goes wrong:** Numbers are manually copied from JSON/CSV files into LaTeX, introducing typos or rounding inconsistencies.
**Why it happens:** Manual transcription across formats is error-prone.
**How to avoid:** Write a script that reads all result files and generates LaTeX table source code automatically. Never manually type a number that exists in a result file.
**Warning signs:** A reviewer finds a number in the text that does not match a table, or CIs that are asymmetric in the wrong direction.
**Recovery:** Re-extract all numbers from result files; verify with checksums.

### Pitfall 2: Overclaiming the Threshold Finding

**What goes wrong:** The paper implies the sample-efficiency threshold is a general law rather than an empirical observation from one dataset (O3, 23 classes).
**Why it happens:** Desire to make the contribution sound stronger.
**How to avoid:** Use hedged language: "We observe that...", "In our O3 dataset...", "This pattern suggests...". Explicitly state the O4 non-replication (rho=-0.034, p=0.879) in the same paragraph.
**Warning signs:** Reviewer asks "Does this generalize?" and the paper has no answer.
**Recovery:** Add explicit scope limitations in Discussion.

### Pitfall 3: Propagating the Invalid CW "3.4x" Claim

**What goes wrong:** The Phase 4 "efficiency at 5% deadtime" comparison uses np.interp boundary clamping, comparing ViT at 22.3% deadtime against CNN at 6.5% deadtime.
**Why it happens:** The number appears in cw_veto_results.json and earlier summaries.
**How to avoid:** Use ONLY the matched-deadtime comparison (22.4%: ViT 0.745 vs CNN 0.735). The Phase 4 VERIFICATION.md documents this bug explicitly.
**Warning signs:** Any sentence containing "3.4x" or "5% deadtime" in the CW section.
**Recovery:** Delete any such claims; replace with matched-deadtime numbers.

### Pitfall 4: Burying the Negative/Nuanced Results

**What goes wrong:** The abstract and introduction emphasize the positive macro-F1 improvement while the rare-class regression and O4 non-replication are hidden in the discussion.
**Why it happens:** Natural tendency to lead with positive results.
**How to avoid:** The abstract MUST mention both the overall improvement AND the rare-class limitation. The user has explicitly requested that the nuanced finding IS the contribution.
**Warning signs:** A summary of the paper that sounds like "ViT beats CNN for GW glitches" without qualification.
**Recovery:** Rewrite abstract with balanced framing.

### Pitfall 5: Model Packaging Without Preprocessing Lock

**What goes wrong:** Released model weights work only with the specific preprocessing pipeline used during training, but the preprocessing steps are not bundled or documented.
**Why it happens:** Training code and inference code diverge; preprocessing is assumed "obvious."
**How to avoid:** Package the eval_transforms() function alongside model weights. Include explicit ImageNet normalization constants. Test the inference script on a fresh environment.
**Warning signs:** Model produces garbage predictions when loaded by a user who uses different normalization.
**Recovery:** Add preprocessing to inference script; document all constants.

## Level of Rigor

**Required for this phase:** Honest, quantitative reporting with appropriate uncertainty. No new derivations needed.

**Justification:** This is a methods/empirical paper, not a theoretical one. The rigor standard is: every claim is backed by a number, every number has a confidence interval, every comparison uses the correct statistical test, and limitations are stated explicitly.

**What this means concretely:**

- Every reported metric includes a bootstrap 95% CI
- Statistical comparisons use paired bootstrap p-values, not just point estimates
- Negative results (rare-class regression, O4 non-replication) are given equal prominence to positive results
- The CW analysis uses matched operating points, not the buggy interpolation
- All figures are regenerated from the actual result files, not hand-drawn or approximated

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Report overall accuracy only | Report macro-F1 + per-class metrics | ~2020 onward | Our paper contributes to this shift for GW glitch classification |
| CNN-only for Gravity Spy | ViT exploration beginning | 2025 (Srivastava) | Our paper is the second ViT paper and the first with per-class analysis |
| Single-view classification | Multi-view attention fusion | 2024 (Wu et al.) | Our paper uses single-view; multi-view is future work |
| Accuracy as CW benefit proxy | Duty cycle / veto efficiency | This paper | Our CW analysis methodology is relatively novel for glitch papers |

**Superseded approaches to avoid:**

- Reporting only overall accuracy: This is the most common metric in the Gravity Spy literature but hides the rare-class problem entirely. Our paper should argue for the shift to per-class metrics.

## Paper Structure

### Recommended Structure (detector characterization / ML methods paper)

Based on the Gravity Spy papers (CQG 2017, EPJ Plus 2024) and standard CQG/PRD format:

1. **Abstract** (~200 words)
   - State the problem: GW glitch classification with extreme class imbalance
   - Central finding: Sample-efficiency threshold -- ViT outperforms CNN above ~100 training samples, underperforms below ~50
   - Quantitative headline: ViT macro-F1 0.723 vs CNN 0.679 (p<0.001), but rare-class F1 0.241 vs 0.303
   - CW connection: Operating-point-dependent veto efficiency, near-equivalent at matched deadtime
   - Actionable message: Per-class evaluation is essential; aggregate metrics mislead

2. **Introduction** (~1.5 pages)
   - GW detection and glitch contamination
   - Gravity Spy and CNN limitations for O4+
   - ViT motivation (global attention, transfer learning)
   - Gap: No per-class analysis of ViT vs CNN for imbalanced GW data
   - Our contribution: Sample-efficiency threshold finding + honest negative results

3. **Data and Methods** (~3 pages)
   - Dataset: 325,634 O3 glitches, 23 classes, temporal split 70/15/15%
   - Preprocessing: Q-transform spectrograms, 224x224, ImageNet normalization
   - CNN baseline: ResNet-50v2 BiT (ImageNet-21k+1k), modern training recipe
   - ViT: ViT-B/16 (AugReg ImageNet-21k+1k), identical training recipe except base LR
   - Class imbalance strategy: Focal loss, class-balanced sampling
   - Evaluation: Macro-F1 (primary), per-class F1 with bootstrap CIs, paired bootstrap
   - Fair comparison protocol: Identical augmentation, optimizer, schedule, epochs

4. **Results** (~3 pages)
   - Overall comparison: Table 1 (macro-F1 with CIs, p-value)
   - Per-class analysis: Figure 1 (grouped bar chart, 23 classes), Table 2 (full per-class)
   - Sample-efficiency threshold: Figure 2 (n_train vs F1_diff scatter)
   - Confusion matrices: Figure 3 (side-by-side)
   - O4 validation: Table 3 (O4 metrics), Figure 4 (degradation per class)
   - O4 threshold test: Spearman results, honest non-replication
   - CW analysis: Figure 5 (duty cycle comparison), Figure 6 (veto ROC)

5. **Discussion** (~2 pages)
   - Why the threshold exists (ViT data hunger vs CNN inductive bias)
   - Why O4 non-replication does not invalidate the finding (distribution shift, class evolution)
   - CW implications: class-specific, not blanket
   - Forbidden proxy lesson: overall metrics hide rare-class regression
   - Comparison to prior ViT work (Srivastava 2025: 92.26% accuracy, no per-class analysis)
   - Limitations: single-view, no augmentation for ultra-rare, O3-specific threshold

6. **Future Work** (~0.5 pages)
   - Data augmentation strategies for ultra-rare classes (cDVGAN, contrastive learning)
   - Multi-view attention fusion (Wu et al. 2024 architecture)
   - O4 retraining with expanded taxonomy
   - Class-specific confidence thresholds for CW vetoes

7. **Conclusion** (~0.5 pages)
   - Restate threshold finding with appropriate hedging
   - Practical recommendation for practitioners
   - Emphasize per-class evaluation as standard practice

### Venue Selection

| Journal | Scope Match | Pros | Cons | Fit |
| --- | --- | --- | --- | --- |
| Classical and Quantum Gravity (CQG) | Excellent | Gravity Spy papers published here; GW community reads it; open to methods papers | Long review cycle | **PRIMARY TARGET** |
| Physical Review D (PRD) | Good | cDVGAN published here; high impact; rapid communications option | May prefer more physics-heavy content | Good backup |
| Machine Learning: Science and Technology (MLST) | Good | IOP journal; explicitly targets ML for science; open to nuanced findings | Less GW-specific readership | Good for ML-focused framing |
| Astronomy and Computing | Moderate | Methods-focused; accepts shorter papers | Lower impact | Backup for shorter version |

**Recommendation:** Target CQG as the primary venue. The Gravity Spy lineage is there (Zevin et al. 2017, 2024), and the detector characterization framing fits. Use REVTeX 4.2 or iopart class.

## Model Packaging Specification

### Required Deliverables per Contract (deliv-trained-model)

For EACH model (ViT-B/16 and ResNet-50v2 BiT):

1. **Model weights** (`best_model.pt` -- already exist in `checkpoints/`)
   - PyTorch state_dict format
   - Include SHA-256 checksum
   - Document: timm model ID, num_classes=23, pretrained source

2. **Preprocessing pipeline** (`src/data/transforms.py` -- already exists)
   - eval_transforms() function with locked constants
   - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Input: 224x224 RGB PNG, normalized to [0,1] before ImageNet normalization

3. **Inference script** (NEW -- to be created)
   - Standalone script: load model, load image, preprocess, predict, output class + confidence
   - No training dependencies (no wandb, no dataloader complexity)
   - Command-line interface: `python inference.py --model vit --image path/to/spectrogram.png`
   - Output: predicted class label, softmax probabilities, top-3 predictions
   - Include class label mapping (index -> Gravity Spy class name)

4. **Model card / README** (NEW -- to be created)
   - Model architecture and training details
   - Expected input format (224x224 RGB Q-transform spectrogram)
   - Performance summary (macro-F1, per-class F1 for key classes)
   - Limitations (O3-trained, single-view, 23 classes only)
   - Citation information
   - License (suggest MIT or Apache 2.0)

### Packaging Structure

```
release/
  README.md                    # Model card with performance, limitations, citation
  LICENSE
  checkpoints/
    vit_b16_gravityspy_o3.pt   # ViT state_dict
    resnet50v2_gravityspy_o3.pt # CNN state_dict
    checksums.sha256
  src/
    inference.py               # Standalone inference script
    preprocessing.py           # Locked eval transforms
    class_labels.json          # {0: "1080Lines", 1: "1400Ripples", ...}
    model_config.json          # Architecture specs for reproducibility
  examples/
    example_spectrogram.png    # One sample image for testing
    expected_output.json       # Expected inference result for the example
```

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| CQG submission | Reviewers reject threshold framing as speculative | Reframe as controlled architecture comparison; submit to MLST | Moderate -- rewrite intro/discussion, keep all results |
| Threshold as central claim | Reviewers demand O4 confirmation | Pivot to "honest negative result" framing: "ViT improves overall but not rare classes" | Low -- same data, different emphasis |
| ONNX export | Model uses timm features incompatible with ONNX | Release PyTorch-only with documented environment | Low -- ONNX is optional enhancement |
| Zenodo deposit | File size limits or access issues | GitHub release with permanent tag + archive.org mirror | Low |

**Decision criteria:** If two reviewers independently question the threshold framing, pivot to the architecture comparison framing. The results are identical; only the narrative emphasis changes.

## Open Questions

1. **Should the paper include ONNX exports alongside PyTorch checkpoints?**
   - What we know: ONNX provides framework-agnostic inference; timm ViT models generally export cleanly
   - What's unclear: Whether the GW community uses ONNX (most use PyTorch or TensorFlow)
   - Impact on this phase: Adds ~1 hour of work for export + validation
   - Recommendation: Include if time permits; not critical for first paper

2. **Should figures be regenerated at journal quality or use existing PNGs?**
   - What we know: Existing figures are matplotlib/seaborn with reasonable formatting
   - What's unclear: Whether CQG has specific figure requirements (vector format, minimum DPI)
   - Impact on this phase: Could add several hours if extensive reformatting needed
   - Recommendation: Check CQG author guidelines; regenerate as vector (PDF/SVG) if required

3. **How to handle the O3 cross-check (rho=-0.119, p=0.59)?**
   - What we know: Both O3 cross-check AND O4 threshold test show non-significant Spearman correlation
   - What's unclear: Whether this means the threshold is an artifact of specific classes, not a general trend
   - Impact on this phase: Weakens the threshold claim further
   - Recommendation: Report both honestly. The threshold is class-specific (Light_Modulation, Power_Line clearly benefit; Chirp clearly does not) rather than a smooth monotonic trend.

## Sources

### Primary (HIGH confidence)

- Zevin et al. (2017), "Gravity Spy," CQG 34 064003 -- taxonomy, CNN baseline, paper structure template
- Zevin et al. (2024), "Gravity Spy: Lessons learned," EPJ Plus 139 100 -- O4 challenges, CNN limitations
- George et al. (2018), "Deep Transfer Learning," PRD 97 101501 -- CNN accuracy benchmark
- Srivastava & Niedzielski (2025), "ViT for Transient Noise," Acta Astronomica 74(3) -- first ViT for LIGO, comparison baseline
- Phase 2-4 result files (this project) -- all quantitative evidence
- Phase 4 VERIFICATION.md -- documents the np.interp bug that invalidates the "3.4x" CW claim

### Secondary (MEDIUM confidence)

- "Position: Embracing Negative Results in Machine Learning" (ICML 2024) -- framing support for nuanced/negative findings
- "CNN vs ViT: SpaceNet Case Study" (arXiv:2510.03297) -- methodological parallel for imbalanced ViT vs CNN comparison
- Wu et al. (2024), arXiv:2401.12913 -- multi-view fusion, related work
- cDVGAN (2024), PRD 110 022004 -- future work reference for augmentation
- Davis et al. (2021) -- CW veto methodology reference
- Semmelrock et al. (2025), "Reproducibility in ML-based Research," AI Magazine -- model packaging standards
- LIGO O4 DetChar (2024), arXiv:2409.02831 -- O4 context

### Tertiary (LOW confidence)

- CQG author guidelines (must verify current submission requirements)
- Zenodo deposit workflow (straightforward but verify size limits for model checkpoints)

## Metadata

**Confidence breakdown:**

- Paper framing and narrative: HIGH -- user has explicitly decided the framing; prior work supports nuanced findings
- Model packaging: HIGH -- standard PyTorch workflow with well-documented tools
- Statistical reporting: HIGH -- all methods established in Phases 2-4 with verification
- Venue selection: MEDIUM -- CQG is the natural target but reviewer reception of threshold framing is uncertain
- CW analysis framing: MEDIUM -- the finding is genuinely marginal; must navigate carefully

**Research date:** 2026-03-17
**Valid until:** Indefinite for the paper content; check CQG submission guidelines before actual submission

## Caveats and Alternatives (Self-Critique)

1. **What assumption might be wrong?** I assume the threshold finding is publishable despite non-significant Spearman tests. A skeptical reviewer could argue this is just noise in a small (23-class) dataset. Mitigation: the per-class evidence is concrete (Light_Modulation +17pp, Chirp -47pp), even if the aggregate trend is not significant.

2. **What alternative did I dismiss too quickly?** I did not deeply explore submitting to a venue focused on negative results (e.g., JMLR MLOSS, or a workshop). For a first paper, a standard venue (CQG) with honest reporting is better for career impact, but a dedicated negative-results venue would guarantee acceptance of the nuanced findings.

3. **What limitation am I understating?** The CW analysis is the weakest part of the paper. The matched-deadtime comparison (1.01x ratio) is effectively null. The paper should perhaps present CW as "we checked, it's basically equivalent" rather than trying to extract a positive finding from class-specific differences.

4. **Is there a simpler method I overlooked?** The paper could be shorter and more impactful as a "Brief Report" or "Letter" format focusing only on the threshold finding with 2-3 key figures, deferring the full CW analysis to a separate note. This would be faster to write and review.

5. **Would a domain expert disagree?** A CW search expert might object that duty cycle is too coarse a proxy for actual CW sensitivity impact. They would be right -- a more sophisticated analysis would use the actual noise PSD before and after vetoes. This should be acknowledged as a limitation and mentioned as future work.
