# Transformer-Based Rare Glitch Classification for LIGO Continuous Wave Searches

## What This Is

A machine learning research project comparing Vision Transformer (ViT-B/16) and CNN (ResNet-50v2 BiT) architectures on LIGO gravitational wave detector glitch spectrograms. The central finding is that architecture preference is **class-dependent**: ViTs excel on classes with distinctive spectral features (e.g., Power_Line: +0.507pp F1) while rare-class evidence is insufficient to conclude either architecture is superior. CW veto analysis shows matched-deadtime efficiency is approximately equal, with class-specific advantages. Deliverables are both trained models with inference pipelines and a CQG research paper.

## Current State (after v1.1)

Both milestones complete. The CQG manuscript is submission-ready with all 9 referee issues resolved. Key established results:

- **Architecture preference is class-dependent**, not systematic: ViT excels on Power_Line (+0.507pp F1), CNN more robust on ultra-rare classes
- **Rare-class comparison is underpowered:** aggregate power 0.20 at observed effect (-0.062 macro-F1); framed as "insufficient evidence"
- **Temporal split explains accuracy gap:** random-split CNN accuracy 95.4% vs temporal-split 91.8% (+3.6pp), within published 95-99% range
- **CW benefit is class-specific:** matched-deadtime efficiency ~equal; Power_Line is the dominant ViT CW advantage
- **O3→O4 generalization robust:** CNN -1.7%, ViT -7.4%, both <20% degradation

## Next Milestone Goals

- Submit revised manuscript to Classical and Quantum Gravity
- Consider Paper II directions: rare-class augmentation (cDVGAN, contrastive), frequency-resolved PSD for CW, per-class architecture routing

## Core Research Question

Can a Vision Transformer trained on LIGO O3 spectrograms outperform Gravity Spy's CNN at classifying rare glitch morphologies, and can this improvement translate to better data quality for continuous gravitational wave searches?

## Scoping Contract Summary

### Contract Coverage

- **claim-rare-improvement**: ViT achieves higher F1/recall on rare glitch classes than Gravity Spy CNN — verified by statistically significant improvement (p < 0.05) on held-out O3 test set
- **claim-cw-benefit**: Improved classification translates to measurable CW search data quality improvement — verified by duty cycle or upper limit comparison
- **False progress to reject**: Overall accuracy gain that doesn't extend to rare classes; qualitative CW benefit argument without quantitative demonstration

### User Guidance To Preserve

- **User-stated observables:** Per-class F1/recall on rare glitch classes (the smoking gun); CW sensitivity improvement metric
- **User-stated deliverables:** Trained ViT model with inference pipeline; research paper with method description, experimental results, CW analysis, and Gravity Spy comparison
- **Must-have references / prior outputs:** Gravity Spy CNN (Zevin et al. 2017) as the quantitative baseline
- **Stop / rethink conditions:** ViT matches but doesn't beat CNN on rare-class F1 after hyperparameter tuning; O4 validation shows catastrophic degradation; rare classes have <10 labeled examples even after augmentation

### Scope Boundaries

**In scope**

- ViT-based classifier for LIGO glitch spectrograms (Q-transform / omega scans)
- Training on O3 GWOSC data with Gravity Spy labels
- Evaluation focused on rare/low-count glitch classes (F1/recall metrics)
- O4 data as distribution-shift validation set
- Data quality veto generation for CW searches
- Line artifact flagging in CW frequency bands
- Trained model pipeline as deployable artifact
- Research paper documenting method and results

**Out of scope**

- Time-series transformers operating directly on strain data
- Real-time deployment or low-latency inference pipeline
- Full CW search pipeline implementation (Viterbi, Sideband, etc.)
- Non-LIGO detector data (Virgo, KAGRA)
- Compact binary coalescence detection or parameter estimation

### Active Anchor Registry

- **ref-gravity-spy**: Zevin et al., Classical and Quantum Gravity, 2017 (Gravity Spy)
  - Why it matters: Defines the CNN baseline classifier and glitch taxonomy that this project must beat on rare classes
  - Carry forward: planning, execution, verification, writing
  - Required action: read, compare, cite

### Carry-Forward Inputs

- O3 Gravity Spy labeled dataset (Zenodo 5649212): 325,634 glitches, 23 classes
- Both trained model checkpoints: ViT-B/16 (983MB), ResNet-50v2 BiT (270MB)
- O4a evaluation data: 38,587 glitches

### Skeptical Review (Updated after v1.0)

- **Weakest anchor:** CNN accuracy 91.81% below published ~97% — plausible O3/temporal-split explanation but no ablation confirms
- **Validated assumptions:** O3 catalog has sufficient labeled examples for most classes; O4 noise similar enough for <20% degradation
- **Confirmed competing explanation:** CNNs with modern training match or beat transformers on rare classes — the rare-class problem IS fundamentally data-limited
- **Confirmed disconfirming observation:** ViT shows no rare-class improvement; forbidden proxy scenario detected and documented
- **False progress rejected:** Overall accuracy improvement caught as forbidden proxy in Phase 3

### Open Contract Questions (for next milestone)

- Would data augmentation (cDVGAN, contrastive) resolve the rare-class gap?
- What morphological features make Power_Line especially suited to ViT attention?
- Is there a principled way to select architecture per class (ensemble or routing)?

## Requirements

### Validated

- [x] DATA-01: Download O3 Gravity Spy dataset (325,634 samples, 23 classes) -- v1.0
- [x] DATA-02: Generate spectrograms (pre-made images, Option C) -- v1.0
- [x] DATA-03: Temporal train/val/test split (70/15/15, 60s gap) -- v1.0
- [x] DATA-04: Rare-class audit (Chirp flagged: 19 total, 11 train) -- v1.0
- [x] MODL-01: CNN baseline (ResNet-50v2 BiT, macro-F1=0.6786) -- v1.0
- [x] MODL-02: Train ViT-B/16 (macro-F1=0.7230) -- v1.0
- [x] MODL-03: Focal loss + class-balanced sampling (identical recipe) -- v1.0
- [x] EVAL-01: Per-class F1 with bootstrap CIs for all 23 classes -- v1.0
- [x] EVAL-02: O4 validation <20% degradation (CNN: -1.7%, ViT: -7.4%) -- v1.0
- [x] DELV-01: Both models packaged with inference pipeline -- v1.0

### Partial (honest negative results)

- [~] EVAL-03: Rare-class significance test ran correctly; result p=0.88 (honest FAIL) -- v1.0
- [~] CWSS-01: CW benefit is class-specific, not blanket; matched-deadtime ~equal -- v1.0

### Validated (v1.1)

- [x] COMP-01: Random-split ablation confirms temporal split explains accuracy gap (95.4% vs 91.8%) -- v1.1
- [x] STAT-01: Power analysis shows all 8 rare-class comparisons underpowered (aggregate power 0.20) -- v1.1
- [x] STAT-02: Rare-class narrative reframed to "insufficient evidence" with quantitative power support -- v1.1
- [x] TEXT-01 through TEXT-05: All text revisions applied (class-dependent reframing, CW proxy, Davis citation, seed limitation, degradation) -- v1.1
- [x] TABL-01: CW Table 3 restructured with separate F1 Adv and DC Adv columns -- v1.1
- [x] TABL-02: O4 rounding corrected (ViT macro-F1 0.670) -- v1.1
- [x] DELV-02: Submission-ready CQG manuscript with all 9 referee issues resolved -- v1.1

### Active

- [ ] Rare-class data augmentation (cDVGAN, contrastive pretraining) for ultra-rare classes
- [ ] Frequency-resolved PSD analysis for Power_Line CW benefit
- [ ] Paper submission to Classical and Quantum Gravity

### Out of Scope

- Time-series transformers on raw strain -- different approach, separate study
- Real-time / low-latency inference pipeline -- research focus, not deployment
- Full CW search pipeline (Viterbi, Sideband) -- only quantify data quality impact
- Non-LIGO detectors (Virgo, KAGRA) -- different noise characteristics
- CBC detection or parameter estimation -- different signal type
- Rare-class augmentation beyond paper scope -- deferred to Paper II

## Research Questions

### Answered

- [x] Does a ViT outperform CNN on rare glitch class F1? -- **No.** ViT rare-class macro-F1 (0.2412) < CNN (0.3028), p=0.88. Architecture preference is class-morphology-dependent, not systematic.
- [x] How robust is the ViT to O3→O4 distribution shift? -- **Both models robust.** CNN: −1.7%, ViT: −7.4%, both <20%.
- [x] Does improved classification improve CW sensitivity? -- **Class-specific only.** Matched-deadtime efficiency ~equal; Power_Line is the main ViT CW advantage (+0.394 F1 on O4).
- [x] What class balancing strategy works best? -- **Focal loss + class-balanced sampling** is the standard; insufficient for ultra-rare classes (<50 samples).
- [x] Is the sample-efficiency threshold a robust finding? -- **No.** Spearman ρ=−0.034, p=0.879 on O4; the "threshold" was class-specific, not monotonic.

### Active

- [ ] Would rare-class-specific augmentation resolve the gap?
- [ ] What morphological features explain Power_Line's strong ViT advantage?
- [ ] Would a two-stage model (ViT common + specialized rare) be viable?

### Out of Scope

- Can transformers improve CBC event detection? — different signal morphology and search pipeline
- Does this generalize to Virgo/KAGRA data? — different detector noise characteristics, separate study

## Research Context

### Physical System

LIGO interferometric gravitational wave detectors (Hanford and Livingston). The data are strain time series h(t) contaminated by non-Gaussian noise transients ("glitches") of various morphological types. These glitches are classified by their time-frequency appearance in spectrograms. Rare glitch morphologies are poorly classified by existing methods and can contaminate long-duration continuous gravitational wave searches for signals from spinning neutron stars.

### Theoretical Framework

Signal processing and machine learning applied to gravitational wave detector characterization. The underlying physics is GW detector noise modeling, with glitches arising from environmental coupling, instrumental artifacts, and unknown sources. CW signals are quasi-monochromatic with frequency evolution governed by neutron star spin-down: f(t) = f₀ + ḟt + ½f̈t² + Doppler modulation from Earth's motion.

### Key Parameters and Scales

| Parameter | Symbol | Regime | Notes |
| --------- | ------ | ------ | ----- |
| Strain sensitivity | h | ~10⁻²³ /√Hz | O3 design sensitivity band |
| CW frequency range | f | 20-2000 Hz | Typical CW search band |
| Glitch duration | τ | 0.01-10 s | Varies by morphology |
| Spectrogram resolution | Δt × Δf | TBD | Q-transform parameter choice |
| Training set size | N_train | ~10⁵ | O3 Gravity Spy labeled glitches |
| Rare class threshold | N_rare | TBD | Definition of "rare" class |
| Model parameters | θ | ~10⁷-10⁸ | ViT-Base to ViT-Large scale |

### Known Results

- Gravity Spy CNN classifier achieves ~97% overall accuracy on common glitch classes — Zevin et al. (2017)
- Rare/low-count classes have significantly lower classification performance
- iDQ provides complementary data quality information but different approach (not morphological classification)
- Various CNN and transfer learning approaches have been applied to GW glitch classification

### What Is New

Applying Vision Transformers — which have global self-attention over the full spectrogram — to glitch classification, specifically targeting the rare-class tail where CNNs underperform. The explicit connection to CW search sensitivity improvement through better vetoes and line artifact removal is also novel.

### Target Venue

To be determined during literature review — candidates include Classical and Quantum Gravity, Physical Review D, or Machine Learning: Science and Technology.

### Computational Environment

- Local workstation: RTX 5090 (32GB VRAM), 48GB system RAM, 12 CPU cores
- Budget: local compute preferred over cloud
- Software: PyTorch, HuggingFace Transformers (for ViT), GWpy/PyCBC (for LIGO data), Gravity Spy tools

## Notation and Conventions

See `.gpd/CONVENTIONS.md` for all notation and sign conventions.
See `.gpd/NOTATION_GLOSSARY.md` for symbol definitions.

## Unit System

SI units for physical quantities (strain in 1/√Hz, frequency in Hz, time in seconds). Dimensionless for ML metrics (F1, recall, accuracy).

## Requirements (Archived)

v1.0 requirements archived to `.gpd/milestones/v1.0-REQUIREMENTS.md`. See Requirements section above for validated/active status. Next milestone will define fresh requirements via `/gpd:new-milestone`.

## Key References

- **ref-gravity-spy**: Zevin et al., Classical and Quantum Gravity, 2017 — Gravity Spy CNN baseline and glitch taxonomy (benchmark anchor)
- Additional literature anchors TBD via literature review

## Constraints

- **Computational resources**: Single RTX 5090 GPU, 32GB VRAM — constrains model size and batch size
- **Data availability**: Dependent on GWOSC public data releases for O3 and O4
- **Label availability**: Gravity Spy labels may have annotation biases; rare classes by definition have few examples
- **Reproducibility**: Must use only publicly available data and open-source tools

## Context

Completed v1.0 (5 phases) and v1.1 (2 phases). The original hypothesis (ViT outperforms CNN on rare classes) was **disconfirmed** — ViT rare-class macro-F1 (0.2412) is below CNN (0.3028), but a power analysis (v1.1) showed this comparison is severely underpowered (aggregate power 0.20), so the result is "insufficient evidence" rather than "definitive regression." The per-class analysis revealed a nuanced finding: architecture preference is class-dependent. ViT excels on Power_Line (+0.507pp O3, +0.394pp O4) while CNN is more robust for morphologically ambiguous rare classes. A random-split ablation (v1.1) confirmed the 5pp accuracy gap from published benchmarks is explained by temporal splitting (95.4% random vs 91.8% temporal). CW veto analysis showed matched-deadtime efficiency is approximately equal, with Power_Line being the only strong ViT CW advantage. The CQG manuscript has been revised to address all 9 referee issues with honest, quantitatively supported claims.

## Key Decisions

| Decision | Rationale | Outcome |
| -------- | --------- | ------- |
| ViT on spectrograms (not time-series) | Global attention captures long-range glitch morphology | **Good** — ViT works well for spectrally distinctive classes; fails on ambiguous rare ones |
| O3 train / O4 validate split | O3 has mature labels; O4 tests distribution shift | **Good** — caught that threshold finding was O3-specific |
| Focus on rare-class F1 over accuracy | Rare tail is the value proposition; accuracy is forbidden proxy | **Good** — caught forbidden proxy scenario in Phase 3 |
| Pre-made Gravity Spy images (Option C) | Simpler pipeline; no strain data dependency | **Good** — 325K samples downloaded successfully |
| ResNet-50v2 BiT as CNN baseline | Modern pretrained CNN with identical recipe | **Good** — fair comparison established |
| Rare threshold=200 (not protocol's 25) | 4 rare classes give more robust evaluation than 1 | **Revisit** — dual threshold should be better documented |
| Re-scope to "class-morphology-dependent" | Original threshold claim not supported by O4 data | **Good** — more honest and still publishable |
| Matched-deadtime CW comparison | 3.4x bug at 5% deadtime was misleading | **Good** — corrected to fair comparison |
| "class-dependent" over morphological complexity proxy | Simpler reframing is more honest; no unsupported causal mechanism | **Good** — referee option (b) cleaner than (a) |
| Simulation-based power analysis | Parametric formulas inappropriate for F1 on n=6-7 samples | **Good** — revealed all rare classes underpowered |
| CW table dual F1/DC columns | Single "Advantage" column mixed incompatible metrics | **Good** — resolved internal inconsistency |

---

_Last updated: 2026-03-18 after v1.1 milestone completion_
