# Research Digest: v1.1 Paper Revision for CQG Submission

Generated: 2026-03-18
Milestone: v1.1
Phases: 6-7

## Narrative Arc

The v1.0 milestone established that ViT architecture preference over CNN is class-dependent rather than systematic, with an honest negative result on rare classes. However, the paper draft contained three analytical gaps flagged by mock peer review: (1) no quantitative explanation for the 5pp accuracy gap between our temporal-split CNN (91.8%) and published random-split benchmarks (95-99%), (2) no statistical power analysis to justify the "insufficient evidence" framing for rare-class comparisons, and (3) internal inconsistencies in the CW advantage table mixing F1-based and duty-cycle-based judgments. Phase 6 closed the first two gaps computationally — a random-split CNN ablation confirmed that temporal splitting (not a pipeline deficiency) explains the accuracy discrepancy, and a simulation-based power analysis demonstrated that all 8 rare-class comparisons are severely underpowered (aggregate power 0.20) at the observed effect size. Phase 7 integrated these quantitative results into the manuscript, reframed the narrative from "class-morphology-dependent" to "class-dependent" throughout, restructured the CW table with separate F1 and DC advantage columns, and resolved all 5 minor referee issues. The result is a submission-ready CQG manuscript with honest, quantitatively supported claims.

## Key Results

| Phase | Result | Equation / Value | Validity Range | Confidence |
| ----- | ------ | ---------------- | -------------- | ---------- |
| 6 | Random-split CNN accuracy | 0.9544 [0.9525, 0.9563] | Single seed=42 | HIGH |
| 6 | Random-split macro-F1 | 0.7507 [0.7273, 0.7716] | Single seed=42 | HIGH |
| 6 | Accuracy gap (random − temporal) | +3.63 pp | — | HIGH |
| 6 | Aggregate rare-class power | 0.20 at δ=−0.062 | α=0.05, 10K sim | HIGH |
| 6 | Chirp power at observed effect | 0.035 (n=7) | — | HIGH |
| 6 | Wandering_Line power | 0.017 (n=6) | — | HIGH |
| 6 | 1080Lines power | 0.310 (n=6) | — | HIGH |
| 6 | Helix power | 0.057 (n=14) | — | HIGH |
| 7 | All 9 referee issues resolved | REF-001 through REF-009 | — | HIGH |
| 7 | Zero "morphology-dependent" instances | 0 (was 4) | — | HIGH |

## Methods Employed

- **Phase 6:** Stratified random split — same class proportions as temporal split but without temporal gap enforcement
- **Phase 6:** Simulation-based power analysis — 10K iterations with permutation test critical values, more appropriate than parametric formulas for F1 on tiny samples
- **Phase 6:** Paired permutation test — for comparing random-split vs temporal-split macro-F1
- **Phase 7:** Targeted text revision — systematic search-and-replace with context-aware reframing
- **Phase 7:** Table restructuring — CW table split into dual F1/DC advantage columns

## Convention Evolution

| Phase | Convention | Description | Status |
| ----- | ---------- | ----------- | ------ |
| 1 | Natural units | SI: strain dimensionless, frequency Hz, time s | Active |
| 1 | Coordinate system | Q-transform spectrograms: 10-2048 Hz log, 224×224 px, min-max [0,1] after SNR clip [0,25.5] | Active |
| 1 | Coupling convention | Macro-F1 primary; accuracy forbidden proxy; bootstrap ≥10K, p<0.05 | Active |
| 6 | Power analysis | α=0.05, power threshold 0.80, simulation-based (10K iterations) | Active |
| 7 | CW advantage reporting | Must distinguish F1-based and DC-based advantage separately | Active |
| 7 | Rare-class claims | Must cite power analysis numbers (aggregate power 0.20) | Active |

## Figures and Data Registry

| File | Phase | Description | Paper-ready? |
| ---- | ----- | ----------- | ------------ |
| results/06-computation-statistical-analysis/random_split_ablation.json | 6 | Random-split accuracy, macro-F1, per-class F1 with CIs | Data only |
| results/06-computation-statistical-analysis/power_analysis.json | 6 | Per-class MDE, power at observed effect, power curves | Data only |
| results/06-computation-statistical-analysis/ablation_summary.txt | 6 | Human-readable combined summary | Data only |
| paper/main.tex | 7 | Revised CQG manuscript (submission-ready) | Yes |
| paper/tables/table_cw.tex | 7 | CW Table 3 with F1 Adv and DC Adv columns | Yes |
| paper/tables/table_overall.tex | 7 | Table 1 with corrected O4 rounding | Yes |

## Open Questions

1. Would rare-class-specific interventions (augmentation, few-shot, contrastive learning) resolve the rare-class gap?
2. Is frequency-resolved PSD analysis needed to quantify Power_Line CW benefit more precisely?
3. Would a multi-seed ablation (3-5 seeds) tighten the random-split accuracy CI meaningfully?
4. Could a more sophisticated power analysis (incorporating FP and class correlations) change the underpowering conclusion for medium-sized rare classes?

## Dependency Graph

    Phase 6 "Computation & Statistical Analysis"
      provides: random-split accuracy (0.9544), per-class MDE, aggregate power (0.20)
      requires: Phase 2 CNN baseline, Phase 3 ViT results, Phase 5 paper draft
    -> Phase 7 "Paper Text & Table Revision"
      provides: revised manuscript (all 9 REF issues), corrected tables
      requires: Phase 6 ablation + power results, Phase 4 CW data, Phase 5 original manuscript

## Mapping to Original Objectives

| Requirement | Status | Fulfilled by | Key Result |
| ----------- | ------ | ------------ | ---------- |
| TEXT-01: Class-dependent reframing | Complete | Phase 7 | Zero "morphology-dependent" instances; hedged hypotheses |
| TEXT-02: CW proxy validity statement | Complete | Phase 7 | Noise PSD stationarity assumption stated in §2.4 |
| TEXT-03: Davis et al. citation | Complete | Phase 7 | Cited twice in §2.4 |
| TEXT-04: Single-run limitation | Complete | Phase 7 | Limitation statement added |
| TEXT-05: Asymmetric degradation discussion | Complete | Phase 7 | 7.4% vs 1.7% discussed in §4.2 |
| TABL-01: CW table restructure | Complete | Phase 7 | Separate F1 Adv and DC Adv columns |
| TABL-02: O4 rounding fix | Complete | Phase 7 | ViT macro-F1 corrected to 0.670 |
| COMP-01: Random-split ablation | Complete | Phase 6 | 95.4% accuracy, +3.63pp gap |
| STAT-01: Power analysis | Complete | Phase 6 | Aggregate power 0.20, all 8 classes underpowered |
| STAT-02: Rare-class reframing | Complete | Phase 7 | "Insufficient evidence" with power numbers throughout |
