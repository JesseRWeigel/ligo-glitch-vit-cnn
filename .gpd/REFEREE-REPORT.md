---
reviewed: 2026-03-18T21:00:00Z
scope: manuscript
target_journal: Classical and Quantum Gravity
recommendation: major_revision
confidence: high
major_issues: 5
minor_issues: 8
---

# Referee Report

**Scope:** Full manuscript review -- "When do Vision Transformers help? Class-dependent architecture preferences for gravitational-wave glitch classification"
**Date:** 2026-03-18
**Target Journal:** Classical and Quantum Gravity (CQG)

## Summary

This paper presents a controlled comparison of ViT-B/16 and ResNet-50v2 BiT for 23-class gravitational-wave glitch classification on LIGO O3 Gravity Spy data. The central claim is that architecture preference is class-dependent: ViT significantly improves overall macro-F1 (0.723 vs. 0.679, p=0.0002) but the rare-class comparison is statistically underpowered (aggregate power = 0.20). A secondary contribution connects classifier performance to continuous gravitational-wave search data quality through a sample-count-based duty-cycle proxy. Both models are evaluated zero-shot on O4 data, where the per-class pattern does not replicate as a monotonic sample-size trend. The paper's statistical methodology is above the field standard: temporal split with random-split control, paired bootstrap testing, and explicit power analysis for rare classes.

The manuscript has genuine merit as a CQG detector-characterization methods paper. The identical-recipe comparison, honest negative results, and CW veto connection are all useful contributions to the community. However, the paper currently overclaims in several specific areas. Most critically, the introduction and conclusion treat the rare-class CNN advantage as a directional finding despite the paper's own power analysis showing this comparison is underpowered, and the physical-interpretation claims (self-attention capturing harmonics, CNN inductive bias as regularization) lack direct evidence. Additionally, the novelty framing needs narrowing -- Wu et al. (CQG 2025) already reported per-class metrics -- and a canonical review article (Cuoco et al., Living Reviews in Relativity, 2025) is missing from the bibliography. These issues are all fixable through targeted text revision without new experiments, though multi-seed training (3-5 seeds) would substantially strengthen the per-class claims.

The mathematics and numerics are sound. The math stage independently verified all key quantities from source JSON to machine precision, finding only a minor double-rounding error in the ViT O4 macro-F1 value (reported 0.670, correct 0.669). The CW duty-cycle computation is correctly implemented as a union-based veto. The sign convention (ViT minus CNN) is consistent throughout.

## Panel Evidence

| Stage | Artifact | Assessment | Key blockers or concerns |
| ----- | -------- | ---------- | ------------------------ |
| Read | .gpd/review/STAGE-reader.json | strong | Intro/conclusion overclaim rare-class direction (RDR-001); physical-interp claims unsupported (RDR-002); CW framing tension (RDR-003); single-seed limitation (RDR-004) |
| Literature | .gpd/review/STAGE-literature.json | strong | Novelty claim too broad vs. Wu et al. (LIT-001); missing Cuoco et al. 2025 Living Reviews (LIT-003); Wu et al. arXiv citation should be CQG published version (LIT-006) |
| Math | .gpd/review/STAGE-math.json | strong | Double-rounding in ViT O4 macro-F1 (MATH-001); "8 rare classes" labeling inconsistency (MATH-002); Spearman power caveat (MATH-005) |
| Physics | .gpd/review/STAGE-physics.json | strong | Rare-class causal narrative unsupported (PHY-003, blocking); Power_Line interp asymmetrically hedged (PHY-001); CW abstract conflates veto eff. with DC (PHY-002); 20% threshold unmotivated (PHY-005) |
| Significance | .gpd/review/STAGE-interestingness.json | adequate | Central finding not surprising (SIG-001); CW analysis undersold vs. class-dependent preference (SIG-003); forbidden-proxy lesson is known ML principle (SIG-002); venue fit adequate after claim recalibration (SIG-005) |

## Recommendation

**MAJOR REVISION**

The paper is publishable at CQG after targeted text revision. The core empirical work -- identical-recipe comparison with temporal split, power analysis, CW veto proxy, and honest O4 non-replication -- is sound and useful. However, the current framing overclaims in three areas that require substantive revision: (1) the rare-class CNN advantage is presented as a directional finding in the introduction and conclusion despite being underpowered, (2) physical-interpretation claims about self-attention and inductive bias lack direct evidence and are asymmetrically hedged, and (3) the novelty claims need narrowing against Wu et al. (CQG 2025) and the CW "first quantitative assessment" needs downgrading to match its proxy nature. None of these requires new experiments, but they require honest reframing throughout the manuscript, not just in one section. Multi-seed training (3-5 seeds) is strongly recommended but not strictly required. The significance is adequate for CQG as a methods/benchmark paper after claim recalibration but would not meet the bar for a more selective venue.

## Evaluation

### Strengths

1. **Statistical hygiene above field standard.** Power analysis for rare-class comparison (Sec 3.1), explicit underpowering acknowledgment, paired bootstrap testing with 10K resamples, and temporal split validation with random-split control (Sec 3.2) are practices that most GW ML papers omit.
2. **Honest negative results.** The O4 non-replication of per-class patterns (Sec 3.5) and the non-significant Spearman correlation (Sec 3.3) are reported straightforwardly rather than buried or omitted.
3. **CW veto connection.** Despite being proxy-level, the duty-cycle analysis (Sec 3.6, Table 3) connects glitch classification to downstream CW search sensitivity in a way that prior Gravity Spy architecture comparisons did not.
4. **Temporal split methodology.** The GPS-time split with 60-second gaps (Sec 2.2) and the random-split control experiment (Sec 3.2) quantify the effect of temporal data leakage on accuracy benchmarks for the first time in the Gravity Spy literature.
5. **Reproducibility infrastructure.** The reproducibility manifest, number macros from a single JSON source of truth, and documented execution pipeline (reproducibility-manifest.json) are exemplary.
6. **Number pipeline integrity.** All 37 LaTeX number macros trace to paper_numbers.json; all key quantities verified to machine precision by the math stage.

### Major Issues

#### Issue 1: Rare-class narrative overclaimed in Introduction and Conclusion

**ID:** REF-001
**Dimension:** significance / correctness
**Severity:** Major revision required
**Location:** paper/main.tex:193, paper/main.tex:693-695

**Description:** The introduction (line 193) states "yet rare-class performance declines" without qualification. The conclusion (lines 693-695) states "CNN retains advantages on rare classes where limited training data appears insufficient for learning effective attention patterns," converting an underpowered observation into a causal narrative. The paper's own power analysis shows the rare-class comparison has aggregate power of only 0.20 and p=0.884 -- the data provide insufficient statistical evidence to determine even the direction of the rare-class effect. The abstract is appropriately hedged; the introduction and conclusion are not.

**Quoted claim:** "CNN retains advantages on rare classes where limited training data appears insufficient for learning effective attention patterns." (Conclusion, line 694)

**Missing evidence:** Statistical significance of the rare-class direction (p=0.884 is consistent with no difference or ViT advantage); multi-seed confirmation that the Chirp 0.000 result is stable.

**Impact:** The headline "class-dependent architecture preference" claim partially depends on rare classes favoring CNN. If the rare-class direction is uncertain, the story reduces to "ViT is better overall and on specific common classes; rare-class direction is unknown" -- still publishable but requiring honest framing throughout.

**Suggested fix:** Harmonize all rare-class language with the abstract's careful phrasing: "insufficient evidence to determine whether ViT improves or degrades rare-class performance." The introduction, discussion (Sec 4.1), and conclusion must use this framing consistently. The inductive-bias regularization interpretation (Sec 4.1) should be explicitly conditioned on future studies confirming the rare-class direction.

#### Issue 2: Novelty claim too broad against Wu et al. (CQG 2025)

**ID:** REF-002
**Dimension:** novelty / literature_context
**Severity:** Major revision required
**Location:** paper/main.tex:167-170

**Description:** The manuscript states "none of these studies has conducted a controlled per-class comparison under identical training conditions." Wu et al. (arXiv:2401.12913, now published as CQG 42, 165015) report class-wise accuracy tables and confusion matrices comparing their attention-fusion model against the legacy Gravity Spy CNN on both O3 and O4 data. The present paper's novelty is more precisely: single-view identical-recipe comparison with paired bootstrap significance testing per class and simulation-based power analysis. This is a genuine advance, but the current framing implies no prior per-class analysis exists.

**Quoted claim:** "none of these studies has conducted a controlled per-class comparison under identical training conditions" (Introduction, line 167-170)

**Missing evidence:** Acknowledgment that Wu et al. reported per-class metrics (accuracy, confusion matrices).

**Impact:** A CQG referee familiar with the Wu et al. paper (published in the same journal) will flag this overclaim. It is easily fixable but currently undermines credibility.

**Suggested fix:** Narrow to: "none of these studies has conducted an identical-recipe paired comparison with statistical testing of per-class differences." Explicitly note that Wu et al. report per-class accuracy but use a different architecture (multi-view fusion) trained under different conditions without paired statistical testing or power analysis.

#### Issue 3: Missing canonical review citation (Cuoco et al. 2025)

**ID:** REF-003
**Dimension:** literature_context
**Severity:** Major revision required
**Location:** paper/main.tex:130-170

**Description:** Cuoco, Cavaglia et al., "Applications of machine learning in gravitational-wave research with current interferometric detectors," Living Reviews in Relativity 28, 2 (2025), is the current canonical review of ML applications in GW research, covering glitch classification extensively. It is not cited. CQG referees working in detector characterization will expect this reference. Additionally, Wu et al. is cited as an arXiv preprint but has been published in CQG (vol. 42, 165015) -- the published version should be cited, especially since CQG is the target journal.

**Impact:** Missing a canonical review in the target journal's subject area signals inadequate literature engagement to referees.

**Suggested fix:** Add Cuoco et al. (2025) to the introduction when discussing ML for GW. Update Wu et al. citation to the published CQG version and verify whether the published version contains updated per-class results.

#### Issue 4: Physical-interpretation claims asymmetrically hedged

**ID:** REF-004
**Dimension:** technical_soundness / completeness
**Severity:** Major revision required
**Location:** paper/main.tex:339-340, paper/main.tex:523-534

**Description:** The Power_Line self-attention interpretation is presented as explanatory ("benefits from ViT's global self-attention," line 339-340; "natural interpretation in terms of architectural inductive biases," Sec 4.1) while the parallel rare-class CNN interpretation is hedged as "speculative" (line 541). Neither interpretation has direct evidence: no attention maps, no ablations, no controlled experiments with synthetic patterns. For CQG, physical-mechanism claims carry weight, and the asymmetric hedging creates an impression of established physics where only empirical pattern-matching exists.

**Quoted claim:** "Power_Line glitches exhibit harmonics of 60 Hz spanning the full frequency range -- a pattern that local convolutional filters may struggle to integrate." (Sec 4.1, line 528-530)

**Missing evidence:** Attention maps showing ViT attends to harmonic structure; ablation removing specific frequency bands; comparison with CNN using larger receptive fields.

**Suggested fix:** Apply consistent "speculative" or "hypothetical" qualification to both the Power_Line and rare-class interpretations. In Sec 4.1, explicitly state that all inductive-bias interpretations are hypotheses without attention-map or ablation evidence. Alternatively, add attention-map visualizations.

#### Issue 5: CW abstract framing conflates veto efficiency with duty cycle

**ID:** REF-005
**Dimension:** technical_soundness / significance
**Severity:** Major revision required
**Location:** paper/main.tex:117-118, paper/main.tex:480-483

**Description:** The abstract reports "veto efficiency for continuous-wave searches is approximately equal (ViT 0.745 vs. CNN 0.735)" without mentioning that the operationally relevant quantity -- duty cycle -- shows a statistically significant CNN advantage (delta_DC = -0.051, 95% CI [-0.054, -0.048], does not contain zero). For CW searches, livetime directly enters sensitivity through integrated observation time. The abstract creates the impression that the CW implications are neutral, when CNN retains 5 percentage points more observation time. Additionally, calling this the "first quantitative assessment" (Sec 1, line 188-190) overstates a sample-count proxy.

**Quoted claim:** "veto efficiency for continuous-wave searches is approximately equal (ViT 0.745 vs. CNN 0.735)" (Abstract, line 117-118)

**Missing evidence:** The duty-cycle difference is presented in Table 3 and Sec 3.6 but not the abstract.

**Impact:** CW practitioners reading only the abstract would conclude the architectures are interchangeable for CW purposes, missing the significant duty-cycle difference.

**Suggested fix:** The abstract must mention the duty-cycle difference alongside veto efficiency. Replace "first quantitative assessment" with "first proxy-level comparison" in the introduction.

### Minor Issues

#### Issue 6: Double-rounding error in ViT O4 macro-F1

**ID:** REF-006
**Dimension:** correctness
**Severity:** Minor revision
**Location:** paper/main.tex:55, paper/tables/table_overall.tex:11

**Description:** The source JSON stores ViT O4 macro-F1 = 0.6694683. Standard 3dp rounding gives 0.669, not the reported 0.670. The error arises from double-rounding (0.66947 -> 0.6695 -> 0.670). The discrepancy is +0.001 and does not change qualitative conclusions.

**Suggested fix:** Correct \vitMacroFOfour from 0.670 to 0.669.

#### Issue 7: "8 rare classes" labeling inconsistency

**ID:** REF-007
**Dimension:** clarity
**Severity:** Minor revision
**Location:** paper/main.tex:354

**Description:** Section 3.1 states "all 8 rare classes are underpowered" but Section 2.1 defines rare as n_train < 200 (only 4 classes). The power analysis tested 8 classes including non-rare ones.

**Suggested fix:** Change "all 8 rare classes" to "all 8 classes with small test sets."

#### Issue 8: Section heading states absence of correlation as positive finding

**ID:** REF-008
**Dimension:** correctness / clarity
**Severity:** Minor revision
**Location:** paper/main.tex:389

**Description:** "Architecture Preference Is Not a Monotonic Function of Sample Size" states evidence of absence. With n=23, the Spearman test has power ~0.08 to detect rho=-0.12.

**Suggested fix:** Soften to "No significant evidence that architecture preference is a monotonic function of sample size."

#### Issue 9: 20% O4 degradation threshold is unmotivated

**ID:** REF-009
**Dimension:** technical_soundness
**Severity:** Minor revision
**Location:** paper/main.tex:448-450

**Description:** The "<20% relative degradation" criterion is ad hoc. ViT degrades 7.4% vs. CNN 1.7% (4.4x ratio) but both "pass." The asymmetric degradation is physically meaningful.

**Suggested fix:** Either motivate the threshold from literature or remove pass/fail framing.

#### Issue 10: "Forbidden-proxy lesson" not cited from imbalanced-learning literature

**ID:** REF-010
**Dimension:** literature_context / novelty
**Severity:** Minor revision
**Location:** paper/main.tex:592-607

**Description:** Per-class evaluation under class imbalance is established (He & Garcia 2009; Buda et al. 2018). The paper frames this as novel without citing this literature.

**Suggested fix:** Add 1-2 imbalanced-learning citations. Reframe as GW demonstration of known principle.

#### Issue 11: ViT interpretability literature not cited

**ID:** REF-011
**Dimension:** literature_context
**Severity:** Minor revision
**Location:** paper/main.tex:523-534

**Description:** Attention-mechanism claims not grounded in ViT interpretability literature (e.g., Raghu et al. 2021).

**Suggested fix:** Cite 1-2 ViT interpretability papers in Sec 4.1.

#### Issue 12: One-sided p-value for rare-class with opposite-direction effect

**ID:** REF-012
**Dimension:** correctness
**Severity:** Minor revision
**Location:** paper/tables/table_overall.tex:10

**Description:** One-sided test (ViT > CNN) gives p=0.884 when effect is ViT < CNN. Two-sided p=0.232, still non-significant.

**Suggested fix:** Switch to two-sided test or add explanatory footnote.

#### Issue 13: Wu et al. cited as arXiv, now published in CQG

**ID:** REF-013
**Dimension:** presentation_quality
**Severity:** Minor revision
**Location:** paper/main.tex:774-777

**Description:** Wu et al. is now CQG 42, 165015.

**Suggested fix:** Update citation to published version.

### Suggestions

1. **Multi-seed training (3-5 seeds).** Would substantially strengthen per-class claims. Estimated effort: medium.
2. **Attention-map visualizations for Power_Line.** Direct evidence for harmonic-structure interpretation. Estimated effort: small.
3. **Restructure narrative emphasis for CQG.** Foreground CW veto findings; the F1-vs-DC disconnect is more CQG-relevant than class-dependent preference. Estimated effort: small.
4. **Two-sided bootstrap tests throughout.** Avoids awkward p near 1.0. Estimated effort: trivial.

## Detailed Evaluation

### 1. Novelty: ADEQUATE (after repair)

The core novelty -- identical-recipe ViT vs. CNN comparison with paired bootstrap testing and power analysis -- is genuine and not preempted. Wu et al. (CQG 2025) reported per-class metrics but with different architectures and no statistical testing. The CW veto connection is novel for Gravity Spy comparisons. After narrowing the novelty claims, the paper is adequate for CQG.

### 2. Correctness: MOSTLY CORRECT

All key quantities verified from source JSON. One minor double-rounding error (ViT O4 macro-F1: 0.670 should be 0.669). Bootstrap CI ordering, dataset totals, per-class differences, and CW delta_DC all check out.

### 3. Clarity: GOOD

Well-organized with clear logical flow. Consistent notation. ASSERT_CONVENTION header documents sign convention. Main weakness: inconsistent hedging between abstract (careful) and introduction/conclusion (overclaimed).

### 4. Completeness: MOSTLY COMPLETE

All promised results delivered. Bootstrap CIs on all primary metrics. Power analysis for rare classes. Gap: single-seed limitation acknowledged but not propagated into claim strength.

### 5. Significance: ADEQUATE (for CQG, after reframing)

Central finding is descriptively true but not surprising. Better characterized as a careful benchmark. CW veto connection and temporal-split methodology are most venue-appropriate. Adequate for CQG as methods paper after reframing.

### 6. Reproducibility: FULLY REPRODUCIBLE

All parameters stated. Seeds documented. Full 16-step execution pipeline. Zenodo DOIs for data. SHA-256 checksums. Code availability promised.

### 7. Literature Context: INCOMPLETE (repairable)

Missing Cuoco et al. 2025 (canonical review). Wu et al. cited as arXiv not CQG. Imbalanced-learning and ViT interpretability literature not cited. Fixable with 4-6 additional citations.

### 8. Presentation Quality: NEEDS POLISHING

Good structure. Appropriate figures and tables. Number macros prevent transcription errors. Issues: inconsistent claim strength, double-rounding, labeling inconsistency.

### 9. Technical Soundness: MOSTLY SOUND

Appropriate methodology: identical recipe, temporal split, focal loss, paired bootstrap, power analysis. CW proxy valid first-order with acknowledged limitations. Main concern: single seed.

### 10. Publishability: MAJOR REVISION

Publishable at CQG after targeted revision. CQG is the correct venue.

## Physics Checklist

| Check | Status | Notes |
| ----- | ------ | ----- |
| Dimensional analysis | pass | All metrics dimensionless ratios |
| Limiting cases | pass | Random-split control recovers published accuracy range |
| Symmetry preservation | N/A | ML benchmark |
| Conservation laws | N/A | ML benchmark |
| Error bars present | pass | 95% bootstrap CIs on all primary metrics |
| Approximations justified | pass | Stationarity assumption stated; coarse proxy acknowledged |
| Convergence demonstrated | partial | Bootstrap 10K adequate; no multi-seed convergence |
| Literature comparison | partial | CNN accuracy gap explained; Wu et al. per-class comparison missing |
| Reproducible | pass | Full pipeline documented with seeds and checksums |

---

### Actionable Items

```yaml
actionable_items:
  - id: "REF-001"
    finding: "Rare-class narrative overclaimed in Introduction and Conclusion"
    severity: "major"
    specific_file: "paper/main.tex"
    specific_change: "Harmonize lines 193 and 693-695 with abstract hedging: use 'insufficient evidence' not directional claims throughout."
    estimated_effort: "small"
    blocks_publication: true

  - id: "REF-002"
    finding: "Novelty claim too broad against Wu et al."
    severity: "major"
    specific_file: "paper/main.tex"
    specific_change: "Narrow lines 167-170 to specify identical-recipe paired comparison with statistical testing. Acknowledge Wu et al. reported per-class accuracy."
    estimated_effort: "small"
    blocks_publication: true

  - id: "REF-003"
    finding: "Missing canonical review citation and outdated Wu et al. citation"
    severity: "major"
    specific_file: "paper/main.tex"
    specific_change: "Add Cuoco et al. (2025) LRR to introduction. Update Wu et al. to CQG 42, 165015."
    estimated_effort: "small"
    blocks_publication: true

  - id: "REF-004"
    finding: "Physical-interpretation claims asymmetrically hedged"
    severity: "major"
    specific_file: "paper/main.tex"
    specific_change: "Apply consistent speculative qualification to Power_Line interpretation (lines 339-340, 523-534)."
    estimated_effort: "small"
    blocks_publication: true

  - id: "REF-005"
    finding: "CW abstract framing conflates veto efficiency with duty cycle"
    severity: "major"
    specific_file: "paper/main.tex"
    specific_change: "Abstract: add duty-cycle difference. Introduction: replace 'first quantitative assessment' with 'first proxy-level comparison'."
    estimated_effort: "small"
    blocks_publication: true

  - id: "REF-006"
    finding: "Double-rounding error in ViT O4 macro-F1"
    severity: "minor"
    specific_file: "paper/main.tex"
    specific_change: "Change vitMacroFOfour from 0.670 to 0.669."
    estimated_effort: "trivial"
    blocks_publication: false

  - id: "REF-007"
    finding: "8 rare classes labeling inconsistency"
    severity: "minor"
    specific_file: "paper/main.tex"
    specific_change: "Line 354: change 'all 8 rare classes' to 'all 8 classes with small test sets'."
    estimated_effort: "trivial"
    blocks_publication: false

  - id: "REF-008"
    finding: "Section heading overstates absence of correlation"
    severity: "minor"
    specific_file: "paper/main.tex"
    specific_change: "Soften Sec 3.3 heading."
    estimated_effort: "trivial"
    blocks_publication: false

  - id: "REF-009"
    finding: "20% O4 degradation threshold unmotivated"
    severity: "minor"
    specific_file: "paper/main.tex"
    specific_change: "Motivate threshold or remove pass/fail framing."
    estimated_effort: "trivial"
    blocks_publication: false

  - id: "REF-010"
    finding: "Forbidden-proxy lesson not cited from imbalanced-learning literature"
    severity: "minor"
    specific_file: "paper/main.tex"
    specific_change: "Add He and Garcia 2009, Buda et al. 2018 citations in Sec 4.4."
    estimated_effort: "trivial"
    blocks_publication: false

  - id: "REF-011"
    finding: "ViT interpretability literature not cited"
    severity: "minor"
    specific_file: "paper/main.tex"
    specific_change: "Add Raghu et al. 2021 in Sec 4.1."
    estimated_effort: "trivial"
    blocks_publication: false

  - id: "REF-012"
    finding: "One-sided p-value with opposite-direction effect"
    severity: "minor"
    specific_file: "paper/tables/table_overall.tex"
    specific_change: "Switch to two-sided test for rare-class row or add footnote."
    estimated_effort: "trivial"
    blocks_publication: false

  - id: "REF-013"
    finding: "Wu et al. cited as arXiv, now published in CQG"
    severity: "minor"
    specific_file: "paper/main.tex"
    specific_change: "Update bibitem to CQG 42, 165015."
    estimated_effort: "trivial"
    blocks_publication: false
```

### Confidence Self-Assessment

| Dimension | Confidence | Notes |
|-----------|-----------|-------|
| Novelty | HIGH | Literature search verified Wu et al. comparison |
| Correctness | HIGH | All key quantities independently recomputed |
| Clarity | HIGH | Direct reading assessment |
| Completeness | HIGH | Full manuscript and pipeline reviewed |
| Significance | MEDIUM | CQG venue fit requires domain judgment |
| Reproducibility | HIGH | Manifest independently reviewed |
| Literature context | MEDIUM | May miss very recent preprints |
| Presentation quality | HIGH | Direct reading |
| Technical soundness | HIGH | Methodology within ML/statistics competence |
| Publishability | MEDIUM | Final editorial judgment depends on referee pool |

---

_Reviewed: 2026-03-18T21:00:00Z_
_Reviewer: AI assistant (gpd-referee)_
_Disclaimer: This is an AI-generated mock referee report. It supplements but does not replace expert peer review._
