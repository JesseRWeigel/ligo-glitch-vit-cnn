# Referee Response

**Journal:** Classical and Quantum Gravity
**Manuscript:** "When do Vision Transformers help? Class-dependent architecture preferences for gravitational-wave glitch classification"
**Round:** 1
**Decision:** Major revision
**Date:** 2026-03-18

## Summary of Changes

We thank the reviewer for the careful and constructive report. All 5 major and 8 minor issues have been addressed through targeted text revisions. No new experiments were required. Below we provide point-by-point responses.

### Major Changes

1. **REF-001:** Harmonized all rare-class language throughout the manuscript to match the abstract's careful hedging ("insufficient statistical evidence"), removing directional claims from Introduction and Conclusion.
2. **REF-002:** Narrowed the novelty claim to specify "identical-recipe paired comparison with statistical testing and power analysis," acknowledging Wu et al.'s per-class results.
3. **REF-003:** Added Cuoco et al. (2025) Living Reviews in Relativity citation; updated Wu et al. to published CQG version.
4. **REF-004:** Applied consistent speculative qualification to all inductive-bias interpretations (Power_Line and rare-class). Added Raghu et al. (2021) citation for ViT interpretability context.
5. **REF-005:** Abstract now reports duty-cycle difference alongside veto efficiency. "First quantitative assessment" downgraded to "first proxy-level comparison."

### Minor Changes

6. **REF-006:** Corrected ViT O4 macro-F1 from 0.670 to 0.669 (double-rounding fix).
7. **REF-007:** Changed "all 8 rare classes" to "all 8 classes with small test sets."
8. **REF-008:** Softened Section 3.3 heading to "No Significant Evidence That..."
9. **REF-009:** Removed ad-hoc 20% pass/fail threshold; now reports asymmetric degradation directly.
10. **REF-010:** Added He & Garcia (2009) and Buda et al. (2018) citations; reframed as GW demonstration of known imbalanced-learning principle.
11. **REF-011:** Added Raghu et al. (2021) citation in Discussion Sec 4.1.
12. **REF-012:** Added footnote to Table I explaining one-sided test direction and two-sided p-value.
13. **REF-013:** Updated Wu et al. citation from arXiv to CQG 42, 165015.

---

## Point-by-Point Responses

### REF-001: Rare-class narrative overclaimed in Introduction and Conclusion

> *"The introduction (line 193) states 'yet rare-class performance declines' without qualification. The conclusion states 'CNN retains advantages on rare classes where limited training data appears insufficient for learning effective attention patterns,' converting an underpowered observation into a causal narrative."*

We agree completely. The rare-class comparison has aggregate power of only 0.20, and the data provide insufficient evidence to determine even the direction of the effect. We have harmonized all rare-class language throughout the manuscript:

**Changes:**
- Introduction (line 205): Replaced "yet rare-class performance declines" with "yet the rare-class comparison is statistically underpowered (aggregate power = 0.20), yielding insufficient evidence to determine whether ViT improves or degrades rare-class performance."
- Discussion Sec 4.1 (line 561): Changed "data scarcity appears to prevent" to "if the tentative point estimates are confirmed by future multi-seed studies, data scarcity may prevent."
- Conclusion (line 723): Replaced directional CNN-advantage claim with "the rare-class comparison remains statistically underpowered (aggregate power = 0.20), leaving insufficient evidence to determine whether CNN's inductive bias provides a meaningful advantage for data-scarce categories."

### REF-002: Novelty claim too broad against Wu et al.

> *"The manuscript states 'none of these studies has conducted a controlled per-class comparison under identical training conditions.' Wu et al. (CQG 42, 165015) report class-wise accuracy tables and confusion matrices."*

We agree that our novelty claim was too broad. Wu et al. do report per-class metrics; our advance is the identical-recipe paired comparison with statistical testing and power analysis.

**Changes:**
- Introduction (line 173-177): Added clarifying sentence noting Wu et al.'s per-class results and narrowed the novelty claim to "identical-recipe paired comparison with statistical testing of per-class differences and simulation-based power analysis."

### REF-003: Missing canonical review citation

> *"Cuoco et al. (2025) Living Reviews in Relativity is the current canonical review of ML for GW research. Additionally, Wu et al. is cited as arXiv but now published in CQG."*

We have added both citations.

**Changes:**
- Introduction (line 161): Added Cuoco et al. (2025) LRR citation.
- Bibliography: Updated Wu et al. to CQG 42, 165015 (line 803).

### REF-004: Physical-interpretation claims asymmetrically hedged

> *"The Power_Line self-attention interpretation is presented as explanatory while the rare-class CNN interpretation is hedged as speculative. Neither has direct evidence."*

We agree and have applied consistent speculative qualification throughout.

**Changes:**
- Discussion Sec 4.1 (line 543): Reframed Power_Line interpretation as explicit hypothesis: "We hypothesize that this benefits classes with spatially distributed, multi-frequency structure...but we lack direct evidence (attention maps or ablation studies) to confirm this mechanism."
- Added Raghu et al. (2021) citation for context (line 550).

### REF-005: CW abstract conflates veto efficiency with duty cycle

> *"The abstract reports 'approximately equal' veto efficiency without mentioning the statistically significant CNN duty-cycle advantage."*

We agree this was misleading.

**Changes:**
- Abstract (line 116): Now reads "veto efficiency...is comparable (ViT 0.745 vs. CNN 0.735), though CNN retains a statistically significant duty-cycle advantage."
- Introduction (line 198): Changed "first quantitative assessment" to "first proxy-level comparison."

### REF-006 through REF-013: Minor Issues

All 8 minor issues have been addressed as specified in the Summary of Changes above. Each change is marked with a `% REVISED: REF-XXX` comment in the LaTeX source.

---

## New Citations Added

1. Cuoco, E., Cavaglia, M. et al. (2025), "Applications of machine learning in gravitational-wave research with current interferometric detectors," Living Reviews in Relativity, **28**, 2.
2. He, H. and Garcia, E. A. (2009), "Learning from Imbalanced Data," IEEE TKDE, **21**(9), 1263-1284.
3. Buda, M. et al. (2018), "A systematic study of the class imbalance problem in CNNs," Neural Networks, **106**, 249-259.
4. Raghu, M. et al. (2021), "Do Vision Transformers See Like Convolutional Neural Networks?," NeurIPS 2021.
5. Wu et al. updated from arXiv to CQG **42**, 165015 (2025).
