# Author Response Tracker

**Round:** 1
**Source:** Staged peer-review panel (`.gpd/REFEREE-REPORT.md`)
**Recommendation:** major_revision
**Blocking issues:** REF-001, REF-002, REF-003, REF-004, REF-005

## Issue Tracker

### REF-001: Rare-class narrative overclaimed in Introduction and Conclusion
- **Severity:** major (blocking)
- **Classification:** fixed
- **Sections affected:** Abstract, Introduction, Discussion (Sec 4.1), Conclusion
- **Change:** Harmonize all rare-class language with abstract hedging. Replace directional claims ("rare-class performance declines", "CNN retains advantages") with "insufficient statistical evidence to determine whether ViT improves or degrades rare-class performance." Condition inductive-bias interpretation on future confirmation.

### REF-002: Novelty claim too broad against Wu et al.
- **Severity:** major (blocking)
- **Classification:** fixed
- **Sections affected:** Introduction
- **Change:** Narrow "none of these studies has conducted a controlled per-class comparison" to "none has conducted an identical-recipe paired comparison with statistical testing of per-class differences and simulation-based power analysis." Acknowledge Wu et al. per-class results.

### REF-003: Missing canonical review citation and outdated Wu et al.
- **Severity:** major (blocking)
- **Classification:** fixed
- **Sections affected:** Introduction, Bibliography
- **Change:** Add Cuoco et al. (2025) LRR citation. Update Wu et al. to CQG 42, 165015.

### REF-004: Physical-interpretation claims asymmetrically hedged
- **Severity:** major (blocking)
- **Classification:** fixed
- **Sections affected:** Results (Sec 3.1), Discussion (Sec 4.1)
- **Change:** Apply consistent speculative language to Power_Line interpretation. Both inductive-bias interpretations framed as hypotheses requiring attention-map or ablation evidence.

### REF-005: CW abstract framing conflates veto efficiency with duty cycle
- **Severity:** major (blocking)
- **Classification:** fixed
- **Sections affected:** Abstract, Introduction
- **Change:** Abstract: add duty-cycle difference alongside veto efficiency. Introduction: replace "first quantitative assessment" with "first proxy-level comparison."

### REF-006: Double-rounding error in ViT O4 macro-F1
- **Severity:** minor
- **Classification:** fixed
- **Sections affected:** Number macros (line 55)
- **Change:** Correct \vitMacroFOfour from 0.670 to 0.669.

### REF-007: "8 rare classes" labeling inconsistency
- **Severity:** minor
- **Classification:** fixed
- **Sections affected:** Results (Sec 3.1)
- **Change:** Change "all 8 rare classes" to "all 8 classes with small test sets."

### REF-008: Section heading overstates absence of correlation
- **Severity:** minor
- **Classification:** fixed
- **Sections affected:** Results (Sec 3.3)
- **Change:** Soften heading to "No Significant Evidence That Architecture Preference Is a Monotonic Function of Sample Size"

### REF-009: 20% O4 degradation threshold unmotivated
- **Severity:** minor
- **Classification:** fixed
- **Sections affected:** Results (Sec 3.5)
- **Change:** Remove pass/fail framing; report degradation values without ad-hoc threshold.

### REF-010: Forbidden-proxy lesson not cited from imbalanced-learning literature
- **Severity:** minor
- **Classification:** fixed
- **Sections affected:** Discussion (Sec 4.4)
- **Change:** Add He & Garcia (2009) and Buda et al. (2018) citations. Reframe as GW demonstration of known principle.

### REF-011: ViT interpretability literature not cited
- **Severity:** minor
- **Classification:** fixed
- **Sections affected:** Discussion (Sec 4.1)
- **Change:** Cite Raghu et al. (2021) in Sec 4.1.

### REF-012: One-sided p-value with opposite-direction effect
- **Severity:** minor
- **Classification:** fixed
- **Sections affected:** Table I footnote or Sec 2.4
- **Change:** Add footnote explaining one-sided test direction and noting two-sided p=0.232.

### REF-013: Wu et al. cited as arXiv, now published in CQG
- **Severity:** minor
- **Classification:** fixed
- **Sections affected:** Bibliography
- **Change:** Update bibitem to CQG 42, 165015.
