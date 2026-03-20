# Manuscript Consistency Report

**Project:** When do Vision Transformers help? Class-dependent architecture preferences for gravitational-wave glitch classification
**Scope:** Full manuscript and supporting artifacts
**Checked:** 2026-03-18

## Cross-Phase Convention Consistency

| Convention | Declaration | Manuscript | Tables | Figures | Status |
|-----------|------------|-----------|--------|---------|--------|
| Primary metric | macro-F1 (ASSERT_CONVENTION) | Sec 2.4: macro-F1 primary | All tables use macro-F1 | Fig 1: per-class F1 | CONSISTENT |
| Delta sign | ViT minus CNN (ASSERT_CONVENTION) | Sec 3.1: Delta F1 = ViT - CNN | table_perclass: ViT-CNN | Fig 2: ViT-CNN | CONSISTENT |
| Units | SI (Hz, s) | Sec 2.1: 10-2048 Hz | -- | -- | CONSISTENT |
| Statistical test | bootstrap >= 10K, p < 0.05 | Sec 2.4: 10K resamples | Table 1: p-values | -- | CONSISTENT |
| Forbidden proxy | overall accuracy not primary | Sec 2.4: explicitly forbidden | Table 1 reports accuracy secondarily | -- | CONSISTENT |

## Number Macro Consistency

All 37 LaTeX number macros in paper/main.tex trace to paper/data/paper_numbers.json. The math review stage verified all key quantities to machine precision.

**One discrepancy found:**
- `\vitMacroFOfour` = 0.670 in main.tex, but source JSON value 0.6694683 rounds to 0.669 (REF-006). This is a double-rounding error, not a data pipeline failure.

## Equation Numbering Consistency

No numbered equations in the manuscript (metrics are defined inline). Cross-references to figures (5), tables (3), and sections are all resolved correctly.

## Notation Consistency

| Symbol | First use | Later uses | Status |
|--------|----------|-----------|--------|
| macro-F1 | Sec 2.4 (defined) | Throughout | CONSISTENT |
| F1_c | Sec 2.4 | Sec 3.1, Table 2 | CONSISTENT |
| Delta F1 | Sec 3.1 | Table 2, Fig 2 | CONSISTENT |
| Delta_DC | Sec 3.6 | Table 3 | CONSISTENT |
| N_train | Sec 2.1 | Sec 3.1, Table 2, Fig 2 | CONSISTENT |
| rho_s (Spearman) | Sec 3.3 | Abstract | CONSISTENT |

## Result Dependency Validation

| Consumed result | Source | Consumer | Match | Status |
|----------------|--------|----------|-------|--------|
| Per-class F1 values | results/03-vit-rare-class/ | paper_numbers.json | Verified by math stage | CONSISTENT |
| O4 evaluation | results/04-o4-validation/ | paper_numbers.json | Verified | CONSISTENT |
| CW veto results | results/04-o4-validation/cw_veto_results.json | paper_numbers.json | Verified; O(1e-4) bootstrap-mean discrepancy expected (MATH-003) | CONSISTENT |
| Power analysis | results/06-computation-statistical-analysis/power_analysis.json | paper text | Verified | CONSISTENT |
| Random-split ablation | results/06-computation-statistical-analysis/random_split_ablation.json | paper text | Verified | CONSISTENT |

## Cross-Artifact Alignment

| Artifact pair | Alignment check | Status |
|--------------|----------------|--------|
| REFEREE-REPORT.md vs REVIEW-LEDGER.json | Same 13 issue IDs, same severities, same blocking status | ALIGNED |
| REFEREE-REPORT.md vs REFEREE-DECISION.json | Same recommendation (major_revision), confidence (high), issue counts (5 major, 8 minor), blocking IDs | ALIGNED |
| REFEREE-REPORT.md vs REFEREE-REPORT.tex | Same recommendation, issue IDs, action matrix, evaluation ratings | ALIGNED |
| REVIEW-LEDGER.json vs REFEREE-DECISION.json | All 5 blocking ledger issues appear in decision blocking_issue_ids | ALIGNED |

## Summary

The manuscript's internal number pipeline is sound: all values trace from source data through paper_numbers.json to LaTeX macros with one minor double-rounding error (REF-006). Conventions (sign, units, primary metric) are consistent throughout. Cross-artifact alignment between the referee report, ledger, and decision is verified.
