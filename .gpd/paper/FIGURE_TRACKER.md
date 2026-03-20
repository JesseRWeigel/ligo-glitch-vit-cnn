---
figure_registry:
  - id: fig-per-class-f1
    label: "Fig. 1"
    kind: figure
    role: smoking_gun
    path: paper/figures/fig_per_class_f1.pdf
    contract_ids: [claim-threshold, deliv-paper]
    decisive: true
    has_units: false
    has_uncertainty: true
    referenced_in_text: true
    caption_self_contained: true
    colorblind_safe: true
  - id: fig-threshold-scatter
    label: "Fig. 2"
    kind: figure
    role: comparison
    path: paper/figures/fig_threshold_scatter.pdf
    contract_ids: [claim-threshold, deliv-paper]
    decisive: true
    has_units: false
    has_uncertainty: false
    referenced_in_text: true
    caption_self_contained: true
    colorblind_safe: true
  - id: fig-confusion-matrices
    label: "Fig. 3"
    kind: figure
    role: comparison
    path: paper/figures/fig_confusion_matrices.pdf
    contract_ids: [deliv-paper]
    decisive: false
    has_units: false
    has_uncertainty: false
    referenced_in_text: true
    caption_self_contained: true
    colorblind_safe: true
  - id: fig-o4-degradation
    label: "Fig. 4"
    kind: figure
    role: benchmark
    path: paper/figures/fig_o4_degradation.pdf
    contract_ids: [claim-threshold, deliv-paper]
    decisive: true
    has_units: false
    has_uncertainty: true
    referenced_in_text: true
    caption_self_contained: true
    colorblind_safe: true
  - id: fig-cw-veto
    label: "Fig. 5"
    kind: figure
    role: comparison
    path: paper/figures/fig_cw_veto.pdf
    contract_ids: [claim-cw-benefit, deliv-paper]
    decisive: true
    has_units: false
    has_uncertainty: true
    referenced_in_text: true
    caption_self_contained: true
    colorblind_safe: true
---

# Figure Tracker: When do Vision Transformers help?

**Total figures:** 5 planned, 5 complete
**Target journal:** Classical and Quantum Gravity (CQG)

## Format Requirements

- **File format:** PDF (vector) for all figures
- **Column width:** single-column: 3.4 in; double-column: 7.0 in
- **Font size:** axis labels >= 10 pt
- **Color:** colorblind-safe palette via seaborn
- **Style file:** paper/scripts/generate_figures.py

## Figure Registry

### Fig. 1: Per-class F1 comparison (ViT vs CNN)

| Field        | Value |
| ------------ | ----- |
| Type         | Grouped bar chart |
| Role         | smoking_gun |
| Source phase | Phase 5 (data from Phases 2-3) |
| Source file  | paper/scripts/generate_figures.py |
| Data file(s) | results/03-vit-rare-class/comparison_table.csv |
| Size         | Double-column |
| Status       | Draft |
| Last updated | 2026-03-17 |

### Fig. 2: Training set size vs F1 difference scatter

| Field        | Value |
| ------------ | ----- |
| Type         | Scatter |
| Role         | comparison |
| Source phase | Phase 5 (data from Phases 3-4) |
| Source file  | paper/scripts/generate_figures.py |
| Data file(s) | results/03-vit-rare-class/comparison_table.csv, results/04-o4-validation/o4_threshold_test.json |
| Size         | Single-column |
| Status       | Draft |
| Last updated | 2026-03-17 |

### Fig. 3: Confusion matrices (ViT and CNN)

| Field        | Value |
| ------------ | ----- |
| Type         | Heatmap |
| Role         | comparison |
| Source phase | Phase 5 (data from Phase 3) |
| Source file  | paper/scripts/generate_figures.py |
| Data file(s) | results/03-vit-rare-class/comparison_table.csv |
| Size         | Double-column |
| Status       | Draft |
| Last updated | 2026-03-17 |

### Fig. 4: O4 degradation analysis

| Field        | Value |
| ------------ | ----- |
| Type         | Bar chart |
| Role         | benchmark |
| Source phase | Phase 5 (data from Phase 4) |
| Source file  | paper/scripts/generate_figures.py |
| Data file(s) | results/04-o4-validation/o4_comparison_table.csv |
| Size         | Single-column |
| Status       | Draft |
| Last updated | 2026-03-17 |

### Fig. 5: CW veto efficiency comparison

| Field        | Value |
| ------------ | ----- |
| Type         | Bar chart |
| Role         | comparison |
| Source phase | Phase 5 (data from Phase 4) |
| Source file  | paper/scripts/generate_figures.py |
| Data file(s) | results/04-o4-validation/cw_veto_results.json, results/04-o4-validation/cw_duty_cycle_comparison.csv |
| Size         | Single-column |
| Status       | Draft |
| Last updated | 2026-03-17 |

## Table Registry

### Table I: Overall comparison (O3 and O4)

| Field        | Value |
| ------------ | ----- |
| Source phase | Phase 5 (data from Phases 2-4) |
| Source file  | paper/tables/generate_tables.py |
| Status       | Draft |
| Last updated | 2026-03-17 |

### Table II: Per-class F1 for all 23 classes

| Field        | Value |
| ------------ | ----- |
| Source phase | Phase 5 (data from Phase 3) |
| Source file  | paper/tables/generate_tables.py |
| Status       | Draft |
| Last updated | 2026-03-17 |

### Table III: CW-critical class analysis

| Field        | Value |
| ------------ | ----- |
| Source phase | Phase 5 (data from Phase 4) |
| Source file  | paper/tables/generate_tables.py |
| Status       | Draft |
| Last updated | 2026-03-17 |

## Progress Summary

| Figure  | Data Ready | Script Written | Draft Plot | Polished | Final |
| ------- | ---------- | -------------- | ---------- | -------- | ----- |
| Fig. 1  | [x]        | [x]            | [x]        | [ ]      | [ ]   |
| Fig. 2  | [x]        | [x]            | [x]        | [ ]      | [ ]   |
| Fig. 3  | [x]        | [x]            | [x]        | [ ]      | [ ]   |
| Fig. 4  | [x]        | [x]            | [x]        | [ ]      | [ ]   |
| Fig. 5  | [x]        | [x]            | [x]        | [ ]      | [ ]   |
| Table I | [x]        | [x]            | [x]        | [ ]      | [ ]   |
| Table II| [x]        | [x]            | [x]        | [ ]      | [ ]   |
| Table III| [x]       | [x]            | [x]        | [ ]      | [ ]   |
