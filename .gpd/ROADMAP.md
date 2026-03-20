# Roadmap: Transformer-Based Rare Glitch Classification for LIGO CW Searches

## Milestones

- **v1.0 Transformer-Based Rare Glitch Classification** -- Phases 1-5 (completed 2026-03-18)
- **v1.1 Paper Revision for CQG Submission** -- Phases 6-7 (completed 2026-03-18)

## Phases

<details>
<summary>v1.0 Transformer-Based Rare Glitch Classification (Phases 1-5) -- COMPLETED 2026-03-18</summary>

- [x] Phase 1: Data Pipeline & Experimental Design (2/2 plans) -- completed 2026-03-16
- [x] Phase 2: CNN Baseline Reproduction (1/1 plan) -- completed 2026-03-17
- [x] Phase 3: ViT Training & Per-Class Comparison (2/2 plans) -- completed 2026-03-17
- [x] Phase 4: O4 Validation & CW Sensitivity (2/2 plans) -- completed 2026-03-17
- [x] Phase 5: Paper & Model Packaging (2/2 plans) -- completed 2026-03-17

**Key findings:** Architecture preference is class-dependent. ViT excels on Power_Line (+0.507pp) but regresses on ultra-rare classes. CW veto efficiency approximately equal at matched deadtime. Paper drafted for CQG.

**Full archive:** `.gpd/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>v1.1 Paper Revision for CQG Submission (Phases 6-7) -- COMPLETED 2026-03-18</summary>

- [x] Phase 6: Computation & Statistical Analysis (1/1 plan) -- completed 2026-03-18
- [x] Phase 7: Paper Text & Table Revision (1/1 plan) -- completed 2026-03-18

**Key findings:** Random-split ablation (95.4% accuracy) confirmed temporal split explains the benchmark accuracy gap. Power analysis showed all 8 rare-class comparisons underpowered (aggregate power 0.20). All 9 referee issues resolved. Manuscript submission-ready for CQG.

**Full archive:** `.gpd/milestones/v1.1-ROADMAP.md`

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
| ----- | --------- | -------------- | ------ | --------- |
| 1. Data Pipeline | v1.0 | 2/2 | Complete | 2026-03-16 |
| 2. CNN Baseline | v1.0 | 1/1 | Complete | 2026-03-17 |
| 3. ViT Training | v1.0 | 2/2 | Complete | 2026-03-17 |
| 4. O4 & CW | v1.0 | 2/2 | Complete | 2026-03-17 |
| 5. Paper & Models | v1.0 | 2/2 | Complete | 2026-03-17 |
| 6. Computation & Analysis | v1.1 | 1/1 | Complete | 2026-03-18 |
| 7. Paper Revision | v1.1 | 1/1 | Complete | 2026-03-18 |

---

_Created: 2026-03-18_
_Last updated: 2026-03-18 after v1.1 milestone completion_
