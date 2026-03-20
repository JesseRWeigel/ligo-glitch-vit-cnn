# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-03-18)

**Machine-readable scoping contract:** `.gpd/state.json` field `project_contract`

**Core research question:** Can a Vision Transformer trained on LIGO O3 spectrograms outperform Gravity Spy's CNN at classifying rare glitch morphologies, and can this improvement translate to better data quality for continuous gravitational wave searches?
**Current focus:** All milestones complete. Next: CQG submission or `/gpd:new-milestone`

## Current Position

**Current Phase:** 7
**Current Phase Name:** Paper Text & Table Revision
**Total Phases:** 7 (5 from v1.0, 2 from v1.1)
**Status:** Milestone complete
**Last Activity:** 2026-03-18
**Last Activity Description:** v1.1 milestone completed and archived

**Progress:** [██████████] 100% (11/11 plans complete across all milestones)

## Active Calculations

None.

## Intermediate Results

- 325,634 O3 Gravity Spy glitches after ml_confidence > 0.9 filter (H1: 121,627, L1: 204,007)
- 23 classes (22 glitch + No_Glitch, NOTA excluded)
- 1 rare class: Chirp (19 total, 11 train)
- Temporal split: 227,943 train / 48,844 val / 48,845 test (70/15/15%)
- **CNN test macro-F1:** 0.6786 [0.6598, 0.6944]
- **CNN rare-class macro-F1:** 0.3028 [0.2085, 0.3751]
- **ViT test macro-F1:** 0.7230 [0.7018, 0.7413] -- significantly above CNN (p=0.0002)
- **ViT rare-class macro-F1:** 0.2412 [0.2019, 0.2957] -- BELOW CNN baseline
- **O4 evaluation:** 38,587 O4a glitches, all 23 classes
- **CNN macro-F1 O4:** 0.6674 [0.6567, 0.6765] -- degradation: -1.7% relative (PASS)
- **ViT macro-F1 O4:** 0.6695 [0.6555, 0.6816] -- degradation: -7.4% relative (PASS)
- **Power_Line:** ViT F1 0.725 vs CNN 0.331, diff = +0.394 (strongest ViT advantage, both O3 and O4)
- **CW veto at matched deadtime (22.4%):** ViT efficiency 0.745 vs CNN 0.735 (~equal)
- **CW-critical classes:** 2/7 favor ViT (Power_Line, Violin_Mode), 5/7 favor CNN
- **Random-split CNN accuracy:** 0.9544 [0.9525, 0.9563] -- +3.63pp above temporal-split (0.9181)
- **Aggregate rare-class power:** 0.20 (severely underpowered; conclusion: insufficient evidence)

## Performance Metrics

| Metric | Value | 95% CI |
|--------|-------|--------|
| CNN macro-F1 (O3) | 0.6786 | [0.6598, 0.6944] |
| ViT macro-F1 (O3) | 0.7230 | [0.7018, 0.7413] |
| CNN rare-class macro-F1 | 0.3028 | [0.2085, 0.3751] |
| ViT rare-class macro-F1 | 0.2412 | [0.2019, 0.2957] |
| CNN macro-F1 (O4) | 0.6674 | [0.6567, 0.6765] |
| ViT macro-F1 (O4) | 0.6695 | [0.6555, 0.6816] |
| CW veto eff. (ViT) | 0.745 | -- |
| CW veto eff. (CNN) | 0.735 | -- |

## Open Questions

- Would rare-class-specific interventions (augmentation, few-shot, contrastive learning) resolve the rare-class gap?
- Is frequency-resolved PSD analysis needed to quantify Power_Line CW benefit more precisely?
- Would a two-stage model (ViT common + specialized rare) be viable?

## Accumulated Context

### Decisions

Full log: `.gpd/DECISIONS.md`

### Active Approximations

- ml_confidence > 0.9 label filter (~25% data discarded)
- 60s temporal gap (minimum heuristic; some instrumental correlations persist longer)
- Pre-made Gravity Spy images used (bilinear resize to 224x224 if needed)
- Duty cycle as CW sensitivity proxy (valid when noise floor comparable across segments)

**Convention Lock:**

- Natural units: SI units: strain dimensionless, frequency Hz, time s
- Coordinate system: Q-transform spectrograms: 10-2048 Hz log axis, 224x224 px, min-max [0,1] after SNR clip [0,25.5]
- Coupling convention: Macro-F1 primary metric; overall accuracy is forbidden proxy; bootstrap >= 10K resamples, p < 0.05

### Propagated Uncertainties

- O4 Gravity Spy labels may be noisier than O3 (newer classifier, fewer volunteer classifications)
- Duty cycle is a coarse CW proxy; frequency-resolved PSD would be more precise

### Pending Todos

None.

### Blockers/Concerns

None active.

## Session Continuity

**Last session:** 2026-03-18
**Stopped at:** v1.1 milestone complete. Manuscript ready for CQG submission.
**Resume file:** --
