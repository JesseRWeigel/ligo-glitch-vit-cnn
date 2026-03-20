# Milestones

## v1.0 Transformer-Based Rare Glitch Classification (Shipped: 2026-03-18)

**Phases completed:** 5 phases, 9 plans, 0 tasks

**Key accomplishments:**
- Downloaded Gravity Spy O3 metadata (325K glitches, 23 classes) and launched async image downloader for 1.3M pre-made spectrogram images at 224x224 RGB
- Constructed temporal train/val/test split (70/15/15) with 60s gap enforcement and locked experimental protocol with macro-F1 as decisive metric
- Trained ResNet-50 CNN baseline on Gravity Spy O3: macro-F1=0.6786, rare-class gap=50.5pp, establishing the per-class F1 floor the ViT must beat
- Trained ViT-B/16 on Gravity Spy O3 spectrograms with layer-wise LR decay -- val macro-F1=0.7810 at epoch 20, surpassing CNN val baseline (0.6618)
- ViT-B/16 test evaluation reveals forbidden proxy scenario: overall macro-F1 improves (0.7230 vs 0.6786, p<0.001) but rare-class macro-F1 regresses (0.2412 vs 0.3028, p=0.88) -- backtracking trigger ACTIVATED
- CW veto analysis: at matched deadtime (22.4%) ViT efficiency=0.745 vs CNN=0.735 (~equal); overall delta_DC=-0.051 (ViT vetoes more aggressively); benefit is class-specific, not blanket
- Packaged ViT-B/16 and ResNet-50v2 BiT as standalone inference artifacts with locked preprocessing, validated against Phase 3 predictions with zero drift
- Complete CQG paper draft with number extraction pipeline, 5 journal-quality figures, 3 LaTeX tables, and honest framing of class-morphology-dependent architecture preferences including O4 non-replication and quantitative CW analysis

---


## v1.1 Paper Revision for CQG Submission (Shipped: 2026-03-18)

**Phases completed:** 2 phases (6-7), 2 plans, 4 tasks

**Key accomplishments:**
- Random-split CNN ablation (95.4% accuracy, +3.6pp gap) confirms temporal split explains benchmark discrepancy; power analysis shows all 8 rare-class comparisons underpowered at the observed -0.062 macro-F1 effect
- Revised CQG manuscript addressing all 9 referee issues: reframed 'class-morphology-dependent' to 'class-dependent', restructured CW table with dual F1/DC advantage columns, integrated Phase 6 power analysis (aggregate power 0.20) and random-split ablation (95.4% accuracy)

---

