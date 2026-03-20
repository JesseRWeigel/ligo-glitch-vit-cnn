# Vision Transformer vs CNN for LIGO Gravitational Wave Glitch Classification

A controlled comparison of Vision Transformer (ViT-B/16) and CNN (ResNet-50v2 BiT) architectures for classifying gravitational wave detector glitches in LIGO data, with a focus on rare glitch classes and continuous gravitational wave (CW) search sensitivity.

## Key Findings

| Metric | CNN | ViT | Δ | p-value |
|--------|-----|-----|---|---------|
| Overall macro-F1 | 0.679 | 0.723 | **+0.044** | 0.0002 |
| Rare-class macro-F1 | 0.303 | 0.241 | −0.062 | 0.884 |
| Overall accuracy | 91.8% | 93.4% | +1.6pp | — |

- **Architecture preference is class-dependent, not sample-size-dependent** (Spearman ρ = −0.034, p = 0.879 on O4)
- **Power_Line is the standout CW-relevant finding**: ViT F1 advantage of +0.507, consistent across O3→O4
- **Both models generalize to O4**: CNN −1.7%, ViT −7.4% relative macro-F1 degradation
- **Temporal splitting reveals prior evaluations overestimated accuracy by ~3.6pp** due to data leakage
- **Rare-class comparison is statistically underpowered** (aggregate power = 0.20) — framed as "insufficient evidence" rather than directional claim

## Methodology

- **Dataset**: Gravity Spy O3 (Zenodo 5649212), 325,632 glitches across 23 classes, filtered to ml_confidence > 0.9
- **Split**: Temporal train/val/test (70/15/15%) with 60-second gap enforcement — prevents data leakage from temporally correlated glitches
- **Training recipe**: Identical for both architectures (AdamW, cosine schedule, focal loss γ=2.0, class-balanced sampling, fp16)
- **Evaluation**: Macro-F1 as primary metric (overall accuracy explicitly forbidden as primary), paired bootstrap with 10,000 resamples
- **Validation**: O4 distribution shift test, CW veto efficiency analysis at matched operating points

## Repository Structure

```
├── scripts/                    # Numbered pipeline scripts (01-18)
│   ├── 01_download_metadata.py    # Fetch Gravity Spy metadata from Zenodo
│   ├── 02_parse_metadata.py       # Parse and filter O3 data
│   ├── 03_download_spectrograms.py # Download spectrogram images
│   ├── 06_temporal_split.py       # Temporal train/val/test split
│   ├── 07_verify_split.py         # Programmatic split verification
│   ├── 09_train_cnn_baseline.py   # Train ResNet-50v2 BiT
│   ├── 11_train_vit.py            # Train ViT-B/16
│   ├── 13_comparison_deliverables.py # Paired bootstrap comparison
│   ├── 16_cw_veto_analysis.py     # CW sensitivity analysis
│   ├── 17_random_split_ablation.py # Temporal vs random split ablation
│   └── 18_power_analysis.py       # Statistical power analysis
├── src/                        # Model and training code
│   ├── models/                    # ViT and CNN model definitions
│   ├── training/                  # Training loop, focal loss
│   ├── evaluation/                # Metrics, bootstrap testing
│   ├── data/                      # Dataset loaders, transforms
│   └── losses/                    # Focal loss implementation
├── paper/                      # LaTeX manuscript
│   ├── main.tex                   # CQG-formatted paper
│   ├── main.pdf                   # Compiled PDF
│   ├── figures/                   # 5 publication-quality figures
│   ├── tables/                    # 3 LaTeX tables
│   └── data/paper_numbers.json    # Auto-extracted numbers (zero manual transcription)
├── results/                    # All experimental results
│   ├── 02-cnn-baseline/           # CNN metrics, confusion matrix
│   ├── 03-vit-rare-class/         # ViT metrics, bootstrap results
│   ├── 04-o4-validation/          # O4 generalization results
│   └── 06-random-split-cnn/       # Ablation study results
├── release/                    # Model release package
│   ├── checkpoints/               # Model weights (see HuggingFace)
│   ├── src/                       # Standalone inference code
│   └── README.md                  # Model card
├── configs/                    # Training configurations
├── docs/                       # Experimental protocol
│   └── experimental_protocol.md   # Locked evaluation protocol
├── figures/                    # Generated analysis figures
└── .gpd/                       # GPD research project artifacts
    ├── PROJECT.md                 # Research context
    ├── CONVENTIONS.md             # Locked conventions
    ├── ROADMAP.md                 # Phase structure
    ├── phases/                    # Per-phase plans, research, summaries
    ├── research/                  # Literature survey
    ├── milestones/                # Archived milestone artifacts
    └── REFEREE-REPORT.md          # Mock peer review
```

## Reproducing Results

### Prerequisites

- Python 3.12+
- NVIDIA GPU with ≥12GB VRAM (trained on RTX 5090 32GB)
- ~70GB disk space for dataset

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Full Pipeline

```bash
# Phase 1: Data acquisition and preparation
python scripts/01_download_metadata.py
python scripts/02_parse_metadata.py
python scripts/03_download_spectrograms.py
python scripts/04_class_distribution.py
python scripts/06_temporal_split.py
python scripts/07_verify_split.py     # Must pass all 5 checks

# Phase 2: CNN baseline
python scripts/09_train_cnn_baseline.py
python scripts/10_evaluate_cnn_baseline.py

# Phase 3: ViT training and comparison
python scripts/11_train_vit.py
python scripts/12_evaluate_vit.py
python scripts/13_comparison_deliverables.py  # Paired bootstrap

# Phase 4: O4 validation and CW analysis
python scripts/15_evaluate_o4.py
python scripts/16_cw_veto_analysis.py
python scripts/16_threshold_test.py

# Phase 6: Ablation and power analysis
python scripts/17_random_split_ablation.py
python scripts/18_power_analysis.py
```

### Pre-trained Models

Model weights are available on [HuggingFace](https://huggingface.co/JesseWeigel/ligo-glitch-vit-cnn) or can be reproduced from scratch using the scripts above.

## Dataset

The Gravity Spy O3 dataset is publicly available:
- **Metadata**: [Zenodo record 5649212](https://zenodo.org/records/5649212) — ML classifications for O1-O3b
- **Spectrograms**: Downloaded via Gravity Spy API (url columns in metadata CSV)
- **O4 validation data**: Fetched via Gravity Spy API for distribution shift testing

## Built With

This research was conducted using [Get Physics Done (GPD)](https://github.com/psi-oss/get-physics-done), an AI copilot for physics research. GPD provided structured workflow management, convention enforcement, verification, and peer review across all 7 phases of the project.

## Citation

If this work is useful to your research, please cite:

```bibtex
@article{weigel2026vit_glitch,
  author = {Weigel, Jesse R.},
  title = {Vision Transformer vs.\ CNN for Gravitational Wave Glitch Classification: Class-Dependent Architecture Preferences and Implications for Continuous Wave Searches},
  year = {2026},
  note = {Manuscript submitted to Classical and Quantum Gravity}
}
```

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt this work with attribution.

## Acknowledgments

- The [Gravity Spy](https://www.zooniverse.org/projects/zooniverse/gravity-spy) project for the labeled glitch dataset
- LIGO is funded by the U.S. National Science Foundation
- [Physical Superintelligence PBC](https://www.psi.inc) for the GPD research framework
