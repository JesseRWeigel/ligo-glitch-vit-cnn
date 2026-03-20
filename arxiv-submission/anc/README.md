# Ancillary Files

Computational scripts for reproducing the results in this paper.

## Training Pipeline

| Script | Description |
|--------|-------------|
| 01_download_metadata.py | Download Gravity Spy metadata from Zenodo |
| 02_parse_metadata.py | Parse and filter metadata (ml_confidence > 0.9) |
| 03_download_spectrograms.py | Download spectrogram images |
| 04_class_distribution.py | Analyze class distribution |
| 05_finalize_after_download.sh | Post-download verification |
| 06_temporal_split.py | GPS-time-based temporal train/val/test split |
| 07_verify_split.py | Verify split integrity and temporal non-overlap |
| 08_split_visualization.py | Visualize split statistics |
| 09_train_cnn_baseline.py | Train ResNet-50 CNN baseline |
| 10_evaluate_cnn_baseline.py | Evaluate CNN on test set with bootstrap CIs |
| 11_train_vit.py | Train ViT-B/16 with AugReg pre-training |
| 12_evaluate_vit.py | Evaluate ViT on test set with bootstrap CIs |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torchvision, timm, scikit-learn, pandas, matplotlib

## Relationship to Paper Results

- Scripts 01-08 produce the dataset described in Section 2
- Scripts 09-10 produce CNN results in Tables 1-3
- Scripts 11-12 produce ViT results in Tables 1-3
- Figure generation: see `generate_figures.py` (in paper/scripts/)
