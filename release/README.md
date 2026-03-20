---
language: en
license: cc-by-4.0
library_name: timm
tags:
  - gravitational-waves
  - ligo
  - vision-transformer
  - image-classification
  - glitch-classification
  - gravity-spy
  - physics
  - deep-learning
  - spectrograms
  - continuous-gravitational-waves
  - resnet
  - detector-characterization
datasets:
  - custom
metrics:
  - f1
  - accuracy
pipeline_tag: image-classification
model-index:
  - name: ViT-B/16 Gravity Spy Glitch Classifier
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: Gravity Spy O3 (temporal split)
          type: custom
        metrics:
          - name: Macro-F1
            type: f1
            value: 0.723
          - name: Accuracy
            type: accuracy
            value: 0.934
          - name: Rare-class Macro-F1
            type: f1
            value: 0.241
  - name: ResNet-50v2 BiT Gravity Spy Glitch Classifier
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: Gravity Spy O3 (temporal split)
          type: custom
        metrics:
          - name: Macro-F1
            type: f1
            value: 0.679
          - name: Accuracy
            type: accuracy
            value: 0.918
          - name: Rare-class Macro-F1
            type: f1
            value: 0.303
---

# Gravity Spy Glitch Classifier: ViT-B/16 and ResNet-50v2 BiT

Two deep learning models for classifying LIGO gravitational-wave detector
glitch morphologies from Q-transform spectrograms, trained on Gravity Spy O3
data (Zevin et al. 2017, CQG 34 064003).

**Paper**: Manuscript submitted to Classical and Quantum Gravity
**Code**: [GitHub](https://github.com/JesseRWeigel/ligo-glitch-vit-cnn)

## Model Overview

| Property | ViT-B/16 | ResNet-50v2 BiT |
|---|---|---|
| Architecture | Vision Transformer (patch 16, 224px) | Big Transfer ResNet-50v2 |
| Pretrained weights | AugReg ImageNet-21k + ImageNet-1k | ImageNet-21k + ImageNet-1k |
| Parameters | ~86M | ~25M |
| Framework | timm | timm |

Both models classify 224x224 RGB Q-transform spectrograms into 23 Gravity Spy
glitch classes (22 glitch morphologies + No_Glitch). The class taxonomy follows
Zevin et al. (2017).

## Key Finding

Architecture preference is **class-dependent**: ViT excels on spectrally distinctive classes (Power_Line: +0.507 F1) but shows insufficient evidence of improvement on rare classes (< 200 training samples). Neither architecture uniformly dominates.

## Training Data

- **Source:** Gravity Spy O3 (H1 + L1), filtered to ml_confidence > 0.9
- **Total samples:** 227,943 training / 48,844 validation / 48,845 test
- **Split:** Temporal split (70/15/15%) with 60-second gap enforcement (prevents data leakage)
- **Classes:** 23 (see `src/class_labels.json`)
- **Preprocessing:** Q-transform spectrograms resized to 224x224, normalized with ImageNet statistics
- **Rare classes:** Chirp (11 train), Wandering_Line (30), Helix (33), Light_Modulation (142)

## Performance (O3 Test Set)

**Primary metric: macro-F1** (averaged equally across all 23 classes).

| Metric | ViT-B/16 | ResNet-50v2 BiT |
|---|---|---|
| **Macro-F1 (PRIMARY)** | **0.7230** [0.7031, 0.7397] | **0.6786** [0.6598, 0.6944] |
| Rare-class macro-F1 | 0.2412 [0.2019, 0.2957] | 0.3028 [0.2085, 0.3751] |
| Overall accuracy (secondary) | 0.9343 | 0.9181 |

*95% bootstrap confidence intervals (10,000 resamples). Macro-F1 difference is
statistically significant (p = 0.0002).*

> **Note:** Overall accuracy is reported as a secondary sanity check only.
> It is not the primary metric because it masks rare-class performance
> differences (see "Limitations" below).

### Per-Class Highlights

| Class | ViT F1 | CNN F1 | Difference | Favors |
|---|---|---|---|---|
| Power_Line | 0.742 | 0.235 | **+0.507** | ViT |
| Paired_Doves | 0.613 | 0.099 | +0.514 | ViT |
| Scratchy | 0.875 | 0.503 | +0.372 | ViT |
| Light_Modulation | 0.859 | 0.691 | +0.168 | ViT |
| Chirp | 0.000 | 0.471 | **−0.471** | CNN |
| Violin_Mode | 0.544 | 0.683 | −0.139 | CNN |
| Scattered_Light | 0.719 | 0.811 | −0.092 | CNN |

## O4 Generalization

Both models were evaluated on 38,587 O4a Gravity Spy spectrograms:

| Metric | ViT-B/16 | ResNet-50v2 BiT |
|---|---|---|
| O4 macro-F1 | 0.6695 [0.6555, 0.6816] | 0.6674 [0.6567, 0.6765] |
| Relative degradation from O3 | −7.4% | −1.7% |

Both models pass the <20% degradation threshold.

## Limitations

- **O3-trained only:** Not fine-tuned on O4 data. Performance may degrade on later observing runs.
- **Single-view:** Uses only the 1.0-second duration Q-transform view (Gravity Spy uses four).
- **23 classes:** New glitch morphologies in O4+ will be misclassified into existing categories.
- **Rare-class:** ViT's macro-F1 advantage does not extend to rare classes (< 200 training samples). The rare-class comparison is statistically underpowered (aggregate power = 0.20).
- **Single seed:** Results from one training run per architecture. No seed variance reported.
- **Label quality:** Training labels from ml_confidence > 0.9 filtering of Gravity Spy citizen science classifications.

## Usage

```bash
pip install torch timm albumentations numpy Pillow

# Classify a spectrogram with ViT
python src/inference.py --model vit --image path/to/spectrogram.png

# Classify with CNN, showing top-5 predictions
python src/inference.py --model cnn --image path/to/spectrogram.png --top-k 5

# JSON output
python src/inference.py --model vit --image path/to/spectrogram.png --json
```

### Input Format

- PNG or JPG image of a Q-transform spectrogram
- Any resolution (automatically resized to 224x224)
- RGB color (grayscale images are converted to 3-channel)

## File Structure

```
checkpoints/
  vit_b16_gravityspy_o3.pt        # ViT-B/16 weights (~983 MB)
  resnet50v2_gravityspy_o3.pt     # ResNet-50v2 BiT weights (~270 MB)
  checksums.sha256                # SHA-256 checksums
src/
  inference.py                    # Standalone CLI inference script
  preprocessing.py                # Locked evaluation transforms
  class_labels.json               # 23-class index-to-label mapping
  model_config.json               # Architecture and training config
examples/
  expected_output.json            # Validated predictions for test images
```

## Citation

```bibtex
@article{weigel2026vit_glitch,
  author = {Weigel, Jesse R.},
  title = {Vision Transformer vs.\ CNN for Gravitational Wave Glitch Classification:
           Class-Dependent Architecture Preferences and Implications for
           Continuous Wave Searches},
  year = {2026},
  note = {Manuscript submitted to Classical and Quantum Gravity}
}
```

## License

CC BY 4.0

## References

- Zevin, M. et al. (2017). "Gravity Spy: integrating advanced LIGO detector characterization, machine learning, and citizen science." *Classical and Quantum Gravity*, 34(6), 064003.
- Wu, Z. et al. (2025). "Multi-view Attention Fusion for Gravitational Wave Glitch Classification." *Classical and Quantum Gravity*, 42, 165015.
- Srivastava, A. and Niedzielski, T. (2025). "Vision Transformer for Transient Noise Classification in Gravitational Wave Data." *Acta Astronomica*, 74(3).
