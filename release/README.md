# Gravity Spy Glitch Classifier: ViT-B/16 and ResNet-50v2 BiT

Two deep learning models for classifying LIGO gravitational-wave detector
glitch morphologies from Q-transform spectrograms, trained on Gravity Spy O3
data (Zevin et al. 2017, CQG 34 064003).

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

## Training Data

- **Source:** Gravity Spy O3 (H1 + L1), filtered to ml_confidence > 0.9
- **Total samples:** 227,943 training / 48,844 validation / 48,845 test
- **Split:** Temporal split (70/15/15%) with 60-second gap enforcement
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

Architecture preference is **class-morphology-dependent**, not a uniform
advantage for either model:

| Class | ViT F1 | CNN F1 | Difference | Favors |
|---|---|---|---|---|
| Power_Line | 0.742 | 0.235 | **+0.507** | ViT |
| Light_Modulation | 0.859 | 0.691 | +0.168 | ViT |
| Scratchy | 0.875 | 0.503 | +0.372 | ViT |
| Chirp | 0.000 | 0.471 | **+0.471** | CNN |
| Scattered_Light | 0.719 | 0.811 | +0.092 | CNN |
| Paired_Doves | 0.613 | 0.099 | +0.514 | ViT |
| Violin_Mode | 0.544 | 0.683 | +0.139 | CNN |

The ViT achieves higher overall macro-F1 but performs **worse on rare classes**
(macro-F1 0.241 vs CNN 0.303). This is a forbidden-proxy scenario: overall
accuracy improvement does not extend to rare classes.

## O4 Generalization

Both models were evaluated on 38,587 O4a Gravity Spy spectrograms:

| Metric | ViT-B/16 | ResNet-50v2 BiT |
|---|---|---|
| O4 macro-F1 | 0.6695 [0.6555, 0.6816] | 0.6674 [0.6567, 0.6765] |
| Relative degradation from O3 | -7.4% | -1.7% |

Both models pass the <20% degradation threshold. The CNN shows better
generalization stability across observing runs.

## Limitations

- **O3-trained only:** Models have not been fine-tuned on O4 data.
  Performance may degrade further on later observing runs with new
  instrumental configurations.
- **Single-view:** Models use only the 1.0-second duration Q-transform view.
  Gravity Spy's full pipeline uses four duration views (0.5s, 1.0s, 2.0s, 4.0s).
- **23 classes:** New glitch morphologies appearing in O4+ will be
  misclassified into existing categories.
- **Rare-class regression for ViT:** The ViT's macro-F1 advantage does not
  extend to rare classes (< 200 training samples). The CNN outperforms on
  Chirp, Helix, and Violin_Mode.
- **Label quality:** Training labels are from ml_confidence > 0.9 filtering
  of Gravity Spy citizen science classifications, not expert-reviewed labels.

## Usage

```bash
# Install dependencies
pip install torch timm albumentations numpy Pillow

# Classify a spectrogram with ViT
python release/src/inference.py --model vit --image path/to/spectrogram.png

# Classify with CNN, showing top-5 predictions
python release/src/inference.py --model cnn --image path/to/spectrogram.png --top-k 5

# JSON output
python release/src/inference.py --model vit --image path/to/spectrogram.png --json
```

### Input Format

- PNG or JPG image of a Q-transform spectrogram
- Any resolution (automatically resized to 224x224)
- RGB color (grayscale images are converted to 3-channel)

### Output

```
Model: ViT-B/16
Image: spectrogram.png

Predictions:
Rank   Class                     Probability
-------------------------------------------
1      Blip                      0.9823
2      Blip_Low_Frequency        0.0091
3      Koi_Fish                  0.0034
```

## File Structure

```
release/
  README.md                              # This file (model card)
  checkpoints/
    vit_b16_gravityspy_o3.pt             # ViT-B/16 weights (~983 MB)
    resnet50v2_gravityspy_o3.pt          # ResNet-50v2 BiT weights (~270 MB)
    checksums.sha256                     # SHA-256 checksums for integrity
  src/
    inference.py                         # Standalone CLI inference script
    preprocessing.py                     # Locked evaluation transforms
    class_labels.json                    # 23-class index-to-label mapping
    model_config.json                    # Architecture and training config
  examples/
    expected_output.json                 # Validated predictions for test images
```

## Verifying Checkpoint Integrity

```bash
cd release/checkpoints
sha256sum -c checksums.sha256
```

## Citation

```bibtex
@misc{gravitspy_vit_cnn_2026,
  title={Vision Transformer vs. CNN for LIGO Glitch Classification:
         Class-Morphology-Dependent Architecture Preferences},
  author={[Authors]},
  year={2026},
  note={Models trained on Gravity Spy O3 data (Zevin et al. 2017, CQG 34 064003)}
}
```

## License

MIT

## References

- Zevin, M. et al. (2017). "Gravity Spy: integrating advanced LIGO detector
  characterization, machine learning, and citizen science." *Classical and
  Quantum Gravity*, 34(6), 064003.
