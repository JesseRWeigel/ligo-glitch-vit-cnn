# ViT vs CNN Statistical Comparison -- Phase 3 Results

## Primary Result (Rare-Class Macro-F1)

- CNN: 0.3028 [0.2085, 0.3751]
- ViT: 0.2412 [0.2019, 0.2957]
- Difference (ViT - CNN): -0.0617 [-0.1463, 0.0408]
- Paired bootstrap p-value: 0.884200 (H0: ViT <= CNN)
- **Verdict: ViT WORSE (rare-class F1 lower than CNN, no improvement)**

## Per-Class Rare Results

- Chirp (11 train, 7 test): CNN F1=0.4706, ViT F1=0.0000, diff=-0.4706
- Wandering_Line (30 train, 6 test): CNN F1=0.0000, ViT F1=0.0000, diff=0.0000
- Helix (33 train, 14 test): CNN F1=0.0494, ViT F1=0.1053, diff=0.0559
- Light_Modulation (142 train, 66 test): CNN F1=0.6914, ViT F1=0.8594, diff=0.1680

## Secondary Results

- Overall macro-F1: CNN=0.6786, ViT=0.7230 (diff=0.0444, p=0.000200)
- Overall accuracy (SANITY CHECK): CNN=0.9181, ViT=0.9343

**Note:** The ViT achieves significantly higher overall macro-F1 (p=0.0002) and overall accuracy,
but this improvement comes entirely from common classes. This is exactly the forbidden proxy scenario
(fp-overall-accuracy): overall accuracy improvement without rare-class F1 improvement.

## Contract Assessment

- **claim-rare-improvement: FAIL**
- test-rare-f1-improvement: FAIL (ViT rare-class F1 <= CNN)
- test-metric-consistency: PASS (sklearn vs torchmetrics diff < 1e-6 for both models)
- test-paired-bootstrap-valid: PASS (10K resamples, paired indices, matching test hash, correct rare classes)
- test-same-test-set: PASS (SHA-256 hash matches: c8f3865d1bf36e1c...)
- fp-overall-accuracy enforcement: CONFIRMED -- rare-class macro-F1 is primary metric throughout; overall accuracy labeled SANITY CHECK in all outputs
- **Backtracking trigger: TRIGGERED -- ViT rare-class macro-F1 (0.2412) is BELOW CNN baseline (0.3028)**

## Interpretation

The ViT-B/16 trained with identical recipe (focal loss, class-balanced sampling, identical augmentation)
achieves substantially higher overall macro-F1 than the CNN (0.7230 vs 0.6786, +4.4pp, p < 0.001).
However, this improvement is concentrated in common classes. On the rare classes that are the project's
primary concern, the ViT actually performs slightly worse:

- Chirp: CNN=0.471, ViT=0.000
- Wandering_Line: CNN=0.000, ViT=0.000
- Helix: CNN=0.049, ViT=0.105
- Light_Modulation: CNN=0.691, ViT=0.859

The ViT's attention mechanism may help with common-class disambiguation but does not provide
sufficient advantage for rare classes with very few training examples (11-142 samples).
Rare-class classification likely requires targeted interventions beyond architecture change:

- Data augmentation specific to rare morphologies
- Few-shot learning or meta-learning approaches
- Synthetic data generation for rare classes
- Contrastive learning to learn rare-class representations from limited examples
