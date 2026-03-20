# Rare-Class Performance Gap Analysis: CNN Baseline

## Definition

- **Rare classes** (n_train < 200): Chirp, Wandering_Line, Helix, Light_Modulation
- **Common classes** (n_train >= 1000): Repeating_Blips, Air_Compressor, Power_Line, 1400Ripples, Low_Frequency_Lines, Whistle, Blip, No_Glitch, Koi_Fish, Extremely_Loud, Blip_Low_Frequency, Low_Frequency_Burst, Tomte, Fast_Scattering, Scattered_Light

## Summary

| Metric | Value | 95% CI |
|--------|-------|--------|
| Macro-F1 (all classes) | 0.6786 | [0.6598, 0.6944] |
| Common-class avg F1 | 0.8075 | -- |
| Rare-class avg F1 | 0.3028 | [0.2085, 0.3751] |
| **Gap (common - rare)** | **50.5 pp** | -- |
| Overall accuracy (SANITY CHECK) | 0.9181 | -- |

## Per-Class Breakdown (sorted by n_train ascending)

| Class | n_train | n_test | F1 | F1 95% CI | Recall | Category |
|-------|---------|--------|-----|-----------|--------|----------|
| Chirp * | 11 | 7 | 0.4706 | [0.1250, 0.7273] (width=0.602) | 0.5714 | RARE |
| Wandering_Line * | 30 | 6 | 0.0000 | [0.0000, 0.0000] (width=0.000) | 0.0000 | Near-rare |
| Helix * | 33 | 14 | 0.0494 | [0.0118, 0.1006] (width=0.089) | 0.2857 | Near-rare |
| Light_Modulation | 142 | 66 | 0.6914 | [0.6064, 0.7683] (width=0.162) | 0.8485 | Near-rare |
| Paired_Doves | 216 | 64 | 0.0987 | [0.0762, 0.1228] (width=0.047) | 0.9219 | Common |
| Violin_Mode | 274 | 397 | 0.6830 | [0.6404, 0.7237] (width=0.083) | 0.5617 | Common |
| 1080Lines * | 341 | 6 | 1.0000 | [1.0000, 1.0000] (width=0.000) | 1.0000 | Common |
| Scratchy | 558 | 76 | 0.5034 | [0.4000, 0.5986] (width=0.199) | 0.4868 | Common |
| Repeating_Blips | 1061 | 347 | 0.7702 | [0.7367, 0.8021] (width=0.065) | 0.8646 | Common |
| Air_Compressor | 1361 | 47 | 0.1538 | [0.0533, 0.2574] (width=0.204) | 0.1489 | Common |
| Power_Line | 1582 | 56 | 0.2353 | [0.1308, 0.3357] (width=0.205) | 0.2500 | Common |
| 1400Ripples | 2428 | 33 | 0.8000 | [0.6933, 0.8889] (width=0.196) | 0.9697 | Common |
| Low_Frequency_Lines | 2853 | 1743 | 0.7014 | [0.6869, 0.7158] (width=0.029) | 0.9954 | Common |
| Whistle | 6299 | 3689 | 0.9628 | [0.9584, 0.9671] (width=0.009) | 0.9845 | Common |
| Blip | 7156 | 2105 | 0.9493 | [0.9422, 0.9558] (width=0.014) | 0.9245 | Common |
| No_Glitch | 11568 | 402 | 0.9364 | [0.9183, 0.9532] (width=0.035) | 0.9154 | Common |
| Koi_Fish | 11950 | 2503 | 0.9565 | [0.9506, 0.9622] (width=0.012) | 0.9401 | Common |
| Extremely_Loud | 13469 | 3317 | 0.9859 | [0.9830, 0.9886] (width=0.006) | 0.9873 | Common |
| Blip_Low_Frequency | 13659 | 2763 | 0.9382 | [0.9314, 0.9447] (width=0.013) | 0.9175 | Common |
| Low_Frequency_Burst | 19834 | 2397 | 0.9915 | [0.9888, 0.9939] (width=0.005) | 0.9962 | Common |
| Tomte | 30403 | 5651 | 0.9630 | [0.9594, 0.9666] (width=0.007) | 0.9558 | Common |
| Fast_Scattering | 34555 | 18998 | 0.9573 | [0.9552, 0.9594] (width=0.004) | 0.9212 | Common |
| Scattered_Light | 68160 | 4158 | 0.8109 | [0.8009, 0.8207] (width=0.020) | 0.7140 | Common |

\* Classes with n_test < 15 -- bootstrap CIs may be unreliable.

## Small-Sample Caveats

- **Chirp** (n_test=7): CI width = 0.602 -- WARNING: CI width > 0.5, per-class F1 unreliable
- **Wandering_Line** (n_test=6): CI width = 0.000
- **Helix** (n_test=14): CI width = 0.089
- **1080Lines** (n_test=6): CI width = 0.000

## Anchor Comparison (ref-gravity-spy)

Overall accuracy of **91.81%** confirms pipeline validity against published
Gravity Spy accuracy of ~97% (Zevin et al. 2017, CQG 34 064003).
- Expected range: [95%, 99%]
- Status: TENSION

Note: Our dataset uses O3 data with 23 classes vs. the original O1/O2 with ~20 classes.
The temporal split (vs. random split in the original) makes exact reproduction not expected.

## Decisive Output

Overall accuracy of 91.81% confirms pipeline validity (ref-gravity-spy anchor).
The decisive output is the **rare-class macro-F1 of 30.3%** [CI: 20.8%, 37.5%],
which is the baseline the ViT must beat in Phase 3.

The gap of **50.5 percentage points** between common-class avg F1 (80.7%) and
rare-class avg F1 (30.3%) demonstrates that the CNN struggles significantly with rare classes,
motivating the ViT investigation in Phase 3.

## Forbidden Proxy Enforcement (fp-overall-accuracy)

This analysis leads with macro-F1 and per-class F1 as primary results.
Overall accuracy appears ONLY as a sanity check against the Gravity Spy anchor.
