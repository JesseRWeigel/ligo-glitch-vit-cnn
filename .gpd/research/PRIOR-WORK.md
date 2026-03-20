# Prior Work: ML-Based LIGO Glitch Classification

**Surveyed:** 2026-03-16
**Domain:** Gravitational-wave detector characterization, machine learning for physics
**Confidence:** MEDIUM-HIGH

## Key Results

| Result | Expression / Value | Conditions | Source | Year | Confidence |
|---|---|---|---|---|---|
| Gravity Spy CNN overall accuracy | >97% (22 classes, multi-duration spectrograms) | O1 data, human-labeled training set, ~7700 samples over 19 initial classes | Zevin et al., CQG 34(6) 064003 | 2017 | HIGH |
| Deep Transfer Learning accuracy | >98.8% overall, perfect precision-recall on 8/22 classes | Gravity Spy O1 dataset, ImageNet-pretrained CNNs fine-tuned | George, Shen & Huerta, PRD 97 101501 | 2018 | HIGH |
| ViT-B/32 glitch classification | 92.26% (24 classes: 22 original + 2 O3a classes) | Pre-trained ViT-B/32, combined Gravity Spy + O3a data | Srivastava & Niedzielski, arXiv:2510.06273 | 2025 | MEDIUM |
| Multi-view fusion + attention (O4) | Improved over baseline CNN (specific numbers in paper) | O4-era data, label smoothing, multi-time-window fusion | Raza et al., arXiv:2401.12913 | 2024 | MEDIUM |
| Unsupervised clustering alignment | Up to 90.9% clustering accuracy | Truncated CNN features + UMAP/t-SNE clustering | George, Shen & Huerta, PRD 97 101501 | 2018 | HIGH |
| Gravity Spy O3 total classified glitches | ~614K (234K Hanford + 380K Livingston) across 23 classes | O1+O2+O3a+O3b, ML-classified | Glanzer et al., Zenodo 5649212 | 2022 | HIGH |

## Foundational Work

### Zevin et al. (2017) -- Gravity Spy: Integrating Advanced LIGO Detector Characterization, ML, and Citizen Science

**Key contribution:** Established the Gravity Spy framework combining CNN-based classification with citizen-science labeling via Zooniverse. Defined the initial morphological taxonomy of LIGO glitches (19 classes at launch, expanded to 22 for O1). Used Q-transform spectrograms at four time durations (0.5s, 1.0s, 2.0s, 4.0s) as the canonical image representation.
**Method:** CNN trained on multi-duration Q-transform spectrograms of hand-labeled glitches. Citizen scientists classify images into pre-identified morphological classes and discover new classes as detectors evolve.
**Limitations:** CNN architecture relatively simple; poor generalization to new glitch morphologies not in training set; performance degrades on rare classes (Wandering_Line, Paired_Doves) with few training samples and on heterogeneous classes (None_of_the_Above, No_Glitch). Does not natively handle multi-view fusion across time scales.
**Relevance:** This is the direct baseline our ViT must beat. The multi-duration spectrogram representation and glitch taxonomy are the community standard we build on.
**Reference:** arXiv:1611.04596, CQG 34(6) 064003

### George, Shen & Huerta (2018) -- Deep Transfer Learning for LIGO Glitch Classification

**Key contribution:** Demonstrated that ImageNet-pretrained deep CNNs (VGG, Inception, ResNet) fine-tuned on Gravity Spy data achieve >98.8% accuracy, lowering the previous error rate by >60%. Showed that truncated pretrained networks serve as excellent feature extractors for unsupervised clustering, enabling discovery of new glitch morphologies without labeled data. Crucially demonstrated few-shot capability: new glitch types classifiable with few labeled examples.
**Method:** Transfer learning from ImageNet to Gravity Spy spectrograms. Unsupervised clustering on CNN-extracted features to discover novel classes.
**Limitations:** Still CNN-based, so limited in capturing long-range spatial dependencies across the spectrogram. Unsupervised clustering accuracy (90.9%) significantly below supervised. Few-shot claims not systematically benchmarked against modern few-shot learning baselines.
**Relevance:** Sets the accuracy bar at >98.8% for CNN-based approaches. The transfer learning + unsupervised clustering pipeline is the state of the art for handling novel/rare classes. Our ViT approach should demonstrate superior feature extraction for rare classes to be justified.
**Reference:** arXiv:1706.07446, PRD 97 101501

### Bahaadini et al. (2018) -- Machine Learning for Gravity Spy: Glitch Classification and Dataset

**Key contribution:** Formalized the Gravity Spy dataset as a benchmark. Documented the class distribution, showing severe imbalance: Wandering_Line had only 25 training samples while common classes like Blip had thousands. Established the multi-view (4 time-duration) input format.
**Limitations:** The class imbalance problem was documented but not solved -- performance on rare classes remained significantly below the overall average.
**Relevance:** Directly quantifies the rare-class problem we aim to solve. The Wandering_Line class (25 training samples) and Paired_Doves class (124 training samples) are canonical examples of the rare-class challenge.
**Reference:** Information Sciences 444, 172-186 (2018)

### Zevin, Jackson, Coughlin et al. (2024) -- Gravity Spy: Lessons Learned and a Path Forward

**Key contribution:** Retrospective analysis of the Gravity Spy project through O3. Documented taxonomy expansion from 19 to 23 classes. Training set grew to ~10,000 labeled glitches across 23 classes (with None_of_the_Above removed for O3). Identified key limitations: the CNN architecture's simplicity leads to poor generalization, inability to handle multi-time-window inputs effectively, and challenges with the growing number of classes across observing runs.
**Method:** Review and analysis of ML + citizen science integration.
**Limitations:** Explicitly acknowledged that the existing CNN architecture needs replacement for O4 and beyond.
**Relevance:** Directly motivates our project. The paper explicitly calls for architectural improvements, particularly for multi-view fusion and handling of evolving glitch taxonomies.
**Reference:** arXiv:2308.15530, Eur. Phys. J. Plus 139, 100 (2024)

## Recent Developments

| Paper | Authors | Year | Advance | Impact on Our Work |
|---|---|---|---|---|
| Vision Transformer for Transient Noise Classification | Srivastava & Niedzielski | 2025 | First ViT application to LIGO glitch classification; ViT-B/32 on 24 classes achieves 92.26% | **Directly relevant competitor/predecessor.** Shows ViT is viable but underperforms CNN baseline. Suggests naive ViT transfer is insufficient; need better training strategy or architecture adaptation. |
| Advancing Glitch Classification in Gravity Spy (O4) | Raza et al. | 2024 | Multi-view fusion with attention modules for O4; label smoothing for noisy labels; interpretability via attention weights | Key architectural ideas: multi-view fusion strategy and attention-based interpretability. Should inform our ViT design. |
| GraviBERT: Transformer-based inference for GW time series | (authors) | 2025 | BERT-style self-supervised pretraining + transformer encoder for GW parameter estimation; 31% MAE reduction; 6.6x convergence speedup | Demonstrates that transformer + self-supervised pretraining works for GW data. Architecture ideas (multi-scale CNN frontend + transformer encoder) directly transferable. Not classification but the pretraining paradigm applies. |
| Flexible GW Parameter Estimation with Transformers (Dingo-T1) | (authors) | 2025 | Transformer encoder for heterogeneous GW data; handles variable-length inputs via token dropping | Shows transformers can handle the multi-scale nature of GW data. Token-dropping strategy relevant for robustness. |
| Attention U-Net for all-sky CW searches | (authors) | 2025 | Attention mechanisms for continuous wave detection | Directly relevant to downstream CW search application -- attention-based models improving CW search sensitivity. |
| Cross-Temporal Spectrogram Autoencoder (CTSAE) | (authors) | 2024 | Unsupervised dimensionality reduction for clustering GW glitches | Alternative unsupervised approach for glitch grouping; potential comparison point. |

## Known Limiting Cases

| Limit | Known Result | Source | Verified By |
|---|---|---|---|
| Abundant training data, common classes (Blip, Koi_Fish, Scattered_Light) | CNN achieves near-perfect precision/recall | Zevin et al. 2017, George et al. 2018 | Multiple groups |
| Rare classes with <100 training samples (Wandering_Line, Paired_Doves) | CNN precision/recall significantly degrades | Zevin et al. 2017, Bahaadini et al. 2018 | Documented in original Gravity Spy papers |
| Morphologically heterogeneous classes (None_of_the_Above) | All classifiers struggle; inherently ill-defined category | Zevin et al. 2017 | Community consensus |
| Naive ViT transfer (ViT-B/32 pretrained on ImageNet) | 92.26% -- below CNN baseline of >97% | Srivastava & Niedzielski 2025 | Single study |
| Deep transfer learning from ImageNet + fine-tuning | >98.8% -- best published CNN result | George et al. 2018 | Published in PRD |

## Open Questions

1. **Can ViT outperform CNNs on rare glitch classes?** -- The only published ViT result (Srivastava 2025) underperforms the CNN baseline overall, but did not specifically optimize for rare classes or use modern ViT training recipes (DeiT-style training, stronger augmentation, self-supervised pretraining). The question of whether ViT's global attention mechanism provides better feature extraction for rare morphologies remains unanswered.

2. **What is the optimal multi-view fusion strategy for spectrograms at different time scales?** -- Gravity Spy uses 4 time durations (0.5s, 1.0s, 2.0s, 4.0s). The Raza et al. 2024 paper explores fusion strategies but within a CNN framework. ViT-native multi-scale fusion (e.g., via cross-attention between scale-specific tokens) has not been explored.

3. **Does self-supervised pretraining on unlabeled LIGO glitches improve rare-class performance?** -- GraviBERT (2025) showed self-supervised pretraining dramatically helps for GW parameter estimation. The analogous experiment for glitch classification has not been published. Given the massive unlabeled glitch dataset (~600K+ classified glitches, plus orders of magnitude more unclassified), this is a high-potential direction.

4. **How does glitch classification accuracy translate to CW search sensitivity?** -- The operational link between better glitch classification and improved continuous wave search sensitivity is asserted in detector characterization literature but not quantified for specific rare-class improvements. Glitches primarily affect transient GW searches; for CW searches, persistent spectral lines are the dominant artifact, but glitch-induced spectral contamination also matters.

5. **What is the right evaluation metric?** -- Overall accuracy obscures rare-class performance. Macro-averaged F1 (treating all classes equally) or per-class F1 for rare classes should be the primary metric, but no published work has systematically used this as the optimization target for LIGO glitch classifiers.

## Notation Conventions in the Literature

| Quantity | Standard Symbol(s) | Variations | Our Choice | Reason |
|---|---|---|---|---|
| Strain time series | h(t) | x(t), s(t) | h(t) | LIGO convention |
| Q-transform time-frequency map | Q-scan, Q-transform, spectrogram | Omega scan (older) | Q-transform spectrogram | Current Gravity Spy convention |
| Signal-to-noise ratio | SNR, rho | S/N | SNR | Standard in detector characterization |
| Glitch confidence | ml_confidence | confidence, score | ml_confidence | Gravity Spy metadata field name |
| Observing run | O1, O2, O3a, O3b, O4a, O4b, O4c | Run 1, S1 (initial LIGO) | O1/O2/O3/O4 | LVK convention |

## Key Context: Glitches vs. Lines for CW Searches

A critical distinction for the project's CW search motivation:

- **Glitches** (transient noise): Broadband, short-duration. Primarily impact transient GW searches (CBC, bursts). Gravity Spy classifies these.
- **Lines** (persistent noise): Narrowband, long-duration. Primarily impact continuous wave searches. The LVK maintains a separate curated line catalog.
- **The connection:** Some glitch classes (e.g., Scattered_Light, Violin_Mode, 1080Lines, Low_Frequency_Lines) have persistent or quasi-periodic character that contaminates CW search frequency bands. Better classification of these boundary cases could improve CW data quality products. Additionally, excess glitch rates in specific frequency bands trigger data quality vetoes that reduce CW search livetime.

This nuance means our project's CW search impact is strongest for glitch classes that have spectral-line-like characteristics, not for all 23 classes equally.

## The Gap Our Project Addresses

The literature reveals a clear gap:

1. **CNN ceiling reached:** The Gravity Spy CNN architecture was acknowledged as insufficient for O4+ by the project's own team (Zevin et al. 2024). The >98.8% accuracy from deep transfer learning (George et al. 2018) sets a high bar but was measured on O1 data with 22 classes; performance on the expanded O3 taxonomy with more classes and evolving morphologies is worse.

2. **ViT not yet competitive:** The single published ViT attempt (Srivastava 2025, 92.26%) used a straightforward fine-tuning approach without modern ViT training techniques. This represents an opportunity, not a negative result for ViT in general.

3. **Rare-class performance never optimized:** No published work specifically targets rare-class F1 as the primary metric. All report overall accuracy, which is dominated by common classes.

4. **Self-supervised pretraining unexplored for classification:** GraviBERT showed transformers + self-supervised pretraining excel for GW data, but nobody has applied this to glitch classification where the unlabeled data pool is enormous.

## Sources

- Zevin et al. (2017), arXiv:1611.04596, CQG 34(6) 064003 -- Gravity Spy foundational paper, CNN baseline, glitch taxonomy
- George, Shen & Huerta (2018), arXiv:1706.07446, PRD 97 101501 -- Deep transfer learning, >98.8% accuracy benchmark
- Bahaadini et al. (2018), Information Sciences 444 -- Gravity Spy dataset formalization, class distribution statistics
- Zevin et al. (2024), arXiv:2308.15530, Eur. Phys. J. Plus 139 100 -- Lessons learned, O3 taxonomy, acknowledged need for architectural upgrade
- Srivastava & Niedzielski (2025), arXiv:2510.06273 -- First ViT application to LIGO glitches, 92.26% accuracy
- Raza et al. (2024), arXiv:2401.12913 -- Multi-view attention fusion for O4 Gravity Spy
- GraviBERT (2025), arXiv:2512.21390 -- Transformer + self-supervised pretraining for GW inference
- Glanzer et al. (2022), Zenodo 5649212 -- Gravity Spy ML classifications O1-O3b, dataset release
- Keats et al. (2012), arXiv:1201.5244 -- F-statistic line veto for CW searches
- LIGO DetChar O4 (2024), arXiv:2409.02831 -- O4 detector characterization, data quality products
- Chatterji (2004), MIT PhD thesis -- Q-transform for GW burst searches, foundational signal processing
