# Phase 1: Data Pipeline & Experimental Design - Research

**Researched:** 2026-03-16
**Domain:** Gravitational-wave detector characterization / data engineering / experimental methodology
**Confidence:** HIGH

## Summary

Phase 1 builds the foundation for all downstream model training and evaluation. The core task is acquiring the Gravity Spy labeled dataset from Zenodo (record 5649212), generating or reusing Q-transform spectrograms, implementing a temporal train/val/test split that prevents data leakage from temporally correlated glitches, auditing rare-class label quality, and locking an experimental protocol that ensures fair model comparison.

The Gravity Spy dataset is well-documented and publicly available. The Zenodo record contains CSV files with ML classifications for ~614K glitches across O1-O3b, with metadata including GPS times, detector labels, class labels, and per-class ML confidence scores. Pre-made spectrogram images (4 durations per glitch, 170x140 px PNG) are available from the training set Zenodo record (1486046), or custom spectrograms can be generated via GWpy's `TimeSeries.q_transform()`. The critical design decision for this phase is the temporal splitting strategy: LIGO glitches cluster temporally due to shared instrumental state, so random splitting causes data leakage that inflates test metrics. A time-based split with >= 60s gap between partitions is mandatory.

**Primary recommendation:** Use pre-made Gravity Spy spectrogram images from Zenodo for rapid pipeline validation, then generate custom 224x224 Q-transform spectrograms via GWpy for the production dataset. Implement temporal splitting using GPS time boundaries with 60s buffer gaps. Filter training labels to ml_confidence > 0.9. Lock macro-F1 and per-class recall as primary metrics before any model training begins.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| ref-gravity-spy: Zenodo 5649212 | benchmark dataset | Source of all glitch labels, GPS times, detector metadata; defines the 23-class taxonomy for O3 | Download, validate integrity, extract metadata CSV | plan, execution, verification |
| ref-gravity-spy: Zenodo 1486046 | training set images | Pre-made spectrogram images for rapid prototyping; defines Gravity Spy visual conventions | Download for validation/comparison; use as rapid-prototyping dataset | execution, verification |
| fp-overall-accuracy | forbidden proxy | Must NOT be used as primary metric; evaluation protocol must lock macro-F1 and per-class recall | Encode in experimental protocol document | plan, execution, all downstream phases |

**Missing or weak anchors:** The exact per-class sample counts for the O3 training set are not available from web search alone -- they must be computed directly from the downloaded CSV data as one of the first pipeline tasks. The Gravity Spy image normalization convention (normalized energy, related to SNR) is documented qualitatively but the exact mapping to pixel values needs empirical verification against downloaded images.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Spectrogram method | Q-transform (constant-Q) via GWpy | STFT, CWT, mel-spectrogram | Gravity Spy standard (Zevin et al. 2017) |
| Frequency range | 10--2048 Hz (log scale) | Linear scale; narrower bands | Gravity Spy / Omicron trigger convention |
| Time windows | 0.5s, 1.0s, 2.0s, 4.0s per glitch | Single duration | Gravity Spy 4-view standard |
| Image resolution | 224x224 px (upscaled from 170x140) | Native 170x140; 384x384 | ViT-B/16 input standard |
| Upscaling method | Bilinear interpolation | Bicubic, nearest-neighbor | Avoid nearest-neighbor (aliasing) |
| Normalization | Min-max to [0, 1] after SNR clipping to [0, 25.5] | Per-image z-score; per-dataset standardization | Project SUMMARY.md convention |
| Q-range | (4, 150) | Narrower ranges | Gravity Spy default |
| SNR threshold | > 7.5 (Omicron trigger threshold) | Higher thresholds | Gravity Spy / Omicron standard |
| Sample rate | 4 kHz | 16 kHz | 4 kHz sufficient for f < 2048 Hz |
| Primary metric | Macro-averaged F1 | Overall accuracy (FORBIDDEN as primary) | Project contract: fp-overall-accuracy |
| Label quality filter | ml_confidence > 0.9 | > 0.5; > 0.95; no filter | SUMMARY.md recommendation |
| Train/val/test split | Temporal (GPS time-based) with 60s gap | Random stratified (FORBIDDEN) | Pitfall prevention: temporal leakage |

**CRITICAL: All downstream phases inherit these conventions. Changing normalization or resolution after model training invalidates all results.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| Q(t, f) = integral h(t') w(t-t', f) e^{-2pi i f t'} dt' | Q-transform definition | Chatterji et al. (2004), CQG | Core spectrogram generation |
| E_norm(t,f) = \|Q(t,f)\|^2 / <\|Q\|^2>_noise | Normalized energy (Gravity Spy color scale) | Zevin et al. (2017) | Spectrogram pixel values |
| F1_macro = (1/N_cls) sum_c 2*P_c*R_c/(P_c+R_c) | Macro-averaged F1 score | Standard | Primary evaluation metric |
| gap >= 60s between splits | Temporal buffer constraint | Project requirement | Data leakage prevention |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Q-transform via GWpy | Time-frequency decomposition with constant-Q | Spectrogram generation from strain data | GWpy docs; Chatterji et al. (2004) |
| Whitening | Removes detector noise curve shape before Q-transform | Pre-processing strain segments | GWpy `TimeSeries.whiten()` |
| Temporal splitting | Partitions data by GPS time, not randomly | Train/val/test split construction | scikit-learn `TimeSeriesSplit` with gap |
| Stratified auditing | Ensures rare classes are adequately represented in validation | Split verification | scikit-learn stratification utilities |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| Bilinear upscaling 170x140 to 224x224 | Pixel interpolation error | Smooth spectrograms (most glitches) | Sub-pixel; negligible for classification | Regenerate at 224x224 natively via GWpy |
| SNR clipping at 25.5 | Fraction of glitches with SNR > 25.5 | Most glitches (median SNR ~ 10-15) | Saturates extremely loud glitches; acceptable | Logarithmic SNR scaling |
| ml_confidence > 0.9 filter | Fraction of discarded low-confidence labels | High-quality training labels | Discards ~30-50% of total dataset; enriches label quality | Lower threshold (0.5) keeps more data but noisier |

## Standard Approaches

### Approach 1: Pre-made Gravity Spy Images + Custom Temporal Split (RECOMMENDED for Phase 1)

**What:** Download pre-made spectrogram images from Zenodo training set (record 1486046) and ML classifications CSV (record 5649212). Resize images to 224x224. Implement temporal train/val/test split using GPS times from the metadata CSV. Audit class distributions and rare-class label quality.

**Why standard:** This is the fastest path to a validated dataset. Pre-made images guarantee consistency with the Gravity Spy visual conventions. The temporal split is the critical novel element that standard Gravity Spy work does not always enforce rigorously.

**Track record:** All published Gravity Spy papers use these pre-made images. Srivastava (2025) used them for ViT training. Wu et al. (2024) used them for O4 multi-view classifier.

**Key steps:**

1. Download Zenodo 5649212 (ML classifications CSV: GPS times, labels, confidence scores, per-class ML scores)
2. Download Zenodo 1486046 (pre-made spectrogram PNG images organized by class)
3. Parse metadata CSV; verify all required columns present (gravityspy_id, event_time, ifo, ml_label, ml_confidence, snr, peak_frequency)
4. Filter to O3a + O3b data; filter to ml_confidence > 0.9
5. Compute per-class sample counts; identify rare classes (< N_rare training examples)
6. Implement temporal split: sort by GPS time, define split boundaries, enforce 60s gap
7. Verify no GPS time overlap between splits (programmatic check)
8. Resize images to 224x224 via bilinear interpolation
9. Create dataset manifest CSVs (train/val/test) with file paths, labels, metadata
10. Write experimental protocol document

**Known difficulties at each step:**

- Step 3: CSV may have inconsistent column names across O1/O2/O3 sub-files; need to standardize
- Step 5: Some classes may have < 25 examples even after confidence filtering; these are high-risk
- Step 6: Temporal split may create uneven class distributions between splits; need stratification-aware temporal splitting or post-split verification
- Step 8: Gravity Spy images are 170x140 with specific color mapping; verify that resizing preserves morphological features

### Approach 2: Custom GWpy Spectrogram Generation from Strain Data (PRODUCTION UPGRADE)

**What:** Download O3 strain data from GWOSC at 4 kHz, generate Q-transform spectrograms at 224x224 native resolution using GWpy, with full control over normalization and frequency range.

**When to use:** After validating the pipeline with pre-made images. Required for the final production dataset if custom normalization, resolution, or frequency range is needed.

**Tradeoffs:** Full control over spectrogram parameters but requires 300-400 GB strain data download and 5-10 hours of CPU-parallel spectrogram generation. Only necessary if pre-made images prove insufficient.

**Key steps:**

1. Use GWOSC Python client to discover O3 4 kHz strain file URLs for H1 and L1
2. Download strain files for time segments containing labeled glitches (not all O3 data)
3. For each glitch GPS time: load strain segment, whiten, Q-transform at 4 durations
4. Normalize spectrograms (min-max after SNR clip to 25.5)
5. Save as 224x224 PNG images
6. Merge with Gravity Spy labels and proceed as in Approach 1

### Anti-Patterns to Avoid

- **Random stratified splitting:** Splits temporally adjacent glitches into different partitions. These share instrumental state and noise realization, inflating test metrics. Discovery after model training invalidates ALL results.
  - _Example:_ Two Scattered_Light glitches 5 seconds apart during the same seismic event end up in train and test. The model memorizes the noise background, not the glitch morphology.

- **Using overall accuracy as a selection metric:** A model ignoring all rare classes can achieve > 95% accuracy due to extreme class imbalance (Blip alone may be 10-20% of data).
  - _Example:_ Predict "Blip" for everything and report 18% accuracy. Predict the top-5 classes correctly and report 90%+ accuracy while having 0% recall on 15+ classes.

- **Skipping the ml_confidence filter:** Low-confidence Gravity Spy labels are effectively noisy annotations from the ML-human feedback loop. Training on them degrades rare-class performance disproportionately.

- **Generating spectrograms without whitening:** Unwhitened spectrograms are dominated by the detector noise curve shape (steeply rising at low frequencies), obscuring glitch morphology. The Q-transform output is physically meaningless without whitening.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Gravity Spy 23-class O3 taxonomy | 1080Lines, 1400Ripples, Air_Compressor, Blip, Blip_Low_Frequency, Chirp, Extremely_Loud, Fast_Scattering, Helix, Koi_Fish, Light_Modulation, Low_Frequency_Burst, Low_Frequency_Lines, No_Glitch, None_of_the_Above, Paired_Doves, Power_Line, Repeating_Blips, Scattered_Light, Scratchy, Tomte, Violin_Mode, Wandering_Line, Whistle | Glanzer et al. 2023 (arXiv:2208.12849) | Adopt directly as class labels |
| O3 total ML-classified glitches | ~234K H1 + ~380K L1 = ~614K total | Zenodo 5649212 | Starting pool before confidence filter |
| Omicron trigger threshold | SNR > 7.5, peak frequency 10-2048 Hz | Gravity Spy pipeline | Do not change; consistency with labels requires same threshold |
| Q-transform parameters | qrange=(4,150), frange=(10,2048), 4 durations | Gravity Spy documentation | Match exactly for label-spectrogram consistency |
| Spectrogram image dimensions | 170x140 px per view, 4 views per glitch | Zenodo 1486046 | Resize to 224x224 for ViT input |
| Pre-made spectrogram naming convention | {ifo}_{gravityspy_id}_spectrogram_{duration}.png | Zenodo 1486046 | Parse file names to link images to metadata |

**Key insight:** The Gravity Spy dataset, taxonomy, and spectrogram conventions are established infrastructure. Re-implementing the Q-transform pipeline from scratch risks subtle normalization differences that break label-spectrogram consistency. Use pre-made images for the primary dataset; only regenerate if custom normalization is scientifically justified.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| Gravity Spy CNN ~97% overall accuracy | Performance ceiling for basic CNN on common classes | Zevin et al. 2017 | On their original test split (not temporal) |
| O3 training set: ~9,631 samples across 23 classes | Approximate dataset size after confidence filtering | Wu et al. 2024 (arXiv:2401.12913) | Depends on confidence threshold |
| Ensemble classifier 98.21% accuracy | State-of-art on standard (non-temporal) split | Wu et al. 2024 | Standard test set, not temporal split |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Gravity Spy: Integrating Advanced LIGO DetChar, ML, and Citizen Science | Zevin et al. | 2017 | Original CNN baseline, class definitions, spectrogram conventions | Class taxonomy, Q-transform parameters, baseline accuracy |
| Machine learning for Gravity Spy: Glitch classification and dataset | Bahaadini et al. | 2018 | Dataset formalization, class distribution statistics | Training/validation/test split sizes, per-class counts |
| Data quality up to the third observing run of Advanced LIGO: Gravity Spy glitch classifications | Glanzer et al. | 2023 | O3 data release, updated taxonomy (23 classes), total counts | O3-specific class definitions, ML classification counts by detector |
| Gravity Spy: Lessons learned and a path forward | Zevin et al. | 2024 | CNN limitations, label quality issues, O4 challenges | Known failure modes, rare-class difficulties, None_of_the_Above handling |
| Advancing Glitch Classification in Gravity Spy: Multi-view Fusion | Wu/Jarov et al. | 2024 | O4 classifier architecture, training set details, multi-view fusion | Training set composition, attention-based fusion design |
| Vision Transformer for Transient Noise Classification | Srivastava & Niedzielski | 2025 | ViT-B/32 on 24 Gravity Spy classes, 92.26% accuracy | Data preprocessing pipeline, ViT training recipe, baseline to exceed |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| GWpy | >= 3.0; `gwpy.timeseries.TimeSeries.q_transform()` | Q-transform spectrogram generation from strain data | Same library used by Gravity Spy; ensures convention consistency |
| gwosc | >= 0.7; `gwosc.api.fetch_run_json()` | GWOSC strain file discovery and download | Official LIGO open data API client |
| pandas | >= 2.1 | Gravity Spy metadata CSV parsing, split construction | Standard data manipulation |
| scikit-learn | >= 1.4; `TimeSeriesSplit`, `GroupKFold` | Temporal splitting with gap parameter | Standard ML evaluation toolkit |
| Pillow / PIL | latest | Image resizing (170x140 to 224x224 bilinear) | Standard image processing |
| matplotlib | >= 3.8 | Spectrogram visualization, class distribution plots | Standard visualization |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| h5py | Direct HDF5 strain file access if needed beyond GWpy | Bulk data processing optimization |
| tqdm | Progress bars for large-scale spectrogram generation | Always (400K+ spectrograms) |
| ligo-segments | GPS segment arithmetic for temporal split boundaries | Computing detector uptime segments, split boundaries |
| requests / wget | Bulk file download from GWOSC and Zenodo | Data acquisition |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
| --- | --- | --- |
| Pre-made Gravity Spy images | Custom GWpy spectrograms from strain | Full control but 10x longer setup; only needed for custom normalization |
| pandas CSV parsing | SQLite / DuckDB | Better for very large datasets; overkill for ~614K rows |
| PIL bilinear resize | torchvision transforms | Either works; PIL is simpler for offline preprocessing |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Download Zenodo 5649212 (ML CSV) | ~5 min | Network speed | Small file (~100-200 MB) |
| Download Zenodo 1486046 (training images) | 1-2 hours | Network speed; ~5-10 GB | wget with resume |
| Download O3 strain data (4 kHz, H1+L1) | 6-24 hours | Network; 300-400 GB | Only if generating custom spectrograms |
| Custom spectrogram generation (100K x 4 views) | 5-10 hours | CPU-bound Q-transform | 12-core parallelism via joblib |
| Temporal split computation | < 1 min | Sorting by GPS time | Trivial |
| Image resizing (pre-made to 224x224) | ~30 min | Disk I/O | Embarrassingly parallel |
| Class distribution analysis | < 1 min | None | Trivial |

**Installation / Setup:**
```bash
# Core data pipeline packages
pip install gwpy>=3.0.0 gwosc>=0.7.0
pip install pandas>=2.1 scikit-learn>=1.4
pip install matplotlib>=3.8 Pillow tqdm h5py
pip install ligo-segments  # GPS segment arithmetic
```

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| GPS time uniqueness | No duplicate glitches in dataset | `assert len(df.event_time.unique()) == len(df)` | All GPS times unique (or near-unique within detector) |
| Temporal split gap verification | No data leakage from temporal correlation | For each test sample, verify nearest train sample is >= 60s away | Zero violations |
| Class label coverage | All expected classes present in each split | Count unique labels per split | All 23 classes present (some rare classes may be absent from val/test) |
| Image-metadata linkage | Every metadata row has corresponding spectrogram images | Check file existence for all 4 durations per glitch | 100% linkage (or documented missing images) |
| Spectrogram visual verification | Generated spectrograms match Gravity Spy website | Visual comparison of 10+ examples per class | Morphological match |
| Detector label consistency | H1 and L1 labels match expected GPS time ranges | Cross-reference GPS times with known detector uptime segments | All H1 times within H1 science segments; same for L1 |
| Confidence filter verification | ml_confidence filter applied correctly | Check that min(ml_confidence) >= 0.9 in filtered dataset | No sub-threshold samples |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| Total O3 ML-classified glitches | All confidence levels | ~614K (234K H1 + 380K L1) | Zenodo 5649212 |
| High-confidence training set | ml_confidence > 0.9 | ~8,000-10,000 samples (estimated) | Wu et al. 2024 |
| Rarest classes | After confidence filter | Wandering_Line, Paired_Doves: likely < 50-100 samples | To be determined from data |
| Most common classes | After confidence filter | Blip, Scattered_Light, Koi_Fish: likely > 1,000 samples each | To be determined from data |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
| --- | --- | --- | --- |
| Total glitch count (O3, H1+L1) | Sum rows in CSV for O3a + O3b | Within 5% of published value | ~614K |
| Class count | Count unique labels | Exactly 23 for O3 (or 24 if None_of_the_Above included) | Glanzer et al. 2023 |
| Temporal ordering | Verify GPS times are monotonically increasing within each detector | Exact | Physical requirement |
| Train/val/test ratio | Count samples per split | Approximately 70/15/15 after temporal split | Design target |
| Image dimensions after resize | Check output image shape | Exactly 224x224x3 | ViT-B/16 input requirement |

### Red Flags During Computation

- If the total number of glitches is far from ~614K (> 20% deviation), the download or parsing may be incomplete
- If any class has 0 samples after confidence filtering, the confidence threshold may be too aggressive for that class -- consider a lower per-class threshold
- If the temporal split produces severely unbalanced class distributions between train and test (> 3x ratio for any class), the glitch population has strong temporal clustering that may require per-class temporal stratification
- If pre-made spectrogram images are missing for > 5% of metadata entries, the Zenodo download may be incomplete or the image set covers a different observing run subset than the CSV
- If GPS times show non-monotonic ordering or duplicate events across detectors, the metadata may have cross-contamination issues

## Common Pitfalls

### Pitfall 1: Temporal Data Leakage

**What goes wrong:** Naive random train/test splitting places temporally adjacent glitches in different partitions. These glitches share instrumental state, PSD shape, and often identical morphology (e.g., a burst of Scattered_Light glitches during a seismic event). The model memorizes instrumental conditions, not glitch features.

**Why it happens:** Standard ML practice uses random stratified splits. LIGO glitches are not i.i.d. samples -- they cluster in time with shared noise backgrounds.

**How to avoid:** Sort all glitches by GPS time. Define split boundaries at fixed GPS times. Enforce >= 60s gap between the last training sample and the first validation/test sample. Verify programmatically: for every test GPS time, the nearest train GPS time must be >= 60s away.

**Warning signs:** Test performance significantly exceeds published literature values. Performance drops sharply on a temporally later held-out set.

**Recovery:** If discovered after model training, ALL results must be discarded and recomputed. There is no shortcut.

### Pitfall 2: Unbalanced Temporal Split

**What goes wrong:** Temporal splitting may concentrate rare classes in one time period, leaving the test set with zero samples of some classes. This happens because rare glitch types often appear in short bursts tied to specific instrumental conditions.

**Why it happens:** Rare classes are rare AND temporally clustered. A single O3a maintenance event might produce most Paired_Doves glitches.

**How to avoid:** After temporal splitting, verify per-class counts in all three partitions. If a rare class is absent from the test set, consider: (a) shifting split boundaries to include at least some samples, (b) using a smaller temporal gap, or (c) documenting the class as "not testable" and excluding it from macro-F1 computation.

**Warning signs:** Per-class counts show zero or near-zero for multiple classes in test set. Macro-F1 computation includes undefined (0/0) class F1 values.

**Recovery:** Adjust split boundaries; document any class exclusions explicitly.

### Pitfall 3: Normalization Mismatch Between Pre-made and Custom Spectrograms

**What goes wrong:** Pre-made Gravity Spy images use normalized energy (related to SNR) as the color scale. Custom GWpy spectrograms may use a different normalization (raw energy, whitened amplitude, etc.). Models trained on one normalization fail on the other.

**Why it happens:** The `q_transform()` function has a `norm` parameter that defaults to `'median'`. Gravity Spy images use a specific normalized-energy color mapping. These are not identical.

**How to avoid:** If using pre-made images, do not mix with custom spectrograms. If generating custom spectrograms, match the Gravity Spy normalization exactly, or commit to a different normalization for the entire pipeline (and document the deviation).

**Warning signs:** Visual comparison of custom vs. pre-made spectrograms shows different color scales or contrast for the same glitch.

**Recovery:** Regenerate all spectrograms with consistent normalization. Retrain all models.

### Pitfall 4: Missing or Inconsistent Metadata Across Observing Runs

**What goes wrong:** The Zenodo CSV files for O1, O2, O3a, and O3b may have different column names, different numbers of class columns (22 vs. 23 vs. 24), or different metadata fields. Naively concatenating them produces parsing errors or silent data corruption.

**Why it happens:** The Gravity Spy taxonomy evolved: O1/O2 had 22 classes, O3 added Blip_Low_Frequency and Fast_Scattering, and None_of_the_Above handling changed.

**How to avoid:** Parse each sub-run CSV separately. Verify column names. Use only O3a + O3b data for this project (24/23-class taxonomy). Document which taxonomy version is used.

**Warning signs:** Unexpected NaN values in class confidence columns. Mismatched row counts between metadata and image files.

**Recovery:** Re-parse with run-specific column mappings.

### Pitfall 5: Ignoring the None_of_the_Above Class

**What goes wrong:** The None_of_the_Above (NOTA) class contains glitches that do not fit any defined morphological category. Including NOTA in training confuses the classifier because it has no consistent morphology. Excluding it means genuinely novel glitches get forced into known categories at inference time.

**Why it happens:** NOTA is a catch-all class with heterogeneous morphology. The Gravity Spy team removed it from the O3 classifier training (Glanzer et al. 2023).

**How to avoid:** Exclude NOTA from training, following standard practice. Document this decision. Note that the trained classifier has no mechanism for rejecting unknown glitch types -- this is a known limitation (out-of-distribution detection is out of scope for this project).

**Warning signs:** If NOTA is included, the class will have poor precision and recall and will drag down macro-F1.

**Recovery:** Remove NOTA samples and retrain.

## Level of Rigor

**Required for this phase:** Data engineering rigor (deterministic, reproducible, programmatically verified)

**Justification:** This is a data pipeline phase, not a theoretical derivation. The rigor standard is: every data processing step must be reproducible from a script, every split must be verifiable programmatically, and every design decision must be documented before model training begins.

**What this means concretely:**

- All data processing must be scripted (no manual steps except rare-class visual audit)
- The temporal split must be verified by a programmatic check (not "eyeballed")
- Per-class sample counts must be computed automatically and documented
- The experimental protocol must be written as a document, not left as implicit choices
- Spectrogram normalization must be documented with exact parameters

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Random stratified split | Temporal split with gap | ~2020-2022 (community recognition) | Prevents inflated metrics; required for publication credibility |
| 22-class O1/O2 taxonomy | 23-24 class O3 taxonomy | O3 data release (2022) | Must use updated taxonomy for O3 data |
| Gravity Spy CNN (Zevin 2017) | Multi-view attention fusion (Wu/Jarov 2024) | 2024 | New state-of-art baseline; uses attention across 4 duration views |
| Manual volunteer labeling only | ML + citizen science + confidence filtering | 2017-present | Use ml_confidence > 0.9 to get high-quality training labels |

**Superseded approaches to avoid:**

- **Random splitting on Gravity Spy data:** Still used in some papers but recognized as flawed. Inflates test metrics by 3-10% depending on temporal clustering severity.
- **Using O1/O2 22-class taxonomy for O3 data:** Misses Blip_Low_Frequency and Fast_Scattering classes that are significant in O3.

## Open Questions

1. **Exact per-class sample counts after ml_confidence > 0.9 filtering on O3 data**
   - What we know: Total ~614K glitches, ~8-10K high-confidence training samples
   - What's unclear: Exact distribution across 23 classes; how many classes fall below 25 samples
   - Impact on this phase: Determines which classes are "rare" and "high-risk"
   - Recommendation: Compute directly from downloaded data as first pipeline task; this resolves immediately upon data acquisition

2. **Optimal temporal split boundaries for balanced class representation**
   - What we know: Must sort by GPS time with 60s gap
   - What's unclear: Whether a 70/15/15 temporal split preserves adequate rare-class representation in val/test
   - Impact on this phase: May need iterative boundary adjustment
   - Recommendation: Implement split, check per-class counts, adjust boundaries if needed; document final boundaries

3. **Whether pre-made Gravity Spy images at 170x140 are sufficient or if 224x224 native generation is needed**
   - What we know: Bilinear upscaling introduces interpolation artifacts; ViT-B/16 expects 224x224
   - What's unclear: Whether interpolation artifacts affect rare-class classification
   - Impact on this phase: Determines whether custom spectrogram generation is required
   - Recommendation: Start with pre-made images; benchmark against custom spectrograms if rare-class performance is unexpectedly poor in Phase 2

4. **None_of_the_Above handling strategy**
   - What we know: NOTA was removed from O3 training by the Gravity Spy team
   - What's unclear: Whether NOTA samples should be retained as a separate OOD validation set
   - Impact on this phase: Affects dataset construction
   - Recommendation: Exclude from training and evaluation; optionally retain as OOD detection test set for future work

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Zenodo 5649212 download | Server downtime, data corruption | Use GravitySpy Python API to query database directly | Low -- API provides same metadata |
| Pre-made spectrogram images | Images missing, wrong format, Zenodo 1486046 unavailable | Generate custom spectrograms from strain data via GWpy | Medium -- requires 300-400 GB strain download + 5-10 hours generation |
| ml_confidence > 0.9 filter | Too few rare-class samples survive | Lower threshold to 0.7 for rare classes only; document class-specific thresholds | Low -- re-filter and re-split |
| Temporal split produces empty rare classes | Severe temporal clustering | Use blocked cross-validation instead of single split; or use leave-one-month-out strategy | Medium -- changes evaluation protocol |

**Decision criteria:** If any rare class has < 10 samples in the test set after temporal splitting, the split boundaries need adjustment or the evaluation must use cross-validation instead of a single split.

## Caveats and Alternatives

**Self-critique:**

1. **Assumption that may be wrong:** I assume the Zenodo 5649212 CSV contains all necessary metadata (GPS time, detector, confidence) in a consistent format across O3a and O3b. If the format changed between sub-runs, additional parsing logic is needed.

2. **Alternative dismissed too quickly:** Generating all spectrograms from scratch (Approach 2) gives full control and avoids any normalization ambiguity. I recommended pre-made images for speed, but if this project values reproducibility and normalization control above rapid prototyping, starting with custom spectrograms is defensible.

3. **Understated limitation:** The 60s temporal gap is a heuristic. Some instrumental conditions (e.g., seismic storms, detector lock losses) persist for hours and create correlated glitches across much longer timescales. A 60s gap prevents the most obvious leakage but does not guarantee statistical independence.

4. **Simpler method overlooked:** For the temporal split, instead of custom GPS-time sorting, one could simply use O3a for training and O3b for testing. This provides a natural temporal split with months of gap but may create class distribution shift between the two sub-runs.

5. **Domain specialist might disagree on:** Whether to include the No_Glitch class. Some practitioners argue it should be included for completeness (the model needs to distinguish real glitches from noise); others exclude it because it is fundamentally different from glitch classification. The project requirements list it as part of the 23-class taxonomy, so include it.

## Sources

### Primary (HIGH confidence)

- [Glanzer et al. 2023, "Data quality up to the third observing run of Advanced LIGO: Gravity Spy glitch classifications," CQG 40 065004, arXiv:2208.12849](https://arxiv.org/abs/2208.12849) -- O3 dataset release, class taxonomy, ML classification pipeline
- [Zenodo record 5649212: Gravity Spy ML Classifications O1-O3b](https://zenodo.org/records/5649212) -- Primary dataset source
- [Zenodo record 1486046: Gravity Spy Training Set](https://zenodo.org/records/1486046) -- Pre-made spectrogram images
- [Zevin et al. 2017, "Gravity Spy: Integrating Advanced LIGO Detector Characterization, Machine Learning, and Citizen Science," CQG 34 064003, arXiv:1611.04596](https://arxiv.org/abs/1611.04596) -- Original Gravity Spy paper, CNN baseline, Q-transform conventions
- [GWpy documentation: Q-transform spectrogram generation](https://gwpy.github.io/docs/stable/examples/timeseries/qscan/) -- Q-transform API reference

### Secondary (MEDIUM confidence)

- [Wu/Jarov et al. 2024, "Advancing Glitch Classification in Gravity Spy: Multi-view Fusion," arXiv:2401.12913](https://arxiv.org/abs/2401.12913) -- O4 classifier, training set composition, multi-view architecture
- [Srivastava & Niedzielski 2025, "Vision Transformer for Transient Noise Classification," arXiv:2510.06273](https://arxiv.org/abs/2510.06273) -- ViT-B/32 baseline, 92.26% accuracy, data preprocessing
- [Zevin et al. 2024, "Gravity Spy: Lessons learned and a path forward," EPJ Plus 139 100, arXiv:2308.15530](https://arxiv.org/abs/2308.15530) -- CNN limitations, label quality issues
- [scikit-learn TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) -- Temporal cross-validation with gap parameter

### Tertiary (LOW confidence)

- [Chatterji et al. 2004, "Multiresolution techniques for the detection of gravitational-wave bursts," CQG](https://arxiv.org/abs/gr-qc/0412119) -- Q-transform algorithm (theoretical foundation, not directly consulted for implementation)
- Gravity Spy class distribution estimates (approximate counts from literature; exact values must be computed from downloaded data)

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH -- Q-transform and temporal splitting are well-understood, deterministic operations
- Standard approaches: HIGH -- Gravity Spy dataset access, GWpy spectrogram generation, and temporal splitting are all well-documented and widely used
- Computational tools: HIGH -- GWpy, pandas, scikit-learn are mature, stable libraries with extensive documentation
- Validation strategies: HIGH -- Programmatic checks for temporal gap, class coverage, and data integrity are straightforward to implement

**Research date:** 2026-03-16
**Valid until:** Stable -- Gravity Spy dataset and GWpy API are unlikely to change significantly. Check for new Zenodo releases or GWpy major version changes.
