---
phase: 07-paper-text-table-revision
plan: 01
depth: full
one-liner: "Revised CQG manuscript addressing all 9 referee issues: reframed 'class-morphology-dependent' to 'class-dependent', restructured CW table with dual F1/DC advantage columns, integrated Phase 6 power analysis (aggregate power 0.20) and random-split ablation (95.4% accuracy)"
subsystem: [paper-writing, analysis]
tags: [referee-response, manuscript-revision, power-analysis, ablation, cw-veto, gravity-spy]

requires:
  - phase: 06-computation-statistical-analysis
    provides: ["Random-split ablation accuracy 0.9544, power analysis aggregate power 0.20, per-class power values"]
  - phase: 04-o4-validation-cw-sensitivity
    provides: ["CW duty cycle data, per-class F1 on O4, delta_DC values"]
  - phase: 05-paper-model-packaging
    provides: ["Original manuscript paper/main.tex, table files"]
provides:
  - "Revised manuscript with all 9 referee issues resolved (REF-001 through REF-009)"
  - "CW Table 3 with separate F1 Adv and DC Adv columns"
  - "Table 1 with corrected O4 ViT macro-F1 rounding (0.670)"
  - "Rare-class narrative reframed as 'insufficient evidence' with power analysis numbers"
  - "Random-split ablation result (95.4%) integrated into Section 3.2"
affects: [submission, author-response-letter]

methods:
  added: [targeted-text-revision, table-restructuring]
  patterns: [referee-issue-tracking, claim-reframing-with-quantitative-evidence]

key-files:
  modified:
    - paper/main.tex
    - paper/tables/table_cw.tex
    - paper/tables/table_overall.tex

key-decisions:
  - "Used 'class-dependent' throughout instead of computing a morphological complexity proxy (option b from referee report)"
  - "Whistle DC Adv marked '---' since delta_DC CI spans zero (point estimate +0.0004 is not meaningfully different from zero)"
  - "Kept one instance of 'underperformance' in the power analysis paragraph where it describes what should NOT be concluded"
  - "Corrected ViT O4 CI lower bound from 0.655 to 0.656 (standard rounding of 0.6555)"

patterns-established:
  - "Rare-class claims must cite power analysis numbers (aggregate power 0.20)"
  - "CW advantage must distinguish F1-based and DC-based judgments"

conventions:
  - "macro-F1 = primary metric"
  - "3 decimal places for F1/accuracy, 1 decimal for percentages"
  - "delta_DC sign: positive = ViT retains more livetime"
  - "bootstrap: 10K resamples, seed=42, 95% CI"

plan_contract_ref: ".gpd/phases/07-paper-text-table-revision/07-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-class-dependent-reframe:
      status: passed
      summary: "All 4 instances of 'class-morphology-dependent' replaced with 'class-dependent'. Power_Line and Chirp examples reframed as illustrative hypotheses with hedged language ('may capture', 'appears insufficient', 'we present as illustrative hypotheses, not established mechanisms')."
      linked_ids: [deliv-revised-manuscript, test-no-morphology-dependent, test-hypothesis-language, ref-gravity-spy]
    claim-rare-reframe:
      status: passed
      summary: "Rare-class narrative reframed from 'underperforms' to 'insufficient evidence' in abstract, Sec 3.1, Sec 3.2, Sec 4.1, and conclusion. Power analysis numbers (aggregate power 0.20, Chirp 0.035, Wandering_Line 0.017, 1080Lines 0.310) integrated throughout."
      linked_ids: [deliv-revised-manuscript, test-rare-reframe-complete, test-power-numbers-present, ref-gravity-spy]
    claim-cw-table-fixed:
      status: passed
      summary: "CW Table 3 restructured with separate F1 Adv and DC Adv columns. All 7 rows internally consistent: delta_DC sign matches DC Adv for all rows. Violin_Mode and Power_Line correctly show F1 Adv=ViT but DC Adv=CNN."
      linked_ids: [deliv-revised-cw-table, test-cw-columns-consistent]
    claim-minor-issues-resolved:
      status: passed
      summary: "All 5 minor issues resolved: (1) CW proxy validity statement with noise PSD stationarity assumption in Sec 2.4, (2) Davis et al. citation in Sec 2.4 CW methods paragraph, (3) single-training-run limitation statement added, (4) O4 ViT macro-F1 corrected to 0.670, (5) asymmetric degradation discussion (7.4% vs 1.7%) added in Sec 4.2."
      linked_ids: [deliv-revised-manuscript, deliv-revised-overall-table, test-five-minor-resolved, ref-davis-2021]
    claim-phase6-integrated:
      status: passed
      summary: "Random-split ablation accuracy 95.4% [95.3%, 95.6%] cited in Sec 3.2 with Gravity Spy benchmark comparison. Power analysis aggregate power 0.20 and per-class power values integrated into Sec 3.1, 3.2, 4.1, and conclusion."
      linked_ids: [deliv-revised-manuscript, test-ablation-in-paper, test-power-in-paper, ref-gravity-spy]
  deliverables:
    deliv-revised-manuscript:
      status: passed
      path: "paper/main.tex"
      summary: "Revised CQG manuscript with all 9 referee issues addressed. 92 lines added, 27 removed. Zero 'morphology-dependent' instances. Rare-class narrative uses 'insufficient evidence' throughout."
      linked_ids: [claim-class-dependent-reframe, claim-rare-reframe, claim-minor-issues-resolved, claim-phase6-integrated]
    deliv-revised-cw-table:
      status: passed
      path: "paper/tables/table_cw.tex"
      summary: "CW Table 3 with 8 columns including separate F1 Adv and DC Adv. Caption explains disagreement mechanism."
      linked_ids: [claim-cw-table-fixed, test-cw-columns-consistent]
    deliv-revised-overall-table:
      status: passed
      path: "paper/tables/table_overall.tex"
      summary: "Table 1 with O4 ViT macro-F1 corrected from 0.669 to 0.670 and CI lower bound from 0.655 to 0.656."
      linked_ids: [claim-minor-issues-resolved, test-five-minor-resolved]
  acceptance_tests:
    test-no-morphology-dependent:
      status: passed
      summary: "grep -c 'morphology-dependent' paper/main.tex returns 0. Zero instances remain."
      linked_ids: [claim-class-dependent-reframe, deliv-revised-manuscript]
    test-hypothesis-language:
      status: passed
      summary: "Power_Line described with 'may capture multi-frequency harmonic structure'. Chirp described with 'appears insufficient'. Sec 3.3 explicitly states 'illustrative hypotheses, not established mechanisms'. No unhedged causal claims."
      linked_ids: [claim-class-dependent-reframe, deliv-revised-manuscript]
    test-rare-reframe-complete:
      status: passed
      summary: "One instance of 'underperformance' remains in the power analysis paragraph ('rather than a definitive finding of ViT underperformance') -- this is the reframing itself, not the paper's own claim. No instance of 'underperforms' used to assert ViT rare-class regression."
      linked_ids: [claim-rare-reframe, deliv-revised-manuscript]
    test-power-numbers-present:
      status: passed
      summary: "Aggregate power 0.20 appears in abstract, Sec 3.2, and conclusion. Chirp power 0.035 and Wandering_Line power 0.017 appear in Sec 3.1 and Sec 4.1. 1080Lines power 0.310 appears in Sec 3.1."
      linked_ids: [claim-rare-reframe, deliv-revised-manuscript]
    test-cw-columns-consistent:
      status: passed
      summary: "All 7 data rows verified: delta_DC sign matches DC Adv column. Violin_Mode (delta_DC=-0.0043) -> DC Adv=CNN. Power_Line (delta_DC=-0.0144) -> DC Adv=CNN. Low_Frequency_Burst (delta_DC=+0.0057) -> DC Adv=ViT. Whistle (delta_DC=+0.0004, CI spans zero) -> DC Adv='---'."
      linked_ids: [claim-cw-table-fixed, deliv-revised-cw-table]
    test-five-minor-resolved:
      status: passed
      summary: "(1) 'approximately stationary' in Sec 2.4 -- present. (2) Davis2021 cited twice in Sec 2.4 -- present. (3) 'Single training run' limitation -- present. (4) O4 ViT macro-F1 = 0.670 -- present. (5) 'asymmetric degradation' in Sec 4.2 -- present."
      linked_ids: [claim-minor-issues-resolved, deliv-revised-manuscript, deliv-revised-overall-table]
    test-ablation-in-paper:
      status: passed
      summary: "Random-split accuracy 95.4% [95.3%, 95.6%] appears in Sec 3.2 with explicit comparison to published 95-99% range and Zevin2017 citation."
      linked_ids: [claim-phase6-integrated, deliv-revised-manuscript]
    test-power-in-paper:
      status: passed
      summary: "Power analysis results appear in Sec 3.1 (per-class power values, aggregate 0.20), Sec 3.2 (aggregate power in context of rare-class comparison), Sec 4.1 (Chirp and Wandering_Line power), and conclusion."
      linked_ids: [claim-phase6-integrated, deliv-revised-manuscript]
  references:
    ref-gravity-spy:
      status: completed
      completed_actions: [cite, compare]
      missing_actions: []
      summary: "Zevin et al. 2017 cited in Sec 3.2 random-split ablation paragraph. Random-split accuracy (95.4%) compared against published 95-99% range."
    ref-davis-2021:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Davis et al. 2021 cited twice in Sec 2.4: once in the veto efficiency framework introduction, once in the proxy validity statement."
  forbidden_proxies:
    fp-keep-morphology-dependent:
      status: rejected
      notes: "Zero instances of 'morphology-dependent' remain in the manuscript. All replaced with 'class-dependent'."
    fp-mixed-advantage-column:
      status: rejected
      notes: "Single 'Advantage' column replaced with separate 'F1 Adv' and 'DC Adv' columns. No mixing of F1-based and DC-based judgments."
    fp-rare-underperforms-without-power:
      status: rejected
      notes: "No instance of 'underperforms' used without power analysis context. Rare-class narrative consistently uses 'insufficient evidence' with aggregate power 0.20."
    fp-overall-accuracy:
      status: rejected
      notes: "Overall accuracy cited only in the random-split ablation paragraph as a benchmark comparison, not as a primary metric. macro-F1 remains primary throughout."
  uncertainty_markers:
    weakest_anchors:
      - "Power analysis uses single-seed ablation; multi-seed would strengthen the claim"
      - "CW proxy validity statement is an assertion, not a validated condition"
    unvalidated_assumptions:
      - "Noise PSD stationarity assumption for duty-cycle proxy (stated but not verified)"
    competing_explanations:
      - "If a morphological complexity metric were computed and showed correlation with delta-F1, the 'class-dependent' reframing would be too conservative"
    disconfirming_observations: []

duration: 20min
completed: 2026-03-18
---

# Phase 7 Plan 01: Paper Text and Table Revision Summary

**Revised CQG manuscript addressing all 9 referee issues: reframed 'class-morphology-dependent' to 'class-dependent', restructured CW table with dual F1/DC advantage columns, integrated Phase 6 power analysis (aggregate power 0.20) and random-split ablation (95.4% accuracy)**

## Performance

- **Duration:** 20 min
- **Started:** 2026-03-18T15:00:00Z
- **Completed:** 2026-03-18T15:20:00Z
- **Tasks:** 2
- **Files modified:** 3

## Key Results

- All 9 referee issues (REF-001 through REF-009) resolved in paper files [CONFIDENCE: HIGH]
- Zero instances of "morphology-dependent" remain in manuscript (was 4) [CONFIDENCE: HIGH]
- CW Table 3 restructured with separate F1 Adv and DC Adv columns; all rows internally consistent with delta_DC signs [CONFIDENCE: HIGH]
- Rare-class narrative uses "insufficient evidence" framing with power analysis numbers (aggregate power 0.20) in 5 locations [CONFIDENCE: HIGH]
- Random-split ablation (95.4% accuracy) integrated into Sec 3.2 with Gravity Spy benchmark comparison [CONFIDENCE: HIGH]
- O4 ViT macro-F1 corrected from 0.669 to 0.670 [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Text revisions -- reframing, new content, and Phase 6 integration** - `fb614ae` (document)
2. **Task 2: Table corrections -- CW dual-advantage columns and O4 rounding** - `d328583` (document)

## Files Created/Modified

- `paper/main.tex` -- Revised manuscript with all text-based referee issues resolved (92 lines added, 27 removed)
- `paper/tables/table_cw.tex` -- CW Table 3 restructured with F1 Adv and DC Adv columns
- `paper/tables/table_overall.tex` -- O4 ViT macro-F1 corrected to 0.670

## Next Phase Readiness

- Manuscript is ready for author-response letter drafting
- All numerical claims are traceable to Phase 6 results (random_split_ablation.json, power_analysis.json)
- No further computational work needed for this revision cycle

## Contract Coverage

- Claim IDs: claim-class-dependent-reframe -> passed, claim-rare-reframe -> passed, claim-cw-table-fixed -> passed, claim-minor-issues-resolved -> passed, claim-phase6-integrated -> passed
- Deliverable IDs: deliv-revised-manuscript -> passed, deliv-revised-cw-table -> passed, deliv-revised-overall-table -> passed
- Acceptance test IDs: test-no-morphology-dependent -> passed, test-hypothesis-language -> passed, test-rare-reframe-complete -> passed, test-power-numbers-present -> passed, test-cw-columns-consistent -> passed, test-five-minor-resolved -> passed, test-ablation-in-paper -> passed, test-power-in-paper -> passed
- Reference IDs: ref-gravity-spy -> cited + compared, ref-davis-2021 -> cited
- Forbidden proxies: fp-keep-morphology-dependent -> rejected, fp-mixed-advantage-column -> rejected, fp-rare-underperforms-without-power -> rejected, fp-overall-accuracy -> rejected

## Validations Completed

- `grep -c 'morphology-dependent' paper/main.tex` returns 0 (REF-001)
- CW Table 3 all rows: delta_DC sign matches DC Adv column (REF-002)
- Random-split ablation 95.4% present in Sec 3.2 with Zevin2017 citation (REF-003)
- Power analysis numbers present in abstract, Sec 3.1, 3.2, 4.1, conclusion (REF-004)
- CW proxy validity statement present in Sec 2.4 (REF-005)
- Davis2021 cited in Sec 2.4 methods paragraph (REF-006)
- Single-training-run limitation present (REF-007)
- O4 ViT macro-F1 = 0.670 in table (REF-008)
- Asymmetric degradation discussion present in Sec 4.2 (REF-009)
- No existing verified numerical values altered (only framing language and rounding fix)

## Decisions Made

- Chose option (b) from referee: reframe as "class-dependent" rather than computing morphological complexity proxy. This is more honest given the evidence.
- Whistle DC Adv marked "---" rather than "ViT" because delta_DC CI [-0.0004, +0.0012] spans zero.
- One instance of "underperformance" retained in the power analysis paragraph where it describes what should NOT be concluded ("rather than a definitive finding of ViT underperformance"). This is the reframing itself.

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## Self-Check: PASSED

- [x] paper/main.tex exists and contains all edits
- [x] paper/tables/table_cw.tex exists with F1 Adv and DC Adv columns
- [x] paper/tables/table_overall.tex exists with 0.670
- [x] All git commits found (fb614ae, d328583)
- [x] No forbidden proxy used
- [x] All contract IDs accounted for
- [x] Zero "morphology-dependent" instances
- [x] Power analysis numbers present in 5 locations
- [x] Random-split ablation present in Sec 3.2

---

_Phase: 07-paper-text-table-revision_
_Completed: 2026-03-18_
