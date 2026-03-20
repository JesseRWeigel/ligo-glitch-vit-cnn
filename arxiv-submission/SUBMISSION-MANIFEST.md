# arXiv Submission Manifest

**Generated:** 2026-03-18 15:55
**Tarball:** arxiv-submission.tar.gz (160K)
**Tarball location:** /home/jesse/Projects/gw-research/arxiv-submission.tar.gz

## Contents

| File | Size | Description |
|------|------|-------------|
| main.tex | 43K | Main LaTeX file (all \input flattened, bibliography inlined) |
| 00README.XXX | 1K | arXiv file manifest |
| figures/fig_per_class_f1.pdf | 30K | Per-class F1 comparison bar chart (Figure 1) |
| figures/fig_threshold_scatter.pdf | 24K | Training-set threshold scatter (Figure 2) |
| figures/fig_confusion_matrices.pdf | 42K | Side-by-side confusion matrices (Figure 3) |
| figures/fig_o4_degradation.pdf | 26K | O3→O4 temporal degradation (Figure 4) |
| figures/fig_cw_veto.pdf | 21K | CW veto efficiency comparison (Figure 5) |
| anc/ | 287K | Ancillary files: 27 Python scripts for full reproducibility |

## Checks Passed

- LaTeX compilation: PASS (16 pages, no errors)
- Bibliography flattened: PASS (thebibliography inlined, natbib-compatible)
- Figures validated: PASS (5 PDF figures, arXiv-compatible)
- Abstract length: PASS (1587/1920 chars)
- \pdfoutput=1: PASS (added to first line)
- Total size: PASS (160K, limit: 50MB)
- No RESULT PENDING placeholders: PASS
- No MISSING: citation markers: PASS
- No TODO/FIXME comments: PASS

## Regenerate

To regenerate the tarball from source:

```bash
/gpd:arxiv-submission paper/
```
