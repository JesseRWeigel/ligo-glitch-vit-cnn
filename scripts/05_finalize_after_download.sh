#!/bin/bash
# Run this after 03_download_spectrograms.py completes to:
# 1. Re-compute class distribution with actual spectrogram coverage
# 2. Validate a sample of downloaded images
#
# Usage: bash scripts/05_finalize_after_download.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "=== Finalizing after download ==="
echo ""

# Check download progress
if [ -f "$PROJECT_ROOT/data/spectrograms/download_progress.json" ]; then
    echo "Download progress:"
    $PYTHON -c "
import json
with open('$PROJECT_ROOT/data/spectrograms/download_progress.json') as f:
    p = json.load(f)
s = p.get('stats', {})
print(f'  Completed: {s.get(\"completed\", \"?\")}/{s.get(\"total\", \"?\")}')
print(f'  Coverage: {s.get(\"coverage_pct\", \"?\"):.1f}%')
print(f'  Failed: {s.get(\"failed\", \"?\")}')
"
    echo ""
fi

# Re-run class distribution with spectrogram coverage check
echo "=== Re-computing class distribution with spectrogram coverage ==="
$PYTHON "$SCRIPT_DIR/04_class_distribution.py"

echo ""
echo "=== Validating sample of downloaded images ==="
$PYTHON "$SCRIPT_DIR/03_download_spectrograms.py" --validate-only

echo ""
echo "=== Done ==="
echo "Check data/metadata/class_distribution_raw.json for final distribution"
echo "Check figures/class_distribution_o3.png for updated bar chart"
