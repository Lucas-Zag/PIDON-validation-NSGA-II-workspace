#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "Python environment:"
python -c "import sys; print(sys.executable)"

echo ""
echo "=== 0. Prepare and validate cases ==="
python "$HERE/prepare_cases.py"

echo ""
echo "=== 1. FE reference calculations ==="
python "$HERE/run_fe_cases.py"

echo ""
echo "=== 2. XPIDON predictions at all selected positions ==="
python "$HERE/run_xpidon_cases.py"

echo ""
echo "=== 3. Metrics and paper-style figures ==="
python "$HERE/compare_and_plot.py"

echo ""
echo "Completed."
echo "Outputs:"
echo "  $HERE/outputs"
echo "Figures:"
echo "  $HERE/figures"
