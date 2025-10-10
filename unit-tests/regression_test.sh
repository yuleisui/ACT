#!/bin/bash
#
# Quick regression test runner for bounds propagation
# Run this script after making changes to bounds_propagation.py
#

set -e

# Get the directory where this script is located (unit-tests folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 ACT Bounds Propagation Regression Tests"
echo "==========================================="
echo ""

# Check if baseline exists (now in same directory)
if [ ! -f "regression_baselines/performance_baseline.json" ]; then
    echo "📊 No baseline found. Capturing baseline first..."
    python test_bounds_prop_regression.py --capture-baseline
    echo ""
fi

# Run regression tests with auto-update on improvements
echo "🔄 Running regression tests with smart auto-update..."
if python test_bounds_prop_regression.py --test-regression; then
    echo ""
    echo "✅ All regression tests passed!"
    echo "🎉 Your changes to bounds_propagation.py are good to go!"
else
    echo ""
    echo "❌ Regression detected!"
    echo "🔔 Options:"
    echo "   1. Review your changes for unintended side effects"
    echo "   2. If changes are intentional, update baseline:"
    echo "      python test_bounds_prop_regression.py --update-baseline"
fi