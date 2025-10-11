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

#!/bin/bash
#
# Enhanced regression test runner for bounds propagation APIs
# Usage: ./regression_test.sh [options]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located (unit-tests folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}🧪 ACT Bounds Propagation Test Suite${NC}"
echo "==========================================="
echo ""

# Check if we have the required files
if [ ! -f "test_configs.py" ]; then
    echo -e "${RED}❌ test_configs.py not found. Please ensure shared configurations are set up.${NC}"
    exit 1
fi

if [ ! -f "test_bounds_propagation.py" ]; then
    echo -e "${RED}❌ test_bounds_propagation.py not found.${NC}"
    exit 1
fi

# Parse command line arguments
SAVE_BASELINE=false
TEST_MODE="normal"

while [[ $# -gt 0 ]]; do
    case $1 in
        --save-baseline)
            SAVE_BASELINE=true
            shift
            ;;
        --quick)
            TEST_MODE="quick"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --save-baseline  Save current performance as baseline"
            echo "  --quick         Run only fast tests"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Check Python dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
if ! python -c "import torch, numpy, unittest" 2>/dev/null; then
    echo -e "${RED}❌ Required dependencies not available.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Dependencies OK${NC}"

# Run shared configuration validation
echo ""
echo -e "${YELLOW}🧩 Validating shared configurations...${NC}"
if python -c "from test_configs import MockFactory, get_unit_test_configs; print('✅ Shared configs loaded successfully')"; then
    echo -e "${GREEN}✅ Shared configuration system working${NC}"
else
    echo -e "${RED}❌ Shared configuration system failed${NC}"
    exit 1
fi

# Save baseline if requested
if [[ "$SAVE_BASELINE" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}📊 Saving performance baseline...${NC}"
    python test_bounds_prop_regression.py --capture-baseline
    echo -e "${GREEN}✅ Baseline saved${NC}"
fi

# Run main tests
echo ""
echo -e "${YELLOW}🧪 Running bounds propagation tests...${NC}"
case $TEST_MODE in
    "quick")
        echo "Running quick test suite..."
        python -m unittest test_bounds_propagation.TestBoundsPropagateLayers.test_reference_linear_implementation -v
        ;;
    *)
        echo "Running full test suite..."
        python -m unittest test_bounds_propagation -v
        ;;
esac

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✅ Bounds propagation tests passed${NC}"
else
    echo -e "${RED}❌ Bounds propagation tests failed${NC}"
    exit 1
fi

# Run regression tests if the file exists
if [ -f "test_bounds_prop_regression.py" ]; then
    echo ""
    echo -e "${YELLOW}🔄 Running regression tests...${NC}"
    if python test_bounds_prop_regression.py --test-regression; then
        echo -e "${GREEN}✅ Regression tests passed${NC}"
    else
        echo -e "${RED}❌ Regression tests failed${NC}"
        echo ""
        echo -e "${YELLOW}📋 Regression Test Failure Information:${NC}"
        echo "  • Detailed failure reasons are printed above"
        echo "  • Performance regressions indicate slower execution (>20% threshold)"
        echo "  • Correctness regressions indicate changed numerical results"
        echo ""
        echo -e "${BLUE}🔧 Next Steps:${NC}"
        echo "  • Review the detailed failure messages above"
        echo "  • If changes are intentional, update baseline: python test_bounds_prop_regression.py --capture-baseline"
        echo "  • If unintentional, investigate algorithmic or numerical changes"
        exit 1
    fi
fi

# Summary
echo ""
echo -e "${BLUE}🎉 Test Summary${NC}"
echo "==========================================="
echo -e "${GREEN}✅ All tests completed successfully!${NC}"
echo ""
echo "📋 What was tested:"
echo "  • Shared configuration system"
echo "  • Reference implementations"
echo "  • Bounds propagation APIs"
if [ -f "test_bounds_prop_regression.py" ]; then
    echo "  • Performance regression"
fi
echo ""
echo "🔧 Next steps:"
echo "  • Run with --save-baseline to update performance baseline"
echo "  • Run with --quick for faster testing during development"