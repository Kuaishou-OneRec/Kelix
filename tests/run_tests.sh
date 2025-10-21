#!/bin/bash
# Script to run layer unit tests

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running Muse Layer Unit Tests${NC}"
echo "=================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install it with: pip install pytest"
    exit 1
fi

# Check if torch is installed
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: PyTorch is not installed${NC}"
    echo "Install it with: pip install torch"
    exit 1
fi

echo -e "${GREEN}All dependencies found${NC}"
echo ""

# Run tests based on argument
if [ $# -eq 0 ]; then
    # Run all tests
    echo "Running all layer tests..."
    pytest tests/ -v --tb=short
elif [ "$1" == "quick" ]; then
    # Run quick tests (skip FlashAttention)
    echo "Running quick tests (CPU only)..."
    pytest tests/ -v --tb=short -m "not flash"
elif [ "$1" == "coverage" ]; then
    # Run with coverage
    echo "Running tests with coverage..."
    pytest tests/ -v --cov=muse/layers --cov-report=html --cov-report=term
else
    # Run specific test file
    echo "Running $1..."
    pytest tests/$1 -v --tb=short
fi

echo ""
echo -e "${GREEN}Tests completed!${NC}"

