#!/bin/bash
# Test ONNX Export Script
# This script tests the ONNX export functionality in a safe manner

set -e

echo "========================================"
echo "ONNX Export Test Script"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: Not running in a virtual environment${NC}"
    echo "Consider activating venv: source venv/bin/activate"
    echo ""
fi

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"
echo ""

# Check if required packages are installed
echo "Checking required packages..."
REQUIRED_PACKAGES=(
    "torch"
    "sentence_transformers"
    "onnx"
    "onnxruntime"
    "numpy"
    "scipy"
)

MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓ $package installed${NC}"
    else
        echo -e "${RED}✗ $package not installed${NC}"
        MISSING_PACKAGES+=("$package")
    fi
done

echo ""

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing packages detected!${NC}"
    echo "Install with: pip install -r scripts/requirements-onnx.txt"
    echo ""
    read -p "Install missing packages now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing packages..."
        pip install -r scripts/requirements-onnx.txt
        echo -e "${GREEN}✓ Packages installed${NC}"
        echo ""
    else
        echo -e "${RED}Cannot proceed without required packages${NC}"
        exit 1
    fi
fi

# Create models directory
echo "Creating models directory..."
mkdir -p models
echo -e "${GREEN}✓ Models directory ready${NC}"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✓ GPU available: $GPU_NAME (Count: $GPU_COUNT)${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected - using CPU (slower)${NC}"
fi
echo ""

# Check disk space
echo "Checking disk space..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 2 ]; then
    echo -e "${RED}✗ Insufficient disk space: ${AVAILABLE_SPACE}GB available${NC}"
    echo "Need at least 2GB for model export"
    exit 1
else
    echo -e "${GREEN}✓ Sufficient disk space: ${AVAILABLE_SPACE}GB available${NC}"
fi
echo ""

# Run a dry-run test with minimal iterations
echo "========================================"
echo "Running ONNX Export (Dry Run)"
echo "========================================"
echo ""

# Export with minimal benchmarking for quick test
python scripts/export_model_to_onnx.py \
    --benchmark-iterations 10 \
    2>&1 | tee /tmp/onnx_export_test.log

# Check if export was successful
if [ -f "models/sbert_optimized.onnx" ]; then
    echo ""
    echo -e "${GREEN}✓ ONNX export successful${NC}"

    # Get file size
    FILE_SIZE=$(du -h models/sbert_optimized.onnx | awk '{print $1}')
    echo "  Model size: $FILE_SIZE"

    # Check validation results
    if grep -q "Accuracy validation PASSED" /tmp/onnx_export_test.log; then
        echo -e "${GREEN}✓ Accuracy validation passed${NC}"
    else
        echo -e "${YELLOW}⚠ Accuracy validation may have issues${NC}"
    fi

    # Check benchmark results
    if grep -q "Speedup:" /tmp/onnx_export_test.log; then
        SPEEDUP=$(grep "Speedup:" /tmp/onnx_export_test.log | awk '{print $2}')
        echo "  Performance speedup: $SPEEDUP"
    fi

    echo ""
    echo -e "${GREEN}========================================"
    echo "ONNX Export Test PASSED"
    echo "========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review results in models/onnx_export_results.txt"
    echo "2. Run full benchmark: python scripts/export_model_to_onnx.py --benchmark-iterations 100"
    echo "3. Convert to TensorRT for production use"

else
    echo ""
    echo -e "${RED}✗ ONNX export failed${NC}"
    echo "Check logs in /tmp/onnx_export_test.log"
    exit 1
fi

# Verify ONNX model
echo ""
echo "Verifying ONNX model..."
python -c "
import onnx
model = onnx.load('models/sbert_optimized.onnx')
onnx.checker.check_model(model)
print('✓ ONNX model is valid')
" 2>&1

echo ""
echo -e "${GREEN}All tests passed!${NC}"
