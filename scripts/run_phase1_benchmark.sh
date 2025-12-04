#!/bin/bash
# =============================================================================
# Phase 1: Run Tensor Core Benchmark
# =============================================================================

set -e

PROJECT_ROOT="/home/devuser/workspace/hackathon-tv5"
BIN_DIR="$PROJECT_ROOT/bin"
RESULTS_DIR="$PROJECT_ROOT/results/phase1"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "==================================================================="
echo "Phase 1: Tensor Core Performance Benchmark"
echo "==================================================================="
echo ""

# Check if benchmark exists
if [ ! -f "$BIN_DIR/tensor_core_benchmark" ]; then
    echo "ERROR: Benchmark not found. Please run compile_phase1.sh first"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU may not be available"
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu \
           --format=csv,noheader
echo ""

# Run benchmark
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.log"

echo "Running benchmark..."
echo "Results will be saved to: $RESULTS_FILE"
echo ""

cd "$BIN_DIR"
./tensor_core_benchmark 2>&1 | tee "$RESULTS_FILE"

# Parse results
echo ""
echo "==================================================================="
echo "Quick Results Summary"
echo "==================================================================="

if grep -q "Speedup:" "$RESULTS_FILE"; then
    SPEEDUP=$(grep "Speedup:" "$RESULTS_FILE" | awk '{print $2}')
    echo "✓ Speedup achieved: $SPEEDUP"

    # Check if target met
    SPEEDUP_NUM=$(echo $SPEEDUP | sed 's/x//')
    if (( $(echo "$SPEEDUP_NUM >= 8.0" | bc -l) )); then
        echo "✓ SUCCESS: Target 8-10x speedup ACHIEVED!"
    elif (( $(echo "$SPEEDUP_NUM >= 5.0" | bc -l) )); then
        echo "⚠ PARTIAL: Good improvement, but below 8x target"
    else
        echo "✗ FAILED: Speedup below expectations"
    fi
else
    echo "✗ Could not parse benchmark results"
fi

if grep -q "Average error:" "$RESULTS_FILE"; then
    ERROR=$(grep "Average error:" "$RESULTS_FILE" | awk '{print $3}')
    echo "✓ Accuracy error: $ERROR"

    ERROR_NUM=$(echo $ERROR)
    if (( $(echo "$ERROR_NUM < 0.01" | bc -l) )); then
        echo "✓ Accuracy within tolerance (< 1%)"
    else
        echo "⚠ WARNING: Accuracy error exceeds 1%"
    fi
fi

echo ""
echo "Full results saved to: $RESULTS_FILE"
echo "==================================================================="
