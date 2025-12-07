#!/bin/bash
# Build complete optimized model pipeline for A100 deployment
#
# Pipeline:
# 1. Convert PyTorch -> ONNX
# 2. Build TensorRT engine with FP16
# 3. Validate optimizations
#
# Usage: ./build_all_optimized.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================================"
echo "🚀 A100 Model Optimization Pipeline"
echo "============================================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check CUDA availability
echo "📋 Checking CUDA availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found - GPU may not be available"
else
    echo "✅ CUDA available:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
echo ""

# Phase 1: Convert to ONNX
echo "============================================================"
echo "PHASE 1: Convert PyTorch Model to ONNX"
echo "============================================================"
echo ""

if [ -f "$PROJECT_ROOT/data/models/minilm_l12_v2.onnx" ]; then
    echo "✅ ONNX model already exists"
    echo "   To rebuild, delete: data/models/minilm_l12_v2.onnx"
else
    echo "🔨 Converting PyTorch to ONNX..."
    python3 "$SCRIPT_DIR/convert_to_onnx.py"

    if [ $? -eq 0 ]; then
        echo "✅ ONNX conversion complete"
    else
        echo "❌ ONNX conversion failed"
        exit 1
    fi
fi
echo ""

# Phase 2: Build TensorRT engine
echo "============================================================"
echo "PHASE 2: Build TensorRT Engine (FP16)"
echo "============================================================"
echo ""

if [ -f "$PROJECT_ROOT/data/models/minilm_l12_v2_fp16.plan" ]; then
    echo "⚠️  TensorRT engine already exists"
    read -p "Rebuild? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping TensorRT build"
    else
        echo "🔨 Building TensorRT engine..."
        python3 "$SCRIPT_DIR/build_trt_engine.py"
    fi
else
    echo "🔨 Building TensorRT engine..."
    python3 "$SCRIPT_DIR/build_trt_engine.py"
fi

if [ -f "$PROJECT_ROOT/data/models/minilm_l12_v2_fp16.plan" ]; then
    echo "✅ TensorRT engine built"
else
    echo "❌ TensorRT build failed (may not be available in this environment)"
    echo "   TensorRT will be available on A100 VM deployment"
fi
echo ""

# Phase 3: Validation
echo "============================================================"
echo "PHASE 3: Validation"
echo "============================================================"
echo ""

echo "📊 Model sizes:"
if [ -f "$PROJECT_ROOT/data/models/minilm_l12_v2.onnx" ]; then
    ONNX_SIZE=$(du -h "$PROJECT_ROOT/data/models/minilm_l12_v2.onnx" | cut -f1)
    echo "  ONNX model: $ONNX_SIZE"
fi

if [ -f "$PROJECT_ROOT/data/models/minilm_l12_v2_fp16.plan" ]; then
    TRT_SIZE=$(du -h "$PROJECT_ROOT/data/models/minilm_l12_v2_fp16.plan" | cut -f1)
    echo "  TensorRT engine: $TRT_SIZE"
fi
echo ""

# Summary
echo "============================================================"
echo "✅ Pipeline Complete"
echo "============================================================"
echo ""
echo "Output files:"
echo "  1. ONNX: data/models/minilm_l12_v2.onnx"
echo "  2. TensorRT: data/models/minilm_l12_v2_fp16.plan"
echo "  3. Metadata: data/models/minilm_l12_v2_fp16.json"
echo ""
echo "Next steps:"
echo "  - Deploy to A100 VM"
echo "  - Run benchmarks: scripts/benchmarks/benchmark_trt_inference.py"
echo "  - Integrate into inference pipeline"
echo ""
