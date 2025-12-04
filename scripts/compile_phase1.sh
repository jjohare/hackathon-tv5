#!/bin/bash
# =============================================================================
# Phase 1: Compile Tensor Core Optimized Kernels
# =============================================================================

set -e

echo "==================================================================="
echo "Phase 1: Compiling Tensor Core Optimized CUDA Kernels"
echo "==================================================================="

# Project directories
PROJECT_ROOT="/home/devuser/workspace/hackathon-tv5"
SRC_DIR="$PROJECT_ROOT/src/cuda"
BUILD_DIR="$PROJECT_ROOT/build/cuda"
BIN_DIR="$PROJECT_ROOT/bin"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$BIN_DIR"

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit"
    exit 1
fi

echo "CUDA Version:"
nvcc --version | grep "release"
echo ""

# Detect GPU compute capability
echo "Detecting GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found, assuming T4 (sm_75)"
fi
echo ""

# Compilation flags
CUDA_ARCH="-arch=sm_75"  # T4 GPU
CUDA_FLAGS="-O3 -use_fast_math -Xcompiler -fopenmp"
CUDA_INCLUDES="-I$SRC_DIR/include"
CUDA_LIBS="-lcudart -lcublas"

echo "==================================================================="
echo "Step 1: Compiling Original Kernel (Scalar Operations)"
echo "==================================================================="

nvcc $CUDA_ARCH $CUDA_FLAGS \
    -c "$SRC_DIR/kernels/semantic_similarity_fp16.cu" \
    -o "$BUILD_DIR/semantic_similarity_fp16.o"

echo "✓ Original kernel compiled"
echo ""

echo "==================================================================="
echo "Step 2: Compiling Optimized Kernel (Tensor Cores)"
echo "==================================================================="

nvcc $CUDA_ARCH $CUDA_FLAGS \
    -c "$SRC_DIR/kernels/semantic_similarity_fp16_tensor_cores.cu" \
    -o "$BUILD_DIR/semantic_similarity_fp16_tensor_cores.o"

echo "✓ Tensor core kernel compiled"
echo ""

echo "==================================================================="
echo "Step 3: Compiling Benchmark"
echo "==================================================================="

nvcc $CUDA_ARCH $CUDA_FLAGS \
    "$SRC_DIR/benchmarks/tensor_core_test.cu" \
    "$BUILD_DIR/semantic_similarity_fp16.o" \
    "$BUILD_DIR/semantic_similarity_fp16_tensor_cores.o" \
    -o "$BIN_DIR/tensor_core_benchmark" \
    $CUDA_LIBS

echo "✓ Benchmark compiled"
echo ""

echo "==================================================================="
echo "Compilation Summary"
echo "==================================================================="
echo "Object files:"
ls -lh "$BUILD_DIR"/*.o
echo ""
echo "Executable:"
ls -lh "$BIN_DIR/tensor_core_benchmark"
echo ""

echo "==================================================================="
echo "✓ Phase 1 Compilation Complete!"
echo "==================================================================="
echo ""
echo "To run benchmark:"
echo "  cd $BIN_DIR"
echo "  ./tensor_core_benchmark"
echo ""
