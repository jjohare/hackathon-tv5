#!/bin/bash
# =============================================================================
# Build Unified GPU Pipeline - All 3 Phases
# =============================================================================
set -e

echo "=========================================="
echo "Building Unified GPU Pipeline"
echo "=========================================="

# Configuration
CUDA_ARCH="sm_75"  # T4 GPU (Turing architecture)
OPTIMIZATION="-O3 --use_fast_math"
PROJECT_ROOT="/home/devuser/workspace/hackathon-tv5"
CUDA_DIR="$PROJECT_ROOT/src/cuda"
BUILD_DIR="$CUDA_DIR/build"
LIB_DIR="$PROJECT_ROOT/target/release"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$LIB_DIR"

cd "$CUDA_DIR/kernels"

echo ""
echo "Step 1: Compiling Phase 1 (Tensor Cores)..."
echo "--------------------------------------------"
nvcc -c semantic_similarity_fp16_tensor_cores.cu \
    -arch=$CUDA_ARCH \
    $OPTIMIZATION \
    -Xptxas -v \
    --ptxas-options=-v \
    -lineinfo \
    -I. \
    -o "$BUILD_DIR/tensor_cores.o"

if [ $? -eq 0 ]; then
    echo "✓ Phase 1 compiled successfully"
else
    echo "✗ Phase 1 compilation failed"
    exit 1
fi

echo ""
echo "Step 2: Compiling Phase 2 (Memory Optimization)..."
echo "--------------------------------------------"
nvcc -c sorted_similarity.cu \
    -arch=$CUDA_ARCH \
    $OPTIMIZATION \
    -Xptxas -v \
    -I. \
    -o "$BUILD_DIR/memory_opt.o"

if [ $? -eq 0 ]; then
    echo "✓ Phase 2 compiled successfully"
else
    echo "✗ Phase 2 compilation failed"
    exit 1
fi

echo ""
echo "Step 3: Compiling Phase 3 (Advanced Indexing)..."
echo "--------------------------------------------"

# LSH
nvcc -c lsh_gpu.cu \
    -arch=$CUDA_ARCH \
    $OPTIMIZATION \
    -I. \
    -o "$BUILD_DIR/lsh_gpu.o"

# Product Quantization
nvcc -c product_quantization.cu \
    -arch=$CUDA_ARCH \
    $OPTIMIZATION \
    -I. \
    -o "$BUILD_DIR/pq.o"

# HNSW (header-only, will be included)

if [ $? -eq 0 ]; then
    echo "✓ Phase 3 compiled successfully"
else
    echo "✗ Phase 3 compilation failed"
    exit 1
fi

echo ""
echo "Step 4: Compiling Unified Pipeline..."
echo "--------------------------------------------"
nvcc -c unified_pipeline.cu \
    -arch=$CUDA_ARCH \
    $OPTIMIZATION \
    -Xptxas -v \
    -I. \
    -o "$BUILD_DIR/unified_pipeline.o"

if [ $? -eq 0 ]; then
    echo "✓ Unified pipeline compiled successfully"
else
    echo "✗ Unified pipeline compilation failed"
    exit 1
fi

echo ""
echo "Step 5: Linking shared library..."
echo "--------------------------------------------"
nvcc \
    "$BUILD_DIR/tensor_cores.o" \
    "$BUILD_DIR/memory_opt.o" \
    "$BUILD_DIR/lsh_gpu.o" \
    "$BUILD_DIR/pq.o" \
    "$BUILD_DIR/unified_pipeline.o" \
    -arch=$CUDA_ARCH \
    -shared \
    -lcudart \
    -lcublas \
    -o "$BUILD_DIR/libunified_gpu.so"

if [ $? -eq 0 ]; then
    echo "✓ Linking successful"
else
    echo "✗ Linking failed"
    exit 1
fi

echo ""
echo "Step 6: Copying library to Rust target..."
echo "--------------------------------------------"
cp "$BUILD_DIR/libunified_gpu.so" "$LIB_DIR/"

if [ $? -eq 0 ]; then
    echo "✓ Library copied to $LIB_DIR"
else
    echo "✗ Failed to copy library"
    exit 1
fi

echo ""
echo "Step 7: Verifying library..."
echo "--------------------------------------------"
if [ -f "$LIB_DIR/libunified_gpu.so" ]; then
    ldd "$LIB_DIR/libunified_gpu.so" || true
    echo ""
    ls -lh "$LIB_DIR/libunified_gpu.so"
    echo "✓ Library verification complete"
else
    echo "✗ Library not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Build Summary"
echo "=========================================="
echo "Phase 1: Tensor Core Acceleration - ✓"
echo "Phase 2: Memory Optimization - ✓"
echo "Phase 3: Advanced Indexing - ✓"
echo "Unified Pipeline - ✓"
echo ""
echo "Library: $LIB_DIR/libunified_gpu.so"
echo ""
echo "Expected Performance:"
echo "  - Tensor Cores: 8-10x speedup"
echo "  - Memory Opt: 4-5x speedup"
echo "  - Indexing: 10-100x candidate reduction"
echo "  - Combined: 300-500x vs baseline"
echo ""
echo "Next steps:"
echo "  1. Run tests: cargo test --release"
echo "  2. Run benchmark: cargo bench"
echo "  3. Profile: nsys profile ./target/release/benchmark"
echo ""
echo "=========================================="
echo "Build Complete! 🚀"
echo "=========================================="
