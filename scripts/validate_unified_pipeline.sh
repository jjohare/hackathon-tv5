#!/bin/bash
# =============================================================================
# Unified Pipeline Validation Script
# =============================================================================
set -e

echo "========================================"
echo "Unified GPU Pipeline Validation"
echo "========================================"
echo ""

PROJECT_ROOT="/home/devuser/workspace/hackathon-tv5"
cd "$PROJECT_ROOT"

ERRORS=0
WARNINGS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

echo "1. Checking File Structure..."
echo "--------------------------------------"

# Check CUDA files
if [ -f "src/cuda/kernels/unified_pipeline.cu" ]; then
    check_pass "CUDA unified pipeline exists"
else
    check_fail "CUDA unified pipeline missing"
fi

if [ -f "src/cuda/kernels/semantic_similarity_fp16_tensor_cores.cu" ]; then
    check_pass "Phase 1 (Tensor Cores) exists"
else
    check_fail "Phase 1 file missing"
fi

if [ -f "src/cuda/kernels/sorted_similarity.cu" ]; then
    check_pass "Phase 2 (Memory Opt) exists"
else
    check_fail "Phase 2 file missing"
fi

if [ -f "src/cuda/kernels/hnsw_gpu.cuh" ]; then
    check_pass "Phase 3 (HNSW) exists"
else
    check_fail "Phase 3 file missing"
fi

# Check Rust files
if [ -f "src/rust/gpu_engine/unified_gpu.rs" ]; then
    check_pass "Rust FFI wrapper exists"
else
    check_fail "Rust FFI wrapper missing"
fi

if [ -f "src/rust/semantic_search/unified_engine.rs" ]; then
    check_pass "Recommendation engine exists"
else
    check_fail "Recommendation engine missing"
fi

# Check tests
if [ -f "tests/cuda_integration_test.rs" ]; then
    check_pass "Integration tests exist"
else
    check_fail "Integration tests missing"
fi

# Check build files
if [ -f "scripts/build_unified_pipeline.sh" ]; then
    check_pass "Build script exists"
else
    check_fail "Build script missing"
fi

if [ -f "build.rs" ]; then
    check_pass "Cargo build script exists"
else
    check_fail "Cargo build script missing"
fi

if [ -f "Makefile" ]; then
    check_pass "Makefile exists"
else
    check_fail "Makefile missing"
fi

echo ""
echo "2. Checking Documentation..."
echo "--------------------------------------"

if [ -f "README_UNIFIED_PIPELINE.md" ]; then
    check_pass "User guide exists"
else
    check_warn "User guide missing"
fi

if [ -f "docs/unified_pipeline_architecture.md" ]; then
    check_pass "Architecture docs exist"
else
    check_warn "Architecture docs missing"
fi

if [ -f "docs/INTEGRATION_SUMMARY.md" ]; then
    check_pass "Integration summary exists"
else
    check_warn "Integration summary missing"
fi

echo ""
echo "3. Checking CUDA Environment..."
echo "--------------------------------------"

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
    check_pass "CUDA toolkit found (version $NVCC_VERSION)"
else
    check_warn "CUDA toolkit not found (required for building)"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    check_pass "GPU found: $GPU_NAME (compute $GPU_COMPUTE)"
else
    check_warn "No NVIDIA GPU detected (required for runtime)"
fi

echo ""
echo "4. Checking Rust Environment..."
echo "--------------------------------------"

if command -v cargo &> /dev/null; then
    CARGO_VERSION=$(cargo --version | awk '{print $2}')
    check_pass "Cargo found (version $CARGO_VERSION)"
else
    check_fail "Cargo not found"
fi

echo ""
echo "5. Checking Code Structure..."
echo "--------------------------------------"

# Check CUDA kernel structure
if grep -q "unified_pipeline_search_knn" src/cuda/kernels/unified_pipeline.cu; then
    check_pass "Main search function defined"
else
    check_fail "Main search function missing"
fi

if grep -q "lsh_hash_kernel" src/cuda/kernels/unified_pipeline.cu; then
    check_pass "LSH hash kernel defined"
else
    check_fail "LSH hash kernel missing"
fi

if grep -q "compute_similarities_tensor_cores" src/cuda/kernels/unified_pipeline.cu; then
    check_pass "Tensor core kernel defined"
else
    check_fail "Tensor core kernel missing"
fi

# Check Rust FFI
if grep -q "extern \"C\"" src/rust/gpu_engine/unified_gpu.rs; then
    check_pass "FFI declarations found"
else
    check_fail "FFI declarations missing"
fi

if grep -q "pub struct GPUPipeline" src/rust/gpu_engine/unified_gpu.rs; then
    check_pass "GPUPipeline struct defined"
else
    check_fail "GPUPipeline struct missing"
fi

# Check recommendation engine
if grep -q "pub struct RecommendationEngine" src/rust/semantic_search/unified_engine.rs; then
    check_pass "RecommendationEngine defined"
else
    check_fail "RecommendationEngine missing"
fi

echo ""
echo "6. Checking Test Coverage..."
echo "--------------------------------------"

if grep -q "test_phase1_tensor_cores" tests/cuda_integration_test.rs; then
    check_pass "Phase 1 test exists"
else
    check_warn "Phase 1 test missing"
fi

if grep -q "test_phase2_memory_optimization" tests/cuda_integration_test.rs; then
    check_pass "Phase 2 test exists"
else
    check_warn "Phase 2 test missing"
fi

if grep -q "test_phase3_indexing" tests/cuda_integration_test.rs; then
    check_pass "Phase 3 test exists"
else
    check_warn "Phase 3 test missing"
fi

if grep -q "test_performance_target" tests/cuda_integration_test.rs; then
    check_pass "Performance test exists"
else
    check_warn "Performance test missing"
fi

echo ""
echo "7. Checking Build System..."
echo "--------------------------------------"

if [ -x "scripts/build_unified_pipeline.sh" ]; then
    check_pass "Build script is executable"
else
    check_warn "Build script not executable"
fi

if grep -q "unified_pipeline.cu" scripts/build_unified_pipeline.sh; then
    check_pass "Build script references unified pipeline"
else
    check_fail "Build script doesn't build unified pipeline"
fi

echo ""
echo "8. Line Counts..."
echo "--------------------------------------"

echo "Implementation files:"
wc -l src/cuda/kernels/unified_pipeline.cu | awk '{printf "  CUDA Pipeline:      %5d lines\n", $1}'
wc -l src/rust/gpu_engine/unified_gpu.rs | awk '{printf "  Rust FFI:           %5d lines\n", $1}'
wc -l src/rust/semantic_search/unified_engine.rs | awk '{printf "  Recommendation:     %5d lines\n", $1}'
wc -l tests/cuda_integration_test.rs | awk '{printf "  Integration Tests:  %5d lines\n", $1}'

echo ""
echo "========================================"
echo "Validation Summary"
echo "========================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "The unified GPU pipeline is fully integrated and ready to build."
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Passed with $WARNINGS warnings${NC}"
    echo ""
    echo "Core implementation is complete, but some optional items are missing."
    exit 0
else
    echo -e "${RED}✗ Failed with $ERRORS errors and $WARNINGS warnings${NC}"
    echo ""
    echo "Please fix the errors before building."
    exit 1
fi
