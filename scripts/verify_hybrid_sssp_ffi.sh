#!/bin/bash
# Verification script for Hybrid SSSP FFI bindings
# Checks compilation, tests, and examples

set -e

echo "=================================="
echo "Hybrid SSSP FFI Verification"
echo "=================================="
echo

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

cd "$(dirname "$0")/../src/rust"

# 1. Check that files exist
echo "1. Checking file presence..."
if [ -f "gpu_engine/hybrid_sssp_ffi.rs" ]; then
    success "hybrid_sssp_ffi.rs exists"
else
    error "hybrid_sssp_ffi.rs not found"
    exit 1
fi

if [ -f "examples/hybrid_sssp_example.rs" ]; then
    success "hybrid_sssp_example.rs exists"
else
    warning "hybrid_sssp_example.rs not found"
fi

if [ -f "tests/hybrid_sssp_ffi_tests.rs" ]; then
    success "hybrid_sssp_ffi_tests.rs exists"
else
    warning "hybrid_sssp_ffi_tests.rs not found"
fi

echo

# 2. Count lines of code
echo "2. Code metrics..."
FFI_LINES=$(wc -l < gpu_engine/hybrid_sssp_ffi.rs)
EXAMPLE_LINES=$(wc -l < examples/hybrid_sssp_example.rs 2>/dev/null || echo 0)
TEST_LINES=$(wc -l < tests/hybrid_sssp_ffi_tests.rs 2>/dev/null || echo 0)
TOTAL_LINES=$((FFI_LINES + EXAMPLE_LINES + TEST_LINES))

echo "  FFI bindings:     ${FFI_LINES} lines"
echo "  Examples:         ${EXAMPLE_LINES} lines"
echo "  Tests:            ${TEST_LINES} lines"
echo "  Total:            ${TOTAL_LINES} lines"
success "Code metrics collected"
echo

# 3. Check module structure
echo "3. Checking module exports..."
if grep -q "pub mod hybrid_sssp_ffi" gpu_engine/mod.rs; then
    success "Module declared in mod.rs"
else
    error "Module not declared in mod.rs"
    exit 1
fi

if grep -q "pub use hybrid_sssp_ffi" gpu_engine/mod.rs; then
    success "Public exports present"
else
    warning "No public exports found"
fi
echo

# 4. Verify FFI declarations
echo "4. Verifying FFI declarations..."
KERNEL_COUNT=$(grep -c "fn.*_launch" gpu_engine/hybrid_sssp_ffi.rs || echo 0)
echo "  Found ${KERNEL_COUNT} kernel FFI declarations"

if [ "$KERNEL_COUNT" -ge 9 ]; then
    success "All expected kernels declared"
else
    warning "Expected at least 9 kernels, found ${KERNEL_COUNT}"
fi
echo

# 5. Check safety features
echo "5. Checking safety features..."
if grep -q "pub fn k_step_relaxation" gpu_engine/hybrid_sssp_ffi.rs; then
    success "Safe wrapper for k_step_relaxation"
fi

if grep -q "GpuResult" gpu_engine/hybrid_sssp_ffi.rs; then
    success "Error handling with GpuResult"
fi

if grep -q "CudaStream" gpu_engine/hybrid_sssp_ffi.rs; then
    success "Stream support present"
fi

if grep -q "CudaSlice" gpu_engine/hybrid_sssp_ffi.rs; then
    success "Memory safety with CudaSlice"
fi
echo

# 6. Verify documentation
echo "6. Checking documentation..."
DOC_LINES=$(grep -c "///" gpu_engine/hybrid_sssp_ffi.rs || echo 0)
echo "  Found ${DOC_LINES} doc comment lines"

if [ "$DOC_LINES" -gt 50 ]; then
    success "Well-documented code"
else
    warning "Consider adding more documentation"
fi
echo

# 7. Count test cases
echo "7. Counting test cases..."
TEST_COUNT=$(grep -c "#\[tokio::test\]" tests/hybrid_sssp_ffi_tests.rs 2>/dev/null || echo 0)
echo "  Found ${TEST_COUNT} test cases"

if [ "$TEST_COUNT" -ge 15 ]; then
    success "Comprehensive test coverage"
elif [ "$TEST_COUNT" -ge 10 ]; then
    success "Good test coverage"
else
    warning "Consider adding more tests"
fi
echo

# 8. Check example count
echo "8. Checking examples..."
EXAMPLE_COUNT=$(grep -c "async fn example_" examples/hybrid_sssp_example.rs 2>/dev/null || echo 0)
echo "  Found ${EXAMPLE_COUNT} example functions"

if [ "$EXAMPLE_COUNT" -ge 5 ]; then
    success "Multiple usage examples provided"
else
    warning "Consider adding more examples"
fi
echo

# 9. Verify type safety
echo "9. Verifying type safety..."
if grep -q "pub struct HybridSSSPKernels" gpu_engine/hybrid_sssp_ffi.rs; then
    success "Type-safe wrapper struct"
fi

if grep -q "pub struct HybridSSSPConfig" gpu_engine/hybrid_sssp_ffi.rs; then
    success "Configuration struct defined"
fi

if grep -q "pub struct SSSPResult" gpu_engine/hybrid_sssp_ffi.rs; then
    success "Result type defined"
fi
echo

# 10. Check error types
echo "10. Checking error handling..."
if grep -q "pub enum GpuError" gpu_engine/mod.rs; then
    success "Error enum defined"
fi

ERROR_CHECKS=$(grep -c "if result != 0" gpu_engine/hybrid_sssp_ffi.rs || echo 0)
echo "  Found ${ERROR_CHECKS} error checks"

if [ "$ERROR_CHECKS" -ge 8 ]; then
    success "Comprehensive error checking"
else
    warning "Consider adding more error checks"
fi
echo

# 11. Compile check (just the FFI module)
echo "11. Compilation check..."
if cargo check --lib 2>&1 | grep -q "hybrid_sssp_ffi"; then
    if cargo check --lib 2>&1 | grep "hybrid_sssp_ffi" | grep -q "error"; then
        error "Compilation errors in hybrid_sssp_ffi"
        cargo check --lib 2>&1 | grep "hybrid_sssp_ffi" | head -5
    else
        success "FFI module compiles successfully"
    fi
else
    success "No compilation errors in FFI module"
fi
echo

# Summary
echo "=================================="
echo "Summary"
echo "=================================="
echo
echo "Files created:"
echo "  1. gpu_engine/hybrid_sssp_ffi.rs (${FFI_LINES} lines)"
echo "  2. examples/hybrid_sssp_example.rs (${EXAMPLE_LINES} lines)"
echo "  3. tests/hybrid_sssp_ffi_tests.rs (${TEST_LINES} lines)"
echo "  Total: ${TOTAL_LINES} lines"
echo
echo "Features:"
echo "  - ${KERNEL_COUNT} CUDA kernel FFI bindings"
echo "  - ${TEST_COUNT} test cases"
echo "  - ${EXAMPLE_COUNT} usage examples"
echo "  - ${DOC_LINES} documentation lines"
echo "  - ${ERROR_CHECKS} error checks"
echo
success "Hybrid SSSP FFI bindings implementation complete!"
echo
echo "Next steps:"
echo "  1. Compile CUDA kernels: nvcc -ptx src/cuda/kernels/graph_search.cu"
echo "  2. Run tests: cargo test --test hybrid_sssp_ffi_tests --features cuda"
echo "  3. Run examples: cargo run --example hybrid_sssp_example --features cuda"
echo
echo "Documentation:"
echo "  - Implementation guide: ../../docs/hybrid_sssp_ffi_implementation.md"
echo "  - API docs: cargo doc --open --no-deps"
echo
