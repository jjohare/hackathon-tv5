#!/bin/bash
# Build CUDA Kernels to PTX for Rust FFI
# Compiles all critical kernels for T4 GPU (sm_75) with Tensor Core support

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNEL_DIR="$PROJECT_ROOT/src/cuda/kernels"
PTX_OUTPUT_DIR="$PROJECT_ROOT/target/ptx"

echo -e "${BLUE}==================================================================="
echo "CUDA Kernel PTX Build System"
echo "Project: Media Gateway Hackathon"
echo "Target: T4 GPU (sm_75) with Tensor Core support"
echo "===================================================================${NC}"
echo ""

# Check CUDA availability
check_cuda() {
    echo -e "${YELLOW}Checking CUDA installation...${NC}"

    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}✗ nvcc not found${NC}"
        echo "  Please install CUDA toolkit or ensure it's in PATH"
        exit 1
    fi

    local cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}✓ CUDA Compiler: nvcc $cuda_version${NC}"

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        echo -e "${GREEN}✓ GPU Detected: $gpu_name${NC}"
    else
        echo -e "${YELLOW}⚠ nvidia-smi not found (GPU detection skipped)${NC}"
    fi
    echo ""
}

# Check kernel directory
check_kernel_dir() {
    echo -e "${YELLOW}Checking kernel directory...${NC}"

    if [ ! -d "$KERNEL_DIR" ]; then
        echo -e "${RED}✗ Kernel directory not found: $KERNEL_DIR${NC}"
        exit 1
    fi

    local kernel_count=$(find "$KERNEL_DIR" -name "*.cu" -type f | wc -l)
    echo -e "${GREEN}✓ Found $kernel_count CUDA kernel files${NC}"
    echo ""
}

# Create output directory
create_output_dir() {
    echo -e "${YELLOW}Creating PTX output directory...${NC}"
    mkdir -p "$PTX_OUTPUT_DIR"
    echo -e "${GREEN}✓ Output directory: $PTX_OUTPUT_DIR${NC}"
    echo ""
}

# Compile kernels
compile_kernels() {
    echo -e "${BLUE}==================================================================="
    echo "Compiling CUDA kernels to PTX..."
    echo "===================================================================${NC}"
    echo ""

    cd "$KERNEL_DIR"

    # Run make with PTX target
    if make ptx; then
        echo ""
        echo -e "${GREEN}✓ PTX compilation successful${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}✗ PTX compilation failed${NC}"
        return 1
    fi
}

# Verify output
verify_output() {
    echo ""
    echo -e "${YELLOW}Verifying PTX output...${NC}"

    local ptx_count=$(find "$PTX_OUTPUT_DIR" -name "*.ptx" -type f 2>/dev/null | wc -l)

    if [ "$ptx_count" -eq 0 ]; then
        echo -e "${RED}✗ No PTX files generated${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Generated $ptx_count PTX files${NC}"
    echo ""

    # List files with sizes
    echo -e "${BLUE}PTX Files:${NC}"
    ls -lh "$PTX_OUTPUT_DIR"/*.ptx | awk '{printf "  %-40s %8s\n", $9, $5}'
    echo ""

    # Total size
    local total_size=$(du -sh "$PTX_OUTPUT_DIR" | awk '{print $1}')
    echo -e "${GREEN}Total PTX size: $total_size${NC}"
}

# Display critical kernels
display_critical_kernels() {
    echo ""
    echo -e "${BLUE}==================================================================="
    echo "Critical Kernels for Rust FFI:"
    echo "===================================================================${NC}"

    local critical_kernels=(
        "semantic_similarity_fp16_tensor_cores.ptx"
        "graph_search.ptx"
        "ontology_reasoning.ptx"
        "hybrid_sssp.ptx"
    )

    for kernel in "${critical_kernels[@]}"; do
        if [ -f "$PTX_OUTPUT_DIR/$kernel" ]; then
            local size=$(stat -f%z "$PTX_OUTPUT_DIR/$kernel" 2>/dev/null || stat -c%s "$PTX_OUTPUT_DIR/$kernel")
            echo -e "${GREEN}✓${NC} $kernel ($(numfmt --to=iec-i --suffix=B $size))"
        else
            echo -e "${RED}✗${NC} $kernel (missing)"
        fi
    done
    echo ""
}

# Generate manifest
generate_manifest() {
    echo -e "${YELLOW}Generating PTX manifest...${NC}"

    local manifest_file="$PTX_OUTPUT_DIR/manifest.txt"

    cat > "$manifest_file" <<EOF
# PTX Manifest
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Target: T4 GPU (sm_75) with Tensor Core support
# Compiler: $(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)

EOF

    find "$PTX_OUTPUT_DIR" -name "*.ptx" -type f | sort | while read -r ptx_file; do
        local basename=$(basename "$ptx_file")
        local size=$(stat -f%z "$ptx_file" 2>/dev/null || stat -c%s "$ptx_file")
        echo "$basename $size" >> "$manifest_file"
    done

    echo -e "${GREEN}✓ Manifest: $manifest_file${NC}"
    echo ""
}

# Display integration info
display_integration_info() {
    echo -e "${BLUE}==================================================================="
    echo "Rust FFI Integration:"
    echo "===================================================================${NC}"
    echo ""
    echo "PTX files are available at:"
    echo "  $PTX_OUTPUT_DIR"
    echo ""
    echo "Load PTX in Rust using:"
    echo '  use cuda_driver_sys::*;'
    echo '  let ptx = include_str!("../target/ptx/semantic_similarity_fp16_tensor_cores.ptx");'
    echo '  cuModuleLoadData(&mut module, ptx.as_ptr() as *const _);'
    echo ""
    echo "Or use cubin for direct binary loading:"
    echo '  make sass  # Generate cubin files'
    echo ""
}

# Main execution
main() {
    check_cuda
    check_kernel_dir
    create_output_dir

    if compile_kernels; then
        verify_output
        display_critical_kernels
        generate_manifest
        display_integration_info

        echo -e "${GREEN}==================================================================="
        echo "✓ Build complete!"
        echo "===================================================================${NC}"
        exit 0
    else
        echo -e "${RED}==================================================================="
        echo "✗ Build failed!"
        echo "===================================================================${NC}"
        exit 1
    fi
}

# Run main
main "$@"
