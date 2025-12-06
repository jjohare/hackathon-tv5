# Makefile for GPU Graph Search Kernels
# Content Discovery and Recommendation System

# Compiler and flags
NVCC := nvcc
CUDA_ARCH := -arch=sm_70
CUDA_FLAGS := -O3 -use_fast_math -std=c++14
INCLUDE := -I./kernels
LIBS := -lcudart

# Directories
KERNEL_DIR := kernels
EXAMPLE_DIR := examples
BUILD_DIR := build

# Targets
KERNELS := $(KERNEL_DIR)/graph_search.cu
EXAMPLES := $(EXAMPLE_DIR)/graph_search_example.cu

# Build all
all: $(BUILD_DIR)/graph_search_example

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build example
$(BUILD_DIR)/graph_search_example: $(EXAMPLES) $(KERNELS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) $(INCLUDE) \
		$(EXAMPLES) $(KERNELS) \
		-o $@ $(LIBS)
	@echo "Built: $@"

# Compile kernels to PTX (for inspection)
$(BUILD_DIR)/graph_search.ptx: $(KERNELS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) \
		-ptx $(KERNELS) -o $@
	@echo "Generated PTX: $@"

# Compile kernels to cubin (binary)
$(BUILD_DIR)/graph_search.cubin: $(KERNELS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) \
		-cubin $(KERNELS) -o $@
	@echo "Generated cubin: $@"

# Run example
run: $(BUILD_DIR)/graph_search_example
	@echo "Running example..."
	./$(BUILD_DIR)/graph_search_example

# Profile with nvprof
profile: $(BUILD_DIR)/graph_search_example
	nvprof --print-gpu-trace ./$(BUILD_DIR)/graph_search_example

# Profile with nsight compute
nsight: $(BUILD_DIR)/graph_search_example
	ncu --set full ./$(BUILD_DIR)/graph_search_example

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Install (copy headers to system include)
install:
	@echo "Installing headers..."
	cp $(KERNEL_DIR)/graph_search.cuh /usr/local/include/
	@echo "Headers installed to /usr/local/include/"

# Generate documentation
docs:
	@echo "Documentation available at: docs/cuda/graph_search_kernels.md"

# Check CUDA installation
check:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@echo ""
	@nvidia-smi

# Build with debug symbols
debug: CUDA_FLAGS := -g -G -lineinfo
debug: all

# Build optimized for specific GPU architecture
sm_70: CUDA_ARCH := -arch=sm_70
sm_70: all

sm_75: CUDA_ARCH := -arch=sm_75
sm_75: all

sm_80: CUDA_ARCH := -arch=sm_80
sm_80: all

sm_86: CUDA_ARCH := -arch=sm_86
sm_86: all

sm_89: CUDA_ARCH := -arch=sm_89
sm_89: all

# Build for multiple architectures
multi_arch: CUDA_ARCH := -gencode arch=compute_70,code=sm_70 \
                         -gencode arch=compute_75,code=sm_75 \
                         -gencode arch=compute_80,code=sm_80 \
                         -gencode arch=compute_86,code=sm_86
multi_arch: all

# Help
help:
	@echo "GPU Graph Search Kernels - Makefile"
	@echo "===================================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all examples (default)"
	@echo "  run          - Build and run example"
	@echo "  profile      - Profile with nvprof"
	@echo "  nsight       - Profile with nsight compute"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install headers"
	@echo "  docs         - Show documentation location"
	@echo "  check        - Check CUDA installation"
	@echo "  debug        - Build with debug symbols"
	@echo ""
	@echo "Architecture-specific builds:"
	@echo "  sm_70        - Volta (Tesla V100)"
	@echo "  sm_75        - Turing (RTX 20XX)"
	@echo "  sm_80        - Ampere (A100)"
	@echo "  sm_86        - Ampere (RTX 30XX)"
	@echo "  sm_89        - Ada Lovelace (RTX 40XX)"
	@echo "  multi_arch   - Build for multiple architectures"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build default"
	@echo "  make run                # Build and run"
	@echo "  make sm_80              # Build for A100"
	@echo "  make profile            # Profile execution"

.PHONY: all run profile nsight clean install docs check help debug \
        sm_70 sm_75 sm_80 sm_86 sm_89 multi_arch
