# Makefile for GPU Kernels - T4 Optimized
# Google T4 (Turing sm_75) with FP16 Tensor Core Support

# Compiler and flags
NVCC := nvcc

# T4-specific architecture flags (Turing sm_75)
CUDA_ARCH := -arch=sm_75 -gencode arch=compute_75,code=sm_75

# T4 Optimization flags
# - FP16 tensor core usage via -use_fast_math
# - Turing instruction set optimizations
# - Register pressure optimization for 2560 cores
CUDA_FLAGS := -O3 -use_fast_math -std=c++14 \
              --ptxas-options=-v \
              -lineinfo \
              -Xptxas -O3 \
              -Xcompiler -O3,-march=native \
              -maxrregcount=128

# T4-specific tensor core flags
TENSOR_FLAGS := -DUSE_TENSOR_CORES=1 -DUSE_FP16=1

# Include paths
INCLUDE := -I./kernels -I/usr/local/cuda/include

# Libraries
LIBS := -lcudart -lcuda -lcublas -lcudnn -lnccl

# Directories
KERNEL_DIR := kernels
BUILD_DIR := build

# Source files
KERNELS := \
	$(KERNEL_DIR)/semantic_similarity.cu \
	$(KERNEL_DIR)/semantic_similarity_fp16.cu \
	$(KERNEL_DIR)/graph_search.cu \
	$(KERNEL_DIR)/ontology_reasoning.cu \
	$(KERNEL_DIR)/sorted_similarity.cu \
	$(KERNEL_DIR)/memory_layout.cu

# Object files
OBJECTS := $(patsubst $(KERNEL_DIR)/%.cu,$(BUILD_DIR)/%.o,$(KERNELS))

# Build all
all: $(BUILD_DIR)/libkernels_t4.a $(BUILD_DIR)/t4_validation

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build T4 validation program
$(BUILD_DIR)/t4_validation: ../examples/t4_validation.cu | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) $(TENSOR_FLAGS) $(INCLUDE) \
		$< -o $@ $(LIBS)
	@echo "Built T4 validation: $@"

# Compile individual kernels
$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) $(TENSOR_FLAGS) $(INCLUDE) \
		-dc $< -o $@
	@echo "Compiled: $< -> $@"

# Link into static library
$(BUILD_DIR)/libkernels_t4.a: $(OBJECTS)
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) -dlink $(OBJECTS) -o $(BUILD_DIR)/device_link.o
	ar rcs $@ $(OBJECTS) $(BUILD_DIR)/device_link.o
	@echo "Built T4 kernel library: $@"

# Compile to PTX for inspection
ptx: $(KERNELS) | $(BUILD_DIR)
	@for kernel in $(KERNELS); do \
		base=$$(basename $$kernel .cu); \
		$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) $(TENSOR_FLAGS) $(INCLUDE) \
			-ptx $$kernel -o $(BUILD_DIR)/$$base.ptx; \
		echo "Generated PTX: $(BUILD_DIR)/$$base.ptx"; \
	done

# Compile to SASS for inspection
sass: $(KERNELS) | $(BUILD_DIR)
	@for kernel in $(KERNELS); do \
		base=$$(basename $$kernel .cu); \
		$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) $(TENSOR_FLAGS) $(INCLUDE) \
			-cubin $$kernel -o $(BUILD_DIR)/$$base.cubin; \
		cuobjdump -sass $(BUILD_DIR)/$$base.cubin > $(BUILD_DIR)/$$base.sass; \
		echo "Generated SASS: $(BUILD_DIR)/$$base.sass"; \
	done

# Profile with nsight compute (T4-specific)
profile: $(BUILD_DIR)/libkernels_t4.a
	@echo "Profiling T4 kernels with Nsight Compute..."
	ncu --target-processes all --kernel-name-base mangled \
	    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
	    --launch-skip 0 --launch-count 10 \
	    $(BUILD_DIR)/test_t4

# Memory analysis (16GB VRAM constraint)
memory-check: $(BUILD_DIR)/libkernels_t4.a
	@echo "Analyzing T4 memory usage (16GB limit)..."
	cuda-memcheck --leak-check full $(BUILD_DIR)/test_t4

# Benchmark T4 performance
benchmark: $(BUILD_DIR)/libkernels_t4.a
	@echo "Running T4 benchmarks..."
	./benchmarks/t4_benchmarks.sh

# Run T4 validation tests
test-t4: $(BUILD_DIR)/t4_validation
	@echo "Running T4 validation tests..."
	@echo ""
	./$(BUILD_DIR)/t4_validation

# Check register usage (critical for occupancy)
check-registers: $(KERNELS) | $(BUILD_DIR)
	@for kernel in $(KERNELS); do \
		base=$$(basename $$kernel .cu); \
		echo "=== Register usage: $$base ==="; \
		$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) $(TENSOR_FLAGS) $(INCLUDE) \
			--ptxas-options=-v -c $$kernel -o /dev/null 2>&1 | grep -E "(registers|spills|shared)"; \
	done

# Analyze occupancy
occupancy: $(BUILD_DIR)/libkernels_t4.a
	@echo "Analyzing T4 occupancy (2560 cores, 40 SMs)..."
	nvprof --analysis-metrics -o $(BUILD_DIR)/t4_analysis.nvvp $(BUILD_DIR)/test_t4

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Architecture-specific builds
t4: CUDA_ARCH := -arch=sm_75
t4: all

# Multi-GPU build with NCCL
multi-gpu: CUDA_FLAGS += -DMULTI_GPU=1
multi-gpu: LIBS += -lnccl
multi-gpu: all

# FP16-optimized build
fp16: CUDA_FLAGS += -DUSE_FP16=1 -DTENSOR_CORES=1
fp16: all

# Debug build
debug: CUDA_FLAGS := -g -G -lineinfo -DDEBUG=1
debug: all

# Help
help:
	@echo "T4-Optimized GPU Kernel Makefile"
	@echo "================================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build T4-optimized kernel library (default)"
	@echo "  test-t4      - Run T4 validation tests"
	@echo "  ptx          - Generate PTX assembly"
	@echo "  sass         - Generate SASS assembly"
	@echo "  profile      - Profile with Nsight Compute"
	@echo "  memory-check - Check memory usage (16GB constraint)"
	@echo "  benchmark    - Run T4 benchmarks"
	@echo "  check-registers - Analyze register usage"
	@echo "  occupancy    - Analyze kernel occupancy"
	@echo "  clean        - Remove build artifacts"
	@echo ""
	@echo "Specialized builds:"
	@echo "  t4           - Build for T4 (sm_75)"
	@echo "  multi-gpu    - Build with NCCL multi-GPU support"
	@echo "  fp16         - Build with FP16 tensor cores"
	@echo "  debug        - Build with debug symbols"
	@echo ""
	@echo "T4 Specifications:"
	@echo "  Architecture: Turing (sm_75)"
	@echo "  CUDA cores: 2560 (40 SMs × 64 cores/SM)"
	@echo "  Tensor cores: 320 (FP16 only)"
	@echo "  Memory: 16GB GDDR6 @ 320 GB/s"
	@echo "  FP16 peak: 65 TFLOPS"
	@echo "  FP32 peak: 8.1 TFLOPS"
	@echo ""

.PHONY: all ptx sass profile memory-check benchmark check-registers occupancy clean t4 multi-gpu fp16 debug help

# Phase 2 Memory Optimization Targets
phase2-benchmark: ../examples/phase2_benchmark.cu $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) $(TENSOR_FLAGS) $(INCLUDE) \
		$< -o $(BUILD_DIR)/phase2_benchmark $(LIBS)
	@echo "Built Phase 2 benchmark: $(BUILD_DIR)/phase2_benchmark"

phase2-test: phase2-benchmark
	@echo "==================================================================="
	@echo "Running Phase 2 Memory Optimization Benchmark"
	@echo "Expected: 4-5x speedup from coalesced memory access"
	@echo "==================================================================="
	./$(BUILD_DIR)/phase2_benchmark

phase2-profile: phase2-benchmark
	@echo "Profiling Phase 2 memory access patterns..."
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l2_utilization \
	    --target-processes all \
	    ./$(BUILD_DIR)/phase2_benchmark

phase2-bandwidth: phase2-benchmark
	@echo "Analyzing memory bandwidth efficiency..."
	nvprof --metrics dram_read_throughput,dram_write_throughput,gld_efficiency,gst_efficiency \
	    ./$(BUILD_DIR)/phase2_benchmark

.PHONY: phase2-benchmark phase2-test phase2-profile phase2-bandwidth
