# Makefile for Unified GPU Pipeline

.PHONY: all cuda rust test clean help

# Default target
all: cuda rust

# Build CUDA unified pipeline
cuda:
	@echo "Building CUDA unified pipeline..."
	@bash scripts/build_unified_pipeline.sh

# Build Rust with GPU integration
rust: cuda
	@echo "Building Rust project..."
	cargo build --release

# Run tests
test: rust
	@echo "Running Rust tests..."
	cargo test --release
	@echo ""
	@echo "Running CUDA integration tests..."
	cargo test --release cuda_integration_test

# Run benchmarks
bench: rust
	@echo "Running benchmarks..."
	cargo bench

# Run example demo
demo: rust
	@echo "Running unified pipeline demo..."
	cargo run --release --example unified_pipeline_demo

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf src/cuda/build
	rm -f target/release/libunified_gpu.so

# Check CUDA availability
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || (echo "ERROR: CUDA not found"; exit 1)
	@nvidia-smi || (echo "ERROR: No NVIDIA GPU detected"; exit 1)

# Profile with Nsight Systems
profile: rust
	@echo "Profiling with Nsight Systems..."
	nsys profile --stats=true cargo run --release --example unified_pipeline_demo

# Memory check with cuda-memcheck
memcheck: rust
	@echo "Running CUDA memory checker..."
	cuda-memcheck cargo run --release --example unified_pipeline_demo

# Generate documentation
docs:
	@echo "Generating documentation..."
	cargo doc --no-deps --open

# Format code
fmt:
	@echo "Formatting code..."
	cargo fmt

# Lint
lint:
	@echo "Running clippy..."
	cargo clippy --all-targets --all-features

# Help
help:
	@echo "Unified GPU Pipeline - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build CUDA and Rust (default)"
	@echo "  cuda        - Build CUDA unified pipeline only"
	@echo "  rust        - Build Rust project (requires CUDA)"
	@echo "  test        - Run all tests"
	@echo "  bench       - Run benchmarks"
	@echo "  demo        - Run unified pipeline demo"
	@echo "  clean       - Remove build artifacts"
	@echo "  check-cuda  - Verify CUDA installation"
	@echo "  profile     - Profile with Nsight Systems"
	@echo "  memcheck    - Check for memory errors"
	@echo "  docs        - Generate documentation"
	@echo "  fmt         - Format code"
	@echo "  lint        - Run clippy linter"
	@echo "  help        - Show this help message"
