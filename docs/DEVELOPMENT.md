# Development Guide

Comprehensive development environment setup and contribution guide.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Building from Source](#building-from-source)
3. [Running Tests](#running-tests)
4. [Contributing Workflow](#contributing-workflow)
5. [Code Style Guide](#code-style-guide)
6. [Debugging and Profiling](#debugging-and-profiling)
7. [Development Tools](#development-tools)

## Development Environment Setup

### System Requirements

**Hardware**
- CPU: 8+ cores (16+ recommended)
- RAM: 16GB minimum (32GB+ recommended)
- GPU: NVIDIA with CUDA support (T4, RTX 2080+, A100)
- Storage: 50GB free space (SSD recommended)

**Software**
- OS: Linux (Ubuntu 22.04+), macOS 12+, or Windows 11 with WSL2
- CUDA Toolkit 12.2+
- Rust 1.75+ with cargo
- Node.js 18+ with npm
- Git 2.30+
- Docker 24.0+ (optional)

### Installing Dependencies

#### Linux (Ubuntu/Debian)

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install build essentials
sudo apt-get install -y \
  build-essential \
  pkg-config \
  libssl-dev \
  curl \
  git \
  cmake

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Node.js (via nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18

# Install additional tools
sudo apt-get install -y \
  lldb \
  gdb \
  valgrind \
  perf \
  htop \
  ncdu
```

#### macOS

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install \
  rust \
  node \
  git \
  cmake \
  pkg-config \
  openssl

# Note: CUDA not available on macOS
# Use Metal for GPU acceleration (future support)

# Install LLDB debugger
xcode-select --install
```

#### Windows (WSL2)

```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Linux instructions above

# Install NVIDIA CUDA support for WSL2
# Download from: https://developer.nvidia.com/cuda/wsl
```

### IDE Setup

#### Visual Studio Code (Recommended)

**Install VS Code**
```bash
# Linux
sudo snap install code --classic

# macOS
brew install --cask visual-studio-code

# Windows
# Download from https://code.visualstudio.com/
```

**Recommended Extensions**
```json
{
  "recommendations": [
    "rust-lang.rust-analyzer",
    "vadimcn.vscode-lldb",
    "serayuzgur.crates",
    "tamasfe.even-better-toml",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "ms-vscode.cmake-tools",
    "nvidia.nsight-vscode-edition",
    "GitHub.copilot",
    "GitHub.vscode-pull-request-github"
  ]
}
```

**Workspace Settings (.vscode/settings.json)**
```json
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.cargo.features": ["all"],
  "rust-analyzer.inlayHints.enable": true,
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

**Launch Configuration (.vscode/launch.json)**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug unit tests",
      "cargo": {
        "args": ["test", "--no-run", "--lib"],
        "filter": {
          "name": "hackathon-tv5",
          "kind": "lib"
        }
      },
      "args": [],
      "cwd": "${workspaceFolder}"
    },
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug example",
      "cargo": {
        "args": ["build", "--example", "basic_search"],
        "filter": {
          "name": "basic_search",
          "kind": "example"
        }
      },
      "args": [],
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

#### CLion / IntelliJ IDEA (Alternative)

```bash
# Install Rust plugin
# File → Settings → Plugins → Search "Rust"

# Configure Rust toolchain
# File → Settings → Languages & Frameworks → Rust
# Toolchain location: ~/.cargo/bin
```

### Environment Configuration

**Create .env file**
```bash
# Copy template
cp .env.example .env

# Edit configuration
vim .env
```

**.env contents**
```bash
# Application
RUST_LOG=debug
RUST_BACKTRACE=1

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=128
EMBEDDING_DIM=1024

# Database URLs
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=devpassword
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgres://postgres:devpassword@localhost:5432/media_gateway

# API Configuration
API_HOST=127.0.0.1
API_PORT=8080
API_WORKERS=4

# Development
ENABLE_HOT_RELOAD=true
ENABLE_PROFILING=true
```

## Building from Source

### Clone Repository

```bash
git clone https://github.com/agenticsorg/hackathon-tv5.git
cd hackathon-tv5
```

### Build CUDA Kernels

```bash
# Navigate to CUDA source
cd src/cuda/kernels

# Compile tensor core kernel
nvcc -arch=sm_75 -O3 -use_fast_math \
  --compiler-options=-fPIC \
  -o semantic_similarity_tc.o \
  -c semantic_similarity_fp16_tensor_cores.cu

# Compile other kernels
nvcc -arch=sm_75 -O3 -use_fast_math \
  --compiler-options=-fPIC \
  -o ontology_reasoning.o \
  -c ontology_reasoning.cu

nvcc -arch=sm_75 -O3 -use_fast_math \
  --compiler-options=-fPIC \
  -o graph_search.o \
  -c graph_search.cu

# Link into shared library
nvcc -shared -o libmedia_gateway_cuda.so \
  semantic_similarity_tc.o \
  ontology_reasoning.o \
  graph_search.o

# Copy to lib directory
cp libmedia_gateway_cuda.so ../../../target/debug/
cp libmedia_gateway_cuda.so ../../../target/release/

cd ../../..
```

**Or use the build script**
```bash
./scripts/build_cuda.sh
```

### Build Rust Application

**Debug Build (Fast compilation, slower runtime)**
```bash
cargo build

# Build specific workspace member
cargo build -p hackathon-tv5

# Build with specific features
cargo build --features "gpu,neo4j"
```

**Release Build (Optimized)**
```bash
cargo build --release

# With full optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# With link-time optimization
cargo build --release --config profile.release.lto=true
```

**Check Compilation (No binary output)**
```bash
cargo check

# Faster than full build, useful for quick feedback
cargo check --all-targets --all-features
```

### Build Node.js CLI

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Link for local development
npm link

# Test CLI
hackathon --version
```

### Incremental Builds

**Speed up development builds**
```bash
# Enable shared incremental compilation
export CARGO_INCREMENTAL=1

# Use sccache for distributed caching
cargo install sccache
export RUSTC_WRAPPER=sccache

# Use mold linker (Linux only) for faster linking
sudo apt-get install mold
export RUSTFLAGS="-C link-arg=-fuse-ld=mold"

# Combine for maximum speed
export CARGO_INCREMENTAL=1
export RUSTC_WRAPPER=sccache
export RUSTFLAGS="-C link-arg=-fuse-ld=mold"
cargo build
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
cargo test --lib

# Run specific test
cargo test test_semantic_search

# Run with output
cargo test -- --nocapture

# Run with logging
RUST_LOG=debug cargo test -- --nocapture

# Run single-threaded (for debugging)
cargo test -- --test-threads=1
```

### Integration Tests

```bash
# Run all integration tests
cargo test --test integration

# Run specific integration test file
cargo test --test hybrid_integration_tests

# Run with Docker containers (Neo4j, Redis, etc.)
docker-compose -f docker-compose.test.yml up -d
cargo test --test integration
docker-compose -f docker-compose.test.yml down
```

### CUDA Kernel Tests

```bash
# Compile and run CUDA tests
cd src/cuda/benchmarks
nvcc -O3 -arch=sm_75 -o tensor_core_test tensor_core_test.cu
./tensor_core_test

# Or use the script
cd ../../..
./scripts/run_phase1_benchmark.sh
```

### Benchmark Tests

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench hybrid_benchmarks

# Generate HTML report
cargo bench -- --save-baseline main

# Compare against baseline
cargo bench -- --baseline main
```

### Load Testing

```bash
# Start application
cargo run --release --bin load-generator

# In another terminal, run load test
./scripts/load_test.sh

# Or use Apache Bench
ab -n 10000 -c 100 http://localhost:8080/v1/search

# Or use wrk
wrk -t12 -c400 -d30s --latency http://localhost:8080/v1/search
```

### Code Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage

# Open report
xdg-open coverage/index.html  # Linux
open coverage/index.html      # macOS
```

### Test Organization

```
tests/
├── unit/                   # Unit tests (in src/)
├── integration/            # Integration tests
│   ├── api_tests.rs
│   ├── gpu_tests.rs
│   └── db_tests.rs
├── chaos_tests.rs         # Chaos engineering tests
├── load_tests.rs          # Load/stress tests
└── mapper_tests.rs        # Data mapper tests
```

## Contributing Workflow

### Fork and Clone

```bash
# Fork repository on GitHub
# https://github.com/agenticsorg/hackathon-tv5/fork

# Clone your fork
git clone https://github.com/YOUR_USERNAME/hackathon-tv5.git
cd hackathon-tv5

# Add upstream remote
git remote add upstream https://github.com/agenticsorg/hackathon-tv5.git
```

### Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-awesome-feature

# Or bugfix branch
git checkout -b fix/issue-123
```

### Make Changes

```bash
# Write code
vim src/rust/gpu_engine/similarity.rs

# Format code
cargo fmt

# Run linter
cargo clippy --all-targets --all-features

# Run tests
cargo test

# Build
cargo build --release
```

### Commit Changes

**Commit Message Format**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting changes
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**
```bash
git commit -m "feat(gpu): implement tensor core optimization

- Add WMMA operations for 16x16x16 tiles
- Precompute norms for 1.5x speedup
- Optimize shared memory usage

Closes #123"

git commit -m "fix(api): handle null embeddings gracefully

Check for null/empty embeddings before GPU transfer

Fixes #456"
```

### Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/my-awesome-feature

# Create PR on GitHub
# https://github.com/agenticsorg/hackathon-tv5/compare
```

### PR Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: Maintainers review your code
3. **Requested Changes**: Address feedback and push updates
4. **Approval**: Once approved, PR is merged

**PR Checklist**
- [ ] Tests pass locally (`cargo test`)
- [ ] Code formatted (`cargo fmt`)
- [ ] No linter warnings (`cargo clippy`)
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] Commit messages follow convention

### Keeping Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase feature branch
git checkout feature/my-awesome-feature
git rebase upstream/main

# Force push (if needed)
git push --force-with-lease origin feature/my-awesome-feature
```

## Code Style Guide

### Rust Style

**Follow Rust API Guidelines**
- https://rust-lang.github.io/api-guidelines/

**Formatting**
```bash
# Auto-format with rustfmt
cargo fmt

# Check formatting
cargo fmt -- --check
```

**Naming Conventions**
```rust
// Modules: snake_case
mod gpu_engine;
mod semantic_search;

// Types: PascalCase
struct SemanticEngine;
enum QueryType;

// Functions/variables: snake_case
fn compute_similarity() {}
let embedding_dim = 1024;

// Constants: SCREAMING_SNAKE_CASE
const MAX_BATCH_SIZE: usize = 128;
const DEFAULT_THRESHOLD: f32 = 0.85;

// Traits: PascalCase with descriptive names
trait VectorSearch {}
trait Embeddable {}
```

**Documentation**
```rust
/// Computes semantic similarity using GPU acceleration.
///
/// # Arguments
///
/// * `query` - The search query string
/// * `limit` - Maximum number of results to return
/// * `threshold` - Minimum similarity score (0.0-1.0)
///
/// # Returns
///
/// A vector of `SearchResult` structs ordered by descending similarity.
///
/// # Errors
///
/// Returns `Err` if GPU is unavailable or query embedding fails.
///
/// # Examples
///
/// ```
/// use hackathon_tv5::SemanticEngine;
///
/// let engine = SemanticEngine::new()?;
/// let results = engine.search("French documentary", 10, 0.85)?;
/// ```
pub fn search(
    &self,
    query: &str,
    limit: usize,
    threshold: f32,
) -> Result<Vec<SearchResult>> {
    // Implementation
}
```

**Error Handling**
```rust
use anyhow::{Context, Result};
use thiserror::Error;

// Define custom errors
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("CUDA initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Out of GPU memory")]
    OutOfMemory,

    #[error("Kernel execution failed: {0}")]
    KernelError(String),
}

// Use Result with context
pub fn load_model(path: &Path) -> Result<Model> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open model file: {}", path.display()))?;

    let model = Model::from_reader(file)
        .context("Failed to parse model")?;

    Ok(model)
}
```

### TypeScript/JavaScript Style

**Use Prettier**
```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2
}
```

**Naming Conventions**
```typescript
// Interfaces: PascalCase with 'I' prefix (optional)
interface SearchOptions {
  limit?: number;
  threshold?: number;
}

// Classes: PascalCase
class MediaGatewayClient {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async search(options: SearchOptions): Promise<SearchResult[]> {
    // Implementation
  }
}

// Functions: camelCase
function computeSimilarity(a: number[], b: number[]): number {
  // Implementation
}

// Constants: SCREAMING_SNAKE_CASE
const MAX_RETRIES = 3;
const DEFAULT_TIMEOUT_MS = 30000;
```

### CUDA C++ Style

**Follow Google C++ Style Guide**
- https://google.github.io/styleguide/cppguide.html

**Naming Conventions**
```cuda
// Kernels: snake_case with __global__
__global__ void compute_similarity_kernel(
    const half* embeddings,
    float* similarities,
    int num_pairs
) {
    // Implementation
}

// Device functions: snake_case with __device__
__device__ float dot_product(
    const half* a,
    const half* b,
    int dim
) {
    // Implementation
}

// Host functions: snake_case
void launch_similarity_kernel(
    const half* embeddings,
    float* similarities,
    int num_pairs
) {
    // Implementation
}

// Constants: SCREAMING_SNAKE_CASE
#define TILE_SIZE 16
#define BLOCK_SIZE 256
```

**Documentation**
```cuda
/**
 * @brief Computes pairwise cosine similarity using tensor cores.
 *
 * @param embeddings Input embeddings in FP16 format [num_items, dim]
 * @param norms Precomputed L2 norms [num_items]
 * @param pairs Pairs of indices to compare [num_pairs, 2]
 * @param similarities Output similarities [num_pairs]
 * @param num_pairs Number of pairs to compute
 * @param embedding_dim Dimensionality of embeddings
 *
 * @note Requires CUDA compute capability 7.0+ for tensor cores
 * @note Embedding dimension must be multiple of 16
 */
__global__ void compute_multimodal_similarity_tensor_cores(
    const half* embeddings,
    const float* norms,
    const int* pairs,
    float* similarities,
    int num_pairs,
    int embedding_dim
);
```

## Debugging and Profiling

### Rust Debugging

**LLDB (Recommended)**
```bash
# Debug with lldb
rust-lldb target/debug/hackathon-tv5

# Set breakpoint
(lldb) breakpoint set --name main
(lldb) breakpoint set --file lib.rs --line 42

# Run
(lldb) run

# Backtrace
(lldb) bt

# Print variable
(lldb) print my_variable

# Continue
(lldb) continue
```

**GDB**
```bash
# Debug with gdb
rust-gdb target/debug/hackathon-tv5

# Set breakpoint
(gdb) break main
(gdb) break src/rust/lib.rs:42

# Run
(gdb) run

# Backtrace
(gdb) backtrace

# Print variable
(gdb) print my_variable
```

### CUDA Debugging

**CUDA-GDB**
```bash
# Compile with debug symbols
nvcc -g -G -arch=sm_75 -o test tensor_core_test.cu

# Debug with cuda-gdb
cuda-gdb ./test

# Set breakpoint in kernel
(cuda-gdb) break compute_similarity_kernel

# Run
(cuda-gdb) run

# Switch to GPU thread
(cuda-gdb) cuda thread (0,0,0)

# Print GPU variable
(cuda-gdb) print threadIdx.x
```

**Nsight Compute**
```bash
# Profile kernel
ncu --set full -o profile ./test

# View results
ncu-ui profile.ncu-rep
```

**Nsight Systems**
```bash
# Profile application
nsys profile -o timeline --stats=true ./target/release/hackathon-tv5

# View timeline
nsys-ui timeline.qdrep
```

### Performance Profiling

**Flamegraph**
```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --release --bin hackathon-tv5

# Open flamegraph.svg in browser
```

**Perf (Linux)**
```bash
# Record performance data
perf record --call-graph=dwarf ./target/release/hackathon-tv5

# Analyze
perf report

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > perf.svg
```

**Valgrind (Memory profiling)**
```bash
# Check for memory leaks
valgrind --leak-check=full ./target/debug/hackathon-tv5

# Cachegrind (cache profiling)
valgrind --tool=cachegrind ./target/release/hackathon-tv5
```

## Development Tools

### Recommended Tools

```bash
# Cargo extensions
cargo install cargo-watch     # Auto-rebuild on file changes
cargo install cargo-edit       # Add/remove dependencies from CLI
cargo install cargo-outdated   # Check for outdated dependencies
cargo install cargo-audit      # Security vulnerability scanner
cargo install cargo-bloat      # Find what takes space in binary
cargo install cargo-tree       # Visualize dependency tree

# Usage examples
cargo watch -x check          # Auto-check on changes
cargo add tokio              # Add dependency
cargo outdated               # Check outdated deps
cargo audit                  # Security audit
cargo bloat --release        # Analyze binary size
cargo tree                   # Show dependency tree
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**.pre-commit-config.yaml**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: local
    hooks:
      - id: cargo-fmt
        name: Cargo format
        entry: cargo fmt
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-clippy
        name: Cargo clippy
        entry: cargo clippy --all-targets --all-features -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false
```

### Documentation Generation

```bash
# Generate Rust documentation
cargo doc --open --no-deps

# Include private items
cargo doc --document-private-items --open

# Generate mdBook documentation
mdbook build docs/book
mdbook serve docs/book
```

## Next Steps

Continue to:
- **[Quick Start Guide](QUICK_START.md)** - Get up and running
- **[API Guide](API_GUIDE.md)** - API reference
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment

## Support

- **GitHub Issues**: [Report bugs](https://github.com/agenticsorg/hackathon-tv5/issues)
- **Discord**: [Join community](https://discord.agentics.org)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)
