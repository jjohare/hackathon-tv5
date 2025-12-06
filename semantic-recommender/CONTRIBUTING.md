# Contributing to Media Gateway Hackathon

**Thank you for your interest in contributing!** This project is part of the Media Gateway Hackathon presented by the Agentics Foundation with support from TV5 Monde USA, Google & Kaltura.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Standards](#code-standards)
4. [Testing Requirements](#testing-requirements)
5. [Documentation Standards](#documentation-standards)
6. [Code Review Process](#code-review-process)
7. [Release Process](#release-process)
8. [Community Guidelines](#community-guidelines)

---

## Getting Started

### Prerequisites

**Required**:
- **Rust** 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **CUDA Toolkit** 11.0+ (for GPU development)
- **Docker** 24.0+ and Docker Compose
- **Git** 2.30+

**Optional**:
- **Python** 3.10+ (for data processing scripts)
- **Node.js** 18+ (for frontend development)
- **Neo4j Desktop** (for ontology visualization)

### First-Time Setup

```bash
# 1. Fork and clone repository
git clone https://github.com/YOUR_USERNAME/hackathon-tv5.git
cd hackathon-tv5

# 2. Add upstream remote
git remote add upstream https://github.com/agenticsorg/hackathon-tv5.git

# 3. Install development dependencies
cargo install cargo-watch cargo-audit cargo-deny
rustup component add clippy rustfmt

# 4. Build project
cargo build

# 5. Run tests
cargo test --all

# 6. Start infrastructure
docker-compose up -d neo4j milvus

# 7. Create feature branch
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Branch Strategy

We follow **GitHub Flow**:

```
main (protected)
  ↓
  ├─ feature/semantic-search-optimization
  ├─ bugfix/cuda-memory-leak
  ├─ docs/api-documentation
  └─ refactor/gpu-engine-cleanup
```

**Branch Naming**:
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `docs/*` - Documentation only
- `refactor/*` - Code refactoring
- `test/*` - Test improvements
- `perf/*` - Performance improvements

### Making Changes

```bash
# 1. Update main branch
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/my-awesome-feature

# 3. Make changes
# ... edit files ...

# 4. Format code
cargo fmt --all

# 5. Run linter
cargo clippy --all-targets --all-features -- -D warnings

# 6. Run tests
cargo test --all

# 7. Commit changes
git add .
git commit -m "feat: add GPU-accelerated similarity search"

# 8. Push to your fork
git push origin feature/my-awesome-feature

# 9. Open Pull Request on GitHub
```

### Commit Message Format

We follow **Conventional Commits**:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or modifications
- `chore`: Build process or auxiliary tool changes

**Examples**:
```bash
feat(gpu): add tensor core optimization for similarity computation

Implement WMMA operations for 8-10x speedup on Ampere/Hopper GPUs.
Uses FP16 precision with minimal accuracy loss.

Closes #123

---

fix(api): handle timeout errors in vector search

Previously, long-running queries would panic instead of returning
a proper error response.

---

docs(integration): add troubleshooting guide for CUDA errors

Covers common issues like driver version mismatches and OOM errors.
```

---

## Code Standards

### Rust Style Guide

**Follow official Rust style** (`rustfmt.toml`):
```toml
max_width = 100
edition = "2021"
use_field_init_shorthand = true
use_try_shorthand = true
```

**Key Principles**:
- **Safety**: Use safe Rust unless `unsafe` is necessary (GPU FFI)
- **Idiomatic**: Follow Rust idioms and patterns
- **Clear**: Prefer readability over cleverness
- **Documented**: All public APIs must have doc comments

**Example**:
```rust
/// Computes semantic similarity between query and candidate vectors using GPU acceleration.
///
/// # Arguments
///
/// * `query` - Query embedding vector (dimension must match model)
/// * `candidates` - Candidate embeddings to compare against
/// * `top_k` - Number of top results to return
///
/// # Returns
///
/// Vector of `SimilarityResult` sorted by descending score
///
/// # Errors
///
/// Returns `Err` if:
/// - GPU device is unavailable
/// - Vector dimensions don't match
/// - CUDA kernel execution fails
///
/// # Examples
///
/// ```
/// use media_gateway::gpu_engine::GpuEngine;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let engine = GpuEngine::new()?;
/// let query = vec![0.5; 512];
/// let candidates = vec![vec![0.5; 512]; 1000];
///
/// let results = engine.compute_similarity(&query, &candidates, 10).await?;
/// assert_eq!(results.len(), 10);
/// # Ok(())
/// # }
/// ```
pub async fn compute_similarity(
    &self,
    query: &[f32],
    candidates: &[Vec<f32>],
    top_k: usize,
) -> Result<Vec<SimilarityResult>> {
    // Implementation...
}
```

### CUDA Style Guide

**Follow NVIDIA CUDA C++ Programming Guide**:
- Use `snake_case` for function names
- Use `UPPER_CASE` for constants
- Document all kernel launch configurations
- Add error checking after every CUDA call

**Example**:
```cuda
/**
 * @brief Computes cosine similarity using tensor cores
 *
 * @param queries Input query vectors [batch_size, dim]
 * @param candidates Candidate vectors [num_candidates, dim]
 * @param results Output similarity scores [batch_size, num_candidates]
 * @param batch_size Number of queries
 * @param num_candidates Number of candidates
 * @param dim Embedding dimension (must be multiple of 16)
 *
 * @note Requires GPU with compute capability >= 7.0
 * @note Uses FP16 tensor cores for 8-10x speedup vs FP32
 */
__global__ void compute_similarity_tensor_cores(
    const half* queries,
    const half* candidates,
    float* results,
    int batch_size,
    int num_candidates,
    int dim
) {
    // Kernel implementation...
}

// Always check for errors
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    return err;
}
```

### Error Handling

**Use `Result` and `anyhow` for error propagation**:

```rust
use anyhow::{Context, Result, bail};

pub async fn load_model(path: &Path) -> Result<Model> {
    let file = tokio::fs::read(path)
        .await
        .context("Failed to read model file")?;

    if file.is_empty() {
        bail!("Model file is empty");
    }

    Model::from_bytes(&file)
        .context("Failed to deserialize model")
}
```

**Never use `unwrap()` or `expect()` in library code**:
```rust
// ❌ BAD
let device = CudaDevice::new(0).unwrap();

// ✅ GOOD
let device = CudaDevice::new(0)
    .context("Failed to initialize CUDA device 0")?;
```

---

## Testing Requirements

### Test Coverage

**Minimum Requirements**:
- **Unit tests**: Critical functions and edge cases
- **Integration tests**: Component interactions
- **GPU tests**: Kernel correctness and performance
- **API tests**: All endpoints and error cases

### Writing Tests

**Unit Tests** (co-located with code):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_computation() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_gpu_engine_initialization() {
        let engine = GpuEngine::new().unwrap();
        assert!(engine.is_available());
    }
}
```

**Integration Tests** (`/tests` directory):
```rust
// tests/hybrid_search_integration.rs
use media_gateway::SearchEngine;

#[tokio::test]
async fn test_end_to_end_search() {
    // Setup
    let engine = SearchEngine::new_test_instance().await.unwrap();

    // Execute
    let results = engine.semantic_search("action movies", 10).await.unwrap();

    // Verify
    assert!(results.len() > 0);
    assert!(results[0].score > 0.0);
}
```

**GPU Tests**:
```bash
# Run GPU benchmarks
cd src/cuda
./scripts/run_phase1_benchmark.sh

# Expected output includes:
# - Performance metrics (TFLOPS)
# - Accuracy validation
# - Memory usage
```

### Running Tests

```bash
# All tests
cargo test --all

# Specific module
cargo test --package media-gateway --lib gpu_engine

# Integration tests only
cargo test --test '*'

# With output
cargo test -- --nocapture

# Ignored tests (slow/GPU-required)
cargo test -- --ignored

# Coverage report (requires cargo-tarpaulin)
cargo tarpaulin --out Html --output-dir coverage
```

### Continuous Integration

**All PRs must pass**:
- ✅ Compilation (with warnings as errors)
- ✅ Formatting check (`cargo fmt --check`)
- ✅ Linter (`cargo clippy`)
- ✅ Unit tests
- ✅ Integration tests
- ✅ Documentation build
- ✅ Security audit (`cargo audit`)

---

## Documentation Standards

### Code Documentation

**All public APIs require documentation**:

```rust
/// Brief one-line description.
///
/// More detailed explanation if needed. Can span multiple paragraphs.
///
/// # Arguments
///
/// * `arg1` - Description
/// * `arg2` - Description
///
/// # Returns
///
/// Description of return value
///
/// # Errors
///
/// Conditions that cause errors
///
/// # Panics
///
/// Conditions that cause panics (should be rare)
///
/// # Safety
///
/// Required for `unsafe` functions - explain invariants
///
/// # Examples
///
/// ```
/// // Working example
/// ```
pub fn my_function(arg1: Type1, arg2: Type2) -> Result<ReturnType> {
    // ...
}
```

### Markdown Documentation

**File organization**:
```
docs/
├── README.md               # Documentation index
├── INTEGRATION_GUIDE.md    # Component integration
├── TROUBLESHOOTING.md      # Common issues
├── FAQ.md                  # Frequently asked questions
├── api/
│   ├── rest-api.md
│   └── graphql-api.md
├── cuda/
│   ├── kernel-development.md
│   └── performance-tuning.md
└── deployment/
    ├── docker.md
    └── kubernetes.md
```

**Documentation checklist**:
- [ ] Clear, concise language
- [ ] Code examples that compile
- [ ] Step-by-step instructions
- [ ] Troubleshooting section
- [ ] Links to related documentation
- [ ] Updated table of contents

### Building Documentation

```bash
# Build Rust docs
cargo doc --no-deps --open

# Check for broken links
cargo deadlinks

# Build and serve all docs
cd docs
python3 -m http.server 8000
# Open http://localhost:8000
```

---

## Code Review Process

### Before Requesting Review

**Self-review checklist**:
- [ ] Code compiles without warnings
- [ ] All tests pass locally
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Public APIs are documented
- [ ] Commit messages follow conventions
- [ ] PR description explains changes
- [ ] Related issues are referenced

### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog entry added
- [ ] Breaking changes documented

## Related Issues
Closes #123
Relates to #456
```

### Review Guidelines

**For Reviewers**:
- **Be constructive**: Suggest improvements, don't just criticize
- **Be specific**: Point to lines, provide examples
- **Be timely**: Review within 2 business days
- **Focus on**: Correctness, maintainability, performance, security

**Review Checklist**:
- [ ] Code is correct and handles edge cases
- [ ] Tests adequately cover changes
- [ ] No security vulnerabilities introduced
- [ ] Performance implications considered
- [ ] Documentation is clear and accurate
- [ ] Code follows project style
- [ ] No unnecessary complexity

**Approval Requirements**:
- ✅ 1 approval from maintainer
- ✅ All CI checks passing
- ✅ No unresolved conversations
- ✅ Up-to-date with `main` branch

---

## Release Process

### Versioning

We follow **Semantic Versioning** (SemVer):
- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes, backward compatible

### Release Checklist

**For Maintainers**:
1. Update `CHANGELOG.md`
2. Update version in `Cargo.toml`
3. Create release branch: `release/v1.2.3`
4. Run full test suite including GPU benchmarks
5. Build release binaries
6. Create GitHub release with notes
7. Publish to crates.io (if applicable)
8. Update documentation site
9. Announce on Discord

### Changelog Format

```markdown
## [1.2.3] - 2025-12-04

### Added
- GPU-accelerated tensor core similarity computation (#123)
- WebSocket API for real-time recommendations (#145)

### Changed
- Improved HNSW index build performance by 30% (#134)
- Updated Neo4j driver to v5.13 (#156)

### Fixed
- Memory leak in long-running GPU workers (#167)
- Race condition in query cache (#178)

### Security
- Updated dependencies to patch CVE-2025-12345
```

---

## Community Guidelines

### Code of Conduct

**We are committed to providing a welcoming and inclusive environment**.

**Expected Behavior**:
- Be respectful and considerate
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

**Unacceptable Behavior**:
- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Spam or excessive self-promotion
- Other conduct inappropriate in a professional setting

**Enforcement**:
Violations may result in warnings, temporary bans, or permanent bans depending on severity.

### Getting Help

**Questions?**
1. Check [FAQ.md](docs/FAQ.md)
2. Search [closed issues](https://github.com/agenticsorg/hackathon-tv5/issues?q=is%3Aissue+is%3Aclosed)
3. Ask on [Discord](https://discord.agentics.org)
4. Open a [GitHub Discussion](https://github.com/agenticsorg/hackathon-tv5/discussions)

**Bugs?**
1. Check if already reported
2. Include reproduction steps
3. Provide system information (GPU model, CUDA version, etc.)
4. Include relevant logs and error messages

**Feature Requests?**
1. Explain the use case
2. Describe expected behavior
3. Consider implementation complexity
4. Discuss on Discord first for large features

### Recognition

**Contributors will be recognized**:
- In `CONTRIBUTORS.md`
- In release notes
- On the project website
- At hackathon presentation

**Top Contributors**:
- May be invited to join maintainer team
- Receive special role on Discord
- Priority for hackathon prizes

---

## Additional Resources

- **Project Documentation**: [/docs](docs/)
- **Discord Community**: https://discord.agentics.org
- **Hackathon Website**: https://agentics.org/hackathon
- **Rust Book**: https://doc.rust-lang.org/book/
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/

---

**Thank you for contributing to Media Gateway Hackathon! 🚀**

Together, we're building the future of agentic AI for media discovery.
