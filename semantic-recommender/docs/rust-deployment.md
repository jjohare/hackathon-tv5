# Rust Deployment Guide for A100 GPU

Complete deployment guide for the Rust semantic recommender on GCP A100 VMs.

## Prerequisites

### Hardware Requirements
- NVIDIA A100 GPU (40GB or 80GB)
- 16+ CPU cores recommended
- 64GB+ RAM
- 500GB+ SSD storage

### Software Requirements
- Ubuntu 22.04 LTS
- CUDA 11.7 or newer
- NVIDIA Driver 515+
- Rust 1.70+

## Build Configuration

### Cross-Compilation for A100

```bash
# Set CUDA compute capability for A100
export CUDA_COMPUTE_CAP=80

# Build with A100 profile
cargo build --profile a100 --features gpu --target x86_64-unknown-linux-gnu

# Binary location
ls -lh target/x86_64-unknown-linux-gnu/a100/semantic-rec
```

### Feature Flags

```toml
# Cargo.toml features
[features]
default = []
gpu = ["cudarc"]              # CUDA GPU support
cpu-only = []                 # CPU fallback
simd = []                     # CPU SIMD optimizations
full = ["gpu", "simd"]        # All optimizations
```

Build commands:

```bash
# GPU only (production)
cargo build --release --features gpu

# CPU fallback
cargo build --release --features cpu-only

# Full optimizations
cargo build --release --features full
```

## Optimization Profiles

### Profile Comparison

| Profile | Use Case | LTO | Opt Level | Build Time |
|---------|----------|-----|-----------|------------|
| dev | Development | No | 1 | ~2 min |
| release | Production | Fat | 3 | ~15 min |
| a100 | A100 deployment | Fat | 3 | ~15 min |
| bench | Benchmarking | Fat | 3 | ~15 min |

### Build Commands

```bash
# Development (fast compile)
cargo build

# Production release
cargo build --release --features gpu

# A100-optimized
cargo build --profile a100 --features gpu

# Benchmarking (with debug symbols)
cargo build --profile bench --features gpu
```

## Docker Deployment

### Multi-stage Dockerfile

```dockerfile
# Build stage
FROM rust:1.75 as builder

# Install CUDA toolkit
RUN apt-get update && apt-get install -y \\
    cuda-toolkit-11-7 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY crates/ crates/

# Build release binary
RUN cargo build --release --features gpu

# Runtime stage
FROM nvidia/cuda:11.7-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libssl3 \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/target/release/semantic-rec /usr/local/bin/

# Copy data (if bundled)
COPY data/ /data/

# Health check
HEALTH CHECK --interval=30s --timeout=3s \\
    CMD semantic-rec info || exit 1

# Run
CMD ["semantic-rec", "load", "--dataset", "/data/movies.csv"]
```

### Build and Run

```bash
# Build image
docker build -t semantic-rec:latest .

# Run with GPU
docker run --gpus all \\
    -v $(pwd)/data:/data \\
    -p 8080:8080 \\
    semantic-rec:latest
```

## Performance Tuning

### GPU Configuration

```rust
// Configure for A100
let config = GpuConfig {
    device_id: 0,
    batch_size: 512,          // Larger batches for A100
    embedding_dim: 768,
    memory_pool_size: 20_000_000_000,  // 20GB pool
    enable_tensor_cores: true,
    compute_capability: (8, 0),        // A100 = 8.0
};
```

### Memory Pool Sizing

```bash
# Monitor GPU memory
nvidia-smi -l 1

# Tune pool size based on dataset
# Formula: num_embeddings * embedding_dim * 4 bytes * 1.2 (overhead)
# For 62K movies with 768 dims:
# 62,423 * 768 * 4 * 1.2 = ~230MB
```

### Batch Size Tuning

```bash
# Test different batch sizes
for size in 128 256 512 1024; do
    semantic-rec bench --batch-size $size --iterations 1000
done
```

## Benchmarking

### Run All Benchmarks

```bash
# Cargo benchmarks
cargo bench --features gpu

# CLI benchmarks
semantic-rec bench --iterations 10000 --output benchmarks.json

# Compare with Python
semantic-rec compare --queries 1000 --threshold 0.001
```

### Expected Performance (A100)

| Metric | Target | Actual |
|--------|--------|--------|
| Query latency (p50) | < 3ms | TBD |
| Query latency (p99) | < 10ms | TBD |
| Throughput | > 500 QPS | TBD |
| Index time (62K) | < 2s | TBD |
| Memory usage | < 4GB | TBD |

## Monitoring

### Built-in Metrics

```bash
# System info
semantic-rec info

# Real-time monitoring
watch -n 1 'semantic-rec info | grep -A 5 GPU'
```

### Prometheus Metrics

```rust
// TODO: Add Prometheus exporter
// Expose metrics at /metrics endpoint
```

### GPU Metrics

```bash
# NVIDIA SMI monitoring
nvidia-smi dmon -s pucvmet

# NVML integration (coming soon)
semantic-rec monitor --interval 1s
```

## Production Checklist

### Pre-deployment

- [ ] Build with `--release` or `--profile a100`
- [ ] Run full integration test with 62K dataset
- [ ] Benchmark against Python baseline
- [ ] Verify GPU memory usage < 80%
- [ ] Test error handling and recovery
- [ ] Validate result accuracy (< 0.1% difference)

### Security

- [ ] No hardcoded credentials
- [ ] TLS for external connections
- [ ] Rate limiting configured
- [ ] Input validation enabled
- [ ] Audit logging enabled

### Reliability

- [ ] Health check endpoint responding
- [ ] Graceful shutdown implemented
- [ ] Auto-restart on crash
- [ ] Memory leak testing passed
- [ ] Load testing completed

## Troubleshooting

### CUDA Errors

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Test CUDA runtime
semantic-rec info
```

### Memory Issues

```bash
# Check available GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# Reduce batch size
semantic-rec --device cuda query "test" --batch-size 128

# Use CPU fallback
semantic-rec --device cpu query "test"
```

### Performance Issues

```bash
# Profile with perf
perf record -g semantic-rec bench --iterations 1000
perf report

# Flame graph
cargo flamegraph -- semantic-rec bench
```

### Build Errors

```bash
# Clean build
cargo clean
cargo build --release

# Update dependencies
cargo update

# Check workspace
cargo check --workspace --all-features
```

## Scaling

### Multi-GPU Support (Coming Soon)

```rust
// Round-robin across GPUs
let engines: Vec<_> = (0..num_gpus)
    .map(|id| GpuSemanticEngine::new_with_device(id))
    .collect();

// Load balance queries
let engine = &engines[query_id % num_gpus];
```

### Horizontal Scaling

```bash
# Run multiple instances
for i in {0..3}; do
    semantic-rec --port $((8080 + i)) &
done

# Nginx load balancer
upstream semantic_rec {
    server localhost:8080;
    server localhost:8081;
    server localhost:8082;
    server localhost:8083;
}
```

## Backup and Recovery

### Data Backup

```bash
# Backup embeddings
cp data/embeddings.npy backups/embeddings-$(date +%Y%m%d).npy

# Backup configuration
cp config.toml backups/config-$(date +%Y%m%d).toml
```

### Disaster Recovery

```bash
# Restore from backup
cp backups/embeddings-20250107.npy data/embeddings.npy

# Rebuild index
semantic-rec load --dataset data/movies.csv --embeddings data/embeddings.npy
```

## CI/CD Pipeline

### GitHub Actions

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run tests
        run: cargo test --workspace --all-features
      - name: Build release
        run: cargo build --release --features gpu

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Build Docker image
        run: docker build -t semantic-rec:${{ github.sha }} .
      - name: Push to registry
        run: docker push semantic-rec:${{ github.sha }}
```

## Rollback Procedure

```bash
# List available versions
docker images semantic-rec

# Rollback to previous version
docker stop semantic-rec-current
docker run --gpus all --name semantic-rec-current \\
    semantic-rec:previous-version

# Verify
semantic-rec info
```

## Support

- GCP A100 VM setup: `docs/gcp-a100-setup.md`
- CUDA troubleshooting: `docs/cuda-troubleshooting.md`
- Performance tuning: `docs/performance-tuning.md`

---

**Deployment Target**: GCP A100 VM
**Last Updated**: 2025-12-07
**Status**: Production Ready (pending integration tests)
