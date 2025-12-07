# Semantic Recommender - Rust Implementation

High-performance GPU-accelerated hyper-personalization system written in Rust.

## Architecture

```
semantic-recommender-rs/
├── crates/
│   ├── gpu-embeddings/          # User embedding management (15.36 GB GPU)
│   ├── temporal-cache/          # Similarity caching (2.48 GB GPU)
│   ├── attention/               # Multi-head attention (<1 MB GPU)
│   ├── semantic-model/          # ONNX sentence transformer
│   ├── hyper-personalization/   # Integration layer
│   ├── benchmarks/              # Performance benchmarks
│   └── cli/                     # Command-line interface
```

## Performance Targets

- **Latency**: <0.5ms (P95)
- **Throughput**: 500K+ QPS
- **GPU Memory**: ~18 GB / 42 GB (43% utilization)
- **Quality**: +40-60% personalization improvement

## Prerequisites

- Rust 1.75+
- CUDA 11.7+ (for GPU support)
- NVIDIA A100 GPU (recommended)

## Quick Start

```bash
# Build with CUDA support
cargo build --release --features cuda

# Run CLI
cargo run --bin semantic-recommender -- --help

# Run benchmarks
cargo bench --workspace

# CPU-only build
cargo build --release --features cpu-only
```

## Dependencies

### Core
- **cudarc**: Direct CUDA bindings for maximum performance
- **tokio**: Async runtime for concurrent GPU operations
- **ort**: ONNX Runtime for semantic model inference

### Math
- **cuBLAS**: GPU-accelerated matrix operations
- **cuDNN**: GPU-accelerated neural network primitives

### Concurrency
- **dashmap**: Concurrent hashmap for user embeddings
- **parking_lot**: Fast synchronization primitives

## Features

- `cuda` (default): Enable GPU acceleration
- `cpu-only`: CPU fallback mode
- `onnx` (default): ONNX Runtime inference
- `distributed`: Multi-GPU support
- `metrics`: Prometheus metrics export

## Memory Safety

All GPU memory management uses RAII patterns with automatic cleanup:

```rust
pub struct GpuTensor<T> {
    device_ptr: CudaSlice<T>,
    // Automatically freed on drop
}
```

Thread-safe shared state using Arc + RwLock:

```rust
pub struct UserEmbeddings {
    embeddings: Arc<RwLock<GpuTensor<f32>>>,
    user_map: Arc<DashMap<String, usize>>,
}
```

## Profiling

```bash
# Generate flamegraph
cargo flamegraph --bench latency

# Profile with perf
cargo bench --bench throughput -- --profile-time=10

# Memory profiling (requires CUDA tools)
nsys profile ./target/release/semantic-recommender
```

## Testing

```bash
# Run all tests
cargo test --workspace --all-features

# Run GPU tests only
cargo test --features cuda

# Run benchmarks
cargo criterion --workspace
```

## License

MIT
