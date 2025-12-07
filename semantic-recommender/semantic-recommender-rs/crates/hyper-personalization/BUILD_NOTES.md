# Build Notes for Hyper-Personalization

## Build Requirements

### Essential Dependencies

1. **CUDA Toolkit** (for GPU acceleration)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install nvidia-cuda-toolkit
   
   # Verify
   nvcc --version
   ```

2. **ONNX Runtime** (for semantic model)
   - Automatically downloaded by `ort` crate
   - Requires network connection during first build

3. **libtorch** (for attention reranker)
   ```bash
   # Download libtorch
   wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
   unzip libtorch-*.zip
   
   # Set environment
   export LIBTORCH=$(pwd)/libtorch
   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
   ```

### Build Commands

#### Minimal Build (CPU only, no attention)

```bash
cargo build --package hyper-personalization \
    --no-default-features \
    --features onnx,cpu-only
```

#### Development Build (CUDA, no attention)

```bash
cargo build --package hyper-personalization \
    --features cuda,onnx
```

#### Full Production Build

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

cargo build --package hyper-personalization \
    --features full \
    --release
```

## Testing

### Unit Tests (no external dependencies)

```bash
cargo test --package hyper-personalization
```

### Integration Tests (requires test data)

```bash
# Generate test data first
cargo run --bin generate-test-data

# Run integration tests
cargo test --package hyper-personalization \
    --test integration_test \
    -- --ignored
```

## Common Build Issues

### Issue: "Cannot find libtorch"

**Solution:**
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

Or use Python PyTorch:
```bash
pip install torch
export LIBTORCH_USE_PYTORCH=1
```

### Issue: "CUDA not found"

**Solution:**
```bash
# Use CPU-only build
cargo build --package hyper-personalization \
    --no-default-features \
    --features onnx,cpu-only
```

### Issue: "ONNX Runtime download fails"

**Solution:**
- Check network connection
- Try manual download from GitHub releases
- Set `ORT_STRATEGY=system` to use system-installed ONNX Runtime

## Development Environment

### Recommended Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install CUDA (if not already installed)
# See: https://developer.nvidia.com/cuda-downloads

# Install libtorch
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-*.zip -d ~/

# Set environment variables permanently
echo 'export LIBTORCH=~/libtorch' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Build
cd semantic-recommender-rs
cargo build --workspace --features full --release
```

## Feature Flags

| Flag | Description | Dependencies |
|------|-------------|--------------|
| `onnx` | Semantic model (ONNX) | ort crate |
| `cuda` | GPU acceleration | CUDA toolkit |
| `cpu-only` | CPU fallback | None |
| `full` | All features | CUDA + libtorch |
| `metrics-export` | Prometheus metrics | prometheus crate |

## Cross-Compilation

### For different CUDA versions

Edit `Cargo.toml`:
```toml
[features]
cuda = ["cudarc/cuda-12010"]  # For CUDA 12.1
```

### For different architectures

```bash
# For ARM64
cargo build --target aarch64-unknown-linux-gnu

# For x86_64
cargo build --target x86_64-unknown-linux-gnu
```

## Deployment

### Docker Build

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install libtorch
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip && \
    unzip libtorch-*.zip && \
    rm libtorch-*.zip

ENV LIBTORCH=/libtorch
ENV LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Build application
COPY . /app
WORKDIR /app
RUN cargo build --release --features full

CMD ["./target/release/your-binary"]
```

## Benchmarking

```bash
# Run benchmarks
cargo bench --package hyper-personalization

# With specific features
cargo bench --package hyper-personalization --features full

# Output to file
cargo bench --package hyper-personalization > bench_results.txt
```

## Profiling

```bash
# CPU profiling with perf
cargo build --release --features full
perf record --call-graph dwarf ./target/release/your-binary
perf report

# GPU profiling with nsight
nsys profile ./target/release/your-binary
```
