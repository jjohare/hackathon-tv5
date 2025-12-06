# Quick Start Guide

Get up and running with the Media Gateway Hackathon system in 5 minutes.

## Prerequisites

### Required
- **CUDA 12.2+** with compute capability 7.0+ (V100/T4) or 9.0+ (H100)
- **Rust 1.75+** with cargo
- **Node.js 18+** with npm
- **Git** for version control

### Optional
- **Docker** 24.0+ for containerized deployment
- **NVIDIA Container Toolkit** for GPU containers
- **Neo4j 5.x** for knowledge graph (embedded mode available)
- **Python 3.9+** for tooling and scripts

### System Requirements
- **GPU**: NVIDIA T4, RTX 2080+, A100, or H100
- **VRAM**: 4GB minimum, 16GB recommended for production
- **RAM**: 16GB minimum, 64GB recommended
- **Storage**: 20GB for development, 500GB+ for production datasets

## Installation (5 minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/agenticsorg/hackathon-tv5.git
cd hackathon-tv5
```

### Step 2: Install System Dependencies
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Node.js dependencies
npm install
```

### Step 3: Build CUDA Kernels
```bash
# Compile tensor core optimized kernels
cd src/cuda/kernels
nvcc -arch=sm_75 -O3 -use_fast_math \
  -o semantic_similarity_tc.o \
  semantic_similarity_fp16_tensor_cores.cu
cd ../../..
```

### Step 4: Build Rust Application
```bash
# Development build
cargo build

# Or production optimized build
cargo build --release
```

## First API Call (2 minutes)

### Option 1: Using Rust Examples
```bash
# Run basic semantic search example
cargo run --example basic_search

# Run GPU-accelerated reasoning
cargo run --example gpu_reasoning

# Run multi-modal query
cargo run --example multimodal_query
```

### Option 2: Using the CLI Tool
```bash
# Install the hackathon CLI globally
npm install -g agentics-hackathon

# Initialize your project
npx agentics-hackathon init

# Check status
npx agentics-hackathon status
```

### Option 3: Programmatic API
```rust
use hackathon_tv5::{
    gpu_engine::GpuSemanticEngine,
    semantic_search::RecommendationEngine,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU engine
    let engine = GpuSemanticEngine::new(Default::default()).await?;

    // Perform semantic search
    let results = engine.search(
        "French documentary about climate change",
        10,  // limit
        0.85 // similarity threshold
    ).await?;

    // Print results
    for (idx, result) in results.iter().enumerate() {
        println!("{}. {} (score: {:.3})",
            idx + 1,
            result.title,
            result.score
        );
    }

    Ok(())
}
```

## Verify Installation

### Check GPU Availability
```bash
# Run diagnostic script
./scripts/check_gpu.sh

# Expected output:
# ✓ CUDA detected: 12.2
# ✓ GPU: NVIDIA Tesla T4
# ✓ Compute capability: 7.5
# ✓ Tensor cores: Available
```

### Run Benchmark Suite
```bash
# Quick performance test (2 minutes)
cargo run --release --bin load-generator -- \
  --duration 60 \
  --concurrency 10

# Expected results:
# Throughput: 50-100 QPS
# Latency p99: <100ms
# GPU Utilization: 80-95%
```

### Run Integration Tests
```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test integration

# All tests with output
cargo test -- --nocapture
```

## Common Issues and Solutions

### Issue: CUDA Not Found
```bash
# Error: nvcc: command not found

# Solution: Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Issue: Compilation Errors
```bash
# Error: linking with `cc` failed

# Solution: Install build tools
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential pkg-config libssl-dev

# RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install openssl-devel
```

### Issue: GPU Not Detected
```bash
# Error: CUDA_ERROR_NO_DEVICE

# Solution: Check driver installation
nvidia-smi

# If no output, reinstall NVIDIA drivers
# Ubuntu/Debian
sudo apt-get install nvidia-driver-535

# Or download from: https://www.nvidia.com/download/index.aspx
```

### Issue: Out of Memory
```bash
# Error: CUDA out of memory

# Solution: Reduce batch size in config
export BATCH_SIZE=32  # Default is 128
cargo run --release

# Or use smaller embedding dimensions
export EMBEDDING_DIM=512  # Default is 1024
```

### Issue: Slow Performance
```bash
# Symptom: <10x speedup vs CPU

# Check GPU utilization
nvidia-smi -l 1

# Should show 90-100% GPU utilization
# If low (<50%), check:

# 1. Batch size too small
export BATCH_SIZE=128

# 2. Using debug build
cargo build --release  # Use release build!

# 3. CPU bottleneck
# Profile with:
cargo flamegraph --example basic_search
```

### Issue: Neo4j Connection Failed
```bash
# Error: Failed to connect to Neo4j

# Solution: Start Neo4j or use embedded mode
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.13

# Or disable Neo4j in config
export ENABLE_NEO4J=false
```

## Next Steps

### 1. Explore Examples
```bash
# Browse example code
ls src/examples/

# Read documentation
cat docs/API_GUIDE.md
```

### 2. Run Performance Tests
```bash
# Phase 1 tensor core validation
./scripts/run_phase1_benchmark.sh

# Full benchmark suite
cargo bench --features benchmarks
```

### 3. Deploy Locally
```bash
# Start all services with Docker Compose
docker-compose -f deployment/docker-compose.yml up

# Access API at http://localhost:8080
curl http://localhost:8080/health
```

### 4. Read Full Documentation
- **[API Guide](API_GUIDE.md)** - Complete API reference
- **[Development Guide](DEVELOPMENT.md)** - Development environment setup
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment
- **[Architecture](../design/architecture/system-architecture.md)** - System design

## Quick Reference

### Common Commands
```bash
# Development
cargo build                    # Debug build
cargo build --release          # Production build
cargo test                     # Run tests
cargo doc --open              # Generate docs

# CUDA
nvcc --version                # Check CUDA version
nvidia-smi                    # Monitor GPU
nsys profile ./target/release/app  # Profile GPU

# Docker
docker build -t media-gateway . # Build image
docker run --gpus all -p 8080:8080 media-gateway  # Run container

# Kubernetes
kubectl apply -f k8s/         # Deploy to k8s
kubectl get pods              # Check status
```

### Environment Variables
```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0   # Select GPU
export BATCH_SIZE=128           # Batch size
export EMBEDDING_DIM=1024       # Embedding dimensions

# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8080
export LOG_LEVEL=info

# Database Configuration
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
export REDIS_URL=redis://localhost:6379
```

### Performance Expectations

| Configuration | Throughput | Latency (p99) | GPU Util |
|--------------|-----------|---------------|----------|
| Single T4 | 50-100 QPS | 50-100ms | 80-95% |
| Single A100 | 200-400 QPS | 20-40ms | 85-95% |
| 4x A100 | 800-1600 QPS | 15-30ms | 90-98% |

### Resource Usage

| Component | CPU | RAM | VRAM | Storage |
|-----------|-----|-----|------|---------|
| Development | 2 cores | 4GB | 2GB | 10GB |
| Production (1M items) | 8 cores | 16GB | 8GB | 100GB |
| Production (100M items) | 32 cores | 128GB | 40GB | 2TB |

## Support

### Documentation
- **Quick Start**: This guide
- **API Reference**: [API_GUIDE.md](API_GUIDE.md)
- **Development**: [DEVELOPMENT.md](DEVELOPMENT.md)
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)

### Community
- **Discord**: [discord.agentics.org](https://discord.agentics.org)
- **GitHub Issues**: [Report a bug](https://github.com/agenticsorg/hackathon-tv5/issues)
- **Website**: [agentics.org/hackathon](https://agentics.org/hackathon)

### Getting Help
1. Check [Common Issues](#common-issues-and-solutions) above
2. Search [GitHub Issues](https://github.com/agenticsorg/hackathon-tv5/issues)
3. Ask on [Discord](https://discord.agentics.org)
4. Create a [new issue](https://github.com/agenticsorg/hackathon-tv5/issues/new)

## What's Next?

Once you have the system running:

1. **Customize**: Modify examples for your use case
2. **Scale**: Deploy to production with Kubernetes
3. **Optimize**: Run benchmarks and tune performance
4. **Extend**: Add new features and integrations

Continue to the [API Guide](API_GUIDE.md) for complete API reference and examples.
