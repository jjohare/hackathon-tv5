# T4 GPU Quick Reference Card

## Build Commands
```bash
cd /home/devuser/workspace/hackathon-tv5/src/cuda

make t4              # Build for T4 (sm_75)
make test-t4         # Run validation tests
make fp16            # FP16-optimized build
make multi-gpu       # Multi-GPU with NCCL
make check-registers # Register usage analysis
make profile         # Nsight Compute profiling
make benchmark       # Run benchmark suite
make clean           # Clean build artifacts
```

## T4 Specifications
| Component | Value |
|-----------|-------|
| Architecture | Turing (sm_75) |
| CUDA Cores | 2560 (40 SMs) |
| Tensor Cores | 320 (FP16) |
| Memory | 16GB GDDR6 |
| Bandwidth | 320 GB/s |
| PCIe | Gen3 16x |
| FP16 Peak | 65 TFLOPS |
| FP32 Peak | 8.1 TFLOPS |

## Memory Budget (768D embeddings)
| Precision | Max Vectors | Memory |
|-----------|-------------|--------|
| FP32 | 273K | 800 MB |
| FP16 | 546K | 800 MB |

## Performance (1M vectors)
| Config | Latency | Throughput |
|--------|---------|------------|
| 1× T4 | 120 ms | 8 q/s |
| 4× T4 | 35 ms | 29 q/s |
| 8× T4 | 20 ms | 50 q/s |

## Rust Integration
```rust
use gpu_engine::t4_config::T4Config;

let config = T4Config::default();
let budget = config.memory_budget(768, 0.8);
let block_size = config.optimal_block_size(WorkloadType::ComputeBound);
```

## Key Files
- **Makefile**: `src/cuda/kernels/Makefile`
- **FP16 Kernels**: `src/cuda/kernels/semantic_similarity_fp16.cu`
- **Rust Config**: `src/rust/gpu_engine/t4_config.rs`
- **Validation**: `src/cuda/examples/t4_validation.cu`
- **Benchmarks**: `benchmarks/t4_benchmarks.sh`
- **Guide**: `docs/T4_OPTIMIZATION_GUIDE.md`

## Troubleshooting
```bash
# Out of memory
let budget = config.memory_budget(768, 0.7);  # Reduce to 70%

# Low performance
make check-registers  # Check register spills

# Multi-GPU issues
cudaDeviceEnablePeerAccess(peer, 0);  # Enable P2P
```
