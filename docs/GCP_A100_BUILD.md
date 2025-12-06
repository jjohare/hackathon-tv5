# GCP A100 Build Guide

## VM Configuration

**Instance:** semantics-testbed-a100
**Type:** a2-highgpu-1g (NVIDIA A100-SXM4-40GB)
**Zone:** us-central1-a
**Pricing:** SPOT/preemptible (~70% savings)
**Project:** agentics-foundation25lon-1812

## Quick Setup

### 1. Create A100 Instance
```bash
gcloud compute instances create semantics-testbed-a100 \
  --project=agentics-foundation25lon-1812 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=common-cu128-debian-11 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced
```

### 2. SSH and Clone Repository
```bash
gcloud compute ssh semantics-testbed-a100 \
  --project=agentics-foundation25lon-1812 \
  --zone=us-central1-a

cd ~
git clone https://github.com/your-org/hackathon-tv5.git
cd hackathon-tv5
```

### 3. Build A100 Kernels
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

cd semantic-recommender/src/cuda/kernels
make -f Makefile.a100 clean
make -f Makefile.a100 all -j8
```

## Build Outputs

**PTX Files** (Rust FFI): `~/hackathon-tv5/semantic-recommender/target/ptx_a100/`
- semantic_similarity.ptx (94KB)
- semantic_similarity_tf32.ptx (93KB) - TF32 tensor cores
- semantic_similarity_fp16.ptx (53KB)
- semantic_similarity_fp16_tensor_cores.ptx (39KB)
- graph_search.ptx (42KB)
- ontology_reasoning.ptx (39KB)
- hybrid_sssp.ptx (22KB)
- product_quantization.ptx (41KB)

**Binaries**: `build_a100/`
- benchmark_a100 (2.2MB executable)
- *.o object files (6.6MB total)

## Benchmarks

```bash
cd ~/hackathon-tv5/semantic-recommender/src/cuda/kernels
./build_a100/benchmark_a100
```

**Expected Results:**
- Exact k-NN: 8,702 QPS @ 114.9ms
- GPU Utilization: 100%
- Memory: 474MB / 40GB

## Profiling

```bash
# TF32 utilization
make -f Makefile.a100 check-tf32

# Memory bandwidth (target: >1 TB/s)
make -f Makefile.a100 check-bandwidth

# Full profile
make -f Makefile.a100 profile
```

## A100 Optimizations

- **Architecture:** sm_80 (Ampere)
- **Tensor Cores:** TF32/FP16/BF16 (432 cores)
- **Memory:** 40GB HBM2e @ 1.6 TB/s (5× vs T4)
- **CUDA Cores:** 6,912 (2.7× vs T4)
- **L2 Cache:** 40MB (8× vs T4)

## Cleanup

```bash
# Delete VM when done
gcloud compute instances delete semantics-testbed-a100 \
  --project=agentics-foundation25lon-1812 \
  --zone=us-central1-a
```

## Cost Management

- **SPOT pricing:** ~$1.50/hour (vs ~$5/hour on-demand)
- **Automatic shutdown:** SPOT instances stop when preempted
- **Manual stop:** `gcloud compute instances stop semantics-testbed-a100 --zone=us-central1-a`

## Notes

- CUDA 12.8 + Driver 570.195.03 pre-installed
- Deep Learning VM image includes all dependencies
- PTX files are architecture-agnostic (work on any GPU ≥sm_80)
- Build artifacts in `build_a100/` directory are binary (not portable)
