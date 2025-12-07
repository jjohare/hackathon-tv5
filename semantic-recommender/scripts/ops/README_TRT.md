# TensorRT Engine Build for A100 GPU

This directory contains scripts for building optimized TensorRT engines from ONNX models for deployment on NVIDIA A100 GPUs.

## Overview

The `build_trt_engine.py` script converts the ONNX model (`minilm_l12_v2.onnx`) into an optimized TensorRT engine with:

- **FP16 Precision**: Leverages A100's Tensor Cores for 2x speedup
- **Dynamic Shapes**: Supports variable batch sizes and sequence lengths
- **Optimized Profiles**: Configured for typical query patterns (batch=1, seq=32)
- **Dual Build Methods**: Python API with trtexec fallback

## Prerequisites

### TensorRT Installation

**Option 1: NVIDIA Container (Recommended for A100)**
```bash
# Use NVIDIA PyTorch container with TensorRT included
docker pull nvcr.io/nvidia/pytorch:24.01-py3
```

**Option 2: Manual Installation**
```bash
# Download TensorRT from NVIDIA Developer
# https://developer.nvidia.com/tensorrt

# Install Python wheel
pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl

# Or install via pip (limited features)
pip install tensorrt
```

**Option 3: Debian/Ubuntu Package**
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install TensorRT
sudo apt-get install tensorrt
```

## Usage

### 1. Build TensorRT Engine

```bash
# Ensure ONNX model exists first
python scripts/ops/convert_to_onnx.py

# Build TensorRT engine
python scripts/ops/build_trt_engine.py
```

**Expected Output:**
```
🚀 TensorRT Engine Builder for A100 GPU
========================================

ONNX Model: data/models/minilm_l12_v2.onnx
Output Engine: data/models/minilm_l12_v2_fp16.plan

Optimization Profiles:
  Min shape (batch, seq): (1, 1)
  Opt shape (batch, seq): (1, 32)
  Max shape (batch, seq): (16, 128)

TensorRT Config:
  FP16: True
  Workspace: 2048 MB

✅ ONNX model found (45.23 MB)
✅ TensorRT Python API available: v8.6.1
✅ FP16 precision enabled (A100 optimized)
🔨 Building TensorRT engine...

✅ Engine saved (28.15 MB)
✅ ENGINE BUILD COMPLETE
```

### 2. Test Build Process

```bash
# Run validation tests
python scripts/ops/test_trt_builder.py
```

**Tests Include:**
- Dependency checking (TensorRT availability)
- Profile configuration validation
- Metadata generation
- Engine validation (if built)
- Command generation for trtexec

### 3. Verify Engine

```bash
# Check engine file
ls -lh data/models/minilm_l12_v2_fp16.plan

# View build metadata
cat data/models/minilm_l12_v2_fp16.json
```

## Configuration

### Optimization Profiles

Defined in `build_trt_engine.py`:

```python
PROFILE_CONFIG = {
    'min': (1, 1),      # Single token (edge case)
    'opt': (1, 32),     # Typical query: "sci-fi action movie"
    'max': (16, 128)    # Batch inference with long queries
}
```

**Shape Format:** `(batch_size, sequence_length)`

- **Min Shape**: Smallest input supported (1 batch, 1 token)
- **Opt Shape**: Optimized for this size (1 batch, 32 tokens)
- **Max Shape**: Maximum input size (16 batches, 128 tokens)

### TensorRT Settings

```python
TRT_CONFIG = {
    'fp16': True,              # Enable FP16 for A100
    'workspace_size': 2048,    # 2GB workspace memory
    'verbose': True,           # Detailed logging
    'max_batch_size': 16       # Maximum batch size
}
```

## Build Methods

### Method 1: TensorRT Python API (Preferred)

Uses the full TensorRT Python API for fine-grained control:

**Advantages:**
- Programmatic control over all settings
- Better error reporting
- Can inspect network layers
- More optimization options

**Requirements:**
- TensorRT Python package installed
- Python bindings available

### Method 2: trtexec Fallback

Uses NVIDIA's command-line tool:

**Advantages:**
- Simpler installation (comes with TensorRT)
- Widely tested and stable
- Good for CI/CD pipelines

**Requirements:**
- `trtexec` binary in PATH
- Typically at `/usr/src/tensorrt/bin/trtexec`

The script automatically tries Python API first, then falls back to trtexec if unavailable.

## Performance Expectations

### A100 GPU with FP16

Based on similar transformer models (384-dim, 12 layers):

| Metric | Value |
|--------|-------|
| **Latency (batch=1)** | ~1-2ms |
| **Throughput (batch=16)** | ~500-1000 queries/sec |
| **Speedup vs ONNX** | 2-3x |
| **Speedup vs PyTorch** | 5-10x |

### Memory Usage

| Component | Size |
|-----------|------|
| Engine file | ~25-30 MB |
| Device memory | ~100-150 MB |
| Workspace | ~2 GB (configurable) |

## Troubleshooting

### Error: "Module 'tensorrt' not found"

**Solution 1:** Install TensorRT Python package
```bash
pip install tensorrt
```

**Solution 2:** Use NVIDIA container
```bash
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3
```

### Error: "trtexec: command not found"

**Solution:** Add TensorRT bin to PATH
```bash
export PATH=$PATH:/usr/src/tensorrt/bin
```

Or install TensorRT package:
```bash
sudo apt-get install tensorrt
```

### Error: "Failed to parse ONNX model"

**Causes:**
- Corrupted ONNX file
- Incompatible ONNX opset version
- Unsupported operations

**Solution:**
```bash
# Rebuild ONNX with compatible opset
python scripts/ops/convert_to_onnx.py

# Validate ONNX
python -c "import onnx; onnx.checker.check_model('data/models/minilm_l12_v2.onnx')"
```

### Warning: "FP16 not supported on this platform"

**Cause:** GPU doesn't support FP16 (compute capability < 7.0)

**Solution:** Use FP32 instead
```python
# In build_trt_engine.py
TRT_CONFIG = {
    'fp16': False,  # Disable FP16
    ...
}
```

### Error: "Out of memory during build"

**Solution:** Reduce workspace size
```python
TRT_CONFIG = {
    'workspace_size': 1024,  # Reduce to 1GB
    ...
}
```

## Output Files

### 1. TensorRT Engine
**Path:** `data/models/minilm_l12_v2_fp16.plan`

Binary engine file containing optimized CUDA kernels.

### 2. Build Metadata
**Path:** `data/models/minilm_l12_v2_fp16.json`

```json
{
  "onnx_model": "minilm_l12_v2.onnx",
  "engine_path": "minilm_l12_v2_fp16.plan",
  "build_method": "python_api",
  "profile_config": {
    "min": [1, 1],
    "opt": [1, 32],
    "max": [16, 128]
  },
  "trt_config": {
    "fp16": true,
    "workspace_size": 2048
  },
  "validation": {
    "valid": true,
    "num_io_tensors": 2,
    "device_memory_mb": 128.5,
    "file_size_mb": 28.15
  }
}
```

## Next Steps

After building the TensorRT engine:

1. **Test Inference:** Run inference with the engine
   ```bash
   python scripts/benchmarks/benchmark_trt_inference.py
   ```

2. **Deploy to Production:** Copy engine to deployment environment
   ```bash
   scp data/models/minilm_l12_v2_fp16.plan user@a100-server:/models/
   ```

3. **Monitor Performance:** Track latency and throughput metrics

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [NVIDIA A100 Optimization Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [ONNX to TensorRT Conversion](https://github.com/onnx/onnx-tensorrt)

## Support

For issues specific to this implementation:
- Check logs in console output
- Review metadata file for build details
- Run test suite: `python scripts/ops/test_trt_builder.py`

For TensorRT issues:
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/deep-learning/tensorrt/)
- [TensorRT GitHub](https://github.com/NVIDIA/TensorRT)
