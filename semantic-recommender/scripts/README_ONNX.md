# ONNX Model Export for TensorRT Optimization

## Overview

This directory contains scripts for exporting the sentence transformer model (`paraphrase-multilingual-MiniLM-L12-v2`) to ONNX format optimized for TensorRT inference on NVIDIA A100 GPUs.

## Files

- **`export_model_to_onnx.py`**: Main export script with validation and benchmarking
- **`test_onnx_export.sh`**: Automated test script for ONNX export
- **`requirements-onnx.txt`**: Python dependencies for ONNX export

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment (recommended)
source venv/bin/activate

# Install ONNX dependencies
pip install -r scripts/requirements-onnx.txt
```

### 2. Run Export Script

```bash
# Basic export
python scripts/export_model_to_onnx.py

# With custom options
python scripts/export_model_to_onnx.py \
  --benchmark-iterations 100 \
  --output-dir models/custom
```

### 3. Run Test Script (Optional)

```bash
# Automated testing with validation
./scripts/test_onnx_export.sh
```

## What the Script Does

### Phase 1: Model Loading
- Downloads and loads `paraphrase-multilingual-MiniLM-L12-v2`
- Extracts underlying transformer model
- Prepares for ONNX export

### Phase 2: ONNX Export
- Exports model with dynamic axes (batch size, sequence length)
- Uses ONNX opset version 14
- Applies constant folding optimization
- Validates exported model structure

### Phase 3: ONNX Optimization
Applies 16 optimization passes:
- Dead-end elimination
- Identity operation removal
- Dropout elimination (inference mode)
- Constant extraction and folding
- Operator fusion (Conv+BN, MatMul+Bias, etc.)
- Pad and transpose optimization

### Phase 4: Accuracy Validation
- Encodes test sentences with both PyTorch and ONNX
- Computes cosine similarities
- Calculates Mean Squared Error (MSE)
- **Pass criteria**: Average cosine similarity >= 0.999

Test sentences:
```
1. "The cat sits on the mat"
2. "A feline rests on a rug"
3. "Dogs are playing in the park"
4. "Machine learning is fascinating"
5. "Deep learning models require GPUs"
```

### Phase 5: Performance Benchmarking
- Warms up both models (10 iterations)
- Benchmarks PyTorch inference (100 iterations)
- Benchmarks ONNX inference (100 iterations)
- Computes speedup factor

## Expected Results

### Model Output
- **File**: `models/sbert_optimized.onnx`
- **Size**: ~118 MB
- **Format**: ONNX opset 14
- **Dynamic axes**: batch_size, sequence_length

### Validation Metrics
```
Average Cosine Similarity: >= 0.999
Min Cosine Similarity: >= 0.999
Max Cosine Similarity: ~1.000
MSE: < 0.00001
```

### Performance Benchmarks (A100 GPU Expected)
```
PyTorch: 10-15 ms per batch (5 sentences)
ONNX:    6-10 ms per batch
Speedup: 1.5-2.0x
```

### CPU Performance (Reference)
```
PyTorch: 50-80 ms per batch
ONNX:    40-60 ms per batch
Speedup: 1.2-1.5x
```

## Output Files

### 1. `models/sbert_optimized.onnx`
Optimized ONNX model ready for:
- ONNX Runtime inference (CPU/GPU)
- TensorRT conversion (GPU)
- Edge deployment

### 2. `models/onnx_export_results.txt`
Detailed report containing:
- Model information (name, size, path)
- Validation metrics (cosine similarity, MSE)
- Benchmark results (PyTorch vs ONNX)
- Speedup analysis

Example:
```
ONNX Export Results
======================================================================

Model Information:
  Model: paraphrase-multilingual-MiniLM-L12-v2
  ONNX Path: models/sbert_optimized.onnx
  Model Size: 118.45 MB

Validation Metrics:
  avg_cosine_similarity: 0.999847
  min_cosine_similarity: 0.999621
  max_cosine_similarity: 0.999932
  mse: 0.00000234

Benchmark Results:
  pytorch_avg_ms: 12.45
  pytorch_std_ms: 0.87
  onnx_avg_ms: 8.32
  onnx_std_ms: 0.54
  speedup: 1.50
```

## Command-Line Options

```bash
python scripts/export_model_to_onnx.py --help
```

### Options:
- `--model-name`: HuggingFace model identifier (default: paraphrase-multilingual-MiniLM-L12-v2)
- `--output-dir`: Output directory (default: models)
- `--fp16`: Use FP16 optimization (default: True)
- `--benchmark-iterations`: Number of benchmark runs (default: 100)

### Examples:

**Export different model:**
```bash
python scripts/export_model_to_onnx.py \
  --model-name "sentence-transformers/all-MiniLM-L6-v2"
```

**Custom output directory:**
```bash
python scripts/export_model_to_onnx.py \
  --output-dir "models/production"
```

**Extended benchmarking:**
```bash
python scripts/export_model_to_onnx.py \
  --benchmark-iterations 500
```

## Integration with TensorRT

### Step 1: Convert ONNX to TensorRT Engine

```bash
# Install TensorRT
pip install tensorrt

# Convert with FP16 precision
trtexec --onnx=models/sbert_optimized.onnx \
        --saveEngine=models/sbert_fp16.trt \
        --fp16 \
        --workspace=4096 \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:32x128,attention_mask:32x128 \
        --maxShapes=input_ids:256x512,attention_mask:256x512 \
        --verbose
```

### Step 2: Benchmark TensorRT Engine

```bash
# Benchmark TensorRT performance
trtexec --loadEngine=models/sbert_fp16.trt \
        --shapes=input_ids:32x128,attention_mask:32x128 \
        --iterations=1000 \
        --warmUp=100
```

Expected TensorRT performance on A100:
- **Latency**: 3-5 ms per batch (32 sentences)
- **Throughput**: 6000-10000 sentences/second
- **Speedup**: 2-3x over PyTorch baseline

### Step 3: Python Integration

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load TensorRT engine
def load_engine(engine_path):
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(f.read())

# Create execution context
engine = load_engine('models/sbert_fp16.trt')
context = engine.create_execution_context()

# Allocate buffers
def allocate_buffers(engine, batch_size, seq_length):
    inputs, outputs, bindings = [], [], []

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    return inputs, outputs, bindings

# Inference
def infer(context, input_ids, attention_mask, bindings, inputs, outputs, stream):
    # Copy inputs to device
    np.copyto(inputs[0]['host'], input_ids.ravel())
    np.copyto(inputs[1]['host'], attention_mask.ravel())

    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    cuda.memcpy_htod_async(inputs[1]['device'], inputs[1]['host'], stream)

    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copy outputs to host
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    return outputs[0]['host']
```

## Troubleshooting

### Issue 1: Low Accuracy

**Symptom**: Cosine similarity < 0.999

**Possible causes:**
1. PyTorch version mismatch
2. Incorrect mean pooling implementation
3. Missing normalization

**Solutions:**
```bash
# Update PyTorch
pip install --upgrade torch

# Verify ONNX version
pip install --upgrade onnx onnxruntime-gpu

# Re-export with verbose logging
python scripts/export_model_to_onnx.py --verbose
```

### Issue 2: ONNX Export Fails

**Symptom**: Export crashes or produces invalid model

**Solutions:**
```bash
# Try lower opset version (edit script: opset_version=12)
# Disable optimizations temporarily
# Check for unsupported operations

# Test with simpler model first
python scripts/export_model_to_onnx.py \
  --model-name "sentence-transformers/all-MiniLM-L6-v2"
```

### Issue 3: Slow ONNX Inference

**Symptom**: ONNX slower than PyTorch

**Check GPU availability:**
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

**Enable CUDA provider:**
```python
session = ort.InferenceSession(
    'models/sbert_optimized.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### Issue 4: Out of Memory

**Symptom**: CUDA OOM during export or inference

**Solutions:**
```bash
# Reduce batch size
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU memory
nvidia-smi -l 1
```

## Performance Optimization Tips

### 1. Batch Size Tuning
- Test different batch sizes: 8, 16, 32, 64
- Find optimal batch size for your GPU memory
- Larger batches = better throughput (up to memory limit)

### 2. Sequence Length Optimization
- Truncate to minimum required length
- Default: 128 tokens (sufficient for most sentences)
- Shorter sequences = faster inference

### 3. TensorRT Precision
- **FP32**: Highest accuracy, slower
- **FP16**: 2x faster, minimal accuracy loss
- **INT8**: 4x faster, requires calibration

### 4. Multi-Stream Inference
```python
# Use multiple CUDA streams for overlapped execution
stream1 = cuda.Stream()
stream2 = cuda.Stream()

# Alternate between streams for continuous processing
```

## Next Steps

1. **Validate export**: Run test script to verify functionality
2. **Benchmark**: Run full benchmark with 100+ iterations
3. **TensorRT conversion**: Convert ONNX to TensorRT for production
4. **Integration**: Update Rust backend to use TensorRT
5. **Production deployment**: Deploy on A100 GPU with optimized settings

## Performance Targets (A100 GPU)

| Metric | Target | Expected |
|--------|--------|----------|
| Latency (32 sentences) | < 5 ms | 3-5 ms |
| Throughput | > 5000 sent/s | 6000-10000 sent/s |
| Speedup vs PyTorch | > 2x | 2-3x |
| Accuracy (cosine sim) | >= 0.999 | 0.9998+ |

## References

- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Sentence Transformers](https://www.sbert.net/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)

## Support

For issues or questions:
1. Check this README and ONNX_EXPORT_GUIDE.md
2. Review logs in `models/onnx_export_results.txt`
3. Check ONNX Runtime providers: `ort.get_available_providers()`
4. Verify CUDA installation: `nvidia-smi`
