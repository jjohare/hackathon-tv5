# ONNX Export Guide

## Overview

This guide explains how to export the sentence transformer model to ONNX format for TensorRT optimization.

## Prerequisites

```bash
# Install ONNX export dependencies
pip install -r scripts/requirements-onnx.txt
```

## Quick Start

### Basic Export

```bash
python scripts/export_model_to_onnx.py
```

This will:
1. Load `paraphrase-multilingual-MiniLM-L12-v2`
2. Export to `models/sbert_optimized.onnx`
3. Apply ONNX optimizations
4. Validate accuracy (target: cosine similarity >= 0.999)
5. Benchmark performance (PyTorch vs ONNX)

### Custom Model Export

```bash
python scripts/export_model_to_onnx.py \
  --model-name "sentence-transformers/all-MiniLM-L6-v2" \
  --output-dir "models/custom" \
  --benchmark-iterations 200
```

## Script Features

### 1. Model Export

The script exports the transformer model with:
- **Dynamic axes**: Supports variable batch size and sequence length
- **Opset version 14**: Latest stable ONNX opset
- **Constant folding**: Reduces graph size
- **Input validation**: Ensures model correctness

### 2. ONNX Optimizations

Applied optimizations include:
- Dead-end elimination
- Identity operation removal
- Dropout elimination (inference mode)
- Constant extraction
- Operator fusion (Conv+BN, MatMul+Bias, etc.)
- Transpose optimization

### 3. Accuracy Validation

The script validates accuracy by:
- Encoding test sentences with both PyTorch and ONNX
- Computing cosine similarities
- Calculating MSE
- **Pass criteria**: Average cosine similarity >= 0.999

**Test sentences:**
```python
[
    "The cat sits on the mat",
    "A feline rests on a rug",
    "Dogs are playing in the park",
    "Machine learning is fascinating",
    "Deep learning models require GPUs"
]
```

### 4. Performance Benchmarking

Benchmarks measure:
- PyTorch inference time (avg ± std)
- ONNX inference time (avg ± std)
- Speedup factor
- Default: 100 iterations with 10 warm-up runs

## Expected Output

```
============================================================
Starting ONNX Export Pipeline
============================================================

Loading model: paraphrase-multilingual-MiniLM-L12-v2
Model loaded successfully

Starting ONNX export...
Input shape: torch.Size([1, 7])
Model exported to: models/sbert_optimized.onnx
ONNX model verification passed

Optimizing ONNX model...
Applied 16 optimization passes
Optimized model saved to: models/sbert_optimized.onnx
Model size: 118.45 MB

Validating ONNX model accuracy...
Validation Metrics:
  Average Cosine Similarity: 0.999847
  Min Cosine Similarity: 0.999621
  Max Cosine Similarity: 0.999932
  MSE: 0.00000234
✓ Accuracy validation PASSED (similarity >= 0.999)

Benchmarking performance (100 iterations)...
Running warm-up iterations...
Benchmarking PyTorch...
Benchmarking ONNX...

Performance Benchmark Results:
  PyTorch: 12.45 ± 0.87 ms
  ONNX:    8.32 ± 0.54 ms
  Speedup: 1.50x

============================================================
ONNX Export Pipeline Completed
============================================================
ONNX model saved to: models/sbert_optimized.onnx
Results saved to: models/onnx_export_results.txt
```

## Output Files

### 1. `models/sbert_optimized.onnx`
- Optimized ONNX model ready for TensorRT
- Size: ~118 MB
- Supports dynamic batch size and sequence length

### 2. `models/onnx_export_results.txt`
- Detailed validation metrics
- Benchmark results
- Model information

## Integration with TensorRT

### Convert ONNX to TensorRT Engine

```bash
# Install TensorRT (requires NVIDIA GPU)
pip install tensorrt

# Convert to TensorRT engine
trtexec --onnx=models/sbert_optimized.onnx \
        --saveEngine=models/sbert_fp16.trt \
        --fp16 \
        --workspace=4096 \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:32x128,attention_mask:32x128 \
        --maxShapes=input_ids:256x512,attention_mask:256x512
```

### TensorRT Inference (Python)

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Load TensorRT engine
with open('models/sbert_fp16.trt', 'rb') as f:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())

# Create execution context
context = engine.create_execution_context()

# Set input shapes
context.set_binding_shape(0, (batch_size, seq_length))
context.set_binding_shape(1, (batch_size, seq_length))

# Allocate GPU memory
# ... (standard TensorRT inference code)
```

## Troubleshooting

### Issue: Low Accuracy

**Symptoms**: Cosine similarity < 0.999

**Solutions**:
1. Check PyTorch version compatibility
2. Verify ONNX opset version
3. Ensure mean pooling is correctly implemented
4. Check normalization step

### Issue: Slow ONNX Inference

**Symptoms**: ONNX slower than PyTorch

**Solutions**:
1. Enable CUDA execution provider
2. Increase graph optimization level
3. Check for CPU fallback operations
4. Verify GPU availability

```python
# Check execution providers
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

### Issue: ONNX Export Fails

**Symptoms**: Export crashes or produces invalid model

**Solutions**:
1. Update PyTorch to latest version
2. Check for unsupported operations
3. Reduce opset version to 12
4. Disable constant folding

## Performance Expectations

### A100 GPU (Expected)
- PyTorch: ~10-15 ms per batch (5 sentences)
- ONNX: ~6-10 ms per batch
- TensorRT FP16: ~3-5 ms per batch
- **Total speedup**: 2-3x over PyTorch baseline

### CPU (Reference)
- PyTorch: ~50-80 ms per batch
- ONNX: ~40-60 ms per batch
- **Speedup**: 1.2-1.5x

## Advanced Options

### Export with FP16

```bash
python scripts/export_model_to_onnx.py --fp16
```

### Custom Benchmark

```bash
python scripts/export_model_to_onnx.py --benchmark-iterations 500
```

### Different Model

```bash
python scripts/export_model_to_onnx.py \
  --model-name "sentence-transformers/all-mpnet-base-v2" \
  --output-dir "models/mpnet"
```

## Next Steps

1. **TensorRT Conversion**: Convert ONNX to TensorRT for maximum performance
2. **Integration**: Update Rust backend to use ONNX/TensorRT models
3. **Benchmarking**: Run comprehensive A100 benchmarks
4. **Optimization**: Fine-tune batch sizes and sequence lengths

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Sentence Transformers](https://www.sbert.net/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
