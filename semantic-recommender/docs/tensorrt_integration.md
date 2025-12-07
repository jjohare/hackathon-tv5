# TensorRT Integration - Phase 4

## Overview

Integration of TensorRT-optimized encoder into the GPU hyper-personalization system, providing 3-5x faster query encoding while maintaining full backwards compatibility.

## Architecture

### Components

1. **TensorRT Encoder** (`scripts/utils/trt_inference.py`)
   - Drop-in replacement for `SentenceTransformer.encode()`
   - Zero-copy CUDA memory management
   - Automatic fallback to PyTorch if engine unavailable
   - Batch processing support

2. **GPU Hyper-Personalization** (`scripts/utils/gpu_hyper_personalization.py`)
   - Modified `__init__` to accept `use_tensorrt` flag
   - New `_load_encoder()` method for encoder selection
   - Graceful fallback to PyTorch encoder
   - Timing metrics include encoder type

### Integration Points

```python
# Before (PyTorch only)
system = GPUHyperPersonalization()

# After (TensorRT optional)
system = GPUHyperPersonalization(use_tensorrt=True)  # Try TensorRT
system = GPUHyperPersonalization(use_tensorrt=False) # Force PyTorch
```

## Performance

### Expected Improvements

| Metric | PyTorch | TensorRT | Speedup |
|--------|---------|----------|---------|
| Query Encoding | 2-3ms | 0.6-1ms | 3-5x |
| Total Latency | ~5ms | ~3ms | 1.7x |
| GPU Memory | Same | Same | - |
| Accuracy | Baseline | 0.99+ cosine | Equivalent |

### Latency Breakdown

```
Total Search Latency (~3ms with TensorRT):
├─ Query encoding:    0.8ms  (TensorRT) ← Optimized!
├─ User fusion:       0.1ms  (GPU)
├─ GPU similarity:    1.5ms  (GPU matmul)
└─ Attention rerank:  0.6ms  (GPU)
```

## Usage

### 1. Basic Usage (PyTorch Fallback)

```bash
# Without TensorRT engine (uses PyTorch)
python scripts/utils/gpu_hyper_personalization.py --test

# Output:
# [Semantic Model]
#   ⚠️  TensorRT engine not found: data/models/minilm_l12_v2_fp16.plan
#   Using PyTorch encoder
#   ✅ PyTorch encoder active
```

### 2. TensorRT-Accelerated Usage

```bash
# With TensorRT engine
python scripts/utils/gpu_hyper_personalization.py --test --use-tensorrt

# Output:
# [Semantic Model]
#   Loading TensorRT engine: data/models/minilm_l12_v2_fp16.plan
#   ✅ TensorRT encoder active (3-5x faster)
```

### 3. Python API

```python
from gpu_hyper_personalization import GPUHyperPersonalization

# Initialize with TensorRT
system = GPUHyperPersonalization(use_tensorrt=True)

# Personalized search
result = system.personalized_search(
    user_id="user_123",
    query="sci-fi movies with time travel",
    top_k=10,
    context={
        'time_of_day': [0.2, 0.1, 0.7],  # Evening
        'genre_prefs': [0.7, 0.2, 0.1],  # Sci-fi
        'social_signal': [1.0, 0.0]       # Solo
    }
)

# Check encoder type
print(f"Encoder: {result['encoder']}")  # "tensorrt" or "pytorch"
print(f"Query encoding: {result['timing']['query_encoding_ms']:.2f}ms")
```

## Validation

### Automated Testing

```bash
# Run validation suite
python scripts/validate_trt_integration.py --engine data/models/minilm_l12_v2_fp16.plan

# Tests:
# ✅ Test 1: TensorRT encoder loads correctly
# ✅ Test 2: Output matches PyTorch (cosine > 0.99)
# ✅ Test 3: Embeddings stay on GPU
# ✅ Test 4: Batch processing works
# ✅ Test 5: Performance gains measurable
# ✅ Test 6: Backwards compatibility maintained
```

### Manual Validation

```python
# 1. Check output equivalence
from trt_inference import TensorRTEncoder
from sentence_transformers import SentenceTransformer

trt = TensorRTEncoder("model.plan", "paraphrase-multilingual-MiniLM-L12-v2")
pytorch = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

query = "test query"
emb_trt = trt.encode([query], normalize_embeddings=True)
emb_pytorch = pytorch.encode([query], convert_to_tensor=True, normalize_embeddings=True)

similarity = torch.nn.functional.cosine_similarity(emb_trt, emb_pytorch)
assert similarity > 0.99, "Output mismatch!"

# 2. Check GPU memory
assert emb_trt.device.type == 'cuda', "Embeddings not on GPU!"
```

## Backwards Compatibility

### Fallback Behavior

The system maintains full backwards compatibility through multiple fallback layers:

1. **No TensorRT Module**: Falls back to PyTorch
   ```python
   # Warning: TensorRT encoder not available, using PyTorch fallback
   ```

2. **Engine Not Found**: Falls back to PyTorch
   ```python
   # Warning: TensorRT engine not found: path/to/engine.plan
   # Using PyTorch encoder
   ```

3. **TensorRT Load Failure**: Falls back to PyTorch
   ```python
   # Warning: TensorRT failed: [error message]
   # Falling back to PyTorch
   ```

### API Compatibility

All existing code continues to work without modification:

```python
# Old code (still works)
system = GPUHyperPersonalization()
result = system.personalized_search("user_1", "query")

# New code (opt-in)
system = GPUHyperPersonalization(use_tensorrt=True)
result = system.personalized_search("user_1", "query")
```

## Deployment

### Development Environment

```bash
# Install TensorRT (optional)
pip install tensorrt pycuda

# Generate engine (if needed)
python scripts/convert_to_tensorrt.py \
    --model paraphrase-multilingual-MiniLM-L12-v2 \
    --output data/models/minilm_l12_v2_fp16.plan \
    --precision fp16

# Test integration
python scripts/validate_trt_integration.py
```

### Production Environment

```bash
# Option 1: PyTorch only (no changes needed)
python scripts/utils/gpu_hyper_personalization.py --test

# Option 2: TensorRT acceleration
# 1. Copy engine file to production
scp data/models/minilm_l12_v2_fp16.plan prod:/app/data/models/

# 2. Run with TensorRT
python scripts/utils/gpu_hyper_personalization.py --test --use-tensorrt
```

### Docker Deployment

```dockerfile
# Dockerfile with TensorRT support
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Install TensorRT (optional)
RUN apt-get update && apt-get install -y \
    python3-libnvinfer \
    python3-libnvinfer-dev

# Copy engine file
COPY data/models/minilm_l12_v2_fp16.plan /app/data/models/

# Run with TensorRT
CMD ["python", "scripts/utils/gpu_hyper_personalization.py", "--use-tensorrt"]
```

## Troubleshooting

### Issue: TensorRT not available

```
[Warning] TensorRT encoder not available, using PyTorch fallback
```

**Solution**: Install TensorRT and PyCUDA
```bash
pip install tensorrt pycuda
```

### Issue: Engine file not found

```
⚠️  TensorRT engine not found: data/models/minilm_l12_v2_fp16.plan
```

**Solution**: Generate the engine file
```bash
python scripts/convert_to_tensorrt.py \
    --model paraphrase-multilingual-MiniLM-L12-v2 \
    --output data/models/minilm_l12_v2_fp16.plan
```

### Issue: Low cosine similarity

```
❌ FAIL: Output mismatch (mean: 0.95, min: 0.92)
```

**Solution**: Check calibration and precision settings
```bash
# Try FP32 instead of FP16
python scripts/convert_to_tensorrt.py --precision fp32
```

### Issue: Slower than PyTorch

```
⚠️  WARNING: Speedup only 1.2x (expected > 1.5x)
```

**Possible causes**:
1. Small batch size (TensorRT benefits from batching)
2. GPU not fully utilized
3. Engine not optimized for GPU architecture

**Solution**:
```bash
# Rebuild engine for specific GPU
python scripts/convert_to_tensorrt.py --gpu-arch 86  # For RTX 3090/A100
```

## Metrics & Monitoring

### Timing Metrics

```json
{
  "timing": {
    "total_ms": 3.2,
    "query_encoding_ms": 0.8,
    "user_fusion_ms": 0.1,
    "gpu_similarity_ms": 1.5,
    "attention_rerank_ms": 0.6
  },
  "encoder": "tensorrt",
  "device": "cuda:0"
}
```

### Performance Monitoring

```python
# Track encoder performance
encoding_times = []
for query in queries:
    start = time.time()
    result = system.personalized_search(user, query)
    encoding_times.append(result['timing']['query_encoding_ms'])

print(f"Mean encoding time: {np.mean(encoding_times):.2f}ms")
print(f"P95 encoding time: {np.percentile(encoding_times, 95):.2f}ms")
```

## Future Enhancements

1. **Dynamic Batching**: Batch multiple concurrent queries
2. **Mixed Precision**: INT8 quantization for 8x speedup
3. **Multi-GPU**: Distribute encoding across GPUs
4. **Async Inference**: Non-blocking encoding
5. **Engine Caching**: Cache compiled engines for faster startup

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [SentenceTransformers](https://www.sbert.net/)
- [ONNX Export Guide](https://huggingface.co/docs/transformers/serialization)
- [Phase 3: ONNX Export](./onnx_export.md)
- [Phase 5: Benchmarking](./benchmarking.md)
