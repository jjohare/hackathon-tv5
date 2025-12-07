# Model Files Download Guide

## Large Model Files Not in Git

Due to GitHub's 100MB file size limit, TensorRT and ONNX model files are **not included** in the git repository. Download them separately:

## Required Model Files

### TensorRT Engine
- **File**: `data/models/minilm_l12_v2_fp16.plan`
- **Size**: ~226 MB
- **Purpose**: TensorRT FP16 inference engine
- **Build**: `python scripts/ops/build_trt_engine.py`

### ONNX Model Files
- **File**: `models/sentence_transformer.onnx`
- **Size**: ~50 MB
- **File**: `models/sentence_transformer.onnx.data`
- **Size**: ~448 MB
- **Purpose**: ONNX export for TensorRT conversion

## Building Models Locally

```bash
# 1. Navigate to semantic-recommender
cd semantic-recommender

# 2. Export ONNX model (if not already done)
python scripts/ops/export_model_onnx.py \
  --model-name paraphrase-multilingual-MiniLM-L12-v2 \
  --output models/sentence_transformer.onnx

# 3. Build TensorRT engine
python scripts/ops/build_trt_engine.py \
  --onnx models/sentence_transformer.onnx \
  --output data/models/minilm_l12_v2_fp16.plan \
  --fp16 \
  --batch-size 32

# Expected: ~5 minutes on RTX A6000
```

## Alternative: Download Pre-built Models

Contact repository maintainer for pre-built model files or use the build scripts above.

## Git LFS (Future)

Consider migrating to Git LFS for model files:

```bash
git lfs install
git lfs track "*.plan"
git lfs track "*.onnx*"
git lfs track "*.trt"
```

---

**Note**: The system will auto-build missing models on first run if ONNX export tools are available.
