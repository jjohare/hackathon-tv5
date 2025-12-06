# Quick Start Guide

Get semantic-recommender running in 5 minutes.

## Prerequisites

**Check**: Do you have these?
```bash
# NVIDIA GPU
nvidia-smi  # Should show GPU info

# CUDA 12.2+
nvcc --version  # Should output CUDA 12.2 or higher

# Rust 1.75+
rustc --version  # Should output 1.75 or higher

# Make (for CUDA kernels)
make --version
```

If missing, install:
- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-downloads
- **Rust**: https://rustup.rs/
- **Make**: `apt install build-essential` (Linux) or `brew install make` (macOS)

---

## Setup

### Step 1: Build CUDA Kernels (1 minute)

```bash
cd semantic-recommender/src/cuda/kernels
make all
# Output: ✓ Building kernels...
#         ✓ kernel_similarity.ptx
#         ✓ kernel_sssp.ptx
#         ✓ Done!
cd ../../..
```

### Step 2: Build Rust Application (2 minutes)

```bash
cargo build --release
# Output: Compiling agentics-hackathon v1.2.0
#         Finished release [optimized] target(s)
```

### Step 3: Start API Server (1 minute)

```bash
cargo run --release --bin api-server
# Output: [INFO] Starting API server on http://localhost:8080
#         [INFO] GPU initialized (NVIDIA T4)
#         [INFO] Vector DB connected to Qdrant
#         [INFO] Ready to accept requests
```

---

## First Query

In a new terminal:

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "French documentary climate change",
    "limit": 10,
    "threshold": 0.85
  }'
```

**Expected Response** (should arrive in ~12ms):

```json
{
  "results": [
    {
      "id": "doc_12345",
      "title": "Climat: l'Urgence d'Agir",
      "similarity": 0.94,
      "metadata": {
        "language": "fr",
        "genre": "Documentary",
        "duration": 52
      },
      "execution_path": "gpu"
    },
    {
      "id": "doc_12346",
      "title": "La Planète Brûle",
      "similarity": 0.91,
      "metadata": {
        "language": "fr",
        "genre": "Documentary",
        "duration": 90
      },
      "execution_path": "gpu"
    }
  ],
  "query_time_ms": 12,
  "metadata": {
    "execution_path": "gpu",
    "gpu_utilization": 0.92,
    "cache_hit_rate": 0.85
  }
}
```

✅ **Success!** Your system is working.

---

## Next Steps

### Option A: Integrate via REST API

See [API.md](API.md) for:
- Authentication
- Batch search
- Recommendations
- Filtering

### Option B: Use MCP Server for AI Agents

```bash
# Terminal 1: Start MCP server
cargo run --release --bin mcp-server

# Terminal 2: In Claude Code
# Add to claude_desktop_config.json:
{
  "mcpServers": {
    "media-gateway": {
      "command": "cargo",
      "args": ["run", "--release", "--bin", "mcp-server"]
    }
  }
}
```

See [INTEGRATION.md](INTEGRATION.md) for patterns.

### Option C: Deploy to Docker

```bash
# Build image
docker build -t semantic-recommender:latest -f Dockerfile.gpu .

# Run with GPU
docker run --gpus all -p 8080:8080 \
  -e CUDA_VISIBLE_DEVICES=0 \
  semantic-recommender:latest
```

---

## Configuration

### Environment Variables

```bash
# GPU
export CUDA_VISIBLE_DEVICES=0           # Single GPU
export GPU_MEMORY_FRACTION=0.8          # 80% of GPU VRAM

# Vector Database (local default)
export QDRANT_URL=http://localhost:6333

# Knowledge Graph (local default)
export NEO4J_URI=bolt://localhost:7687

# API Server
export API_PORT=8080
export RATE_LIMIT_RPS=1000
```

### Performance Tuning

**For faster queries** (trade-off: less accuracy):
```bash
export SIMILARITY_THRESHOLD=0.75         # Lower threshold
export GPU_BATCH_SIZE=512                # Larger batches
export CACHE_SIZE=1000                   # Increase cache
```

**For more accurate results**:
```bash
export SIMILARITY_THRESHOLD=0.95         # Higher threshold
export GPU_BATCH_SIZE=64                 # Smaller batches
export RERANK_TOP_K=100                  # Rerank more candidates
```

---

## Troubleshooting

### CUDA Version Mismatch
```
Error: CUDA compute capability not supported
```
**Solution**: Check `nvidia-smi` CUDA version matches 12.2+
```bash
nvcc --version  # Should be CUDA 12.2 or higher
```

### GPU Out of Memory
```
Error: cudaMalloc failed: out of memory
```
**Solution**: Reduce batch size
```bash
export GPU_BATCH_SIZE=64
cargo run --release --bin api-server
```

### Vector DB Connection Failed
```
Error: Failed to connect to Qdrant at http://localhost:6333
```
**Solution**: Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Port Already in Use
```
Error: Address already in use: 127.0.0.1:8080
```
**Solution**: Use different port
```bash
export API_PORT=8081
cargo run --release --bin api-server
```

---

## Validation

Run quick validation suite:

```bash
# Check GPU
cargo run --release --bin diagnostics -- gpu
# Output: ✓ GPU: NVIDIA T4
#         ✓ Compute Capability: 7.5
#         ✓ VRAM: 16GB

# Check Vector DB
cargo run --release --bin diagnostics -- db
# Output: ✓ Qdrant: Connected
#         ✓ Collection: 1000 vectors loaded

# Benchmark
cargo bench --bench quickstart
# Output: Simple search:        12ms (P50)
#         Batch search:         120ms (10 queries)
#         Recommendation:       45ms (cold-start)
```

---

## What's Next?

1. **API Integration** → [API.md](API.md)
2. **MCP/Agent Integration** → [INTEGRATION.md](INTEGRATION.md)
3. **Architecture Deep-Dive** → [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Production Deployment** → [ARCHITECTURE.md#deployment](ARCHITECTURE.md#deployment)

---

**Stuck?** Check the [README](../README.md) or examine logs:
```bash
export RUST_LOG=debug
cargo run --release --bin api-server
```
