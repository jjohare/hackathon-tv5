# Reference Documentation

**Version:** 1.0
**Date:** 2025-12-07
**Status:** Production

---

## Overview

Reference material including glossary, configuration options, troubleshooting, dependencies, and version history.

---

## Documents

### 1. [GLOSSARY.md](./GLOSSARY.md) - Technical terminology

Comprehensive glossary of technical terms used throughout the documentation:
- Machine learning concepts (embeddings, similarity, semantic search)
- Graph theory (SSSP, ontology, traversal)
- Performance metrics (latency, throughput, QPS)
- Infrastructure (TensorRT, CUDA, GPU)

---

### 2. [CONFIGURATION.md](./CONFIGURATION.md) - All configuration options

Complete reference for all configurable parameters:
- TensorRT engine settings
- GPU memory allocation
- Batch processing parameters
- API server configuration
- Caching policies
- Logging levels

---

### 3. [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues and solutions

Diagnostic guide for common problems:
- Installation issues (CUDA not found, dependency conflicts)
- Runtime errors (OOM errors, slow queries)
- Performance problems (low throughput, high latency)
- Data quality issues (missing metadata, corrupted embeddings)

---

### 4. [DEPENDENCIES.md](./DEPENDENCIES.md) - Software dependencies

Complete dependency inventory:
- Python packages (versions and purposes)
- System libraries (CUDA, cuDNN, TensorRT)
- External services (Neo4j, optional databases)
- Development tools (build requirements)

---

### 5. [CHANGELOG.md](./CHANGELOG.md) - Version history

Project version history and notable changes:
- Major releases
- Breaking changes
- New features
- Bug fixes
- Performance improvements

---

### 6. [API.md](./API.md) - Legacy API reference

**Note:** This is legacy documentation. For current API reference, see [../api/REST_API.md](../api/REST_API.md).

---

## Quick Reference

### System Requirements

**Minimum:**
- Python 3.10+
- 16GB RAM
- 50GB disk space
- CUDA 11.8+ (for GPU)

**Recommended:**
- Python 3.11
- 32GB RAM
- RTX 3060 or better (12GB VRAM)
- 100GB SSD storage

---

### Key Configuration Files

```
semantic-recommender/
├── .env                          # Environment variables
├── config/
│   ├── tensorrt.yaml            # TensorRT settings
│   ├── server.yaml              # API server config
│   └── logging.yaml             # Logging configuration
└── scripts/server/
    └── config.json              # Query service config
```

---

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `TENSORRT_ENGINE_PATH` | `models/minilm_fp16.plan` | TensorRT engine file |
| `EMBEDDINGS_PATH` | `data/embeddings/tmdb/` | Vector storage path |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

See [CONFIGURATION.md](./CONFIGURATION.md) for complete reference.

---

### Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| `CUDA_ERROR_OUT_OF_MEMORY` | GPU OOM | Reduce batch size, use smaller model |
| `TRT_ENGINE_NOT_FOUND` | Missing model | Download TensorRT engine |
| `INVALID_EMBEDDING_DIM` | Dimension mismatch | Regenerate embeddings with correct model |
| `NEO4J_CONNECTION_ERROR` | Graph DB unavailable | Start Neo4j service |

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed diagnostics.

---

## Related Documentation

### Prerequisites
- [Quick Start](../guides/QUICKSTART.md) - First-time setup
- [Architecture](../architecture/SYSTEM_OVERVIEW.md) - System design

### Implementation
- [API Reference](../api/REST_API.md) - Interface specifications
- [Algorithms](../algorithms/) - Core logic details

### Operations
- [Deployment Guide](../guides/DEPLOYMENT_GUIDE.md) - Production setup
- [Performance Tuning](../guides/PERFORMANCE_TUNING.md) - Optimisation

---

**Last Updated:** 2025-12-07
