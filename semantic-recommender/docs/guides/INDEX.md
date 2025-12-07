# User Guides

**Version:** 1.0
**Date:** 2025-12-07
**Status:** Production

---

## Overview

Concise, practical guides for setup, deployment, and operations. Each guide is focused on specific tasks with step-by-step instructions.

---

## Getting Started

### 1. [QUICKSTART.md](./QUICKSTART.md) - 5-minute setup
**Time:** 5-10 minutes
**Audience:** New users, developers

**Prerequisites:**
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB RAM minimum

**What you'll learn:**
- Clone repository and install dependencies
- Download TensorRT engine
- Run first semantic query
- Verify system performance

---

## Deployment

### 2. [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Production deployment
**Time:** 30-60 minutes
**Audience:** DevOps engineers, system administrators

**What you'll learn:**
- Container orchestration (Docker)
- GPU resource allocation
- Load balancing and scaling
- Monitoring and logging
- Health checks and recovery

---

## Optimisation

### 3. [PERFORMANCE_TUNING.md](./PERFORMANCE_TUNING.md) - Performance optimisation
**Time:** 1-2 hours
**Audience:** Developers, performance engineers

**What you'll learn:**
- TensorRT engine optimisation
- Batch size tuning
- Memory management
- Caching strategies
- Bottleneck analysis

---

## Data Management

### 4. [DATA_ENRICHMENT.md](./DATA_ENRICHMENT.md) - Semantic enrichment
**Time:** 2-4 hours
**Audience:** Data engineers, ML engineers

**What you'll learn:**
- TMDB metadata enrichment
- Embedding generation pipeline
- Ontology graph construction
- Quality validation
- Incremental updates

---

## GPU Setup

### 5. [GPU_SETUP.md](./GPU_SETUP.md) - CUDA/TensorRT configuration
**Time:** 30-45 minutes
**Audience:** ML engineers, system administrators

**What you'll learn:**
- NVIDIA driver installation
- CUDA toolkit setup
- TensorRT installation and verification
- Model conversion (ONNX → TensorRT)
- Performance validation

---

## Common Tasks

### Quick References

**Start the service:**
```bash
cd scripts/server
python query_interface.py
```

**Test query:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "dark thriller", "limit": 5}'
```

**Check GPU usage:**
```bash
nvidia-smi
```

**View logs:**
```bash
tail -f logs/query_service.log
```

---

## Troubleshooting

For common issues and solutions, see:
- [Troubleshooting Guide](../reference/TROUBLESHOOTING.md)
- [Configuration Reference](../reference/CONFIGURATION.md)

---

## Related Documentation

### Prerequisites
- [README](../../README.md) - Project overview
- [Architecture](../architecture/SYSTEM_OVERVIEW.md) - System design

### Deep Dives
- [Algorithms](../algorithms/) - Core logic specifications
- [API Reference](../api/REST_API.md) - Interface documentation

### Reports
- [Benchmark Results](../reports/BENCHMARK_RESULTS.md) - Performance data
- [Validation Report](../reports/VALIDATION_REPORT.md) - Test results

---

**Last Updated:** 2025-12-07
