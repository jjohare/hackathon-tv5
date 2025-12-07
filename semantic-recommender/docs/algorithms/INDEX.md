# Algorithm Documentation

**Version:** 1.0
**Date:** 2025-12-07
**Status:** Production

---

## Overview

This directory contains deep technical specifications for all core algorithms in the semantic recommender system. Each document includes mathematical foundations, complexity analysis, and implementation details.

---

## Documents

### Core Algorithms

1. **[SSSP_ALGORITHMS.md](./SSSP_ALGORITHMS.md)** - Single-Source Shortest Path
   - Dijkstra's algorithm (GPU implementation)
   - Duan SSSP (CPU optimised for large graphs)
   - Adaptive selection criteria (< 10K nodes → GPU, else CPU)
   - Complexity: O((V + E) log V)

2. **[EMBEDDING_PIPELINE.md](./EMBEDDING_PIPELINE.md)** - TensorRT encoding
   - Sentence-BERT model optimisation
   - FP16 quantisation
   - Batch processing strategies
   - Latency: 24ms per query (RTX A6000)

3. **[HYBRID_FUSION.md](./HYBRID_FUSION.md)** - Neuro-symbolic fusion
   - Semantic similarity scoring
   - Ontology reasoning integration
   - Weighted fusion strategies
   - Exploration vs exploitation modes

4. **[GRAPH_REASONING.md](./GRAPH_REASONING.md)** - Ontology reasoning
   - Neo4j graph traversal
   - Semantic relationship extraction
   - Director/actor similarity computation
   - Franchise detection

5. **[SIMILARITY_COMPUTATION.md](./SIMILARITY_COMPUTATION.md)** - GPU vector search
   - Cosine similarity (vectorised)
   - Top-K selection algorithms
   - Memory-efficient batch processing
   - Throughput: 0.32ms for 62K items

6. **[ADAPTIVE_SELECTION.md](./ADAPTIVE_SELECTION.md)** - Algorithm selection logic
   - Graph size heuristics
   - Performance profiling
   - Dynamic switching criteria

---

## Mathematical Notation

All documents use the following conventions:

- **Sets:** Capital letters (V, E, G)
- **Scalars:** Lowercase Greek (α, β, λ)
- **Vectors:** Bold lowercase (v, q)
- **Matrices:** Bold uppercase (M, W)
- **Functions:** f(x), g(x)
- **Complexity:** O(), Θ(), Ω()

---

## Implementation Status

| Algorithm | Status | Performance | Hardware |
|-----------|--------|-------------|----------|
| TensorRT Encoding | ✅ Production | 24ms | RTX A6000 |
| GPU Similarity | ✅ Production | 0.32ms (62K) | RTX A6000 |
| Dijkstra SSSP | ✅ Production | 1.2ms (n<10K) | RTX A6000 |
| Duan SSSP | ✅ Production | 110ms (n>1M) | CPU |
| Hybrid Fusion | ✅ Production | <1ms | CPU |
| Adaptive Selection | ✅ Production | <0.1ms | CPU |

---

## Related Documentation

### Prerequisites
- [Architecture Overview](../architecture/SYSTEM_OVERVIEW.md)
- [Data Architecture](../architecture/DATA_ARCHITECTURE.md)

### Implementation
- [API Reference](../api/REST_API.md) - Query interface
- [Performance Tuning](../guides/PERFORMANCE_TUNING.md) - Optimisation guide

### Reports
- [Benchmark Results](../reports/BENCHMARK_RESULTS.md) - Performance measurements

---

**Last Updated:** 2025-12-07
