# Vector Search Implementation Guide

**Target Audience**: Backend Engineers, ML Engineers
**Prerequisites**: Rust knowledge, understanding of vector embeddings
**Estimated Implementation Time**: 1-2 weeks

---

## Table of Contents

1. [RuVector Integration](#1-ruvector-integration)
2. [FAISS Alternative Setup](#2-faiss-alternative-setup)
3. [Indexing Strategies](#3-indexing-strategies)
4. [Quantization Techniques](#4-quantization-techniques)
5. [Performance Optimization](#5-performance-optimization)
6. [Production Deployment](#6-production-deployment)

---

## 1. RuVector Integration

### 1.1 Installation

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Create project
cargo new --bin recommendation-vector-search
cd recommendation-vector-search
```

### 1.2 Configure Dependencies

```toml
# Cargo.toml
[package]
name = "recommendation-vector-search"
version = "0.1.0"
edition = "2021"

[dependencies]
# Vector database
qdrant-client = "1.7"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"

# Embeddings
ndarray = "0.15"
half = "2.3"  # FP16 support
bincode = "1.3"

# Performance
rayon = "1.7"
dashmap = "5.5"

# Monitoring
prometheus = "0.13"
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
criterion = "0.5"
```

### 1.3 Initialize Qdrant (RuVector Backend)

```rust
// src/vector_db.rs
use qdrant_client::{
    client::QdrantClient,
    qdrant::{
        vectors_config::Config, CreateCollection, Distance,
        VectorParams, VectorsConfig, SearchPoints, PointStruct,
        UpsertPoints, UpsertPointsBuilder, Value,
    },
};
use anyhow::Result;
use std::collections::HashMap;

pub struct VectorDatabase {
    client: QdrantClient,
    collection_name: String,
}

impl VectorDatabase {
    pub async fn new(url: &str, collection_name: &str) -> Result<Self> {
        let client = QdrantClient::from_url(url).build()?;

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
        })
    }

    pub async fn create_collection(
        &self,
        vector_size: u64,
        distance: Distance,
    ) -> Result<()> {
        // HNSW index configuration
        let vectors_config = VectorsConfig {
            config: Some(Config::Params(VectorParams {
                size: vector_size,
                distance: distance.into(),
                hnsw_config: Some(qdrant_client::qdrant::HnswConfigDiff {
                    m: Some(16),  // Connections per node
                    ef_construct: Some(200),  // Construction time search depth
                    full_scan_threshold: Some(10000),
                    max_indexing_threads: Some(0),  // Auto
                    on_disk: Some(false),  // Keep in memory for speed
                    payload_m: None,
                }),
                quantization_config: None,  // Will add later
                on_disk: Some(false),
            })),
        };

        self.client
            .create_collection(&CreateCollection {
                collection_name: self.collection_name.clone(),
                vectors_config: Some(vectors_config),
                ..Default::default()
            })
            .await?;

        Ok(())
    }

    pub async fn upsert_vectors(
        &self,
        vectors: Vec<(u64, Vec<f32>, HashMap<String, String>)>,
    ) -> Result<()> {
        let points: Vec<PointStruct> = vectors
            .into_iter()
            .map(|(id, vector, metadata)| {
                let payload: HashMap<String, Value> = metadata
                    .into_iter()
                    .map(|(k, v)| (k, Value::from(v)))
                    .collect();

                PointStruct::new(id, vector, payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                self.collection_name.clone(),
                points,
            ))
            .await?;

        Ok(())
    }

    pub async fn search(
        &self,
        query_vector: Vec<f32>,
        limit: u64,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<Vec<SearchResult>> {
        let search_result = self.client
            .search_points(&SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: query_vector,
                limit,
                filter,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await?;

        let results = search_result
            .result
            .into_iter()
            .map(|point| SearchResult {
                id: point.id.unwrap().num().unwrap(),
                score: point.score,
                payload: point.payload,
            })
            .collect();

        Ok(results)
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub payload: HashMap<String, Value>,
}
```

### 1.4 Deployment

```bash
# Install Qdrant (RuVector backend)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant:latest

# Verify deployment
curl http://localhost:6333/collections
```

---

## 2. FAISS Alternative Setup

### 2.1 Python Integration (via PyO3)

```toml
# Cargo.toml
[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
```

```rust
// src/faiss_wrapper.rs
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use numpy::{PyArray1, PyArray2};

pub struct FAISSIndex {
    py: Python<'_>,
    index: PyObject,
}

impl FAISSIndex {
    pub fn new<'py>(py: Python<'py>, dimension: usize, index_type: &str) -> PyResult<Self> {
        let faiss = PyModule::import(py, "faiss")?;

        // Create index (e.g., "IVF1024,PQ64" or "HNSW32")
        let index = faiss
            .call_method1("index_factory", (dimension, index_type, faiss.getattr("METRIC_INNER_PRODUCT")?))?
            .into();

        Ok(Self { py, index })
    }

    pub fn train(&self, vectors: &PyArray2<f32>) -> PyResult<()> {
        self.index.call_method1(self.py, "train", (vectors,))?;
        Ok(())
    }

    pub fn add(&self, vectors: &PyArray2<f32>) -> PyResult<()> {
        self.index.call_method1(self.py, "add", (vectors,))?;
        Ok(())
    }

    pub fn search(
        &self,
        queries: &PyArray2<f32>,
        k: usize,
    ) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<i64>>)> {
        let result = self.index.call_method1(self.py, "search", (queries, k))?;
        let (distances, indices) = result.extract::<(Py<PyArray2<f32>>, Py<PyArray2<i64>>)>()?;
        Ok((distances, indices))
    }
}
```

### 2.2 GPU-Accelerated FAISS

```python
# Python helper script: faiss_gpu_setup.py
import faiss
import numpy as np

def create_gpu_index(dimension, index_type="IVF1024,PQ64", gpu_id=0):
    """Create GPU-accelerated FAISS index."""

    # Create CPU index first
    cpu_index = faiss.index_factory(dimension, index_type, faiss.METRIC_INNER_PRODUCT)

    # Transfer to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)

    return gpu_index

def train_and_add(index, vectors):
    """Train index and add vectors."""
    # Train on sample (for IVF)
    index.train(vectors)

    # Add all vectors
    index.add(vectors)

    return index

# Example usage
if __name__ == "__main__":
    dimension = 1024
    n_vectors = 1000000

    # Generate random embeddings (replace with real data)
    vectors = np.random.random((n_vectors, dimension)).astype('float32')

    # Create and populate index
    index = create_gpu_index(dimension)
    index = train_and_add(index, vectors)

    # Search
    query = vectors[:10]  # First 10 as queries
    distances, indices = index.search(query, k=100)

    print(f"Search results shape: {indices.shape}")
    print(f"Top result for query 0: {indices[0, :5]}")
```

---

## 3. Indexing Strategies

### 3.1 HNSW (Hierarchical Navigable Small World)

**Best for**: <10M vectors, low-latency requirements (<10ms)

```rust
// HNSW configuration
use qdrant_client::qdrant::HnswConfigDiff;

let hnsw_config = HnswConfigDiff {
    m: Some(16),              // Connections per layer (higher = more accurate, slower build)
    ef_construct: Some(200),  // Construction-time search depth
    full_scan_threshold: Some(10000),  // Switch to full scan for small collections
    max_indexing_threads: Some(8),
    on_disk: Some(false),     // Keep in memory for speed
    payload_m: None,
};
```

**Performance Characteristics:**
- Insert: O(log n) per vector
- Search: O(log n) per query
- Memory: ~500 bytes per vector (1024-dim, M=16)
- Build time: ~1 hour for 10M vectors

### 3.2 IVF (Inverted File Index)

**Best for**: 10M-1B vectors, batch processing

```python
# FAISS IVF configuration
import faiss

dimension = 1024
n_list = 4096  # Number of clusters (sqrt(n_vectors))

# Coarse quantizer (first stage)
quantizer = faiss.IndexFlatIP(dimension)

# IVF index with product quantization
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    n_list,      # Number of Voronoi cells
    64,          # Subquantizer count (M)
    8,           # Bits per subquantizer
)

# Search parameters
index.nprobe = 32  # Number of cells to visit (speed/accuracy tradeoff)
```

**Performance Characteristics:**
- Insert: O(1) amortized
- Search: O(nprobe × k) per query
- Memory: ~96 bytes per vector (with PQ compression)
- Build time: ~3 hours for 100M vectors

### 3.3 CAGRA (CUDA ANN Graph)

**Best for**: GPU-accelerated billion-scale search

```python
# cuVS CAGRA integration
import cuvs
from cuvs.neighbors import cagra

# Build CAGRA index
build_params = cagra.IndexParams(
    intermediate_graph_degree=64,
    graph_degree=32,
)

index = cagra.build(
    build_params,
    embeddings,  # [N, D] numpy array
)

# Search parameters
search_params = cagra.SearchParams(
    itopk_size=64,
    search_width=2,
    max_iterations=0,
)

distances, indices = cagra.search(
    search_params,
    index,
    queries,
    k=100,
)
```

**Performance Characteristics:**
- Insert: Not supported (rebuild required)
- Search: <10ms for 1B vectors (GPU)
- Memory: ~300 bytes per vector
- Build time: ~40 minutes for 1B vectors (H100)

### 3.4 Hybrid Sharding Strategy

```rust
// Shard by content type for better locality
pub enum ShardStrategy {
    ByContentType,  // Movies, TV, Docs separately
    ByLanguage,     // EN, FR, multilingual
    HashBased,      // Consistent hashing
    Geographic,     // US, EU, APAC
}

pub struct ShardedIndex {
    shards: Vec<VectorDatabase>,
    strategy: ShardStrategy,
}

impl ShardedIndex {
    pub async fn search(
        &self,
        query: &[f32],
        shard_selector: impl Fn(&ShardStrategy) -> Vec<usize>,
    ) -> Result<Vec<SearchResult>> {
        // Select relevant shards
        let shard_indices = shard_selector(&self.strategy);

        // Parallel search across shards
        let searches: Vec<_> = shard_indices
            .iter()
            .map(|&i| self.shards[i].search(query.to_vec(), 100, None))
            .collect();

        let results = futures::future::try_join_all(searches).await?;

        // Merge and re-rank
        let mut merged = results.into_iter().flatten().collect::<Vec<_>>();
        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        merged.truncate(100);

        Ok(merged)
    }
}
```

---

## 4. Quantization Techniques

### 4.1 Scalar Quantization (int8)

**4x compression, 99%+ accuracy**

```rust
// Quantize to int8
pub fn quantize_to_int8(vector: &[f32]) -> (Vec<i8>, f32, f32) {
    let min = vector.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let scale = (max - min) / 255.0;
    let quantized: Vec<i8> = vector
        .iter()
        .map(|&v| ((v - min) / scale).round() as i8)
        .collect();

    (quantized, min, scale)
}

// Dequantize
pub fn dequantize_from_int8(quantized: &[i8], min: f32, scale: f32) -> Vec<f32> {
    quantized
        .iter()
        .map(|&q| q as f32 * scale + min)
        .collect()
}

// Distance computation on quantized vectors (faster)
pub fn int8_dot_product(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}
```

**Qdrant Integration:**
```rust
use qdrant_client::qdrant::{QuantizationConfig, ScalarQuantization};

let quantization_config = QuantizationConfig {
    scalar: Some(ScalarQuantization {
        r#type: 1,  // Int8
        quantile: Some(0.99),  // Use 99th percentile for bounds
        always_ram: Some(true),
    }),
};
```

### 4.2 Product Quantization (PQ)

**32x compression, 95%+ accuracy**

```python
# FAISS Product Quantization
import faiss

dimension = 1024
M = 64  # Number of subquantizers (dimension must be divisible by M)
nbits = 8  # Bits per subquantizer (256 centroids)

# Create PQ index
index = faiss.IndexPQ(dimension, M, nbits)

# Train on representative sample (10K-100K vectors)
training_vectors = embeddings[:100000]
index.train(training_vectors)

# Add all vectors
index.add(embeddings)

# Search
distances, indices = index.search(queries, k=100)
```

**Memory Calculation:**
```
Full precision: 1024 dim × 4 bytes = 4096 bytes
PQ compressed: 64 subquantizers × 1 byte = 64 bytes
Compression: 64x
```

### 4.3 Binary Quantization

**32x compression, 95% accuracy (compatible models)**

```rust
// Binary quantization (1 bit per dimension)
pub fn quantize_to_binary(vector: &[f32]) -> Vec<u8> {
    let mut binary = vec![0u8; (vector.len() + 7) / 8];

    for (i, &value) in vector.iter().enumerate() {
        if value > 0.0 {
            binary[i / 8] |= 1 << (i % 8);
        }
    }

    binary
}

// Hamming distance (XOR + popcount)
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}
```

**Qdrant Binary Quantization:**
```rust
use qdrant_client::qdrant::{QuantizationConfig, BinaryQuantization};

let quantization_config = QuantizationConfig {
    binary: Some(BinaryQuantization {
        always_ram: Some(true),
    }),
};
```

---

## 5. Performance Optimization

### 5.1 Batch Processing

```rust
// Batch upsert for higher throughput
pub async fn batch_upsert(
    &self,
    vectors: Vec<(u64, Vec<f32>, HashMap<String, String>)>,
    batch_size: usize,
) -> Result<()> {
    for chunk in vectors.chunks(batch_size) {
        self.upsert_vectors(chunk.to_vec()).await?;

        // Rate limiting to avoid overwhelming server
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    Ok(())
}
```

**Optimal Batch Sizes:**
- HNSW: 100-1000 vectors/batch
- IVF: 10,000-100,000 vectors/batch
- Network-limited: 1,000 vectors/batch

### 5.2 Parallel Search

```rust
use rayon::prelude::*;

// Parallel search across multiple queries
pub async fn parallel_search(
    &self,
    queries: Vec<Vec<f32>>,
    k: usize,
) -> Result<Vec<Vec<SearchResult>>> {
    let results: Vec<_> = queries
        .par_iter()
        .map(|query| {
            tokio::runtime::Handle::current()
                .block_on(self.search(query.clone(), k as u64, None))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(results)
}
```

### 5.3 Caching Layer

```rust
use dashmap::DashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub struct CachedVectorDatabase {
    db: VectorDatabase,
    cache: DashMap<u64, Vec<SearchResult>>,
    cache_ttl: std::time::Duration,
}

impl CachedVectorDatabase {
    fn query_hash(query: &[f32], k: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        for &v in query {
            v.to_bits().hash(&mut hasher);
        }
        k.hash(&mut hasher);
        hasher.finish()
    }

    pub async fn search_cached(
        &self,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let hash = Self::query_hash(&query, k);

        // Check cache
        if let Some(cached) = self.cache.get(&hash) {
            return Ok(cached.clone());
        }

        // Cache miss - query database
        let results = self.db.search(query.clone(), k as u64, None).await?;

        // Store in cache
        self.cache.insert(hash, results.clone());

        Ok(results)
    }
}
```

### 5.4 Monitoring

```rust
use prometheus::{Counter, Histogram, Registry};

pub struct VectorDBMetrics {
    search_duration: Histogram,
    search_total: Counter,
    cache_hits: Counter,
    cache_misses: Counter,
}

impl VectorDBMetrics {
    pub fn new(registry: &Registry) -> Self {
        let search_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new("vector_search_duration_seconds", "Search latency")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        ).unwrap();

        registry.register(Box::new(search_duration.clone())).unwrap();

        Self {
            search_duration,
            search_total: Counter::new("vector_search_total", "Total searches").unwrap(),
            cache_hits: Counter::new("vector_cache_hits", "Cache hits").unwrap(),
            cache_misses: Counter::new("vector_cache_misses", "Cache misses").unwrap(),
        }
    }

    pub fn observe_search(&self, duration: std::time::Duration) {
        self.search_duration.observe(duration.as_secs_f64());
        self.search_total.inc();
    }
}
```

---

## 6. Production Deployment

### 6.1 High-Availability Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant-1:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data_1:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__NODE_ID=1
      - QDRANT__CLUSTER__PEERS=qdrant-2:6335,qdrant-3:6335

  qdrant-2:
    image: qdrant/qdrant:latest
    ports:
      - "6334:6333"
    volumes:
      - ./qdrant_data_2:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__NODE_ID=2
      - QDRANT__CLUSTER__PEERS=qdrant-1:6335,qdrant-3:6335

  qdrant-3:
    image: qdrant/qdrant:latest
    ports:
      - "6335:6333"
    volumes:
      - ./qdrant_data_3:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__NODE_ID=3
      - QDRANT__CLUSTER__PEERS=qdrant-1:6335,qdrant-2:6335

  # Load balancer
  haproxy:
    image: haproxy:latest
    ports:
      - "6336:6336"
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg
```

### 6.2 Kubernetes Deployment

```yaml
# qdrant-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
spec:
  serviceName: qdrant
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
          name: http
        - containerPort: 6334
          name: grpc
        volumeMounts:
        - name: data
          mountPath: /qdrant/storage
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### 6.3 Backup Strategy

```bash
#!/bin/bash
# backup_qdrant.sh

BACKUP_DIR="/backups/qdrant-$(date +%Y%m%d-%H%M%S)"
QDRANT_URL="http://localhost:6333"

# Create snapshot
curl -X POST "$QDRANT_URL/collections/media_embeddings/snapshots"

# Download snapshot
SNAPSHOT_NAME=$(curl "$QDRANT_URL/collections/media_embeddings/snapshots" | jq -r '.result[0].name')
curl "$QDRANT_URL/collections/media_embeddings/snapshots/$SNAPSHOT_NAME" -o "$BACKUP_DIR/snapshot.tar"

# Backup metadata
curl "$QDRANT_URL/collections" > "$BACKUP_DIR/collections.json"

echo "Backup completed: $BACKUP_DIR"
```

### 6.4 Load Testing

```rust
// Load testing with criterion
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn load_test(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let db = rt.block_on(async {
        VectorDatabase::new("http://localhost:6333", "test_collection")
            .await
            .unwrap()
    });

    let mut group = c.benchmark_group("vector_search");

    for concurrent_queries in [1, 10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrent_queries),
            &concurrent_queries,
            |b, &n| {
                b.iter(|| {
                    rt.block_on(async {
                        let queries: Vec<_> = (0..n)
                            .map(|_| vec![0.5f32; 1024])
                            .collect();

                        db.parallel_search(queries, 100).await
                    })
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, load_test);
criterion_main!(benches);
```

---

## Performance Targets

**Target Metrics (100M vectors, 1024-dim):**
- **Search latency (p99)**: <10ms
- **Throughput**: >10,000 QPS
- **Index build time**: <4 hours
- **Memory usage**: <40GB (with quantization)
- **Recall@100**: >95%

**Optimization Checklist:**
- [ ] HNSW parameters tuned (M=16, ef_construct=200)
- [ ] Quantization enabled (int8 or PQ)
- [ ] Caching layer implemented
- [ ] Parallel search enabled
- [ ] Monitoring metrics exposed
- [ ] Backup strategy configured
- [ ] Load testing passed

---

**Next Steps:**
1. Implement embedding generation pipeline
2. Integrate with ontology reasoning
3. Deploy to production cluster
4. Set up monitoring dashboards

**Related Guides:**
- [GPU Setup Guide](gpu-setup-guide.md)
- [Ontology Reasoning Guide](ontology-reasoning-guide.md)
- [Learning Pipeline Guide](learning-pipeline-guide.md)
