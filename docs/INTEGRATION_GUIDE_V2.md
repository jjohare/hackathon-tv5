# Component Integration Guide

**Media Gateway Hackathon - GPU-Accelerated Semantic Discovery System**

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Integration](#component-integration)
3. [CUDA → Rust FFI Integration](#cuda-rust-ffi-integration)
4. [API → Engine Integration](#api-engine-integration)
5. [Storage Layer Integration](#storage-layer-integration)
6. [Testing Integrated System](#testing-integrated-system)
7. [Performance Validation](#performance-validation)

---

## 1. Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   REST API   │  │   GraphQL    │  │   WebSocket  │          │
│  │   (Axum)     │  │   (async)    │  │   (real-time)│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Rust Orchestration Layer                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Core Engine Coordinator                    │    │
│  │  • GPU Engine Management                                │    │
│  │  • Query Planning & Optimization                        │    │
│  │  • Result Fusion & Ranking                              │    │
│  │  • Memory & Thread Management                           │    │
│  └───┬─────────────┬─────────────────┬────────────────────┘    │
└──────┼─────────────┼─────────────────┼─────────────────────────┘
       │             │                 │
       ▼             ▼                 ▼
┌─────────────┐ ┌──────────────┐ ┌─────────────────┐
│ GPU Engine  │ │   Neo4j      │ │ Vector Database │
│  (CUDA)     │ │   (GMC-O)    │ │ (RuVector/Milvus)│
│             │ │              │ │                 │
│ • Embeddings│ │ • Ontology   │ │ • Content       │
│ • Similarity│ │ • Relations  │ │   Vectors       │
│ • Reasoning │ │ • Inference  │ │ • Fast Search   │
└─────────────┘ └──────────────┘ └─────────────────┘
```

### Data Flow Pipeline

```
1. Query Input
   ↓
2. Query Parsing & Enrichment
   ├─→ Extract entities
   ├─→ Generate embeddings (GPU)
   └─→ Build query context
   ↓
3. Parallel Execution
   ├─→ Vector Search (RuVector/Milvus)
   │   └─→ GPU-accelerated similarity (CUDA)
   ├─→ Ontology Reasoning (Neo4j)
   │   └─→ GPU-accelerated inference (CUDA)
   └─→ Metadata Filtering
   ↓
4. Result Fusion
   ├─→ Merge results by entity ID
   ├─→ De-duplicate
   └─→ Score aggregation
   ↓
5. GPU-Accelerated Ranking
   ├─→ Multi-modal scoring
   ├─→ Personalization (AgentDB)
   └─→ Context-aware boosting
   ↓
6. Response Generation
   ├─→ Explanation text
   ├─→ Confidence scores
   └─→ Related entities
```

---

## 2. Component Integration

### 2.1 CUDA Kernels

**Location**: `/src/cuda/kernels/`

**Components**:
- `semantic_similarity_fp16_tensor_cores.cu` - Core similarity computation
- `ontology_reasoning.cu` - Graph traversal and inference
- `graph_search.cu` - Optimized path finding

**Integration Points**:
```cuda
// Exported C functions for FFI
extern "C" {
    // Similarity computation
    void compute_similarity_batch(
        const half* queries,      // [batch_size, dim]
        const half* candidates,   // [num_candidates, dim]
        float* results,           // [batch_size, num_candidates]
        int batch_size,
        int num_candidates,
        int dim,
        cudaStream_t stream
    );

    // Ontology reasoning
    void ontology_inference(
        const MediaOntologyNode* nodes,
        const int* adjacency_list,
        const float* edge_weights,
        ReasoningResult* results,
        int num_nodes,
        int max_hops,
        cudaStream_t stream
    );
}
```

### 2.2 Rust GPU Engine

**Location**: `/src/rust/gpu_engine/`

**Key Modules**:
- `mod.rs` - Engine initialization and coordination
- `similarity.rs` - FFI bindings for similarity kernels
- `reasoning.rs` - FFI bindings for ontology kernels
- `memory.rs` - GPU memory management

**Integration Pattern**:
```rust
use cudarc::driver::{CudaDevice, CudaStream};

pub struct GpuEngine {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    similarity_kernel: CudaFunction,
    reasoning_kernel: CudaFunction,
}

impl GpuEngine {
    pub async fn compute_similarity(
        &self,
        queries: &[f16],
        candidates: &[f16],
    ) -> Result<Vec<f32>> {
        // 1. Allocate GPU memory
        let d_queries = self.device.htod_sync_copy(queries)?;
        let d_candidates = self.device.htod_sync_copy(candidates)?;

        // 2. Launch kernel
        unsafe {
            self.similarity_kernel.launch(
                (grid_size, 1, 1),
                (block_size, 1, 1),
                &[&d_queries, &d_candidates, &d_results],
                &self.stream,
            )?;
        }

        // 3. Copy results back
        let results = self.device.dtoh_sync_copy(&d_results)?;
        Ok(results)
    }
}
```

### 2.3 Ontology Layer

**Location**: `/src/rust/ontology/`

**Key Modules**:
- `types.rs` - GMC-O entity types
- `reasoner.rs` - Inference engine
- `neo4j_client.rs` - Graph database client
- `cache.rs` - Query result caching

**Integration Pattern**:
```rust
pub struct OntologyReasoner {
    neo4j: Neo4jClient,
    gpu_engine: Arc<GpuEngine>,
    cache: Arc<RwLock<LruCache<String, ReasoningResult>>>,
}

impl OntologyReasoner {
    pub async fn infer_relationships(
        &self,
        entity_id: &str,
        max_depth: u32,
    ) -> Result<Vec<Relationship>> {
        // 1. Check cache
        if let Some(cached) = self.cache.read().await.get(entity_id) {
            return Ok(cached.clone());
        }

        // 2. Query Neo4j for graph structure
        let graph = self.neo4j.get_subgraph(entity_id, max_depth).await?;

        // 3. GPU-accelerated reasoning
        let inferences = self.gpu_engine
            .ontology_inference(&graph.nodes, &graph.edges)
            .await?;

        // 4. Cache results
        self.cache.write().await.insert(
            entity_id.to_string(),
            inferences.clone(),
        );

        Ok(inferences)
    }
}
```

### 2.4 Vector Storage

**Location**: `/src/rust/storage/`

**Supported Backends**:
- RuVector (in-memory, HNSW index)
- Milvus (distributed, production-scale)
- FAISS (high-performance, GPU-enabled)

**Integration Pattern**:
```rust
pub trait VectorStore: Send + Sync {
    async fn insert(&self, id: &str, vector: &[f32]) -> Result<()>;
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    async fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>>;
}

pub struct HybridStorage {
    vector_store: Box<dyn VectorStore>,
    ontology: Arc<OntologyReasoner>,
    gpu_engine: Arc<GpuEngine>,
}

impl HybridStorage {
    pub async fn semantic_search(
        &self,
        query: &str,
        filters: SearchFilters,
    ) -> Result<Vec<RankedResult>> {
        // 1. Generate query embedding (GPU)
        let query_vec = self.gpu_engine.encode_text(query).await?;

        // 2. Vector similarity search
        let vector_results = self.vector_store
            .search(&query_vec, filters.top_k * 2)
            .await?;

        // 3. Ontology enrichment
        let enriched = self.ontology
            .enrich_results(vector_results)
            .await?;

        // 4. GPU-accelerated re-ranking
        let ranked = self.gpu_engine
            .rank_results(query_vec, enriched)
            .await?;

        Ok(ranked.into_iter().take(filters.top_k).collect())
    }
}
```

---

## 3. CUDA → Rust FFI Integration

### 3.1 FFI-Safe Data Structures

**Critical Requirements**:
- `#[repr(C)]` on all shared structs
- No Rust-specific types (Vec, String, etc.)
- Explicit padding and alignment
- Static size assertions

**Example Structure** (`src/rust/models/ontology_ffi.rs`):
```rust
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MediaOntologyNode {
    pub id: u64,
    pub node_type: u32,
    pub confidence: f32,
    pub embedding: [f32; 512],
    pub metadata_offset: u32,
    pub metadata_length: u32,
    _padding: [u8; 8],
}

// Compile-time size/alignment verification
const _: () = assert!(
    std::mem::size_of::<MediaOntologyNode>() == 2072
);
const _: () = assert!(
    std::mem::align_of::<MediaOntologyNode>() == 8
);
```

**Corresponding CUDA Structure** (`src/cuda/kernels/ontology_ffi_check.cuh`):
```cuda
struct MediaOntologyNode {
    uint64_t id;
    uint32_t node_type;
    float confidence;
    float embedding[512];
    uint32_t metadata_offset;
    uint32_t metadata_length;
    uint8_t _padding[8];
};

// Static assertions
static_assert(sizeof(MediaOntologyNode) == 2072, "Size mismatch");
static_assert(alignof(MediaOntologyNode) == 8, "Alignment mismatch");
static_assert(offsetof(MediaOntologyNode, embedding) == 16, "Offset mismatch");
```

### 3.2 Memory Transfer Pattern

```rust
use cudarc::driver::DevicePtr;

pub struct GpuMemoryManager {
    device: Arc<CudaDevice>,
}

impl GpuMemoryManager {
    pub fn upload_nodes(
        &self,
        nodes: &[MediaOntologyNode],
    ) -> Result<DevicePtr<MediaOntologyNode>> {
        // 1. Verify data layout
        assert_eq!(
            std::mem::size_of_val(nodes),
            nodes.len() * std::mem::size_of::<MediaOntologyNode>()
        );

        // 2. Allocate GPU memory
        let d_nodes = unsafe {
            self.device.alloc::<MediaOntologyNode>(nodes.len())?
        };

        // 3. Transfer data
        unsafe {
            self.device.htod_copy(nodes, &d_nodes)?;
        }

        Ok(d_nodes)
    }

    pub fn download_results(
        &self,
        d_results: &DevicePtr<ReasoningResult>,
        count: usize,
    ) -> Result<Vec<ReasoningResult>> {
        let mut results = vec![
            unsafe { std::mem::zeroed() };
            count
        ];

        unsafe {
            self.device.dtoh_copy(d_results, &mut results)?;
        }

        Ok(results)
    }
}
```

### 3.3 Error Handling Across FFI Boundary

```rust
#[repr(C)]
pub struct CudaResult {
    pub status_code: i32,
    pub error_message: [u8; 256],
}

impl CudaResult {
    pub fn to_rust_result<T>(self, value: T) -> Result<T> {
        if self.status_code == 0 {
            Ok(value)
        } else {
            let error_msg = std::str::from_utf8(&self.error_message)
                .unwrap_or("Unknown CUDA error")
                .trim_end_matches('\0');
            Err(anyhow!("CUDA Error {}: {}", self.status_code, error_msg))
        }
    }
}

// CUDA side
extern "C" void launch_kernel_safe(
    const void* input,
    void* output,
    CudaResult* result
) {
    cudaError_t err = cudaSuccess;

    // Kernel execution
    kernel<<<grid, block>>>(input, output);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        result->status_code = static_cast<int>(err);
        strncpy(
            reinterpret_cast<char*>(result->error_message),
            cudaGetErrorString(err),
            255
        );
    } else {
        result->status_code = 0;
    }
}
```

---

## 4. API → Engine Integration

### 4.1 REST API Layer

**Location**: `/src/api/`

**Framework**: Axum (async, high-performance)

**Key Endpoints**:
```rust
use axum::{Router, routing::{get, post}, Json};

pub fn create_api_router(engine: Arc<SearchEngine>) -> Router {
    Router::new()
        // Search endpoints
        .route("/api/v1/search/semantic", post(semantic_search_handler))
        .route("/api/v1/search/hybrid", post(hybrid_search_handler))

        // Recommendation endpoints
        .route("/api/v1/recommend/:user_id", get(get_recommendations))
        .route("/api/v1/recommend/personalized", post(personalized_recommend))

        // Ontology endpoints
        .route("/api/v1/ontology/entity/:id", get(get_entity_details))
        .route("/api/v1/ontology/relationships", post(get_relationships))

        // Status endpoints
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/metrics", get(prometheus_metrics))

        .layer(Extension(engine))
        .layer(TraceLayer::new_for_http())
}
```

### 4.2 Request Handler Example

```rust
use axum::{Extension, Json};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct SemanticSearchRequest {
    pub query: String,
    pub filters: Option<SearchFilters>,
    pub top_k: Option<usize>,
    pub use_gpu: Option<bool>,
}

#[derive(Serialize)]
pub struct SemanticSearchResponse {
    pub results: Vec<RankedResult>,
    pub query_time_ms: u64,
    pub total_candidates: usize,
    pub explanation: Option<String>,
}

pub async fn semantic_search_handler(
    Extension(engine): Extension<Arc<SearchEngine>>,
    Json(request): Json<SemanticSearchRequest>,
) -> Result<Json<SemanticSearchResponse>, ApiError> {
    let start = Instant::now();

    // 1. Validate request
    let top_k = request.top_k.unwrap_or(10).min(100);

    // 2. Execute search
    let results = engine
        .semantic_search(&request.query, top_k)
        .await
        .map_err(|e| ApiError::SearchError(e.to_string()))?;

    // 3. Generate explanation (optional)
    let explanation = if request.filters.is_some() {
        Some(engine.explain_results(&results).await?)
    } else {
        None
    };

    Ok(Json(SemanticSearchResponse {
        results,
        query_time_ms: start.elapsed().as_millis() as u64,
        total_candidates: results.len(),
        explanation,
    }))
}
```

### 4.3 WebSocket Real-Time Integration

```rust
use axum::extract::ws::{WebSocket, WebSocketUpgrade, Message};

pub async fn recommendation_stream_handler(
    ws: WebSocketUpgrade,
    Extension(engine): Extension<Arc<SearchEngine>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_recommendation_stream(socket, engine))
}

async fn handle_recommendation_stream(
    mut socket: WebSocket,
    engine: Arc<SearchEngine>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));

    loop {
        tokio::select! {
            // Periodic recommendations
            _ = interval.tick() => {
                let recommendations = engine
                    .get_trending_content()
                    .await
                    .unwrap_or_default();

                let msg = serde_json::to_string(&recommendations).unwrap();
                if socket.send(Message::Text(msg)).await.is_err() {
                    break;
                }
            }

            // Client messages
            Some(msg) = socket.recv() => {
                if let Ok(Message::Text(text)) = msg {
                    // Handle user interactions
                    let _ = engine.record_interaction(&text).await;
                }
            }
        }
    }
}
```

---

## 5. Storage Layer Integration

### 5.1 Hybrid Storage Architecture

```rust
pub struct HybridStorageManager {
    // Vector storage
    vector_store: Box<dyn VectorStore>,

    // Graph database
    neo4j: Neo4jClient,

    // Reinforcement learning
    agent_db: AgentDbClient,

    // GPU engine
    gpu_engine: Arc<GpuEngine>,

    // Configuration
    config: StorageConfig,
}

impl HybridStorageManager {
    pub async fn initialize() -> Result<Self> {
        // 1. Initialize vector store
        let vector_store = match env::var("VECTOR_BACKEND")?.as_str() {
            "milvus" => Box::new(MilvusStore::connect().await?),
            "ruvector" => Box::new(RuVectorStore::new()),
            "faiss" => Box::new(FaissStore::new()?),
            _ => bail!("Unknown vector backend"),
        };

        // 2. Connect to Neo4j
        let neo4j = Neo4jClient::connect(
            &env::var("NEO4J_URI")?,
            &env::var("NEO4J_USER")?,
            &env::var("NEO4J_PASSWORD")?,
        ).await?;

        // 3. Initialize AgentDB
        let agent_db = AgentDbClient::new(
            &env::var("AGENTDB_PATH")?,
        )?;

        // 4. Initialize GPU engine
        let gpu_engine = Arc::new(GpuEngine::new()?);

        Ok(Self {
            vector_store,
            neo4j,
            agent_db,
            gpu_engine,
            config: StorageConfig::default(),
        })
    }
}
```

### 5.2 Multi-Backend Query Execution

```rust
impl HybridStorageManager {
    pub async fn execute_query(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<RankedResult>> {
        // Parallel query execution
        let (vector_results, ontology_results, rl_scores) = tokio::join!(
            // Vector search
            async {
                let query_vec = self.gpu_engine
                    .encode_text(&query.text)
                    .await?;
                self.vector_store.search(&query_vec, query.top_k).await
            },

            // Ontology reasoning
            async {
                if let Some(entity_id) = &query.entity_context {
                    self.neo4j
                        .find_related_entities(entity_id, query.max_depth)
                        .await
                } else {
                    Ok(vec![])
                }
            },

            // RL-based personalization
            async {
                if let Some(user_id) = &query.user_id {
                    self.agent_db
                        .get_user_preferences(user_id)
                        .await
                } else {
                    Ok(HashMap::new())
                }
            }
        );

        // Merge results
        self.merge_and_rank(
            vector_results?,
            ontology_results?,
            rl_scores?,
        ).await
    }
}
```

### 5.3 Data Synchronization

```rust
pub struct DataSyncManager {
    storage: Arc<HybridStorageManager>,
    sync_interval: Duration,
}

impl DataSyncManager {
    pub async fn start_sync_loop(&self) {
        let mut interval = tokio::time::interval(self.sync_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.sync_all_backends().await {
                error!("Sync error: {}", e);
            }
        }
    }

    async fn sync_all_backends(&self) -> Result<()> {
        // 1. Sync vectors: AgentDB → VectorStore
        let new_embeddings = self.storage.agent_db
            .get_new_embeddings()
            .await?;

        for (id, vector) in new_embeddings {
            self.storage.vector_store
                .insert(&id, &vector)
                .await?;
        }

        // 2. Sync relationships: Neo4j → AgentDB
        let relationships = self.storage.neo4j
            .get_recent_relationships(100)
            .await?;

        self.storage.agent_db
            .update_relationship_graph(relationships)
            .await?;

        info!("Sync completed successfully");
        Ok(())
    }
}
```

---

## 6. Testing Integrated System

### 6.1 Component-Level Tests

**GPU Engine Tests**:
```bash
cd /home/devuser/workspace/hackathon-tv5

# Run CUDA tests
./scripts/compile_phase1.sh
./scripts/run_phase1_benchmark.sh

# Run Rust GPU engine tests
cargo test --package media-gateway --lib gpu_engine -- --nocapture
```

**Ontology Layer Tests**:
```bash
# Run ontology reasoning tests
cargo test --package media-gateway --lib ontology -- --nocapture

# Run Neo4j integration tests (requires running Neo4j)
docker-compose up -d neo4j
cargo test --package media-gateway --test ontology_integration
```

**Storage Layer Tests**:
```bash
# Run vector store tests
cargo test --package media-gateway --lib storage -- --nocapture

# Run hybrid storage integration tests
cargo test --package media-gateway --test hybrid_storage_integration
```

### 6.2 Integration Test Suite

**Location**: `/tests/integration/`

**Key Test Scenarios**:
```rust
#[tokio::test]
async fn test_end_to_end_search_pipeline() {
    // 1. Setup
    let config = TestConfig::default();
    let engine = SearchEngine::new(config).await.unwrap();

    // 2. Index test data
    let test_movies = load_test_movies();
    engine.index_batch(&test_movies).await.unwrap();

    // 3. Execute search
    let query = "action movies with car chases";
    let results = engine.semantic_search(query, 10).await.unwrap();

    // 4. Validate results
    assert!(results.len() > 0);
    assert!(results[0].score > 0.7);

    // 5. Check GPU utilization
    let metrics = engine.get_gpu_metrics().await.unwrap();
    assert!(metrics.utilization > 50.0);
}

#[tokio::test]
async fn test_ontology_enrichment() {
    let engine = SearchEngine::new(TestConfig::default()).await.unwrap();

    // Search for entity
    let entity_id = "movie:123";
    let results = engine
        .find_related_entities(entity_id, 2)
        .await
        .unwrap();

    // Validate relationships
    assert!(results.iter().any(|r| r.relationship_type == "director"));
    assert!(results.iter().any(|r| r.relationship_type == "genre"));
}

#[tokio::test]
async fn test_personalized_recommendations() {
    let engine = SearchEngine::new(TestConfig::default()).await.unwrap();

    // Simulate user interactions
    let user_id = "user:456";
    engine.record_view(user_id, "movie:789").await.unwrap();
    engine.record_rating(user_id, "movie:789", 4.5).await.unwrap();

    // Get recommendations
    let recommendations = engine
        .get_personalized_recommendations(user_id, 10)
        .await
        .unwrap();

    // Validate personalization
    assert!(recommendations.len() == 10);
    assert!(recommendations[0].personalization_score > 0.0);
}
```

### 6.3 Performance Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_gpu_similarity(c: &mut Criterion) {
    let engine = GpuEngine::new().unwrap();
    let query = vec![0.5f16; 512];
    let candidates = vec![vec![0.5f16; 512]; 10000];

    c.bench_function("gpu_similarity_10k", |b| {
        b.iter(|| {
            let _ = engine.compute_similarity(
                black_box(&query),
                black_box(&candidates),
            );
        })
    });
}

criterion_group!(benches, benchmark_gpu_similarity);
criterion_main!(benches);
```

---

## 7. Performance Validation

### 7.1 Latency Metrics

**Target Performance** (from design docs):
- Semantic search (100M vectors): **<10ms p99**
- Ontology reasoning: **<50ms**
- GPU similarity compute: **<5ms**
- End-to-end query: **<100ms p99**

**Measurement**:
```rust
use prometheus::{Histogram, register_histogram};

lazy_static! {
    static ref SEARCH_LATENCY: Histogram = register_histogram!(
        "search_latency_seconds",
        "Search query latency"
    ).unwrap();

    static ref GPU_COMPUTE_LATENCY: Histogram = register_histogram!(
        "gpu_compute_latency_seconds",
        "GPU computation latency"
    ).unwrap();
}

impl SearchEngine {
    pub async fn semantic_search(&self, query: &str, k: usize) -> Result<Vec<RankedResult>> {
        let timer = SEARCH_LATENCY.start_timer();

        // Search logic...

        timer.observe_duration();
        Ok(results)
    }
}
```

### 7.2 Throughput Benchmarks

```bash
# Run throughput tests
cd /home/devuser/workspace/hackathon-tv5

# Single-query latency
cargo run --release --bin benchmark -- --mode latency --queries 1000

# Concurrent throughput
cargo run --release --bin benchmark -- --mode throughput --concurrent 100

# GPU utilization test
cargo run --release --bin benchmark -- --mode gpu-stress --duration 60
```

**Expected Output**:
```
=== Latency Benchmark ===
Queries: 1000
Mean: 8.5ms
P50: 7.2ms
P95: 12.3ms
P99: 18.7ms

=== Throughput Benchmark ===
Concurrent clients: 100
Duration: 60s
Total queries: 45,680
QPS: 761.3
Mean latency: 15.2ms

=== GPU Utilization ===
Average GPU utilization: 87%
Memory used: 4.2GB / 24GB
Compute throughput: 156 TFLOPS
```

### 7.3 Monitoring Setup

**Prometheus Metrics**:
```rust
use prometheus::{Encoder, TextEncoder};

pub async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();

    Response::builder()
        .header("Content-Type", "text/plain; version=0.0.4")
        .body(Body::from(buffer))
        .unwrap()
}
```

**Grafana Dashboard**:
```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3000
```

---

## Next Steps

1. **Build and test each component**:
   ```bash
   ./scripts/compile_phase1.sh
   cargo build --release
   cargo test --all
   ```

2. **Deploy infrastructure**:
   ```bash
   docker-compose up -d neo4j milvus prometheus grafana
   ```

3. **Run integration tests**:
   ```bash
   cargo test --test integration -- --nocapture
   ```

4. **Start API server**:
   ```bash
   cargo run --release --bin api-server
   ```

5. **Validate performance**:
   ```bash
   cargo run --release --bin benchmark
   ```

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## Further Reading

- [FFI Integration Guide](FFI_INTEGRATION_GUIDE.md)
- [CUDA Development Guide](cuda/IMPLEMENTATION_SUMMARY.md)
- [Performance Optimization](T4_OPTIMIZATION_GUIDE.md)
- [Deployment Guide](deployment-guide.md)
