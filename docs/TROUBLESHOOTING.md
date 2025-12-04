# Troubleshooting Guide

**Media Gateway Hackathon - Common Issues and Solutions**

## Table of Contents
1. [CUDA Errors](#cuda-errors)
2. [Rust Compilation Issues](#rust-compilation-issues)
3. [API Errors](#api-errors)
4. [Performance Issues](#performance-issues)
5. [Memory Issues](#memory-issues)
6. [Network and Database Issues](#network-and-database-issues)
7. [FFI Integration Problems](#ffi-integration-problems)
8. [Deployment Issues](#deployment-issues)

---

## 1. CUDA Errors

### Error: `CUDA driver version is insufficient`

**Symptoms**:
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

**Cause**: NVIDIA driver version doesn't support the CUDA toolkit version.

**Solution**:
```bash
# Check current driver version
nvidia-smi

# Check CUDA version requirement
nvcc --version

# Update NVIDIA drivers (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install --reinstall nvidia-driver-535

# Reboot
sudo reboot

# Verify
nvidia-smi
```

**Alternative**: Downgrade CUDA toolkit to match driver:
```bash
# Remove current CUDA
sudo apt-get remove --purge cuda-*

# Install compatible version
sudo apt-get install cuda-toolkit-11-8
```

---

### Error: `out of memory` during kernel execution

**Symptoms**:
```
CUDA error 2: out of memory
```

**Cause**: Trying to allocate more GPU memory than available.

**Solution 1**: Reduce batch size
```rust
// In your code
const BATCH_SIZE: usize = 32; // Reduce from 128
const MAX_CANDIDATES: usize = 1000; // Reduce from 10000
```

**Solution 2**: Enable memory pools and caching
```rust
use cudarc::driver::CudaDevice;

let device = CudaDevice::new(0)?;

// Set memory pool size (in bytes)
unsafe {
    cuda_sys::cuCtxSetLimit(
        cuda_sys::CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE,
        1024 * 1024 * 1024, // 1GB
    );
}
```

**Solution 3**: Use streaming and chunking
```rust
pub async fn process_large_batch(
    &self,
    data: &[f16],
    chunk_size: usize,
) -> Result<Vec<f32>> {
    let mut results = Vec::new();

    for chunk in data.chunks(chunk_size) {
        let chunk_results = self.process_chunk(chunk).await?;
        results.extend(chunk_results);

        // Allow GPU to free memory
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    Ok(results)
}
```

**Solution 4**: Clear GPU memory cache
```bash
# Kill all processes using GPU
sudo fuser -k /dev/nvidia*

# Reset GPU
sudo nvidia-smi --gpu-reset
```

---

### Error: `no kernel image is available for execution`

**Symptoms**:
```
CUDA error 209: no kernel image is available for execution on the device
```

**Cause**: Kernel compiled for wrong GPU architecture.

**Solution**: Check GPU compute capability and recompile:
```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Example output: 8.0 (for A100)

# Recompile with correct architecture
cd /home/devuser/workspace/hackathon-tv5/src/cuda

# For Turing (T4): sm_75
nvcc -arch=sm_75 -o kernel.o kernel.cu

# For Ampere (A100): sm_80
nvcc -arch=sm_80 -o kernel.o kernel.cu

# For Hopper (H100): sm_90
nvcc -arch=sm_90 -o kernel.o kernel.cu

# Or use compilation script
./scripts/compile_phase1.sh
```

**Update build script** (`scripts/compile_phase1.sh`):
```bash
# Auto-detect compute capability
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')

nvcc -arch=sm_${COMPUTE_CAP} \
     -I./include \
     -O3 \
     -use_fast_math \
     -o kernel.o \
     kernel.cu
```

---

### Error: Tensor core operations failing

**Symptoms**:
```
Invalid configuration for WMMA operations
```

**Cause**: Incorrect matrix dimensions or unsupported data types.

**Solution**: Verify tensor core requirements
```cuda
// Tensor cores require specific tile sizes
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Verify dimensions are multiples of tile size
__device__ void verify_dimensions(int M, int N, int K) {
    assert(M % TILE_M == 0);
    assert(N % TILE_N == 0);
    assert(K % TILE_K == 0);
}

// Use correct data types
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
```

**Verify GPU support**:
```bash
# Check if GPU supports tensor cores
nvidia-smi -q | grep "Compute Capability"

# Tensor cores available on:
# - Volta (sm_70): V100
# - Turing (sm_75): T4, RTX 2080
# - Ampere (sm_80): A100, RTX 3090
# - Hopper (sm_90): H100
```

---

## 2. Rust Compilation Issues

### Error: `linking with cc failed`

**Symptoms**:
```
error: linking with `cc` failed: exit status: 1
note: /usr/bin/ld: cannot find -lcudart
```

**Cause**: CUDA libraries not in linker path.

**Solution**:
```bash
# Add CUDA libraries to path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

# Make permanent
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify libraries exist
ls /usr/local/cuda/lib64/libcudart.so

# Rebuild
cargo clean
cargo build --release
```

---

### Error: `cudarc` build failures

**Symptoms**:
```
error: failed to run custom build command for `cudarc`
```

**Solution 1**: Ensure CUDA is properly installed
```bash
# Check CUDA installation
which nvcc
# Should output: /usr/local/cuda/bin/nvcc

# If not found, install CUDA
sudo apt-get install cuda-toolkit-12-3
```

**Solution 2**: Update Cargo.toml dependencies
```toml
[dependencies]
cudarc = { version = "0.10", features = ["cuda-12030"] }

[build-dependencies]
cc = "1.0"
```

**Solution 3**: Clear build cache
```bash
cargo clean
rm -rf target/
cargo build --release
```

---

### Error: FFI struct size mismatch

**Symptoms**:
```
assertion failed: std::mem::size_of::<MediaOntologyNode>() == 2072
```

**Cause**: Struct padding differs between Rust and CUDA.

**Solution**: Verify struct definitions match exactly

**Rust side** (`src/rust/models/ontology_ffi.rs`):
```rust
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MediaOntologyNode {
    pub id: u64,              // Offset: 0
    pub node_type: u32,       // Offset: 8
    pub confidence: f32,      // Offset: 12
    pub embedding: [f32; 512],// Offset: 16
    pub metadata_offset: u32, // Offset: 2064
    pub metadata_length: u32, // Offset: 2068
    _padding: [u8; 8],        // Offset: 2072
}

// Add debug output
const _: () = {
    println!("Size: {}", std::mem::size_of::<MediaOntologyNode>());
    println!("Align: {}", std::mem::align_of::<MediaOntologyNode>());
};
```

**CUDA side** (`src/cuda/kernels/ontology_ffi_check.cuh`):
```cuda
#include <cstdint>
#include <cstdio>

struct MediaOntologyNode {
    uint64_t id;
    uint32_t node_type;
    float confidence;
    float embedding[512];
    uint32_t metadata_offset;
    uint32_t metadata_length;
    uint8_t _padding[8];
} __attribute__((packed));

// Add debug output
static void verify_struct_layout() {
    printf("Size: %zu\n", sizeof(MediaOntologyNode));
    printf("Align: %zu\n", alignof(MediaOntologyNode));
    printf("id offset: %zu\n", offsetof(MediaOntologyNode, id));
    printf("embedding offset: %zu\n", offsetof(MediaOntologyNode, embedding));
}
```

**Run verification**:
```bash
# Rust side
cargo test ontology_ffi::tests::test_struct_layout -- --nocapture

# CUDA side
cd src/cuda
nvcc -run kernels/ontology_ffi_check.cu
```

---

## 3. API Errors

### Error: `Connection refused` to API server

**Symptoms**:
```
Error: Connection refused (os error 111)
```

**Cause**: API server not running or wrong port.

**Solution**:
```bash
# Check if server is running
ps aux | grep api-server

# Check port binding
sudo netstat -tlnp | grep 8080

# Start server
cargo run --release --bin api-server

# Or with custom port
API_PORT=8081 cargo run --release --bin api-server

# Test connection
curl http://localhost:8080/api/v1/health
```

---

### Error: `Request timeout`

**Symptoms**:
```
Error: Request timeout after 30s
```

**Cause**: Query taking too long, possibly GPU initialization.

**Solution 1**: Increase timeout
```rust
// In API client
let client = reqwest::Client::builder()
    .timeout(Duration::from_secs(120)) // Increase from 30s
    .build()?;
```

**Solution 2**: Warm up GPU on startup
```rust
impl SearchEngine {
    pub async fn new() -> Result<Self> {
        let engine = Self { /* ... */ };

        // Warm up GPU
        info!("Warming up GPU...");
        let dummy_query = vec![0.5f16; 512];
        let _ = engine.gpu_engine
            .compute_similarity(&dummy_query, &dummy_query)
            .await?;
        info!("GPU ready");

        Ok(engine)
    }
}
```

**Solution 3**: Add query queueing
```rust
use tokio::sync::Semaphore;

pub struct SearchEngine {
    query_semaphore: Arc<Semaphore>,
    // ...
}

impl SearchEngine {
    pub async fn semantic_search(&self, query: &str) -> Result<Vec<RankedResult>> {
        // Limit concurrent queries
        let _permit = self.query_semaphore.acquire().await?;

        // Execute search
        // ...
    }
}
```

---

### Error: Invalid JSON response

**Symptoms**:
```
Error: expected value at line 1 column 1
```

**Cause**: API returning error HTML instead of JSON.

**Solution**: Add proper error handling
```rust
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

pub enum ApiError {
    SearchError(String),
    DatabaseError(String),
    ValidationError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::SearchError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ApiError::DatabaseError(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
            ApiError::ValidationError(msg) => (StatusCode::BAD_REQUEST, msg),
        };

        let body = Json(serde_json::json!({
            "error": message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}
```

---

## 4. Performance Issues

### Issue: Query latency >100ms

**Symptoms**: Queries taking longer than expected.

**Diagnosis**:
```rust
use tracing::{info, instrument};

#[instrument(skip(self))]
pub async fn semantic_search(&self, query: &str) -> Result<Vec<RankedResult>> {
    let total_start = Instant::now();

    // 1. Embedding generation
    let embed_start = Instant::now();
    let query_vec = self.gpu_engine.encode_text(query).await?;
    info!("Embedding: {:?}", embed_start.elapsed());

    // 2. Vector search
    let search_start = Instant::now();
    let candidates = self.vector_store.search(&query_vec, 100).await?;
    info!("Vector search: {:?}", search_start.elapsed());

    // 3. Ontology enrichment
    let ontology_start = Instant::now();
    let enriched = self.ontology.enrich_results(candidates).await?;
    info!("Ontology: {:?}", ontology_start.elapsed());

    // 4. Ranking
    let rank_start = Instant::now();
    let ranked = self.gpu_engine.rank_results(query_vec, enriched).await?;
    info!("Ranking: {:?}", rank_start.elapsed());

    info!("Total: {:?}", total_start.elapsed());
    Ok(ranked)
}
```

**Solution 1**: Enable query caching
```rust
use lru::LruCache;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct SearchEngine {
    query_cache: Arc<RwLock<LruCache<String, Vec<RankedResult>>>>,
    // ...
}

impl SearchEngine {
    pub async fn semantic_search(&self, query: &str) -> Result<Vec<RankedResult>> {
        // Check cache
        if let Some(cached) = self.query_cache.read().await.get(query) {
            return Ok(cached.clone());
        }

        // Execute search
        let results = self.execute_search(query).await?;

        // Cache results
        self.query_cache.write().await.put(query.to_string(), results.clone());

        Ok(results)
    }
}
```

**Solution 2**: Optimize vector search index
```rust
// Use HNSW instead of flat index
use faiss::{Index, IndexHNSW};

let mut index = IndexHNSW::new(512, 32)?; // dim=512, M=32
index.set_efConstruction(200);
index.set_efSearch(100);
```

**Solution 3**: Batch queries
```rust
pub async fn batch_search(
    &self,
    queries: &[String],
) -> Result<Vec<Vec<RankedResult>>> {
    // Generate embeddings in parallel
    let query_vecs = futures::stream::iter(queries)
        .map(|q| self.gpu_engine.encode_text(q))
        .buffer_unordered(10)
        .collect::<Vec<_>>()
        .await;

    // Single GPU call for all queries
    let results = self.vector_store
        .batch_search(&query_vecs, 10)
        .await?;

    Ok(results)
}
```

---

### Issue: Low GPU utilization

**Symptoms**: GPU utilization <50% during queries.

**Diagnosis**:
```bash
# Monitor GPU in real-time
nvidia-smi dmon -s u

# Expected output for good utilization:
# Idx  GPU  Utilization
#   0    0       85%
```

**Solution 1**: Increase batch size
```rust
// Process queries in larger batches
const OPTIMAL_BATCH_SIZE: usize = 128; // Tune based on GPU

pub async fn process_queries(&self, queries: &[String]) -> Result<Vec<Vec<RankedResult>>> {
    let batches = queries.chunks(OPTIMAL_BATCH_SIZE);

    let results = futures::stream::iter(batches)
        .then(|batch| self.process_batch(batch))
        .collect::<Vec<_>>()
        .await;

    Ok(results)
}
```

**Solution 2**: Use async streams for GPU work
```rust
use cudarc::driver::CudaStream;

pub struct GpuEngine {
    streams: Vec<CudaStream>,
    // ...
}

impl GpuEngine {
    pub async fn parallel_compute(&self, data: &[Vec<f16>]) -> Result<Vec<Vec<f32>>> {
        let stream_count = self.streams.len();

        let futures = data
            .chunks(data.len() / stream_count)
            .zip(self.streams.iter())
            .map(|(chunk, stream)| {
                self.compute_on_stream(chunk, stream)
            });

        futures::future::try_join_all(futures).await
    }
}
```

---

## 5. Memory Issues

### Issue: Memory leak in long-running service

**Symptoms**: Memory usage grows over time.

**Diagnosis**:
```bash
# Monitor memory usage
watch -n 1 'ps aux | grep api-server | grep -v grep'

# Check for GPU memory leaks
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

**Solution 1**: Implement proper cleanup
```rust
impl Drop for GpuEngine {
    fn drop(&mut self) {
        // Synchronize before cleanup
        if let Err(e) = self.device.synchronize() {
            error!("Failed to sync device: {}", e);
        }

        // Free GPU memory
        info!("Cleaning up GPU resources");
    }
}
```

**Solution 2**: Limit cache sizes
```rust
pub struct SearchEngine {
    query_cache: Arc<RwLock<LruCache<String, Vec<RankedResult>>>>,
    embedding_cache: Arc<RwLock<LruCache<String, Vec<f32>>>>,
}

impl SearchEngine {
    pub fn new() -> Self {
        Self {
            query_cache: Arc::new(RwLock::new(LruCache::new(1000))), // Max 1000 queries
            embedding_cache: Arc::new(RwLock::new(LruCache::new(5000))), // Max 5000 embeddings
        }
    }
}
```

**Solution 3**: Periodic cleanup task
```rust
pub async fn start_cleanup_task(engine: Arc<SearchEngine>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 min

        loop {
            interval.tick().await;

            // Clear caches
            engine.query_cache.write().await.clear();
            engine.embedding_cache.write().await.clear();

            // Force GPU cleanup
            if let Err(e) = engine.gpu_engine.synchronize_and_cleanup().await {
                error!("Cleanup failed: {}", e);
            }

            info!("Cleanup completed");
        }
    });
}
```

---

## 6. Network and Database Issues

### Issue: Neo4j connection failures

**Symptoms**:
```
Error: Failed to connect to Neo4j at bolt://localhost:7687
```

**Solution**:
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Start Neo4j
docker-compose up -d neo4j

# Check logs
docker-compose logs neo4j

# Test connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p password

# Verify Rust connection
cargo test --package media-gateway --lib ontology::neo4j_client::tests
```

---

### Issue: Milvus vector search errors

**Symptoms**:
```
Error: Collection 'media_embeddings' not found
```

**Solution**:
```bash
# Check Milvus status
docker-compose ps milvus

# Create collection
docker exec -it milvus python3 << EOF
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
]

schema = CollectionSchema(fields)
collection = Collection("media_embeddings", schema)

print("Collection created successfully")
EOF

# Verify in Rust
cargo test --package media-gateway --lib storage::milvus::tests::test_collection_exists
```

---

## 7. FFI Integration Problems

### Issue: Segmentation fault at FFI boundary

**Symptoms**:
```
Segmentation fault (core dumped)
```

**Diagnosis**:
```bash
# Run with backtrace
RUST_BACKTRACE=full cargo run

# Use gdb
gdb --args target/debug/api-server
(gdb) run
(gdb) bt  # After crash
```

**Common Causes & Solutions**:

**1. Null pointer dereference**:
```rust
// BAD
let ptr = std::ptr::null_mut();
unsafe { cuda_function(ptr) }; // Segfault!

// GOOD
let data = vec![0f32; 512];
let ptr = data.as_ptr();
unsafe { cuda_function(ptr) };
```

**2. Dangling pointer**:
```rust
// BAD
let data = {
    let temp = vec![0f32; 512];
    temp.as_ptr() // temp dropped here!
};
unsafe { cuda_function(data) }; // Segfault!

// GOOD
let data = vec![0f32; 512];
unsafe { cuda_function(data.as_ptr()) };
// data lives until after CUDA call
```

**3. Incorrect lifetime**:
```rust
// GOOD - Pin data while GPU uses it
use std::pin::Pin;

let data = vec![0f32; 512];
let pinned = Pin::new(&data);

unsafe {
    cuda_function(pinned.as_ref().as_ptr());
}
// Data guaranteed to live through CUDA call
```

---

## 8. Deployment Issues

### Issue: Docker build failures

**Symptoms**:
```
ERROR: failed to solve: process "/bin/sh -c cargo build --release" did not complete successfully
```

**Solution**: Multi-stage build with caching
```dockerfile
# Stage 1: Build dependencies
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Cache dependencies
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/lib.rs
RUN cargo build --release
RUN rm -rf src

# Build application
COPY . .
RUN cargo build --release

# Stage 2: Runtime
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

COPY --from=builder /app/target/release/api-server /usr/local/bin/

CMD ["api-server"]
```

---

### Issue: CUDA not available in production

**Symptoms**:
```
Error: No CUDA-capable device found
```

**Solution**: Verify runtime configuration
```bash
# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Update docker-compose.yml
version: '3.8'
services:
  api-server:
    image: media-gateway:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Getting Help

If issues persist:

1. **Check logs**:
   ```bash
   # Application logs
   tail -f /var/log/media-gateway/api-server.log

   # Docker logs
   docker-compose logs -f api-server

   # System logs
   journalctl -u media-gateway -f
   ```

2. **Collect diagnostics**:
   ```bash
   ./scripts/collect_diagnostics.sh
   ```

3. **Report issue** with:
   - Error messages and stack traces
   - GPU model and driver version
   - CUDA toolkit version
   - Rust version (`rustc --version`)
   - Diagnostic output

4. **Community support**:
   - GitHub Issues: https://github.com/agenticsorg/hackathon-tv5/issues
   - Discord: https://discord.agentics.org

---

## See Also

- [Integration Guide](INTEGRATION_GUIDE_V2.md)
- [FFI Integration](FFI_INTEGRATION_GUIDE.md)
- [Performance Tuning](T4_OPTIMIZATION_GUIDE.md)
- [Deployment Guide](deployment-guide.md)
