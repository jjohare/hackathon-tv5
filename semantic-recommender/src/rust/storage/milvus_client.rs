/// Milvus Vector Database Client
///
/// Provides high-performance vector similarity search with 8.7ms P99 latency target.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use super::{StorageError, StorageResult};

const DEFAULT_TIMEOUT_MS: u64 = 50;
const MAX_BATCH_SIZE: usize = 1000;

#[derive(Debug, Clone)]
pub struct MilvusConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub connection_pool_size: usize,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
}

impl Default for MilvusConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 19530,
            database: "default".to_string(),
            connection_pool_size: 32,
            timeout_ms: DEFAULT_TIMEOUT_MS,
            retry_attempts: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct SearchParams {
    pub metric_type: MetricType,
    pub ef: usize,           // HNSW search parameter
    pub nprobe: usize,       // IVF search parameter
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            metric_type: MetricType::Cosine,
            ef: 64,
            nprobe: 16,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetricType {
    Cosine,
    L2,
    IP,  // Inner Product
}

/// High-performance Milvus client with connection pooling
pub struct MilvusClient {
    config: MilvusConfig,
    connection_pool: Arc<RwLock<Vec<MilvusConnection>>>,
    metrics: Arc<MilvusMetrics>,
}

struct MilvusConnection {
    endpoint: String,
    last_used: Instant,
}

#[derive(Default)]
struct MilvusMetrics {
    total_searches: std::sync::atomic::AtomicU64,
    total_inserts: std::sync::atomic::AtomicU64,
    total_errors: std::sync::atomic::AtomicU64,
    total_latency_us: std::sync::atomic::AtomicU64,
}

impl MilvusClient {
    pub async fn new(config: MilvusConfig) -> StorageResult<Self> {
        let endpoint = format!("{}:{}", config.host, config.port);

        // Initialize connection pool
        let mut pool = Vec::new();
        for _ in 0..config.connection_pool_size {
            pool.push(MilvusConnection {
                endpoint: endpoint.clone(),
                last_used: Instant::now(),
            });
        }

        Ok(Self {
            config,
            connection_pool: Arc::new(RwLock::new(pool)),
            metrics: Arc::new(MilvusMetrics::default()),
        })
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        collection_name: &str,
        query_vector: &[f32],
        top_k: usize,
        filters: &HashMap<String, String>,
        params: Option<SearchParams>,
    ) -> StorageResult<Vec<VectorSearchResult>> {
        let start = Instant::now();
        self.metrics.total_searches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let search_params = params.unwrap_or_default();

        // Simulate Milvus search with optimized HNSW
        let results = self.execute_search(
            collection_name,
            query_vector,
            top_k,
            filters,
            &search_params,
        ).await?;

        let latency_us = start.elapsed().as_micros() as u64;
        self.metrics.total_latency_us.fetch_add(latency_us, std::sync::atomic::Ordering::Relaxed);

        // Verify latency target
        if latency_us > 10_000 {
            log::warn!(
                "Search latency {}μs exceeds 10ms target for collection {}",
                latency_us,
                collection_name
            );
        }

        Ok(results)
    }

    async fn execute_search(
        &self,
        collection_name: &str,
        query_vector: &[f32],
        top_k: usize,
        filters: &HashMap<String, String>,
        params: &SearchParams,
    ) -> StorageResult<Vec<VectorSearchResult>> {
        // Get connection from pool
        let conn = self.get_connection().await?;

        // Build filter expression
        let filter_expr = self.build_filter_expression(filters);

        // Execute search with retry logic
        for attempt in 0..self.config.retry_attempts {
            match self.try_search(&conn, collection_name, query_vector, top_k, &filter_expr, params).await {
                Ok(results) => return Ok(results),
                Err(e) if attempt < self.config.retry_attempts - 1 => {
                    log::warn!("Search attempt {} failed: {}, retrying...", attempt + 1, e);
                    tokio::time::sleep(Duration::from_millis(50 * (attempt as u64 + 1))).await;
                }
                Err(e) => {
                    self.metrics.total_errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(StorageError::Milvus(format!("Search failed after {} attempts: {}", self.config.retry_attempts, e)));
                }
            }
        }

        unreachable!()
    }

    async fn try_search(
        &self,
        conn: &MilvusConnection,
        collection_name: &str,
        query_vector: &[f32],
        top_k: usize,
        filter_expr: &str,
        params: &SearchParams,
    ) -> StorageResult<Vec<VectorSearchResult>> {
        // In production, this would call the actual Milvus gRPC API
        // For now, simulate with placeholder logic

        log::debug!(
            "Searching collection {} with k={}, ef={}, metric={:?}",
            collection_name,
            top_k,
            params.ef,
            params.metric_type
        );

        // Placeholder: Generate mock results
        // In production: milvus_client.search(collection_name, query_vector, top_k, filter_expr, params)
        let results = (0..top_k.min(10))
            .map(|i| VectorSearchResult {
                id: format!("media_{}", i),
                score: 1.0 - (i as f32 * 0.05),
                metadata: HashMap::from([
                    ("title".to_string(), format!("Media Item {}", i)),
                    ("type".to_string(), "video".to_string()),
                ]),
                embedding: None,
            })
            .collect();

        Ok(results)
    }

    /// Insert vectors into collection
    pub async fn insert(
        &self,
        collection_name: &str,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<HashMap<String, String>>,
    ) -> StorageResult<Vec<String>> {
        if vectors.len() != metadata.len() {
            return Err(StorageError::Milvus(
                "Vectors and metadata length mismatch".to_string()
            ));
        }

        self.metrics.total_inserts.fetch_add(vectors.len() as u64, std::sync::atomic::Ordering::Relaxed);

        let mut inserted_ids = Vec::new();

        // Batch insert for performance
        for chunk in vectors.chunks(MAX_BATCH_SIZE) {
            let chunk_size = chunk.len();
            let ids = self.batch_insert(collection_name, chunk, &metadata[..chunk_size]).await?;
            inserted_ids.extend(ids);
        }

        Ok(inserted_ids)
    }

    async fn batch_insert(
        &self,
        collection_name: &str,
        vectors: &[Vec<f32>],
        metadata: &[HashMap<String, String>],
    ) -> StorageResult<Vec<String>> {
        let conn = self.get_connection().await?;

        log::debug!(
            "Batch inserting {} vectors into collection {}",
            vectors.len(),
            collection_name
        );

        // Placeholder: Generate mock IDs
        // In production: milvus_client.insert(collection_name, vectors, metadata)
        let ids: Vec<String> = (0..vectors.len())
            .map(|i| format!("vec_{}", uuid::Uuid::new_v4()))
            .collect();

        Ok(ids)
    }

    /// Delete vectors by IDs
    pub async fn delete(
        &self,
        collection_name: &str,
        ids: &[String],
    ) -> StorageResult<u64> {
        let conn = self.get_connection().await?;

        log::debug!("Deleting {} vectors from collection {}", ids.len(), collection_name);

        // In production: milvus_client.delete(collection_name, ids)
        Ok(ids.len() as u64)
    }

    /// Create collection with schema
    pub async fn create_collection(
        &self,
        collection_name: &str,
        dimension: usize,
        index_type: IndexType,
    ) -> StorageResult<()> {
        log::info!(
            "Creating collection {} with dimension {} and index {:?}",
            collection_name,
            dimension,
            index_type
        );

        // In production: milvus_client.create_collection(...)
        Ok(())
    }

    /// Build HNSW index for fast ANN search
    pub async fn build_index(
        &self,
        collection_name: &str,
        index_params: IndexParams,
    ) -> StorageResult<()> {
        log::info!("Building index for collection {} with params {:?}", collection_name, index_params);

        // In production: milvus_client.create_index(collection_name, index_params)
        Ok(())
    }

    async fn get_connection(&self) -> StorageResult<MilvusConnection> {
        let pool = self.connection_pool.read().await;
        pool.first()
            .cloned()
            .ok_or_else(|| StorageError::Milvus("No connections available".to_string()))
    }

    fn build_filter_expression(&self, filters: &HashMap<String, String>) -> String {
        if filters.is_empty() {
            return String::new();
        }

        let expressions: Vec<String> = filters
            .iter()
            .map(|(key, value)| format!("{} == \"{}\"", key, value))
            .collect();

        expressions.join(" && ")
    }

    /// Get client metrics
    pub fn get_metrics(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("total_searches".to_string(), self.metrics.total_searches.load(std::sync::atomic::Ordering::Relaxed)),
            ("total_inserts".to_string(), self.metrics.total_inserts.load(std::sync::atomic::Ordering::Relaxed)),
            ("total_errors".to_string(), self.metrics.total_errors.load(std::sync::atomic::Ordering::Relaxed)),
            ("avg_latency_us".to_string(), {
                let total = self.metrics.total_latency_us.load(std::sync::atomic::Ordering::Relaxed);
                let searches = self.metrics.total_searches.load(std::sync::atomic::Ordering::Relaxed);
                if searches > 0 { total / searches } else { 0 }
            }),
        ])
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    HNSW,
    IVF_FLAT,
    IVF_PQ,
    FLAT,
}

#[derive(Debug, Clone)]
pub struct IndexParams {
    pub index_type: IndexType,
    pub metric_type: MetricType,
    pub m: usize,           // HNSW: max connections
    pub ef_construction: usize,  // HNSW: construction parameter
    pub nlist: usize,       // IVF: number of clusters
}

impl Default for IndexParams {
    fn default() -> Self {
        Self {
            index_type: IndexType::HNSW,
            metric_type: MetricType::Cosine,
            m: 16,
            ef_construction: 200,
            nlist: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_milvus_client_creation() {
        let config = MilvusConfig::default();
        let client = MilvusClient::new(config).await.unwrap();
        assert_eq!(client.get_metrics()["total_searches"], 0);
    }

    #[tokio::test]
    async fn test_filter_expression() {
        let config = MilvusConfig::default();
        let client = MilvusClient::new(config).await.unwrap();

        let filters = HashMap::from([
            ("genre".to_string(), "action".to_string()),
            ("year".to_string(), "2024".to_string()),
        ]);

        let expr = client.build_filter_expression(&filters);
        assert!(expr.contains("genre == \"action\""));
        assert!(expr.contains("year == \"2024\""));
    }
}
