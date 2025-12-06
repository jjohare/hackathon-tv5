use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, warn, error};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaContent {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    Neo4jOnly,
    DualWrite,     // Write both, read from Neo4j
    ShadowMilvus,  // Write both, read from Milvus with Neo4j fallback
    Hybrid,        // Read from Milvus, write to Milvus only
}

impl StorageMode {
    pub fn from_env() -> Self {
        match std::env::var("STORAGE_MODE").as_deref() {
            Ok("dual_write") => StorageMode::DualWrite,
            Ok("shadow_milvus") => StorageMode::ShadowMilvus,
            Ok("hybrid") => StorageMode::Hybrid,
            _ => StorageMode::Neo4jOnly,
        }
    }
}

/// Metrics tracking for dual-write mode
#[derive(Debug, Default, Clone, Serialize)]
pub struct DualWriteMetrics {
    pub neo4j_writes: u64,
    pub neo4j_errors: u64,
    pub milvus_writes: u64,
    pub milvus_errors: u64,
    pub milvus_latency_ms: Vec<f64>,
    pub neo4j_latency_ms: Vec<f64>,
    pub timestamp: i64,
}

impl DualWriteMetrics {
    pub fn average_milvus_latency(&self) -> f64 {
        if self.milvus_latency_ms.is_empty() {
            0.0
        } else {
            self.milvus_latency_ms.iter().sum::<f64>() / self.milvus_latency_ms.len() as f64
        }
    }

    pub fn average_neo4j_latency(&self) -> f64 {
        if self.neo4j_latency_ms.is_empty() {
            0.0
        } else {
            self.neo4j_latency_ms.iter().sum::<f64>() / self.neo4j_latency_ms.len() as f64
        }
    }

    pub fn milvus_success_rate(&self) -> f64 {
        if self.milvus_writes == 0 {
            0.0
        } else {
            (self.milvus_writes - self.milvus_errors) as f64 / self.milvus_writes as f64
        }
    }
}

/// Temporary coordinator during migration: write to both Neo4j and Milvus
pub struct DualWriteCoordinator {
    neo4j: Arc<Neo4jClient>,
    milvus: Option<Arc<MilvusClient>>,
    mode: StorageMode,
    metrics: Arc<Mutex<DualWriteMetrics>>,
}

impl DualWriteCoordinator {
    pub fn new(
        neo4j: Arc<Neo4jClient>,
        milvus: Option<Arc<MilvusClient>>,
        mode: StorageMode,
    ) -> Self {
        Self {
            neo4j,
            milvus,
            mode,
            metrics: Arc::new(Mutex::new(DualWriteMetrics::default())),
        }
    }

    pub async fn insert_embedding(&self, content: &MediaContent) -> Result<()> {
        match self.mode {
            StorageMode::Neo4jOnly => {
                self.write_neo4j(content).await?;
            },
            StorageMode::DualWrite => {
                // Write to both, but fail if Neo4j fails (primary)
                self.write_neo4j(content).await?;
                self.write_milvus_best_effort(content).await;
            },
            StorageMode::ShadowMilvus | StorageMode::Hybrid => {
                // Write to Milvus as primary
                self.write_milvus(content).await?;
                // Neo4j as backup (best effort)
                self.write_neo4j_best_effort(content).await;
            },
        }

        Ok(())
    }

    pub async fn search_similar(&self, query: &[f32], top_k: usize) -> Result<Vec<MediaContent>> {
        match self.mode {
            StorageMode::Neo4jOnly | StorageMode::DualWrite => {
                // Read from Neo4j (primary during migration)
                self.neo4j.search_similar(query, top_k).await
            },
            StorageMode::ShadowMilvus => {
                // Read from Milvus with fallback to Neo4j
                match self.milvus.as_ref() {
                    Some(milvus) => {
                        match milvus.search_similar(query, top_k).await {
                            Ok(results) => Ok(results),
                            Err(e) => {
                                warn!("Milvus search failed, falling back to Neo4j: {}", e);
                                self.neo4j.search_similar(query, top_k).await
                            }
                        }
                    },
                    None => self.neo4j.search_similar(query, top_k).await,
                }
            },
            StorageMode::Hybrid => {
                // Read from Milvus only (post-migration)
                match self.milvus.as_ref() {
                    Some(milvus) => milvus.search_similar(query, top_k).await,
                    None => anyhow::bail!("Milvus not configured in Hybrid mode"),
                }
            },
        }
    }

    pub async fn get_metrics(&self) -> DualWriteMetrics {
        self.metrics.lock().await.clone()
    }

    async fn write_neo4j(&self, content: &MediaContent) -> Result<()> {
        let start = std::time::Instant::now();

        let result = self.neo4j.insert_embedding(content).await;

        let latency = start.elapsed().as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.lock().await;
        metrics.neo4j_writes += 1;
        metrics.neo4j_latency_ms.push(latency);

        if result.is_err() {
            metrics.neo4j_errors += 1;
        }

        result
    }

    async fn write_neo4j_best_effort(&self, content: &MediaContent) {
        if let Err(e) = self.write_neo4j(content).await {
            warn!("Neo4j best-effort write failed: {}", e);
        }
    }

    async fn write_milvus(&self, content: &MediaContent) -> Result<()> {
        let milvus = self.milvus.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Milvus not configured"))?;

        let start = std::time::Instant::now();

        let result = milvus.insert_embedding(content).await;

        let latency = start.elapsed().as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.lock().await;
        metrics.milvus_writes += 1;
        metrics.milvus_latency_ms.push(latency);

        if result.is_err() {
            metrics.milvus_errors += 1;
        }

        result
    }

    async fn write_milvus_best_effort(&self, content: &MediaContent) {
        if let Some(milvus) = &self.milvus {
            let start = std::time::Instant::now();

            match milvus.insert_embedding(content).await {
                Ok(_) => {
                    let latency = start.elapsed().as_secs_f64() * 1000.0;
                    let mut metrics = self.metrics.lock().await;
                    metrics.milvus_writes += 1;
                    metrics.milvus_latency_ms.push(latency);

                    debug!("Milvus shadow write succeeded: {} ({}ms)", content.id, latency);
                },
                Err(e) => {
                    let mut metrics = self.metrics.lock().await;
                    metrics.milvus_errors += 1;

                    warn!("Milvus shadow write failed for {}: {}", content.id, e);
                }
            }
        }
    }
}

// Mock client implementations (replace with actual clients in production)

pub struct Neo4jClient;

impl Neo4jClient {
    pub async fn insert_embedding(&self, content: &MediaContent) -> Result<()> {
        // In production:
        // self.graph.run(
        //     neo4rs::query("MERGE (m:MediaContent {id: $id}) SET m.embedding = $embedding, m.metadata = $metadata")
        //         .param("id", &content.id)
        //         .param("embedding", &content.embedding)
        //         .param("metadata", &content.metadata)
        // ).await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        Ok(())
    }

    pub async fn search_similar(&self, _query: &[f32], _top_k: usize) -> Result<Vec<MediaContent>> {
        // In production: use Neo4j vector index
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok(vec![])
    }
}

pub struct MilvusClient;

impl MilvusClient {
    pub async fn insert_embedding(&self, content: &MediaContent) -> Result<()> {
        // In production:
        // let data = vec![
        //     Field::new("id", vec![content.id.clone()]),
        //     Field::new("embedding", vec![content.embedding.clone()]),
        //     Field::new("metadata", vec![content.metadata.to_string()]),
        // ];
        // self.client.insert("media_embeddings", data).await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        Ok(())
    }

    pub async fn search_similar(&self, _query: &[f32], _top_k: usize) -> Result<Vec<MediaContent>> {
        // In production: use Milvus HNSW search
        tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dual_write_mode() {
        let neo4j = Arc::new(Neo4jClient);
        let milvus = Some(Arc::new(MilvusClient));

        let coordinator = DualWriteCoordinator::new(neo4j, milvus, StorageMode::DualWrite);

        let content = MediaContent {
            id: "test-123".to_string(),
            embedding: vec![0.1; 768],
            metadata: serde_json::json!({"title": "Test"}),
        };

        coordinator.insert_embedding(&content).await.unwrap();

        let metrics = coordinator.get_metrics().await;
        assert_eq!(metrics.neo4j_writes, 1);
        assert_eq!(metrics.milvus_writes, 1);
    }

    #[tokio::test]
    async fn test_shadow_milvus_fallback() {
        let neo4j = Arc::new(Neo4jClient);
        let milvus = None;  // Milvus unavailable

        let coordinator = DualWriteCoordinator::new(neo4j, milvus, StorageMode::ShadowMilvus);

        let query = vec![0.1; 768];
        let results = coordinator.search_similar(&query, 10).await.unwrap();

        // Should fallback to Neo4j
        assert!(results.is_empty());  // Mock returns empty
    }
}
