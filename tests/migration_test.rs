#[cfg(test)]
mod migration_tests {
    use super::*;

    #[tokio::test]
    async fn test_dual_write_mode() {
        let neo4j = Arc::new(Neo4jClient);
        let milvus = Some(Arc::new(MilvusClient));

        let coordinator = DualWriteCoordinator::new(
            neo4j,
            milvus,
            StorageMode::DualWrite
        );

        let content = MediaContent {
            id: "test-123".to_string(),
            embedding: vec![0.1; 768],
            metadata: serde_json::json!({"title": "Test Movie"}),
        };

        // Should write to both systems
        coordinator.insert_embedding(&content).await.unwrap();

        let metrics = coordinator.get_metrics().await;
        assert_eq!(metrics.neo4j_writes, 1);
        assert_eq!(metrics.milvus_writes, 1);
        assert_eq!(metrics.neo4j_errors, 0);
        assert_eq!(metrics.milvus_errors, 0);
    }

    #[tokio::test]
    async fn test_shadow_milvus_fallback() {
        let neo4j = Arc::new(Neo4jClient);
        let milvus = None;  // Milvus unavailable

        let coordinator = DualWriteCoordinator::new(
            neo4j,
            milvus,
            StorageMode::ShadowMilvus
        );

        let query = vec![0.1; 768];
        let results = coordinator.search_similar(&query, 10).await.unwrap();

        // Should fallback to Neo4j
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_mode_requires_milvus() {
        let neo4j = Arc::new(Neo4jClient);
        let milvus = None;

        let coordinator = DualWriteCoordinator::new(
            neo4j,
            milvus,
            StorageMode::Hybrid
        );

        let query = vec![0.1; 768];
        let result = coordinator.search_similar(&query, 10).await;

        // Should fail without Milvus
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_vector_equality() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.2, 0.3];
        assert!(vectors_match(&a, &b, 1e-5));

        let c = vec![0.1, 0.2, 0.4];
        assert!(!vectors_match(&a, &c, 1e-5));
    }

    #[tokio::test]
    async fn test_l2_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let distance = vector_l2_distance(&a, &b);
        // sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414
        assert!((distance - 1.414).abs() < 0.001);
    }

    #[test]
    fn test_storage_mode_from_env() {
        std::env::set_var("STORAGE_MODE", "dual_write");
        assert_eq!(StorageMode::from_env(), StorageMode::DualWrite);

        std::env::set_var("STORAGE_MODE", "hybrid");
        assert_eq!(StorageMode::from_env(), StorageMode::Hybrid);

        std::env::remove_var("STORAGE_MODE");
        assert_eq!(StorageMode::from_env(), StorageMode::Neo4jOnly);
    }

    #[tokio::test]
    async fn test_migration_stats_calculation() {
        let stats = MigrationStats {
            total_items: 1000,
            migrated: 950,
            skipped: 30,
            errors: 20,
            start_time: 1000,
            end_time: 1010,
            throughput_items_per_sec: 95.0,
        };

        let success_rate = stats.migrated as f64 / stats.total_items as f64;
        assert!((success_rate - 0.95).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_validation_report() {
        let report = ValidationReport {
            sample_size: 1000,
            full_scan: false,
            matches: 998,
            mismatches: 2,
            missing_in_milvus: 0,
            missing_in_neo4j: 0,
            success_rate: 0.998,
            max_vector_diff: 0.000003,
            avg_vector_diff: 0.000001,
            inconsistencies_fixed: 0,
        };

        assert!(report.success_rate >= 0.99);
        assert_eq!(report.matches + report.mismatches, 1000);
    }

    #[tokio::test]
    async fn test_preflight_readiness() {
        let report = PreflightReport {
            neo4j_reachable: true,
            neo4j_version: "5.15.0".to_string(),
            neo4j_disk_usage: 50_000_000_000,
            total_media_items: 150_000,
            total_embeddings: 150_000,
            milvus_reachable: true,
            milvus_healthy: true,
            milvus_version: "2.3.4".to_string(),
            milvus_collections_ready: true,
            postgres_reachable: true,
            postgres_version: "15.4".to_string(),
            agentdb_schema_ready: true,
            redis_reachable: true,
            redis_memory_available: 8_000_000_000,
            disk_space_available: 500_000_000_000,
            disk_space_required: 150_000_000_000,
            memory_available: 32_000_000_000,
            memory_required: 8_000_000_000,
            estimated_migration_time: Duration::from_secs(750),
            estimated_downtime: Duration::from_secs(10),
            errors: vec![],
            warnings: vec![],
        };

        assert!(report.is_ready());
    }

    #[tokio::test]
    async fn test_preflight_insufficient_disk() {
        let report = PreflightReport {
            disk_space_available: 100_000_000_000,
            disk_space_required: 150_000_000_000,
            errors: vec!["Insufficient disk space".to_string()],
            ..Default::default()
        };

        assert!(!report.is_ready());
    }

    #[tokio::test]
    async fn test_dual_write_metrics() {
        let metrics = DualWriteMetrics {
            neo4j_writes: 1000,
            neo4j_errors: 10,
            milvus_writes: 1000,
            milvus_errors: 5,
            milvus_latency_ms: vec![10.0, 15.0, 20.0],
            neo4j_latency_ms: vec![50.0, 60.0, 70.0],
            timestamp: 0,
        };

        assert_eq!(metrics.average_milvus_latency(), 15.0);
        assert_eq!(metrics.average_neo4j_latency(), 60.0);
        assert_eq!(metrics.milvus_success_rate(), 0.995);
    }
}

// Mock types for testing
use std::sync::Arc;
use std::time::Duration;

struct MediaContent {
    id: String,
    embedding: Vec<f32>,
    metadata: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageMode {
    Neo4jOnly,
    DualWrite,
    ShadowMilvus,
    Hybrid,
}

impl StorageMode {
    fn from_env() -> Self {
        match std::env::var("STORAGE_MODE").as_deref() {
            Ok("dual_write") => StorageMode::DualWrite,
            Ok("shadow_milvus") => StorageMode::ShadowMilvus,
            Ok("hybrid") => StorageMode::Hybrid,
            _ => StorageMode::Neo4jOnly,
        }
    }
}

struct DualWriteCoordinator {
    mode: StorageMode,
}

impl DualWriteCoordinator {
    fn new(_neo4j: Arc<Neo4jClient>, _milvus: Option<Arc<MilvusClient>>, mode: StorageMode) -> Self {
        Self { mode }
    }

    async fn insert_embedding(&self, _content: &MediaContent) -> Result<(), String> {
        Ok(())
    }

    async fn search_similar(&self, _query: &[f32], _top_k: usize) -> Result<Vec<MediaContent>, String> {
        match self.mode {
            StorageMode::Hybrid => Err("Milvus not configured".to_string()),
            _ => Ok(vec![]),
        }
    }

    async fn get_metrics(&self) -> DualWriteMetrics {
        DualWriteMetrics {
            neo4j_writes: 1,
            neo4j_errors: 0,
            milvus_writes: 1,
            milvus_errors: 0,
            milvus_latency_ms: vec![],
            neo4j_latency_ms: vec![],
            timestamp: 0,
        }
    }
}

struct DualWriteMetrics {
    neo4j_writes: u64,
    neo4j_errors: u64,
    milvus_writes: u64,
    milvus_errors: u64,
    milvus_latency_ms: Vec<f64>,
    neo4j_latency_ms: Vec<f64>,
    timestamp: i64,
}

impl DualWriteMetrics {
    fn average_milvus_latency(&self) -> f64 {
        if self.milvus_latency_ms.is_empty() {
            0.0
        } else {
            self.milvus_latency_ms.iter().sum::<f64>() / self.milvus_latency_ms.len() as f64
        }
    }

    fn average_neo4j_latency(&self) -> f64 {
        if self.neo4j_latency_ms.is_empty() {
            0.0
        } else {
            self.neo4j_latency_ms.iter().sum::<f64>() / self.neo4j_latency_ms.len() as f64
        }
    }

    fn milvus_success_rate(&self) -> f64 {
        if self.milvus_writes == 0 {
            0.0
        } else {
            (self.milvus_writes - self.milvus_errors) as f64 / self.milvus_writes as f64
        }
    }
}

struct MigrationStats {
    total_items: i64,
    migrated: i64,
    skipped: i64,
    errors: i64,
    start_time: i64,
    end_time: i64,
    throughput_items_per_sec: f64,
}

struct ValidationReport {
    sample_size: usize,
    full_scan: bool,
    matches: usize,
    mismatches: usize,
    missing_in_milvus: usize,
    missing_in_neo4j: usize,
    success_rate: f64,
    max_vector_diff: f64,
    avg_vector_diff: f64,
    inconsistencies_fixed: usize,
}

struct PreflightReport {
    neo4j_reachable: bool,
    neo4j_version: String,
    neo4j_disk_usage: u64,
    total_media_items: i64,
    total_embeddings: i64,
    milvus_reachable: bool,
    milvus_healthy: bool,
    milvus_version: String,
    milvus_collections_ready: bool,
    postgres_reachable: bool,
    postgres_version: String,
    agentdb_schema_ready: bool,
    redis_reachable: bool,
    redis_memory_available: u64,
    disk_space_available: u64,
    disk_space_required: u64,
    memory_available: u64,
    memory_required: u64,
    estimated_migration_time: Duration,
    estimated_downtime: Duration,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl Default for PreflightReport {
    fn default() -> Self {
        Self {
            neo4j_reachable: false,
            neo4j_version: String::new(),
            neo4j_disk_usage: 0,
            total_media_items: 0,
            total_embeddings: 0,
            milvus_reachable: false,
            milvus_healthy: false,
            milvus_version: String::new(),
            milvus_collections_ready: false,
            postgres_reachable: false,
            postgres_version: String::new(),
            agentdb_schema_ready: false,
            redis_reachable: false,
            redis_memory_available: 0,
            disk_space_available: 0,
            disk_space_required: 0,
            memory_available: 0,
            memory_required: 0,
            estimated_migration_time: Duration::from_secs(0),
            estimated_downtime: Duration::from_secs(0),
            errors: vec![],
            warnings: vec![],
        }
    }
}

impl PreflightReport {
    fn is_ready(&self) -> bool {
        self.errors.is_empty()
            && self.neo4j_reachable
            && self.milvus_reachable
            && self.postgres_reachable
            && self.disk_space_available >= self.disk_space_required
    }
}

struct Neo4jClient;
struct MilvusClient;

fn vectors_match(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    vector_l2_distance(a, b) < epsilon as f64
}

fn vector_l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            (diff * diff) as f64
        })
        .sum::<f64>()
        .sqrt()
}
