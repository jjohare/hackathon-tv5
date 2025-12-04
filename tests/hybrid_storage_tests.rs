/// Integration Tests for Hybrid Storage
///
/// Tests vector search, graph traversal, and hybrid query execution.

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test Milvus vector search latency
    #[tokio::test]
    async fn test_milvus_vector_search_latency() {
        // This test requires actual Milvus instance
        // Placeholder for CI/CD environment

        let config = hackathon_tv5::storage::MilvusConfig::default();
        let client = hackathon_tv5::storage::MilvusClient::new(config)
            .await
            .expect("Failed to create Milvus client");

        let query_vector = vec![0.1f32; 768];
        let filters = HashMap::new();

        let start = std::time::Instant::now();

        let results = client
            .search("media_embeddings", &query_vector, 10, &filters, None)
            .await;

        let latency_ms = start.elapsed().as_millis();

        // Verify P99 latency target: < 10ms
        assert!(
            latency_ms < 10,
            "Vector search latency {}ms exceeds 10ms target",
            latency_ms
        );

        if let Ok(results) = results {
            assert!(results.len() <= 10, "Should return at most 10 results");
        }
    }

    /// Test Neo4j graph enrichment
    #[tokio::test]
    async fn test_neo4j_graph_enrichment() {
        // Placeholder - requires Neo4j instance

        let config = hackathon_tv5::storage::Neo4jConfig::default();

        // In CI/CD: skip if Neo4j not available
        if std::env::var("NEO4J_URI").is_err() {
            println!("Skipping Neo4j test - NEO4J_URI not set");
            return;
        }

        let client = hackathon_tv5::storage::Neo4jClient::new(config)
            .await
            .expect("Failed to create Neo4j client");

        let content_ids = vec!["content_1".to_string(), "content_2".to_string()];

        let enrichment = client.enrich_batch(&content_ids).await;

        assert!(
            enrichment.is_ok(),
            "Graph enrichment should succeed"
        );

        if let Ok(enrichment) = enrichment {
            assert!(
                enrichment.len() <= content_ids.len(),
                "Should enrich at most the queried content"
            );
        }
    }

    /// Test hybrid query execution
    #[tokio::test]
    async fn test_hybrid_query_execution() {
        // Placeholder - requires full stack

        println!("Hybrid query test placeholder - requires Milvus + Neo4j + PostgreSQL");

        // Mock components for unit testing
        let query = hackathon_tv5::storage::SearchQuery {
            embedding: vec![0.1; 768],
            k: 10,
            user_id: "user_123".to_string(),
            context: "browsing".to_string(),
            metadata_filters: HashMap::new(),
            graph_filters: HashMap::new(),
            include_relationships: true,
            require_genre_filter: false,
            require_cultural_context: false,
            timeout_ms: 100,
        };

        // Test query planner
        let planner_config = hackathon_tv5::storage::query_planner::PlannerConfig::default();
        let planner = hackathon_tv5::storage::QueryPlanner::new(planner_config);

        let plan = planner.plan(&query);

        assert!(
            plan.estimated_latency_ms < 50.0,
            "Hybrid query should be estimated under 50ms"
        );

        assert!(
            plan.requires_milvus,
            "Query with embedding should require Milvus"
        );

        assert!(
            plan.requires_neo4j,
            "Query with relationships should require Neo4j"
        );
    }

    /// Test query strategy selection
    #[test]
    fn test_query_strategy_vector_only() {
        let planner_config = hackathon_tv5::storage::query_planner::PlannerConfig::default();
        let planner = hackathon_tv5::storage::QueryPlanner::new(planner_config);

        let query = hackathon_tv5::storage::SearchQuery {
            embedding: vec![0.1; 768],
            k: 10,
            user_id: String::new(),
            context: String::new(),
            metadata_filters: HashMap::new(),
            graph_filters: HashMap::new(),
            include_relationships: false,
            require_genre_filter: false,
            require_cultural_context: false,
            timeout_ms: 50,
        };

        let plan = planner.plan(&query);

        assert_eq!(
            plan.strategy,
            hackathon_tv5::storage::query_planner::QueryStrategy::VectorOnly,
            "Simple query should use vector-only strategy"
        );

        assert!(
            plan.estimated_latency_ms < 10.0,
            "Vector-only query should be estimated under 10ms"
        );
    }

    /// Test query strategy hybrid parallel
    #[test]
    fn test_query_strategy_hybrid_parallel() {
        let planner_config = hackathon_tv5::storage::query_planner::PlannerConfig::default();
        let planner = hackathon_tv5::storage::QueryPlanner::new(planner_config);

        let query = hackathon_tv5::storage::SearchQuery {
            embedding: vec![0.1; 768],
            k: 50,
            user_id: "user_123".to_string(),
            context: "browsing".to_string(),
            metadata_filters: HashMap::new(),
            graph_filters: HashMap::new(),
            include_relationships: true,
            require_genre_filter: true,
            require_cultural_context: false,
            timeout_ms: 100,
        };

        let plan = planner.plan(&query);

        assert_eq!(
            plan.strategy,
            hackathon_tv5::storage::query_planner::QueryStrategy::HybridParallel,
            "Complex query with relationships should use hybrid parallel"
        );

        assert!(plan.requires_milvus, "Should require Milvus");
        assert!(plan.requires_neo4j, "Should require Neo4j");
        assert!(plan.requires_agentdb, "Should require AgentDB for user policy");
    }

    /// Test migration stats
    #[test]
    fn test_migration_stats() {
        let stats = hackathon_tv5::storage::MigrationStats {
            embeddings_migrated: 10000,
            relationships_preserved: 50000,
            policies_created: 500,
            errors: 10,
            duration_secs: 120.5,
        };

        assert_eq!(stats.embeddings_migrated, 10000);
        assert_eq!(stats.relationships_preserved, 50000);
        assert_eq!(stats.policies_created, 500);
        assert_eq!(stats.errors, 10);
        assert!(stats.duration_secs > 120.0);
    }

    /// Benchmark vector search performance
    #[tokio::test]
    async fn benchmark_vector_search() {
        let config = hackathon_tv5::storage::MilvusConfig::default();
        let client = hackathon_tv5::storage::MilvusClient::new(config)
            .await
            .expect("Failed to create Milvus client");

        let iterations = 100;
        let query_vector = vec![0.1f32; 768];
        let filters = HashMap::new();

        let mut latencies = Vec::new();

        for _ in 0..iterations {
            let start = std::time::Instant::now();

            let _ = client
                .search("media_embeddings", &query_vector, 10, &filters, None)
                .await;

            let latency_us = start.elapsed().as_micros() as u64;
            latencies.push(latency_us);
        }

        latencies.sort();

        let p50 = latencies[iterations / 2];
        let p95 = latencies[(iterations * 95) / 100];
        let p99 = latencies[(iterations * 99) / 100];

        println!("Vector search latency (μs): P50={}, P95={}, P99={}", p50, p95, p99);

        // Verify P99 < 10ms (10,000μs)
        assert!(
            p99 < 10_000,
            "P99 latency {}μs exceeds 10ms target",
            p99
        );
    }

    /// Test cache key generation consistency
    #[test]
    fn test_cache_key_consistency() {
        let query1 = hackathon_tv5::storage::SearchQuery {
            embedding: vec![0.1, 0.2, 0.3],
            k: 10,
            user_id: String::new(),
            context: String::new(),
            metadata_filters: HashMap::new(),
            graph_filters: HashMap::new(),
            include_relationships: false,
            require_genre_filter: false,
            require_cultural_context: false,
            timeout_ms: 100,
        };

        let query2 = hackathon_tv5::storage::SearchQuery {
            embedding: vec![0.1, 0.2, 0.3],
            k: 10,
            user_id: String::new(),
            context: String::new(),
            metadata_filters: HashMap::new(),
            graph_filters: HashMap::new(),
            include_relationships: false,
            require_genre_filter: false,
            require_cultural_context: false,
            timeout_ms: 100,
        };

        // Same queries should generate same cache keys
        // This would require exposing the cache key generation method
        // or testing through the coordinator
    }
}
