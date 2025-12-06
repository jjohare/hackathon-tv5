use recommendation_engine::storage::{HybridStorageCoordinator, MilvusClient, AgentDBCoordinator};
use testcontainers::{clients::Cli, Container, Image, RunnableImage};
use testcontainers::core::WaitFor;
use tokio::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;

mod fixtures;
use fixtures::{media_generator, query_generator, user_generator};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MediaContent {
    id: String,
    title: String,
    embedding: Vec<f32>,
    genres: Vec<String>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct SearchQuery {
    embedding: Vec<f32>,
    user_id: Option<String>,
    k: usize,
    filters: HashMap<String, String>,
}

struct TestEnvironment {
    coordinator: HybridStorageCoordinator,
    _milvus: Container<'static, GenericImage>,
    _neo4j: Container<'static, GenericImage>,
    _postgres: Container<'static, GenericImage>,
    _redis: Container<'static, GenericImage>,
}

async fn setup_test_coordinator() -> Result<TestEnvironment> {
    let docker = Cli::default();

    // Start Milvus
    let milvus = docker.run(
        RunnableImage::from(GenericImage::new("milvusdb/milvus", "v2.4.0"))
            .with_wait_for(WaitFor::message_on_stdout("Milvus Proxy successfully initialized"))
    );
    let milvus_port = milvus.get_host_port_ipv4(19530);

    // Start Neo4j
    let neo4j = docker.run(
        RunnableImage::from(GenericImage::new("neo4j", "5.15"))
            .with_env_var("NEO4J_AUTH", "neo4j/testpassword")
            .with_wait_for(WaitFor::message_on_stdout("Started."))
    );
    let neo4j_port = neo4j.get_host_port_ipv4(7687);

    // Start PostgreSQL with pgvector
    let postgres = docker.run(
        RunnableImage::from(GenericImage::new("pgvector/pgvector", "pg16"))
            .with_env_var("POSTGRES_PASSWORD", "testpassword")
            .with_wait_for(WaitFor::message_on_stdout("database system is ready to accept connections"))
    );
    let postgres_port = postgres.get_host_port_ipv4(5432);

    // Start Redis
    let redis = docker.run(
        RunnableImage::from(GenericImage::new("redis", "7-alpine"))
            .with_wait_for(WaitFor::message_on_stdout("Ready to accept connections"))
    );
    let redis_port = redis.get_host_port_ipv4(6379);

    // Initialize hybrid coordinator
    let config = HybridStorageConfig {
        milvus_endpoint: format!("localhost:{}", milvus_port),
        neo4j_uri: format!("bolt://localhost:{}", neo4j_port),
        neo4j_user: "neo4j".to_string(),
        neo4j_password: "testpassword".to_string(),
        postgres_url: format!("postgres://postgres:testpassword@localhost:{}/test", postgres_port),
        redis_url: format!("redis://localhost:{}", redis_port),
        cache_ttl: Duration::from_secs(300),
    };

    let coordinator = HybridStorageCoordinator::new(config).await?;

    Ok(TestEnvironment {
        coordinator,
        _milvus: milvus,
        _neo4j: neo4j,
        _postgres: postgres,
        _redis: redis,
    })
}

async fn insert_test_data(coordinator: &HybridStorageCoordinator, count: usize) -> Result<Vec<MediaContent>> {
    let media_items = media_generator::generate_media_content(count);

    for media in &media_items {
        coordinator.ingest_content(media).await?;
    }

    // Wait for indexing
    tokio::time::sleep(Duration::from_secs(2)).await;

    Ok(media_items)
}

#[tokio::test]
async fn test_hybrid_search_latency_p99_under_10ms() {
    let env = setup_test_coordinator().await.unwrap();

    // Insert test data (1000 items)
    let test_media = insert_test_data(&env.coordinator, 1000).await.unwrap();

    // Warm up cache
    for _ in 0..10 {
        let query = query_generator::create_random_query();
        let _ = env.coordinator.search_with_context(&query).await;
    }

    // Measure search latency (100 iterations)
    let mut latencies = Vec::with_capacity(100);

    for _ in 0..100 {
        let query = query_generator::create_random_query();
        let start = Instant::now();
        let results = env.coordinator.search_with_context(&query).await.unwrap();
        let elapsed = start.elapsed();

        assert!(!results.is_empty(), "Should return results");
        latencies.push(elapsed);
    }

    // Calculate percentiles
    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p90 = latencies[(latencies.len() as f64 * 0.90) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    let max = latencies.last().unwrap();

    println!("Latency Statistics:");
    println!("  p50: {:?}", p50);
    println!("  p90: {:?}", p90);
    println!("  p99: {:?}", p99);
    println!("  max: {:?}", max);

    // Validate p99 < 10ms target
    assert!(
        p99 < Duration::from_millis(10),
        "p99 latency {:?} exceeds 10ms target",
        p99
    );
}

#[tokio::test]
async fn test_vector_search_accuracy() {
    let env = setup_test_coordinator().await.unwrap();

    // Create known similar items with controlled embeddings
    let base_embedding = vec![1.0; 768];

    let item1 = MediaContent {
        id: "test_1".to_string(),
        title: "Test Movie 1".to_string(),
        embedding: base_embedding.clone(),
        genres: vec!["SciFi".to_string()],
        metadata: HashMap::new(),
    };

    // Similar item (cosine similarity ~0.95)
    let mut similar_embedding = base_embedding.clone();
    for i in 0..10 {
        similar_embedding[i] = 0.95;
    }
    let item2 = MediaContent {
        id: "test_2".to_string(),
        title: "Test Movie 2".to_string(),
        embedding: similar_embedding,
        genres: vec!["SciFi".to_string()],
        metadata: HashMap::new(),
    };

    // Different item (cosine similarity ~0.2)
    let different_embedding = vec![0.2; 768];
    let item3 = MediaContent {
        id: "test_3".to_string(),
        title: "Test Movie 3".to_string(),
        embedding: different_embedding,
        genres: vec!["Romance".to_string()],
        metadata: HashMap::new(),
    };

    // Ingest content
    env.coordinator.ingest_content(&item1).await.unwrap();
    env.coordinator.ingest_content(&item2).await.unwrap();
    env.coordinator.ingest_content(&item3).await.unwrap();

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Search for item1
    let query = SearchQuery {
        embedding: item1.embedding.clone(),
        user_id: None,
        k: 3,
        filters: HashMap::new(),
    };

    let results = env.coordinator.search_with_context(&query).await.unwrap();

    // Verify ranking: item1 (self) > item2 (similar) > item3 (different)
    assert_eq!(results[0].id, "test_1", "First result should be self");
    assert_eq!(results[1].id, "test_2", "Second result should be similar item");
    assert_eq!(results[2].id, "test_3", "Third result should be different item");

    // Verify similarity scores are decreasing
    assert!(results[0].score > results[1].score);
    assert!(results[1].score > results[2].score);
}

#[tokio::test]
async fn test_graph_enrichment_integration() {
    let env = setup_test_coordinator().await.unwrap();

    // Create content with graph relationships in Neo4j
    env.coordinator.neo4j_client.run(neo4rs::query(
        "CREATE (m:MediaContent {id: 'movie_1', title: 'Test SciFi Movie'})
         CREATE (g:Genre {name: 'SciFi'})
         CREATE (a:Actor {name: 'John Doe'})
         CREATE (m)-[:BELONGS_TO]->(g)
         CREATE (a)-[:ACTS_IN]->(m)"
    )).await.unwrap();

    // Insert vector data
    let media = MediaContent {
        id: "movie_1".to_string(),
        title: "Test SciFi Movie".to_string(),
        embedding: vec![1.0; 768],
        genres: vec!["SciFi".to_string()],
        metadata: HashMap::new(),
    };

    env.coordinator.ingest_content(&media).await.unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Search should include graph data
    let query = SearchQuery {
        embedding: vec![1.0; 768],
        user_id: None,
        k: 5,
        filters: HashMap::new(),
    };

    let results = env.coordinator.search_with_context(&query).await.unwrap();

    // Verify graph enrichment
    let result = results.iter().find(|r| r.id == "movie_1").unwrap();
    assert_eq!(result.genres, vec!["SciFi"]);
    assert!(result.metadata.contains_key("actors"));
    assert_eq!(result.metadata["actors"], "John Doe");
}

#[tokio::test]
async fn test_agentdb_policy_integration() {
    let env = setup_test_coordinator().await.unwrap();

    // Create user with learned policy (prefers SciFi, dislikes Romance)
    let policy = Policy {
        agent_id: "user_123".to_string(),
        preferences: vec![
            ("genre_scifi".to_string(), 0.9),
            ("genre_action".to_string(), 0.7),
            ("genre_romance".to_string(), 0.2),
        ].into_iter().collect(),
    };

    env.coordinator.agentdb.set_policy("user_123", &policy).await.unwrap();

    // Insert test content
    let scifi_movie = MediaContent {
        id: "scifi_1".to_string(),
        title: "Space Adventure".to_string(),
        embedding: vec![0.5; 768],
        genres: vec!["SciFi".to_string()],
        metadata: HashMap::new(),
    };

    let romance_movie = MediaContent {
        id: "romance_1".to_string(),
        title: "Love Story".to_string(),
        embedding: vec![0.5; 768], // Same embedding as SciFi
        genres: vec!["Romance".to_string()],
        metadata: HashMap::new(),
    };

    env.coordinator.ingest_content(&scifi_movie).await.unwrap();
    env.coordinator.ingest_content(&romance_movie).await.unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Search with policy re-ranking
    let query = SearchQuery {
        embedding: vec![0.5; 768],
        user_id: Some("user_123".to_string()),
        k: 10,
        filters: HashMap::new(),
    };

    let results = env.coordinator.search_with_context(&query).await.unwrap();

    // Verify SciFi items ranked higher than Romance despite same embedding
    let scifi_rank = results.iter().position(|r| r.id == "scifi_1").unwrap();
    let romance_rank = results.iter().position(|r| r.id == "romance_1").unwrap();

    assert!(
        scifi_rank < romance_rank,
        "SciFi (rank {}) should rank higher than Romance (rank {}) due to user policy",
        scifi_rank,
        romance_rank
    );
}

#[tokio::test]
async fn test_cache_effectiveness() {
    let env = setup_test_coordinator().await.unwrap();

    insert_test_data(&env.coordinator, 100).await.unwrap();

    let query = query_generator::create_random_query();

    // First query (cold cache)
    let start = Instant::now();
    let results1 = env.coordinator.search_with_context(&query).await.unwrap();
    let cold_latency = start.elapsed();

    // Second query (warm cache)
    let start = Instant::now();
    let results2 = env.coordinator.search_with_context(&query).await.unwrap();
    let warm_latency = start.elapsed();

    // Verify cache hit
    assert_eq!(results1.len(), results2.len());
    assert_eq!(results1[0].id, results2[0].id);

    // Warm cache should be significantly faster (at least 50% improvement)
    println!("Cold cache: {:?}, Warm cache: {:?}", cold_latency, warm_latency);
    assert!(
        warm_latency < cold_latency / 2,
        "Cache should provide at least 50% speedup"
    );
}

#[tokio::test]
async fn test_concurrent_writes_and_reads() {
    let env = setup_test_coordinator().await.unwrap();

    let coordinator = Arc::new(env.coordinator);
    let mut handles = vec![];

    // Spawn 10 concurrent writers
    for i in 0..10 {
        let coord = coordinator.clone();
        handles.push(tokio::spawn(async move {
            for j in 0..100 {
                let media = media_generator::generate_single_media(&format!("writer_{}_{}", i, j));
                coord.ingest_content(&media).await.unwrap();
            }
        }));
    }

    // Spawn 10 concurrent readers
    for _ in 0..10 {
        let coord = coordinator.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..100 {
                let query = query_generator::create_random_query();
                let _ = coord.search_with_context(&query).await.unwrap();
            }
        }));
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify data integrity
    let total_count = coordinator.get_total_count().await.unwrap();
    assert_eq!(total_count, 1000, "All writes should succeed");
}

#[tokio::test]
async fn test_filter_pushdown_optimization() {
    let env = setup_test_coordinator().await.unwrap();

    // Insert data with specific genres
    for i in 0..1000 {
        let genre = if i % 2 == 0 { "SciFi" } else { "Romance" };
        let media = MediaContent {
            id: format!("movie_{}", i),
            title: format!("Movie {}", i),
            embedding: vec![0.5; 768],
            genres: vec![genre.to_string()],
            metadata: HashMap::new(),
        };
        env.coordinator.ingest_content(&media).await.unwrap();
    }

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Query with filter
    let query = SearchQuery {
        embedding: vec![0.5; 768],
        user_id: None,
        k: 100,
        filters: vec![("genre".to_string(), "SciFi".to_string())].into_iter().collect(),
    };

    let start = Instant::now();
    let results = env.coordinator.search_with_context(&query).await.unwrap();
    let filtered_latency = start.elapsed();

    // Verify all results match filter
    for result in &results {
        assert!(result.genres.contains(&"SciFi".to_string()));
    }

    // Query without filter (should be similar latency due to filter pushdown)
    let query_no_filter = SearchQuery {
        embedding: vec![0.5; 768],
        user_id: None,
        k: 100,
        filters: HashMap::new(),
    };

    let start = Instant::now();
    let _ = env.coordinator.search_with_context(&query_no_filter).await.unwrap();
    let no_filter_latency = start.elapsed();

    println!("Filtered: {:?}, No filter: {:?}", filtered_latency, no_filter_latency);

    // Filter pushdown should keep latency within 2x
    assert!(
        filtered_latency < no_filter_latency * 2,
        "Filter pushdown should minimize overhead"
    );
}
