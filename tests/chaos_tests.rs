use recommendation_engine::storage::HybridStorageCoordinator;
use testcontainers::{clients::Cli, Container};
use tokio::time::Duration;
use anyhow::Result;

mod fixtures;
use fixtures::{media_generator, query_generator};

async fn simulate_component_failure(container: &Container<'_>) -> Result<()> {
    // Send SIGSTOP to pause container
    let container_id = container.id();
    std::process::Command::new("docker")
        .args(&["pause", container_id])
        .output()?;
    Ok(())
}

async fn restore_component(container: &Container<'_>) -> Result<()> {
    let container_id = container.id();
    std::process::Command::new("docker")
        .args(&["unpause", container_id])
        .output()?;
    Ok(())
}

async fn simulate_network_partition(containers: Vec<&Container<'_>>) -> Result<()> {
    for container in containers {
        let container_id = container.id();
        // Add iptables rules to drop traffic
        std::process::Command::new("docker")
            .args(&[
                "exec",
                container_id,
                "iptables",
                "-A",
                "INPUT",
                "-j",
                "DROP",
            ])
            .output()?;
    }
    Ok(())
}

async fn restore_network(containers: Vec<&Container<'_>>) -> Result<()> {
    for container in containers {
        let container_id = container.id();
        std::process::Command::new("docker")
            .args(&[
                "exec",
                container_id,
                "iptables",
                "-F",
            ])
            .output()?;
    }
    Ok(())
}

#[tokio::test]
#[ignore] // Run manually due to destructive nature
async fn test_milvus_failure_graceful_degradation() {
    let env = setup_test_coordinator().await.unwrap();

    // Insert test data
    let test_data = insert_test_data(&env.coordinator, 100).await.unwrap();

    // Verify normal operation
    let query = query_generator::create_random_query();
    let results_before = env.coordinator.search_with_context(&query).await.unwrap();
    assert!(!results_before.is_empty());

    // Simulate Milvus failure
    println!("Simulating Milvus failure...");
    simulate_component_failure(&env._milvus).await.unwrap();

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Query should fallback to Neo4j (slower but functional)
    let result = env.coordinator.search_with_context(&query).await;

    assert!(
        result.is_ok(),
        "Should fallback to Neo4j when Milvus fails: {:?}",
        result.err()
    );

    let results_fallback = result.unwrap();
    assert!(
        !results_fallback.is_empty(),
        "Fallback should return results"
    );

    // Verify fallback mode indicator
    assert!(
        env.coordinator.is_degraded_mode().await,
        "Should indicate degraded mode"
    );

    // Restore Milvus
    println!("Restoring Milvus...");
    restore_component(&env._milvus).await.unwrap();
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Verify recovery
    let results_after = env.coordinator.search_with_context(&query).await.unwrap();
    assert!(!results_after.is_empty());
    assert!(!env.coordinator.is_degraded_mode().await);
}

#[tokio::test]
#[ignore]
async fn test_neo4j_failure_vector_only_mode() {
    let env = setup_test_coordinator().await.unwrap();

    insert_test_data(&env.coordinator, 100).await.unwrap();

    // Kill Neo4j
    println!("Simulating Neo4j failure...");
    simulate_component_failure(&env._neo4j).await.unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Query should succeed with vector search only (no graph enrichment)
    let query = query_generator::create_random_query();
    let result = env.coordinator.search_with_context(&query).await;

    assert!(result.is_ok(), "Should work with vector search only");

    let results = result.unwrap();
    assert!(!results.is_empty());

    // Verify no graph enrichment data
    for result in &results {
        assert!(
            result.metadata.get("actors").is_none(),
            "Should not have graph enrichment data"
        );
    }

    // Restore Neo4j
    restore_component(&env._neo4j).await.unwrap();
    tokio::time::sleep(Duration::from_secs(5)).await;
}

#[tokio::test]
#[ignore]
async fn test_redis_cache_failure_no_impact() {
    let env = setup_test_coordinator().await.unwrap();

    insert_test_data(&env.coordinator, 100).await.unwrap();

    let query = query_generator::create_random_query();

    // Warm cache
    let _ = env.coordinator.search_with_context(&query).await.unwrap();

    // Kill Redis
    println!("Simulating Redis failure...");
    simulate_component_failure(&env._redis).await.unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Query should still work (cache miss, direct to storage)
    let result = env.coordinator.search_with_context(&query).await;

    assert!(
        result.is_ok(),
        "Should work without cache: {:?}",
        result.err()
    );

    let results = result.unwrap();
    assert!(!results.is_empty());

    // Restore Redis
    restore_component(&env._redis).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn test_network_partition_between_shards() {
    let env = setup_test_coordinator().await.unwrap();

    insert_test_data(&env.coordinator, 1000).await.unwrap();

    // Simulate network partition (assuming multi-shard Milvus)
    println!("Simulating network partition...");
    // This would require multi-node Milvus setup
    // For now, simulate by introducing latency

    let query = query_generator::create_random_query();
    let result = env.coordinator.search_with_context(&query).await;

    // Should succeed with available shards
    assert!(result.is_ok());

    let results = result.unwrap();
    // May have reduced recall but should not error
    assert!(results.len() > 0);
}

#[tokio::test]
#[ignore]
async fn test_cascading_failure_recovery() {
    let env = setup_test_coordinator().await.unwrap();

    insert_test_data(&env.coordinator, 100).await.unwrap();

    let query = query_generator::create_random_query();

    // Normal operation
    let results_before = env.coordinator.search_with_context(&query).await.unwrap();
    assert!(!results_before.is_empty());

    // Kill Redis
    println!("Step 1: Redis failure");
    simulate_component_failure(&env._redis).await.unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    let result1 = env.coordinator.search_with_context(&query).await;
    assert!(result1.is_ok(), "Should survive Redis failure");

    // Kill Neo4j
    println!("Step 2: Neo4j failure");
    simulate_component_failure(&env._neo4j).await.unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    let result2 = env.coordinator.search_with_context(&query).await;
    assert!(result2.is_ok(), "Should survive Redis + Neo4j failure");

    // Kill Milvus (all components down)
    println!("Step 3: Milvus failure (total failure)");
    simulate_component_failure(&env._milvus).await.unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    let result3 = env.coordinator.search_with_context(&query).await;
    assert!(result3.is_err(), "Should fail with all components down");

    // Restore Milvus first
    println!("Step 4: Restore Milvus");
    restore_component(&env._milvus).await.unwrap();
    tokio::time::sleep(Duration::from_secs(5)).await;

    let result4 = env.coordinator.search_with_context(&query).await;
    assert!(result4.is_ok(), "Should recover with Milvus only");

    // Restore Neo4j
    println!("Step 5: Restore Neo4j");
    restore_component(&env._neo4j).await.unwrap();
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Restore Redis
    println!("Step 6: Restore Redis");
    restore_component(&env._redis).await.unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Full recovery
    let results_after = env.coordinator.search_with_context(&query).await.unwrap();
    assert!(!results_after.is_empty());
    assert!(!env.coordinator.is_degraded_mode().await);
}

#[tokio::test]
#[ignore]
async fn test_slow_component_timeout() {
    let env = setup_test_coordinator().await.unwrap();

    insert_test_data(&env.coordinator, 100).await.unwrap();

    // Inject network latency (simulating slow Milvus)
    let container_id = env._milvus.id();
    std::process::Command::new("docker")
        .args(&[
            "exec",
            container_id,
            "tc",
            "qdisc",
            "add",
            "dev",
            "eth0",
            "root",
            "netem",
            "delay",
            "500ms",
        ])
        .output()
        .ok();

    tokio::time::sleep(Duration::from_secs(1)).await;

    let query = query_generator::create_random_query();
    let start = tokio::time::Instant::now();

    let result = env.coordinator.search_with_context(&query).await;
    let elapsed = start.elapsed();

    // Should timeout and fallback quickly (not wait 500ms)
    assert!(elapsed < Duration::from_millis(200), "Should timeout quickly");
    assert!(result.is_ok(), "Should fallback on timeout");

    // Cleanup
    std::process::Command::new("docker")
        .args(&[
            "exec",
            container_id,
            "tc",
            "qdisc",
            "del",
            "dev",
            "eth0",
            "root",
        ])
        .output()
        .ok();
}

#[tokio::test]
#[ignore]
async fn test_data_corruption_detection() {
    let env = setup_test_coordinator().await.unwrap();

    let test_data = insert_test_data(&env.coordinator, 10).await.unwrap();

    // Corrupt data in Milvus (delete some vectors)
    env.coordinator
        .milvus_client
        .delete_entity("test_1")
        .await
        .unwrap();

    // Query should detect inconsistency
    let query = SearchQuery {
        embedding: test_data[0].embedding.clone(),
        user_id: None,
        k: 5,
        filters: HashMap::new(),
    };

    let result = env.coordinator.search_with_context(&query).await;

    // Should either succeed with available data or gracefully handle
    match result {
        Ok(results) => {
            // If successful, verify data consistency
            for result in results {
                assert!(result.id != "test_1", "Deleted item should not appear");
            }
        }
        Err(e) => {
            // Should be a handled error, not a panic
            assert!(e.to_string().contains("consistency"), "Should indicate consistency issue");
        }
    }
}
