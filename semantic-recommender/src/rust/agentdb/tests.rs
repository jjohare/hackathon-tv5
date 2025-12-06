#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio;

    #[tokio::test]
    async fn test_policy_lookup_latency() {
        // Test 5ms policy lookup target
        let postgres_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://agentdb:test@localhost:5432/agentdb_test".to_string());
        let redis_url = std::env::var("TEST_REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

        let coordinator = AgentDBCoordinator::new(&postgres_url, &redis_url, 10, 3600)
            .await
            .expect("Failed to create coordinator");

        let state = State {
            embedding: vec![0.1; 768],
            context: vec![("device".to_string(), "mobile".to_string())].into_iter().collect(),
        };

        // Warm cache
        let _ = coordinator.get_policy("test_agent", "user123", &state).await;

        // Measure cached lookup (target: <5ms)
        let start = std::time::Instant::now();
        let _ = coordinator.get_policy("test_agent", "user123", &state).await;
        let elapsed = start.elapsed();

        println!("Policy lookup latency: {:?}", elapsed);
        assert!(elapsed.as_millis() < 5, "Policy lookup exceeded 5ms target: {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_batch_episode_insert() {
        let postgres_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://agentdb:test@localhost:5432/agentdb_test".to_string());
        let redis_url = std::env::var("TEST_REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

        let coordinator = Arc::new(
            AgentDBCoordinator::new(&postgres_url, &redis_url, 10, 3600)
                .await
                .expect("Failed to create coordinator")
        );

        // Queue 100 episodes
        for i in 0..100 {
            let episode = Episode {
                agent_id: "test_agent".to_string(),
                user_id: format!("user{}", i),
                session_id: format!("session{}", i),
                state_vector: vec![0.1; 768],
                action_taken: vec![("action".to_string(), serde_json::json!("media_123"))].into_iter().collect(),
                reward: 0.8,
                context: HashMap::new(),
            };
            coordinator.record_episode(episode).await.expect("Failed to queue episode");
        }

        // Trigger flush
        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

        println!("Batch episode insert completed");
    }

    #[tokio::test]
    async fn test_thompson_sampling() {
        let postgres_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://agentdb:test@localhost:5432/agentdb_test".to_string());
        let redis_url = std::env::var("TEST_REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

        let coordinator = AgentDBCoordinator::new(&postgres_url, &redis_url, 10, 3600)
            .await
            .expect("Failed to create coordinator");

        let mut policy = coordinator.default_policy("test_agent", "user123", "state_abc");
        policy.q_values.insert("action_a".to_string(), ActionValue { mean: 0.8, variance: 0.1 });
        policy.q_values.insert("action_b".to_string(), ActionValue { mean: 0.6, variance: 0.2 });

        let actions = vec!["action_a".to_string(), "action_b".to_string()];

        // Run Thompson Sampling 100 times
        let mut counts = HashMap::new();
        for _ in 0..100 {
            let selected = coordinator.select_action(&policy, &actions);
            *counts.entry(selected).or_insert(0) += 1;
        }

        println!("Thompson Sampling distribution: {:?}", counts);
        // Action A should be selected more frequently (higher mean)
        assert!(counts.get("action_a").unwrap_or(&0) > counts.get("action_b").unwrap_or(&0));
    }
}
