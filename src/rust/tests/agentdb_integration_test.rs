// Full integration test: AgentDB + Recommendation Engine
#[cfg(test)]
mod agentdb_integration {
    use recommendation_engine::agentdb::{AgentDBCoordinator, State};
    use recommendation_engine::storage::{Episode, Policy, ActionValue};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio;

    async fn setup_test_coordinator() -> Arc<AgentDBCoordinator> {
        let postgres_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://agentdb:test@localhost:5432/agentdb_test".to_string());
        let redis_url = std::env::var("TEST_REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

        Arc::new(
            AgentDBCoordinator::new(&postgres_url, &redis_url, 10, 3600)
                .await
                .expect("Failed to create coordinator"),
        )
    }

    #[tokio::test]
    async fn test_full_recommendation_cycle() {
        let coordinator = setup_test_coordinator().await;

        // Start background worker
        let coordinator_clone = coordinator.clone();
        tokio::spawn(async move {
            coordinator_clone.start_flush_worker().await;
        });

        // User makes initial request (cold start)
        let user_embedding = vec![0.25; 768];
        let state = State {
            embedding: user_embedding.clone(),
            context: vec![
                ("device".to_string(), "mobile".to_string()),
                ("time".to_string(), "evening".to_string()),
            ]
            .into_iter()
            .collect(),
        };

        // Get policy (will be default on first request)
        let policy = coordinator
            .get_policy("recommendation_agent", "test_user_001", &state)
            .await
            .expect("Failed to get policy");

        assert_eq!(policy.total_visits, 0, "New user should have 0 visits");

        // Simulate recommendation selection
        let candidates = vec![
            "documentary_climate_change".to_string(),
            "drama_french_resistance".to_string(),
            "comedy_parisian_life".to_string(),
        ];

        let selected_action = coordinator.select_action(&policy, &candidates);
        println!("Selected: {}", selected_action);

        // User watches content (high engagement)
        let reward = 0.85; // 85% completion rate
        let state_hash = coordinator.hash_state(&user_embedding);

        coordinator
            .update_policy(
                "recommendation_agent",
                "test_user_001",
                &state_hash,
                &selected_action,
                reward,
            )
            .await
            .expect("Failed to update policy");

        // Record episode
        let episode = Episode {
            agent_id: "recommendation_agent".to_string(),
            user_id: "test_user_001".to_string(),
            session_id: "session_001".to_string(),
            state_vector: user_embedding.clone(),
            action_taken: vec![
                ("media_id".to_string(), serde_json::json!(selected_action)),
                ("completion_rate".to_string(), serde_json::json!(0.85)),
            ]
            .into_iter()
            .collect(),
            reward,
            context: vec![
                ("device".to_string(), serde_json::json!("mobile")),
                ("engagement".to_string(), serde_json::json!("high")),
            ]
            .into_iter()
            .collect(),
        };

        coordinator
            .record_episode(episode)
            .await
            .expect("Failed to record episode");

        // Wait for background flush
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Second request should have updated policy
        let updated_policy = coordinator
            .get_policy("recommendation_agent", "test_user_001", &state)
            .await
            .expect("Failed to get updated policy");

        assert_eq!(updated_policy.total_visits, 1, "Policy should reflect 1 visit");
        assert!(
            updated_policy.average_reward > 0.0,
            "Average reward should be updated"
        );
        assert!(
            updated_policy.q_values.contains_key(&selected_action),
            "Q-values should include selected action"
        );

        println!("✅ Full recommendation cycle completed successfully");
        println!("   Policy visits: {}", updated_policy.total_visits);
        println!("   Average reward: {:.2}", updated_policy.average_reward);
        println!("   Q-values: {:?}", updated_policy.q_values);
    }

    #[tokio::test]
    async fn test_multi_user_performance() {
        let coordinator = setup_test_coordinator().await;

        let start = std::time::Instant::now();
        let mut handles = vec![];

        // Simulate 100 concurrent users
        for i in 0..100 {
            let coordinator = coordinator.clone();
            let handle = tokio::spawn(async move {
                let user_id = format!("user_{:03}", i);
                let embedding = vec![i as f32 / 100.0; 768];
                let state = State {
                    embedding,
                    context: HashMap::new(),
                };

                coordinator
                    .get_policy("recommendation_agent", &user_id, &state)
                    .await
                    .expect("Failed to get policy");
            });
            handles.push(handle);
        }

        // Wait for all requests
        for handle in handles {
            handle.await.expect("Task failed");
        }

        let elapsed = start.elapsed();
        let avg_latency = elapsed.as_millis() / 100;

        println!("100 concurrent policy lookups: {:?}", elapsed);
        println!("Average latency: {}ms", avg_latency);

        assert!(
            avg_latency < 50,
            "Average latency should be <50ms, got {}ms",
            avg_latency
        );
    }

    #[tokio::test]
    async fn test_thompson_sampling_convergence() {
        let coordinator = setup_test_coordinator().await;

        let user_embedding = vec![0.5; 768];
        let state_hash = coordinator.hash_state(&user_embedding);

        // Simulate 100 interactions with different rewards
        // Action A has higher average reward (0.8 vs 0.5)
        for i in 0..100 {
            let action = if i % 3 == 0 { "action_a" } else { "action_b" };
            let reward = if action == "action_a" {
                0.8 + (i as f32 % 10.0) * 0.02
            } else {
                0.5 + (i as f32 % 10.0) * 0.02
            };

            coordinator
                .update_policy(
                    "recommendation_agent",
                    "convergence_user",
                    &state_hash,
                    action,
                    reward,
                )
                .await
                .expect("Failed to update policy");
        }

        // Get final policy
        let state = State {
            embedding: user_embedding,
            context: HashMap::new(),
        };

        let final_policy = coordinator
            .get_policy("recommendation_agent", "convergence_user", &state)
            .await
            .expect("Failed to get final policy");

        // Verify action_a has higher mean
        let action_a_mean = final_policy
            .q_values
            .get("action_a")
            .map(|v| v.mean)
            .unwrap_or(0.0);
        let action_b_mean = final_policy
            .q_values
            .get("action_b")
            .map(|v| v.mean)
            .unwrap_or(0.0);

        println!("After 100 iterations:");
        println!("  Action A mean: {:.3}", action_a_mean);
        println!("  Action B mean: {:.3}", action_b_mean);

        assert!(
            action_a_mean > action_b_mean,
            "Action A should have higher mean reward"
        );
        assert!(
            action_a_mean > 0.75,
            "Action A mean should be close to true value (0.8)"
        );
    }
}
