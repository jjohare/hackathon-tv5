// AgentDB Integration Example
use anyhow::Result;
use recommendation_engine::agentdb::{AgentDBCoordinator, State};
use recommendation_engine::storage::Episode;
use std::collections::HashMap;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize AgentDB
    let coordinator = Arc::new(
        AgentDBCoordinator::new(
            "postgresql://agentdb:password@localhost:5432/agentdb",
            "redis://localhost:6379",
            20,    // max connections
            3600,  // cache TTL (1 hour)
        )
        .await?,
    );

    // Start background flush worker
    let coordinator_clone = coordinator.clone();
    tokio::spawn(async move {
        coordinator_clone.start_flush_worker().await;
    });

    // Example: Get personalized recommendation
    let user_embedding = vec![0.15; 768]; // User behavioral embedding
    let state = State {
        embedding: user_embedding.clone(),
        context: vec![
            ("device".to_string(), "mobile".to_string()),
            ("time".to_string(), "evening".to_string()),
        ]
        .into_iter()
        .collect(),
    };

    // Fast policy lookup (5ms cached)
    let policy = coordinator
        .get_policy("recommendation_agent", "user_12345", &state)
        .await?;

    // Thompson Sampling action selection
    let candidate_actions = vec![
        "media_documentary_123".to_string(),
        "media_drama_456".to_string(),
        "media_comedy_789".to_string(),
    ];

    let selected_action = coordinator.select_action(&policy, &candidate_actions);
    println!("Selected media: {} (policy visits: {})", selected_action, policy.total_visits);

    // User watches video (engagement signal)
    let watch_time_sec = 1800; // 30 minutes
    let media_duration_sec = 3600; // 1 hour
    let reward = (watch_time_sec as f32) / (media_duration_sec as f32); // 0.5

    // Update policy with reward
    let state_hash = coordinator.hash_state(&user_embedding);
    coordinator
        .update_policy(
            "recommendation_agent",
            "user_12345",
            &state_hash,
            &selected_action,
            reward,
        )
        .await?;

    // Record episode for offline training
    let episode = Episode {
        agent_id: "recommendation_agent".to_string(),
        user_id: "user_12345".to_string(),
        session_id: "session_abc123".to_string(),
        state_vector: user_embedding,
        action_taken: vec![
            ("media_id".to_string(), serde_json::json!(selected_action)),
            ("watch_time".to_string(), serde_json::json!(watch_time_sec)),
        ]
        .into_iter()
        .collect(),
        reward,
        context: vec![
            ("device".to_string(), serde_json::json!("mobile")),
            ("completion_rate".to_string(), serde_json::json!(0.5)),
        ]
        .into_iter()
        .collect(),
    };

    coordinator.record_episode(episode).await?;

    println!("Episode recorded, policy updated (async)");

    // Wait for background flush
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    Ok(())
}
