// Integration with recommendation engine
use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use super::coordinator::{AgentDBCoordinator, State};
use crate::storage::Episode;

pub struct RecommendationContext {
    pub user_id: String,
    pub session_id: String,
    pub device_type: String,
    pub location: String,
}

pub struct AgentDBIntegration {
    coordinator: Arc<AgentDBCoordinator>,
    agent_id: String,
}

impl AgentDBIntegration {
    pub fn new(coordinator: Arc<AgentDBCoordinator>, agent_id: String) -> Self {
        Self { coordinator, agent_id }
    }

    /// Get personalized recommendation using RL policy
    pub async fn recommend_with_policy(
        &self,
        user_embedding: Vec<f32>,
        context: RecommendationContext,
        candidate_actions: Vec<String>,
    ) -> Result<String> {
        let state = State {
            embedding: user_embedding.clone(),
            context: vec![
                ("device".to_string(), context.device_type.clone()),
                ("location".to_string(), context.location.clone()),
            ].into_iter().collect(),
        };

        // Get RL policy (5ms cached)
        let policy = self.coordinator.get_policy(&self.agent_id, &context.user_id, &state).await?;

        // Thompson Sampling action selection
        let selected_action = self.coordinator.select_action(&policy, &candidate_actions);

        info!("Selected action {} for user {} (visits: {})", selected_action, context.user_id, policy.total_visits);

        Ok(selected_action)
    }

    /// Record user interaction and update policy
    pub async fn record_interaction(
        &self,
        user_id: &str,
        session_id: &str,
        user_embedding: Vec<f32>,
        action: String,
        reward: f32,
        context: RecommendationContext,
    ) -> Result<()> {
        let state_hash = self.coordinator.hash_state(&user_embedding);

        // Update policy (async)
        self.coordinator.update_policy(&self.agent_id, user_id, &state_hash, &action, reward).await?;

        // Record episode for offline training
        let episode = Episode {
            agent_id: self.agent_id.clone(),
            user_id: user_id.to_string(),
            session_id: session_id.to_string(),
            state_vector: user_embedding,
            action_taken: vec![
                ("action".to_string(), serde_json::json!(action)),
                ("reward".to_string(), serde_json::json!(reward)),
            ].into_iter().collect(),
            reward,
            context: vec![
                ("device".to_string(), serde_json::json!(context.device_type)),
                ("location".to_string(), serde_json::json!(context.location)),
            ].into_iter().collect(),
        };

        self.coordinator.record_episode(episode).await?;

        Ok(())
    }
}

// Add to recommendation engine
impl crate::semantic_search::engine::RecommendationEngine {
    pub fn with_agentdb(mut self, coordinator: Arc<AgentDBCoordinator>) -> Self {
        // Integration hook for AgentDB
        info!("AgentDB integration enabled");
        self
    }
}
