// Redis caching layer for AgentDB - 5ms policy lookups
use anyhow::{Context, Result};
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub agent_id: String,
    pub user_id: String,
    pub state_hash: String,
    pub q_values: HashMap<String, ActionValue>,
    pub action_counts: HashMap<String, u32>,
    pub total_visits: u32,
    pub exploration_rate: f32,
    pub average_reward: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionValue {
    pub mean: f32,
    pub variance: f32,
}

pub struct AgentDBCache {
    client: ConnectionManager,
    ttl: Duration,
}

impl AgentDBCache {
    pub async fn new(redis_url: &str, ttl_secs: u64) -> Result<Self> {
        let client = Client::open(redis_url)?;
        let manager = ConnectionManager::new(client).await?;
        Ok(Self {
            client: manager,
            ttl: Duration::from_secs(ttl_secs),
        })
    }

    pub async fn get_policy(&mut self, agent_id: &str, user_id: &str, state_hash: &str) -> Result<Option<Policy>> {
        let key = format!("policy:{}:{}:{}", agent_id, user_id, state_hash);
        let data: Option<String> = self.client.get(&key).await?;
        match data {
            Some(s) => Ok(Some(serde_json::from_str(&s)?)),
            None => Ok(None),
        }
    }

    pub async fn set_policy(&mut self, policy: &Policy) -> Result<()> {
        let key = format!("policy:{}:{}:{}", policy.agent_id, policy.user_id, policy.state_hash);
        let data = serde_json::to_string(policy)?;
        self.client.set_ex(&key, data, self.ttl.as_secs() as usize).await?;
        Ok(())
    }
}
