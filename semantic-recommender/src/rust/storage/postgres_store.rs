// PostgreSQL storage for AgentDB episodes and policies
use anyhow::{Context, Result};
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use tokio_postgres::NoTls;

use super::redis_cache::{Policy, ActionValue};

#[derive(Debug, Clone)]
pub struct Episode {
    pub agent_id: String,
    pub user_id: String,
    pub session_id: String,
    pub state_vector: Vec<f32>,
    pub action_taken: HashMap<String, JsonValue>,
    pub reward: f32,
    pub context: HashMap<String, JsonValue>,
}

pub struct PostgresStore {
    pool: Pool<PostgresConnectionManager<NoTls>>,
}

impl PostgresStore {
    pub async fn new(database_url: &str, max_conn: u32) -> Result<Self> {
        let manager = PostgresConnectionManager::new_from_stringlike(database_url, NoTls)?;
        let pool = Pool::builder().max_size(max_conn).build(manager).await?;
        Ok(Self { pool })
    }

    pub async fn get_policy(&self, agent_id: &str, user_id: &str, state_hash: &str) -> Result<Option<Policy>> {
        let conn = self.pool.get().await?;
        let row = conn.query_opt(
            "SELECT agent_id, user_id, state_hash, q_values, action_counts, total_visits, \
             exploration_rate, average_reward FROM rl_policies \
             WHERE agent_id = $1 AND user_id = $2 AND state_hash = $3",
            &[&agent_id, &user_id, &state_hash],
        ).await?;

        match row {
            Some(r) => {
                let q_values_json: JsonValue = r.try_get("q_values")?;
                let action_counts_json: JsonValue = r.try_get("action_counts")?;
                Ok(Some(Policy {
                    agent_id: r.try_get("agent_id")?,
                    user_id: r.try_get("user_id")?,
                    state_hash: r.try_get("state_hash")?,
                    q_values: serde_json::from_value(q_values_json)?,
                    action_counts: serde_json::from_value(action_counts_json)?,
                    total_visits: r.try_get::<_, i32>("total_visits")? as u32,
                    exploration_rate: r.try_get("exploration_rate")?,
                    average_reward: r.try_get("average_reward")?,
                }))
            }
            None => Ok(None),
        }
    }

    pub async fn upsert_policy(&self, policy: &Policy) -> Result<()> {
        let conn = self.pool.get().await?;
        let q_values_json = serde_json::to_value(&policy.q_values)?;
        let action_counts_json = serde_json::to_value(&policy.action_counts)?;

        conn.execute(
            "INSERT INTO rl_policies (agent_id, user_id, state_hash, q_values, action_counts, \
             total_visits, exploration_rate, average_reward, last_updated) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW()) \
             ON CONFLICT (agent_id, user_id, state_hash) DO UPDATE SET \
             q_values = EXCLUDED.q_values, action_counts = EXCLUDED.action_counts, \
             total_visits = EXCLUDED.total_visits, average_reward = EXCLUDED.average_reward, \
             last_updated = NOW()",
            &[
                &policy.agent_id, &policy.user_id, &policy.state_hash,
                &q_values_json, &action_counts_json, &(policy.total_visits as i32),
                &policy.exploration_rate, &policy.average_reward,
            ],
        ).await?;
        Ok(())
    }

    pub async fn batch_insert_episodes(&self, episodes: &[Episode]) -> Result<()> {
        if episodes.is_empty() {
            return Ok(());
        }
        let conn = self.pool.get().await?;
        let transaction = conn.transaction().await?;

        for ep in episodes {
            let vec_str = format!("[{}]", ep.state_vector.iter()
                .map(|v| v.to_string()).collect::<Vec<_>>().join(", "));
            let action_json = serde_json::to_value(&ep.action_taken)?;
            let context_json = serde_json::to_value(&ep.context)?;

            transaction.execute(
                "INSERT INTO agent_episodes (agent_id, user_id, session_id, state_vector, \
                 action_taken, reward, context) VALUES ($1, $2, $3, $4::vector, $5, $6, $7)",
                &[&ep.agent_id, &ep.user_id, &ep.session_id, &vec_str, &action_json, &ep.reward, &context_json],
            ).await?;
        }
        transaction.commit().await?;
        Ok(())
    }
}
