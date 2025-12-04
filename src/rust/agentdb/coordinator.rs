// AgentDB Coordinator: Fast policy lookups + async RL updates
use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
use serde_json::Value as JsonValue;

use crate::storage::{AgentDBCache, PostgresStore, Policy, Episode, ActionValue};

#[derive(Debug, Clone)]
pub struct State {
    pub embedding: Vec<f32>,
    pub context: HashMap<String, String>,
}

pub enum Update {
    Episode(Episode),
    Policy(Policy),
}

pub struct AgentDBCoordinator {
    postgres: Arc<PostgresStore>,
    redis: Arc<Mutex<AgentDBCache>>,
    update_queue: Arc<Mutex<VecDeque<Update>>>,
    batch_size: usize,
    flush_interval: Duration,
}

impl AgentDBCoordinator {
    pub async fn new(
        postgres_url: &str,
        redis_url: &str,
        max_connections: u32,
        cache_ttl_secs: u64,
    ) -> Result<Self> {
        let postgres = Arc::new(PostgresStore::new(postgres_url, max_connections).await?);
        let redis = Arc::new(Mutex::new(AgentDBCache::new(redis_url, cache_ttl_secs).await?));

        info!("AgentDB coordinator initialized");

        Ok(Self {
            postgres,
            redis,
            update_queue: Arc::new(Mutex::new(VecDeque::new())),
            batch_size: 100,
            flush_interval: Duration::from_millis(100),
        })
    }

    /// Fast path: Get policy from cache (5ms target) with PostgreSQL fallback
    pub async fn get_policy(&self, agent_id: &str, user_id: &str, state: &State) -> Result<Policy> {
        let start = Instant::now();
        let state_hash = self.hash_state(&state.embedding);

        // Try Redis cache first
        let mut redis_guard = self.redis.lock().await;
        if let Some(policy) = redis_guard.get_policy(agent_id, user_id, &state_hash).await? {
            debug!("Policy lookup: {}μs (cache HIT)", start.elapsed().as_micros());
            return Ok(policy);
        }
        drop(redis_guard);

        // Fallback to PostgreSQL
        let policy = self.postgres.get_policy(agent_id, user_id, &state_hash)
            .await?
            .unwrap_or_else(|| self.default_policy(agent_id, user_id, &state_hash));

        // Warm cache
        let mut redis_guard = self.redis.lock().await;
        redis_guard.set_policy(&policy).await?;
        drop(redis_guard);

        debug!("Policy lookup: {}μs (cache MISS, PostgreSQL fallback)", start.elapsed().as_micros());
        Ok(policy)
    }

    /// Async path: Record episode (returns immediately, flushed in background)
    pub async fn record_episode(&self, episode: Episode) -> Result<()> {
        let mut queue = self.update_queue.lock().await;
        queue.push_back(Update::Episode(episode));
        Ok(())
    }

    /// Update policy after reward (invalidates cache, async database write)
    pub async fn update_policy(&self, agent_id: &str, user_id: &str, state_hash: &str, action: &str, reward: f32) -> Result<()> {
        // Invalidate cache immediately
        let mut redis_guard = self.redis.lock().await;
        redis_guard.invalidate_policy(agent_id, user_id, state_hash).await?;
        drop(redis_guard);

        // Get current policy or create new
        let mut policy = self.postgres.get_policy(agent_id, user_id, state_hash)
            .await?
            .unwrap_or_else(|| self.default_policy(agent_id, user_id, state_hash));

        // Bayesian update of Q-value
        let action_value = policy.q_values.entry(action.to_string()).or_insert(ActionValue {
            mean: reward,
            variance: 0.1,
        });

        let visit_count = *policy.action_counts.get(action).unwrap_or(&0);
        let new_mean = (action_value.mean * visit_count as f32 + reward) / (visit_count + 1) as f32;
        let new_variance = 0.9 * action_value.variance + 0.1 * (reward - new_mean).abs();

        action_value.mean = new_mean;
        action_value.variance = new_variance;

        *policy.action_counts.entry(action.to_string()).or_insert(0) += 1;
        policy.total_visits += 1;
        policy.average_reward = (policy.average_reward * (policy.total_visits - 1) as f32 + reward) / policy.total_visits as f32;

        // Queue async database update
        let mut queue = self.update_queue.lock().await;
        queue.push_back(Update::Policy(policy));
        Ok(())
    }

    /// Background worker: Flush updates to PostgreSQL
    pub async fn start_flush_worker(self: Arc<Self>) {
        info!("Starting AgentDB flush worker (batch_size: {}, interval: {:?})", self.batch_size, self.flush_interval);

        loop {
            tokio::time::sleep(self.flush_interval).await;

            let updates = {
                let mut queue = self.update_queue.lock().await;
                queue.drain(..).collect::<Vec<_>>()
            };

            if updates.is_empty() {
                continue;
            }

            let episodes: Vec<Episode> = updates.iter().filter_map(|u| match u {
                Update::Episode(ep) => Some(ep.clone()),
                _ => None,
            }).collect();

            let policies: Vec<Policy> = updates.iter().filter_map(|u| match u {
                Update::Policy(p) => Some(p.clone()),
                _ => None,
            }).collect();

            // Batch insert episodes
            if !episodes.is_empty() {
                if let Err(e) = self.postgres.batch_insert_episodes(&episodes).await {
                    warn!("Failed to flush {} episodes: {}", episodes.len(), e);
                } else {
                    debug!("Flushed {} episodes", episodes.len());
                }
            }

            // Upsert policies
            for policy in policies {
                if let Err(e) = self.postgres.upsert_policy(&policy).await {
                    warn!("Failed to upsert policy: {}", e);
                }
            }
        }
    }

    /// Select action using Thompson Sampling
    pub fn select_action(&self, policy: &Policy, actions: &[String]) -> String {
        use rand::distributions::Distribution;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        let mut best_sample = f32::NEG_INFINITY;
        let mut best_action = actions[0].clone();

        for action in actions {
            let action_value = policy.q_values.get(action).unwrap_or(&ActionValue {
                mean: 0.0,
                variance: 1.0,
            });

            // Sample from Normal distribution
            let normal = Normal::new(action_value.mean, action_value.variance.sqrt()).unwrap();
            let sample = normal.sample(&mut rng);

            if sample > best_sample {
                best_sample = sample;
                best_action = action.clone();
            }
        }

        best_action
    }

    fn hash_state(&self, embedding: &[f32]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        // Hash first 16 dimensions for speed
        for &val in embedding.iter().take(16) {
            hasher.update(val.to_le_bytes());
        }
        
        format!("{:x}", hasher.finalize())
    }

    fn default_policy(&self, agent_id: &str, user_id: &str, state_hash: &str) -> Policy {
        Policy {
            agent_id: agent_id.to_string(),
            user_id: user_id.to_string(),
            state_hash: state_hash.to_string(),
            q_values: HashMap::new(),
            action_counts: HashMap::new(),
            total_visits: 0,
            exploration_rate: 0.1,
            average_reward: 0.0,
        }
    }
}
