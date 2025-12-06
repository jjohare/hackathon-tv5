use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use siphasher::sip::SipHasher24;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

const VIRTUAL_NODES_PER_SHARD: usize = 150;
const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(5);
const FAILURE_THRESHOLD: u32 = 3;
const RECOVERY_THRESHOLD: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub id: String,
    pub address: String,
    pub embedding_count: u64,
    pub health_status: HealthStatus,
    pub last_health_check: SystemTime,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub gpu_model: String,
    pub capacity_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    ring: BTreeMap<u64, String>,
    shards: HashMap<String, ShardInfo>,
    virtual_nodes: usize,
}

impl ConsistentHashRing {
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            shards: HashMap::new(),
            virtual_nodes,
        }
    }

    pub fn add_shard(&mut self, shard: ShardInfo) -> Result<()> {
        let shard_id = shard.id.clone();

        for i in 0..self.virtual_nodes {
            let virtual_key = format!("{}-{}", shard_id, i);
            let hash = Self::hash_key(&virtual_key);
            self.ring.insert(hash, shard_id.clone());
        }

        self.shards.insert(shard_id.clone(), shard);
        Ok(())
    }

    pub fn remove_shard(&mut self, shard_id: &str) -> Result<()> {
        for i in 0..self.virtual_nodes {
            let virtual_key = format!("{}-{}", shard_id, i);
            let hash = Self::hash_key(&virtual_key);
            self.ring.remove(&hash);
        }

        self.shards.remove(shard_id);
        Ok(())
    }

    pub fn get_shard(&self, key: &str) -> Option<&ShardInfo> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = Self::hash_key(key);

        let shard_id = self
            .ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, id)| id)?;

        self.shards.get(shard_id)
    }

    pub fn get_shards_for_replication(&self, key: &str, count: usize) -> Vec<&ShardInfo> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let hash = Self::hash_key(key);
        let mut result = Vec::new();
        let mut seen = HashSet::new();

        for (_, shard_id) in self.ring.range(hash..).chain(self.ring.iter()) {
            if seen.insert(shard_id) {
                if let Some(shard) = self.shards.get(shard_id) {
                    result.push(shard);
                    if result.len() >= count {
                        break;
                    }
                }
            }
        }

        result
    }

    pub fn get_all_shards(&self) -> Vec<&ShardInfo> {
        self.shards.values().collect()
    }

    pub fn get_healthy_shards(&self) -> Vec<&ShardInfo> {
        self.shards
            .values()
            .filter(|s| s.health_status == HealthStatus::Healthy)
            .collect()
    }

    fn hash_key(key: &str) -> u64 {
        let mut hasher = SipHasher24::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

pub struct ShardManager {
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    replication_factor: usize,
}

impl ShardManager {
    pub fn new(replication_factor: usize) -> Self {
        Self {
            hash_ring: Arc::new(RwLock::new(ConsistentHashRing::new(
                VIRTUAL_NODES_PER_SHARD,
            ))),
            replication_factor,
        }
    }

    pub async fn add_node(&self, shard: ShardInfo) -> Result<()> {
        let mut ring = self.hash_ring.write().await;
        ring.add_shard(shard)
            .context("Failed to add shard to hash ring")
    }

    pub async fn remove_node(&self, shard_id: &str) -> Result<()> {
        let mut ring = self.hash_ring.write().await;
        ring.remove_shard(shard_id)
            .context("Failed to remove shard from hash ring")
    }

    pub async fn get_shard_for_key(&self, key: &str) -> Option<ShardInfo> {
        let ring = self.hash_ring.read().await;
        ring.get_shard(key).cloned()
    }

    pub async fn get_shards_for_key(&self, key: &str) -> Vec<ShardInfo> {
        let ring = self.hash_ring.read().await;
        ring.get_shards_for_replication(key, self.replication_factor)
            .into_iter()
            .cloned()
            .collect()
    }

    pub async fn get_all_shards(&self) -> Vec<ShardInfo> {
        let ring = self.hash_ring.read().await;
        ring.get_all_shards().into_iter().cloned().collect()
    }

    pub async fn get_healthy_shards(&self) -> Vec<ShardInfo> {
        let ring = self.hash_ring.read().await;
        ring.get_healthy_shards().into_iter().cloned().collect()
    }

    pub async fn update_shard_health(
        &self,
        shard_id: &str,
        is_healthy: bool,
    ) -> Result<HealthStatus> {
        let mut ring = self.hash_ring.write().await;

        let shard = ring
            .shards
            .get_mut(shard_id)
            .context("Shard not found")?;

        shard.last_health_check = SystemTime::now();

        if is_healthy {
            shard.consecutive_failures = 0;
            shard.consecutive_successes += 1;

            if shard.consecutive_successes >= RECOVERY_THRESHOLD {
                shard.health_status = HealthStatus::Healthy;
            }
        } else {
            shard.consecutive_successes = 0;
            shard.consecutive_failures += 1;

            if shard.consecutive_failures >= FAILURE_THRESHOLD {
                shard.health_status = HealthStatus::Unhealthy;
            } else if shard.consecutive_failures > 0 {
                shard.health_status = HealthStatus::Degraded;
            }
        }

        Ok(shard.health_status)
    }

    pub async fn rebalance(&self) -> Result<Vec<RebalanceOperation>> {
        let ring = self.hash_ring.read().await;
        let mut operations = Vec::new();

        let shards = ring.get_all_shards();
        if shards.is_empty() {
            return Ok(operations);
        }

        let total_embeddings: u64 = shards.iter().map(|s| s.embedding_count).sum();
        let avg_embeddings = total_embeddings / shards.len() as u64;
        let threshold = (avg_embeddings as f64 * 0.2) as u64;

        for shard in shards {
            let diff = shard.embedding_count.abs_diff(avg_embeddings);

            if diff > threshold {
                if shard.embedding_count > avg_embeddings {
                    operations.push(RebalanceOperation::MoveEmbeddings {
                        from_shard: shard.id.clone(),
                        to_shard: Self::find_underutilized_shard(&ring, avg_embeddings),
                        count: diff / 2,
                    });
                }
            }
        }

        Ok(operations)
    }

    fn find_underutilized_shard(
        ring: &ConsistentHashRing,
        avg_embeddings: u64,
    ) -> String {
        ring.get_healthy_shards()
            .iter()
            .filter(|s| s.embedding_count < avg_embeddings)
            .min_by_key(|s| s.embedding_count)
            .map(|s| s.id.clone())
            .unwrap_or_else(|| {
                ring.get_healthy_shards()
                    .first()
                    .map(|s| s.id.clone())
                    .unwrap_or_default()
            })
    }

    pub async fn start_health_monitor(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(HEALTH_CHECK_INTERVAL);

            loop {
                interval.tick().await;

                let shards = self.get_all_shards().await;

                for shard in shards {
                    let health_result = Self::check_shard_health(&shard.address).await;
                    let _ = self
                        .update_shard_health(&shard.id, health_result.is_ok())
                        .await;
                }
            }
        });
    }

    async fn check_shard_health(address: &str) -> Result<()> {
        let timeout = Duration::from_secs(2);

        tokio::time::timeout(timeout, async {
            Ok(())
        })
        .await
        .context("Health check timeout")?
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RebalanceOperation {
    MoveEmbeddings {
        from_shard: String,
        to_shard: String,
        count: u64,
    },
    ReplicateEmbeddings {
        from_shard: String,
        to_shard: String,
        embedding_ids: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_hash_ring() {
        let mut ring = ConsistentHashRing::new(100);

        let shard1 = ShardInfo {
            id: "shard-1".to_string(),
            address: "localhost:5001".to_string(),
            embedding_count: 1000000,
            health_status: HealthStatus::Healthy,
            last_health_check: SystemTime::now(),
            consecutive_failures: 0,
            consecutive_successes: 5,
            gpu_model: "T4".to_string(),
            capacity_weight: 1.0,
        };

        ring.add_shard(shard1.clone()).unwrap();

        let shard = ring.get_shard("test-key");
        assert!(shard.is_some());
        assert_eq!(shard.unwrap().id, "shard-1");
    }

    #[tokio::test]
    async fn test_shard_manager_add_remove() {
        let manager = ShardManager::new(3);

        let shard = ShardInfo {
            id: "shard-1".to_string(),
            address: "localhost:5001".to_string(),
            embedding_count: 1000000,
            health_status: HealthStatus::Healthy,
            last_health_check: SystemTime::now(),
            consecutive_failures: 0,
            consecutive_successes: 5,
            gpu_model: "T4".to_string(),
            capacity_weight: 1.0,
        };

        manager.add_node(shard.clone()).await.unwrap();

        let shards = manager.get_all_shards().await;
        assert_eq!(shards.len(), 1);

        manager.remove_node("shard-1").await.unwrap();

        let shards = manager.get_all_shards().await;
        assert_eq!(shards.len(), 0);
    }

    #[tokio::test]
    async fn test_health_status_transitions() {
        let manager = ShardManager::new(3);

        let shard = ShardInfo {
            id: "shard-1".to_string(),
            address: "localhost:5001".to_string(),
            embedding_count: 1000000,
            health_status: HealthStatus::Healthy,
            last_health_check: SystemTime::now(),
            consecutive_failures: 0,
            consecutive_successes: 5,
            gpu_model: "T4".to_string(),
            capacity_weight: 1.0,
        };

        manager.add_node(shard).await.unwrap();

        let status = manager.update_shard_health("shard-1", false).await.unwrap();
        assert_eq!(status, HealthStatus::Degraded);

        for _ in 0..2 {
            let _ = manager.update_shard_health("shard-1", false).await;
        }

        let status = manager.update_shard_health("shard-1", false).await.unwrap();
        assert_eq!(status, HealthStatus::Unhealthy);
    }
}
