use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDBMigrationStats {
    pub experiences_migrated: i64,
    pub q_values_migrated: i64,
    pub policies_migrated: i64,
    pub errors: i64,
    pub duration_secs: u64,
}

impl AgentDBMigrationStats {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║         AGENTDB MIGRATION SUMMARY                         ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        println!("📊 RESULTS:");
        println!("  • RL Experiences: {:>12}", self.experiences_migrated);
        println!("  • Q-Values:       {:>12}", self.q_values_migrated);
        println!("  • Policies:       {:>12}", self.policies_migrated);
        println!("  • Errors:         {:>12}", self.errors);

        println!("\n⏱️  Duration: {} seconds", self.duration_secs);
        println!();
    }
}

pub async fn migrate_rl_state(skip_history: bool, batch_size: usize) -> Result<AgentDBMigrationStats> {
    let start = std::time::Instant::now();

    info!("Initializing AgentDB PostgreSQL schema...");
    let postgres = connect_postgres().await?;
    create_agentdb_schema(&postgres).await?;

    info!("Connecting to Redis (current RL state)...");
    let redis = connect_redis().await?;

    let mut stats = AgentDBMigrationStats {
        experiences_migrated: 0,
        q_values_migrated: 0,
        policies_migrated: 0,
        errors: 0,
        duration_secs: 0,
    };

    // Migrate RL experiences
    info!("Migrating RL experiences...");
    if !skip_history {
        stats.experiences_migrated = migrate_experiences(&redis, &postgres, batch_size).await?;
    } else {
        info!("Skipping experience history migration");
    }

    // Migrate Q-values
    info!("Migrating Q-values...");
    stats.q_values_migrated = migrate_q_values(&redis, &postgres, batch_size).await?;

    // Migrate policies
    info!("Migrating policies...");
    stats.policies_migrated = migrate_policies(&redis, &postgres).await?;

    stats.duration_secs = start.elapsed().as_secs();

    info!("AgentDB migration completed");
    Ok(stats)
}

async fn create_agentdb_schema(postgres: &PostgresClient) -> Result<()> {
    info!("Creating AgentDB tables...");

    postgres.execute(
        r#"
        CREATE TABLE IF NOT EXISTS rl_experiences (
            id BIGSERIAL PRIMARY KEY,
            agent_id VARCHAR(100) NOT NULL,
            state JSONB NOT NULL,
            action VARCHAR(100) NOT NULL,
            reward FLOAT NOT NULL,
            next_state JSONB NOT NULL,
            done BOOLEAN NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            INDEX idx_agent_timestamp (agent_id, timestamp DESC)
        );

        CREATE TABLE IF NOT EXISTS q_values (
            agent_id VARCHAR(100) NOT NULL,
            state_hash VARCHAR(64) NOT NULL,
            action VARCHAR(100) NOT NULL,
            q_value FLOAT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (agent_id, state_hash, action)
        );

        CREATE TABLE IF NOT EXISTS policies (
            agent_id VARCHAR(100) PRIMARY KEY,
            policy_data BYTEA NOT NULL,
            version INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS agent_metadata (
            agent_id VARCHAR(100) PRIMARY KEY,
            learning_rate FLOAT,
            epsilon FLOAT,
            discount_factor FLOAT,
            total_episodes BIGINT,
            total_rewards FLOAT,
            metadata JSONB,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        "#
    ).await?;

    info!("AgentDB schema created");
    Ok(())
}

async fn migrate_experiences(
    redis: &RedisClient,
    postgres: &PostgresClient,
    batch_size: usize,
) -> Result<i64> {
    let mut total = 0i64;

    // Get all experience keys from Redis
    let keys = redis.scan_keys("rl:experience:*").await?;
    info!("Found {} experience keys", keys.len());

    for chunk in keys.chunks(batch_size) {
        let mut batch = Vec::new();

        for key in chunk {
            match redis.get::<String>(key).await {
                Ok(Some(json)) => {
                    match serde_json::from_str::<Experience>(&json) {
                        Ok(exp) => batch.push(exp),
                        Err(e) => warn!("Failed to parse experience {}: {}", key, e),
                    }
                },
                Ok(None) => {},
                Err(e) => warn!("Failed to fetch {}: {}", key, e),
            }
        }

        if !batch.is_empty() {
            postgres.insert_experiences_batch(&batch).await?;
            total += batch.len() as i64;
            info!("Migrated {} experiences", total);
        }
    }

    Ok(total)
}

async fn migrate_q_values(
    redis: &RedisClient,
    postgres: &PostgresClient,
    batch_size: usize,
) -> Result<i64> {
    let mut total = 0i64;

    let keys = redis.scan_keys("rl:qvalue:*").await?;
    info!("Found {} Q-value keys", keys.len());

    for chunk in keys.chunks(batch_size) {
        let mut batch = Vec::new();

        for key in chunk {
            match redis.get::<String>(key).await {
                Ok(Some(json)) => {
                    match serde_json::from_str::<QValue>(&json) {
                        Ok(qval) => batch.push(qval),
                        Err(e) => warn!("Failed to parse Q-value {}: {}", key, e),
                    }
                },
                Ok(None) => {},
                Err(e) => warn!("Failed to fetch {}: {}", key, e),
            }
        }

        if !batch.is_empty() {
            postgres.insert_q_values_batch(&batch).await?;
            total += batch.len() as i64;
            info!("Migrated {} Q-values", total);
        }
    }

    Ok(total)
}

async fn migrate_policies(
    redis: &RedisClient,
    postgres: &PostgresClient,
) -> Result<i64> {
    let mut total = 0i64;

    let keys = redis.scan_keys("rl:policy:*").await?;
    info!("Found {} policy keys", keys.len());

    for key in keys {
        match redis.get::<Vec<u8>>(key).await {
            Ok(Some(data)) => {
                let agent_id = key.trim_start_matches("rl:policy:");
                postgres.insert_policy(agent_id, &data).await?;
                total += 1;
            },
            Ok(None) => {},
            Err(e) => warn!("Failed to fetch {}: {}", key, e),
        }
    }

    info!("Migrated {} policies", total);
    Ok(total)
}

// Mock types

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Experience {
    agent_id: String,
    state: serde_json::Value,
    action: String,
    reward: f32,
    next_state: serde_json::Value,
    done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QValue {
    agent_id: String,
    state_hash: String,
    action: String,
    q_value: f32,
}

struct PostgresClient;
struct RedisClient;

async fn connect_postgres() -> Result<PostgresClient> {
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(PostgresClient)
}

async fn connect_redis() -> Result<RedisClient> {
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(RedisClient)
}

impl PostgresClient {
    async fn execute(&self, _sql: &str) -> Result<()> {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok(())
    }

    async fn insert_experiences_batch(&self, _batch: &[Experience]) -> Result<()> {
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        Ok(())
    }

    async fn insert_q_values_batch(&self, _batch: &[QValue]) -> Result<()> {
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        Ok(())
    }

    async fn insert_policy(&self, _agent_id: &str, _data: &[u8]) -> Result<()> {
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        Ok(())
    }
}

impl RedisClient {
    async fn scan_keys(&self, _pattern: &str) -> Result<Vec<String>> {
        Ok(vec![])
    }

    async fn get<T>(&self, _key: &str) -> Result<Option<T>>
    where
        T: serde::de::DeserializeOwned,
    {
        Ok(None)
    }
}
