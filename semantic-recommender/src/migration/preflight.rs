use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, warn, error};

#[derive(Debug, Serialize, Deserialize)]
pub struct PreflightReport {
    // Neo4j checks
    pub neo4j_reachable: bool,
    pub neo4j_version: String,
    pub neo4j_disk_usage: u64,
    pub total_media_items: i64,
    pub total_embeddings: i64,

    // Milvus checks
    pub milvus_reachable: bool,
    pub milvus_healthy: bool,
    pub milvus_version: String,
    pub milvus_collections_ready: bool,

    // PostgreSQL checks
    pub postgres_reachable: bool,
    pub postgres_version: String,
    pub agentdb_schema_ready: bool,

    // Redis checks
    pub redis_reachable: bool,
    pub redis_memory_available: u64,

    // Resource checks
    pub disk_space_available: u64,
    pub disk_space_required: u64,
    pub memory_available: u64,
    pub memory_required: u64,

    // Estimates
    pub estimated_migration_time: Duration,
    pub estimated_downtime: Duration,

    // Validation
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl PreflightReport {
    pub fn is_ready(&self) -> bool {
        self.errors.is_empty()
            && self.neo4j_reachable
            && self.milvus_reachable
            && self.postgres_reachable
            && self.disk_space_available >= self.disk_space_required
            && self.memory_available >= self.memory_required
    }

    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║             MIGRATION PREFLIGHT REPORT                    ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        println!("📊 DATA INVENTORY:");
        println!("  • Media items:     {:>10}", self.total_media_items);
        println!("  • Embeddings:      {:>10}", self.total_embeddings);
        println!("  • Neo4j disk:      {:>10} GB", self.neo4j_disk_usage / 1_000_000_000);

        println!("\n🔌 CONNECTIVITY:");
        self.print_status("Neo4j", self.neo4j_reachable, &self.neo4j_version);
        self.print_status("Milvus", self.milvus_reachable, &self.milvus_version);
        self.print_status("PostgreSQL", self.postgres_reachable, &self.postgres_version);
        self.print_status("Redis", self.redis_reachable, "");

        println!("\n💾 RESOURCES:");
        println!("  • Disk available:  {:>10} GB", self.disk_space_available / 1_000_000_000);
        println!("  • Disk required:   {:>10} GB", self.disk_space_required / 1_000_000_000);
        println!("  • Memory available:{:>10} GB", self.memory_available / 1_000_000_000);
        println!("  • Memory required: {:>10} GB", self.memory_required / 1_000_000_000);

        println!("\n⏱️  ESTIMATES:");
        println!("  • Migration time:  {:>10} min", self.estimated_migration_time.as_secs() / 60);
        println!("  • Expected downtime: {:>7} sec", self.estimated_downtime.as_secs());

        if !self.errors.is_empty() {
            println!("\n❌ ERRORS:");
            for err in &self.errors {
                println!("  • {}", err);
            }
        }

        if !self.warnings.is_empty() {
            println!("\n⚠️  WARNINGS:");
            for warn in &self.warnings {
                println!("  • {}", warn);
            }
        }

        println!("\n{}", if self.is_ready() {
            "✅ READY FOR MIGRATION"
        } else {
            "❌ NOT READY - Fix errors above"
        });
        println!();
    }

    fn print_status(&self, name: &str, reachable: bool, version: &str) {
        let status = if reachable { "✅" } else { "❌" };
        let version_str = if !version.is_empty() {
            format!(" ({})", version)
        } else {
            String::new()
        };
        println!("  {} {:<15} {}", status, name, version_str);
    }
}

pub async fn run_preflight_checks(check_connectivity: bool) -> Result<PreflightReport> {
    let mut report = PreflightReport {
        neo4j_reachable: false,
        neo4j_version: String::new(),
        neo4j_disk_usage: 0,
        total_media_items: 0,
        total_embeddings: 0,
        milvus_reachable: false,
        milvus_healthy: false,
        milvus_version: String::new(),
        milvus_collections_ready: false,
        postgres_reachable: false,
        postgres_version: String::new(),
        agentdb_schema_ready: false,
        redis_reachable: false,
        redis_memory_available: 0,
        disk_space_available: 0,
        disk_space_required: 0,
        memory_available: 0,
        memory_required: 0,
        estimated_migration_time: Duration::from_secs(0),
        estimated_downtime: Duration::from_secs(0),
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    // 1. Check Neo4j
    info!("Checking Neo4j connectivity...");
    match check_neo4j().await {
        Ok((version, items, embeddings, disk)) => {
            report.neo4j_reachable = true;
            report.neo4j_version = version;
            report.total_media_items = items;
            report.total_embeddings = embeddings;
            report.neo4j_disk_usage = disk;
            info!("✅ Neo4j OK: {} items, {} embeddings", items, embeddings);
        },
        Err(e) => {
            report.errors.push(format!("Neo4j unreachable: {}", e));
            error!("❌ Neo4j check failed: {}", e);
        }
    }

    // 2. Check Milvus
    if check_connectivity {
        info!("Checking Milvus cluster...");
        match check_milvus().await {
            Ok((version, healthy, collections_ok)) => {
                report.milvus_reachable = true;
                report.milvus_version = version;
                report.milvus_healthy = healthy;
                report.milvus_collections_ready = collections_ok;

                if !healthy {
                    report.warnings.push("Milvus cluster not healthy".to_string());
                }
                if !collections_ok {
                    report.warnings.push("Milvus collections not created - will be auto-created".to_string());
                }

                info!("✅ Milvus OK: healthy={}, collections={}", healthy, collections_ok);
            },
            Err(e) => {
                report.errors.push(format!("Milvus unreachable: {}", e));
                error!("❌ Milvus check failed: {}", e);
            }
        }
    }

    // 3. Check PostgreSQL
    info!("Checking PostgreSQL...");
    match check_postgres().await {
        Ok((version, schema_ready)) => {
            report.postgres_reachable = true;
            report.postgres_version = version;
            report.agentdb_schema_ready = schema_ready;

            if !schema_ready {
                report.warnings.push("AgentDB schema not initialized - will be created".to_string());
            }

            info!("✅ PostgreSQL OK: version={}, schema={}", version, schema_ready);
        },
        Err(e) => {
            report.errors.push(format!("PostgreSQL unreachable: {}", e));
            error!("❌ PostgreSQL check failed: {}", e);
        }
    }

    // 4. Check Redis
    info!("Checking Redis...");
    match check_redis().await {
        Ok(memory) => {
            report.redis_reachable = true;
            report.redis_memory_available = memory;
            info!("✅ Redis OK: {} GB available", memory / 1_000_000_000);
        },
        Err(e) => {
            report.errors.push(format!("Redis unreachable: {}", e));
            error!("❌ Redis check failed: {}", e);
        }
    }

    // 5. Check system resources
    info!("Checking system resources...");
    report.disk_space_available = get_disk_space()?;
    report.memory_available = get_available_memory()?;

    // Calculate requirements
    // Estimate: 1KB per embedding (768 dims * FP16 = 1.5KB, compressed to ~1KB)
    // Milvus: 3x replication factor
    let embedding_size = report.total_embeddings as u64 * 1024;
    report.disk_space_required = embedding_size * 3;  // Milvus replication
    report.memory_required = embedding_size / 10;     // ~10% for HNSW index

    if report.disk_space_available < report.disk_space_required {
        report.errors.push(format!(
            "Insufficient disk space: need {} GB, have {} GB",
            report.disk_space_required / 1_000_000_000,
            report.disk_space_available / 1_000_000_000
        ));
    }

    if report.memory_available < report.memory_required {
        report.warnings.push(format!(
            "Low memory: recommended {} GB, have {} GB",
            report.memory_required / 1_000_000_000,
            report.memory_available / 1_000_000_000
        ));
    }

    // 6. Estimate migration time
    // Throughput: ~1000 embeddings/sec with batch_size=1000
    let migration_secs = report.total_embeddings as u64 / 1000;
    report.estimated_migration_time = Duration::from_secs(migration_secs);

    // Zero-downtime migration, but ~10 sec rolling update
    report.estimated_downtime = Duration::from_secs(10);

    Ok(report)
}

async fn check_neo4j() -> Result<(String, i64, i64, u64)> {
    // Mock implementation - replace with actual neo4rs client
    let neo4j_url = std::env::var("NEO4J_URI")
        .unwrap_or_else(|_| "bolt://localhost:7687".to_string());

    info!("Connecting to Neo4j at {}", neo4j_url);

    // In production, use:
    // let graph = neo4rs::Graph::new(&neo4j_url, user, password).await?;
    // let version = graph.run(query("CALL dbms.components()")).await?;
    // let items = graph.execute(query("MATCH (m:MediaContent) RETURN count(m)")).await?;

    // Mock data for demonstration
    Ok((
        "5.15.0".to_string(),     // version
        150_000,                   // media items
        150_000,                   // embeddings
        50_000_000_000,           // disk usage (50 GB)
    ))
}

async fn check_milvus() -> Result<(String, bool, bool)> {
    let milvus_url = std::env::var("MILVUS_URI")
        .unwrap_or_else(|_| "http://localhost:19530".to_string());

    info!("Connecting to Milvus at {}", milvus_url);

    // In production, use Milvus client SDK
    // let client = milvus::Client::connect(&milvus_url).await?;
    // let version = client.get_version().await?;
    // let health = client.health_check().await?;

    Ok((
        "2.3.4".to_string(),      // version
        true,                      // healthy
        false,                     // collections not created yet
    ))
}

async fn check_postgres() -> Result<(String, bool)> {
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://localhost:5432/agentdb".to_string());

    info!("Connecting to PostgreSQL at {}", db_url);

    // In production:
    // let pool = sqlx::PgPool::connect(&db_url).await?;
    // let version: String = sqlx::query_scalar("SELECT version()").fetch_one(&pool).await?;
    // let schema_check: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'rl_experiences'").fetch_one(&pool).await?;

    Ok((
        "PostgreSQL 15.4".to_string(),  // version
        false,                          // schema not ready
    ))
}

async fn check_redis() -> Result<u64> {
    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://localhost:6379".to_string());

    info!("Connecting to Redis at {}", redis_url);

    // In production:
    // let client = redis::Client::open(redis_url)?;
    // let mut conn = client.get_async_connection().await?;
    // let info: redis::InfoDict = redis::cmd("INFO").query_async(&mut conn).await?;

    Ok(8_000_000_000)  // 8 GB
}

fn get_disk_space() -> Result<u64> {
    // In production, use sys-info or similar crate
    // let info = sys_info::disk_info()?;
    Ok(500_000_000_000)  // 500 GB
}

fn get_available_memory() -> Result<u64> {
    // In production, use sys-info
    // let info = sys_info::mem_info()?;
    Ok(32_000_000_000)  // 32 GB
}
