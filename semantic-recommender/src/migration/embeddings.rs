use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStats {
    pub total_items: i64,
    pub migrated: i64,
    pub skipped: i64,
    pub errors: i64,
    pub start_time: i64,
    pub end_time: i64,
    pub throughput_items_per_sec: f64,
}

impl MigrationStats {
    pub fn print_summary(&self) {
        let duration_secs = self.end_time - self.start_time;
        let duration_mins = duration_secs / 60;

        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║         EMBEDDING MIGRATION SUMMARY                       ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        println!("📊 RESULTS:");
        println!("  • Total items:    {:>12}", self.total_items);
        println!("  • Migrated:       {:>12}", self.migrated);
        println!("  • Skipped:        {:>12}", self.skipped);
        println!("  • Errors:         {:>12}", self.errors);

        println!("\n⏱️  PERFORMANCE:");
        println!("  • Duration:       {:>9} min", duration_mins);
        println!("  • Throughput:     {:>9.0} items/sec", self.throughput_items_per_sec);

        let success_rate = if self.total_items > 0 {
            (self.migrated as f64 / self.total_items as f64) * 100.0
        } else {
            0.0
        };

        println!("\n✅ Success rate: {:.2}%", success_rate);
        println!();
    }
}

pub async fn migrate_embeddings_batch(
    batch_size: usize,
    dry_run: bool,
    resume_from: Option<String>,
) -> Result<MigrationStats> {
    let start_time = chrono::Utc::now().timestamp();

    info!("Connecting to Neo4j...");
    let neo4j = connect_neo4j().await?;

    info!("Connecting to Milvus...");
    let milvus = if !dry_run {
        Some(connect_milvus().await?)
    } else {
        None
    };

    // Get total count
    let total_items = neo4j.count_embeddings().await?;
    info!("Total items to migrate: {}", total_items);

    let mut migrated = 0i64;
    let mut skipped = 0i64;
    let mut errors = 0i64;
    let mut last_id = resume_from;

    // Concurrency limiter (10 concurrent batches)
    let semaphore = Arc::new(Semaphore::new(10));

    let mut batch_num = 0;

    loop {
        batch_num += 1;

        // Fetch batch from Neo4j
        let batch = neo4j.fetch_embeddings_batch(batch_size, last_id.clone()).await?;

        if batch.is_empty() {
            info!("No more items to migrate");
            break;
        }

        info!("Batch {}: fetched {} items", batch_num, batch.len());

        // Update last_id for next iteration
        if let Some(last_item) = batch.last() {
            last_id = Some(last_item.id.clone());
        }

        // Process batch
        let batch_results = if dry_run {
            // Dry run: just count
            batch.len() as i64
        } else {
            let milvus = milvus.as_ref().unwrap().clone();
            let semaphore = semaphore.clone();

            process_batch_parallel(batch, milvus, semaphore).await?
        };

        migrated += batch_results;

        // Progress update
        let progress = (migrated as f64 / total_items as f64) * 100.0;
        info!("Progress: {}/{} ({:.1}%)", migrated, total_items, progress);

        // Save checkpoint
        if !dry_run {
            save_checkpoint(last_id.as_ref().unwrap(), migrated).await?;
        }
    }

    let end_time = chrono::Utc::now().timestamp();
    let duration_secs = (end_time - start_time).max(1);
    let throughput = migrated as f64 / duration_secs as f64;

    Ok(MigrationStats {
        total_items,
        migrated,
        skipped,
        errors,
        start_time,
        end_time,
        throughput_items_per_sec: throughput,
    })
}

async fn process_batch_parallel(
    batch: Vec<EmbeddingItem>,
    milvus: Arc<MilvusClient>,
    semaphore: Arc<Semaphore>,
) -> Result<i64> {
    use futures::stream::{self, StreamExt};

    let results = stream::iter(batch)
        .map(|item| {
            let milvus = milvus.clone();
            let semaphore = semaphore.clone();

            async move {
                let _permit = semaphore.acquire().await.unwrap();

                match milvus.insert_embedding(&item).await {
                    Ok(_) => Ok(1i64),
                    Err(e) => {
                        error!("Failed to migrate {}: {}", item.id, e);
                        Ok(0i64)
                    }
                }
            }
        })
        .buffer_unordered(10)
        .collect::<Vec<_>>()
        .await;

    let total: i64 = results.into_iter().filter_map(|r| r.ok()).sum();
    Ok(total)
}

async fn save_checkpoint(last_id: &str, migrated: i64) -> Result<()> {
    // In production: save to Redis or database
    let checkpoint = serde_json::json!({
        "last_id": last_id,
        "migrated": migrated,
        "timestamp": chrono::Utc::now().timestamp(),
    });

    tokio::fs::write(
        "/tmp/migration_checkpoint.json",
        serde_json::to_string_pretty(&checkpoint)?
    ).await?;

    Ok(())
}

// Mock types and clients

#[derive(Debug, Clone)]
pub struct EmbeddingItem {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: serde_json::Value,
}

struct Neo4jMigrationClient;
struct MilvusClient;

async fn connect_neo4j() -> Result<Arc<Neo4jMigrationClient>> {
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(Arc::new(Neo4jMigrationClient))
}

async fn connect_milvus() -> Result<Arc<MilvusClient>> {
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(Arc::new(MilvusClient))
}

impl Neo4jMigrationClient {
    async fn count_embeddings(&self) -> Result<i64> {
        // In production: MATCH (m:MediaContent) RETURN count(m)
        Ok(150_000)
    }

    async fn fetch_embeddings_batch(
        &self,
        limit: usize,
        after_id: Option<String>,
    ) -> Result<Vec<EmbeddingItem>> {
        // In production:
        // MATCH (m:MediaContent)
        // WHERE m.id > $after_id
        // RETURN m.id, m.embedding, m.metadata
        // ORDER BY m.id
        // LIMIT $limit

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Mock: return empty to signal completion
        Ok(vec![])
    }
}

impl MilvusClient {
    async fn insert_embedding(&self, item: &EmbeddingItem) -> Result<()> {
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        Ok(())
    }
}
