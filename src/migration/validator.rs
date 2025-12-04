use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationReport {
    pub sample_size: usize,
    pub full_scan: bool,
    pub matches: usize,
    pub mismatches: usize,
    pub missing_in_milvus: usize,
    pub missing_in_neo4j: usize,
    pub success_rate: f64,
    pub max_vector_diff: f64,
    pub avg_vector_diff: f64,
    pub inconsistencies_fixed: usize,
}

impl ValidationReport {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║             MIGRATION VALIDATION REPORT                   ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        println!("📊 VALIDATION SCOPE:");
        println!("  • Sample size:     {:>10}", self.sample_size);
        println!("  • Full scan:       {:>10}", if self.full_scan { "YES" } else { "NO" });

        println!("\n✅ RESULTS:");
        println!("  • Matches:         {:>10}", self.matches);
        println!("  • Mismatches:      {:>10}", self.mismatches);
        println!("  • Missing (Milvus):{:>10}", self.missing_in_milvus);
        println!("  • Missing (Neo4j): {:>10}", self.missing_in_neo4j);
        println!("  • Fixed:           {:>10}", self.inconsistencies_fixed);

        println!("\n📈 ACCURACY:");
        println!("  • Success rate:    {:>9.2}%", self.success_rate * 100.0);
        println!("  • Max vector diff: {:>10.6}", self.max_vector_diff);
        println!("  • Avg vector diff: {:>10.6}", self.avg_vector_diff);

        let status = if self.success_rate >= 0.99 {
            "✅ VALIDATION PASSED"
        } else if self.success_rate >= 0.95 {
            "⚠️  VALIDATION WARNING"
        } else {
            "❌ VALIDATION FAILED"
        };

        println!("\n{}", status);
        println!();
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MigrationProgress {
    pub total_items: i64,
    pub migrated_items: i64,
    pub failed_items: i64,
    pub current_batch: usize,
    pub progress_percent: f64,
    pub elapsed_secs: u64,
    pub eta_secs: u64,
    pub throughput_per_sec: f64,
}

impl MigrationProgress {
    pub fn print_live(&self) {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║           MIGRATION PROGRESS (LIVE)                       ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        let bar_width = 50;
        let filled = (self.progress_percent * bar_width as f64) as usize;
        let empty = bar_width - filled;

        println!("Progress: [{}{}] {:.1}%",
            "█".repeat(filled),
            "░".repeat(empty),
            self.progress_percent * 100.0
        );

        println!("\n📊 STATISTICS:");
        println!("  • Total:         {:>12}", self.total_items);
        println!("  • Migrated:      {:>12}", self.migrated_items);
        println!("  • Failed:        {:>12}", self.failed_items);
        println!("  • Current batch: {:>12}", self.current_batch);

        println!("\n⏱️  TIMING:");
        println!("  • Elapsed:       {:>9} min", self.elapsed_secs / 60);
        println!("  • ETA:           {:>9} min", self.eta_secs / 60);
        println!("  • Throughput:    {:>9.0} items/sec", self.throughput_per_sec);

        println!();
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MigrationReport {
    pub start_time: i64,
    pub end_time: i64,
    pub duration_secs: u64,
    pub total_items: i64,
    pub successful: i64,
    pub failed: i64,
    pub validation: ValidationReport,
    pub performance: PerformanceMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub avg_neo4j_latency_ms: f64,
    pub avg_milvus_latency_ms: f64,
    pub throughput_items_per_sec: f64,
    pub peak_memory_mb: u64,
}

impl MigrationReport {
    pub fn to_markdown(&self) -> String {
        format!(
            r#"# Migration Report

## Overview
- **Start**: {}
- **End**: {}
- **Duration**: {} minutes
- **Total Items**: {}
- **Successful**: {} ({:.1}%)
- **Failed**: {} ({:.1}%)

## Validation Results
- **Sample Size**: {}
- **Matches**: {}
- **Success Rate**: {:.2}%
- **Max Vector Diff**: {:.6}

## Performance
- **Neo4j Latency**: {:.2} ms
- **Milvus Latency**: {:.2} ms
- **Throughput**: {:.0} items/sec
- **Peak Memory**: {} MB

## Status
{}
"#,
            chrono::DateTime::from_timestamp(self.start_time, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default(),
            chrono::DateTime::from_timestamp(self.end_time, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default(),
            self.duration_secs / 60,
            self.total_items,
            self.successful,
            (self.successful as f64 / self.total_items as f64 * 100.0),
            self.failed,
            (self.failed as f64 / self.total_items as f64 * 100.0),
            self.validation.sample_size,
            self.validation.matches,
            self.validation.success_rate * 100.0,
            self.validation.max_vector_diff,
            self.performance.avg_neo4j_latency_ms,
            self.performance.avg_milvus_latency_ms,
            self.performance.throughput_items_per_sec,
            self.performance.peak_memory_mb,
            if self.validation.success_rate >= 0.99 {
                "✅ Migration completed successfully"
            } else {
                "⚠️ Migration completed with warnings"
            }
        )
    }
}

pub async fn validate_migration(
    sample_size: usize,
    full_scan: bool,
    fix_inconsistencies: bool,
) -> Result<ValidationReport> {
    info!("Starting validation (sample_size={}, full_scan={}, fix={})",
        sample_size, full_scan, fix_inconsistencies);

    let neo4j = connect_neo4j().await?;
    let milvus = connect_milvus().await?;

    let mut matches = 0;
    let mut mismatches = 0;
    let mut missing_in_milvus = 0;
    let mut missing_in_neo4j = 0;
    let mut inconsistencies_fixed = 0;
    let mut max_diff = 0.0;
    let mut sum_diff = 0.0;
    let mut diff_count = 0;

    // Get sample from Neo4j
    let items = if full_scan {
        neo4j.get_all_embeddings().await?
    } else {
        neo4j.get_random_embeddings(sample_size).await?
    };

    info!("Validating {} items...", items.len());

    for (i, (id, neo4j_embedding)) in items.iter().enumerate() {
        if i % 100 == 0 {
            info!("Progress: {}/{}", i, items.len());
        }

        // Check Milvus has same embedding
        match milvus.get_embedding_by_id(id).await {
            Ok(Some(milvus_embedding)) => {
                let diff = vector_l2_distance(&neo4j_embedding, &milvus_embedding);
                max_diff = max_diff.max(diff);
                sum_diff += diff;
                diff_count += 1;

                if vectors_match(&neo4j_embedding, &milvus_embedding, 1e-5) {
                    matches += 1;
                } else {
                    mismatches += 1;
                    warn!("Embedding mismatch for {}: L2 distance = {:.6}", id, diff);

                    if fix_inconsistencies {
                        // Re-sync from Neo4j (source of truth)
                        milvus.update_embedding(id, &neo4j_embedding).await?;
                        inconsistencies_fixed += 1;
                        info!("Fixed embedding for {}", id);
                    }
                }
            },
            Ok(None) => {
                missing_in_milvus += 1;
                warn!("Missing in Milvus: {}", id);

                if fix_inconsistencies {
                    milvus.insert_embedding(id, &neo4j_embedding).await?;
                    inconsistencies_fixed += 1;
                    info!("Added missing embedding to Milvus: {}", id);
                }
            },
            Err(e) => {
                error!("Error querying Milvus for {}: {}", id, e);
                missing_in_milvus += 1;
            }
        }
    }

    // Check for items in Milvus but not in Neo4j (orphaned)
    if full_scan {
        let milvus_ids = milvus.get_all_ids().await?;
        let neo4j_ids: std::collections::HashSet<_> = items.iter().map(|(id, _)| id).collect();

        for milvus_id in milvus_ids {
            if !neo4j_ids.contains(&milvus_id) {
                missing_in_neo4j += 1;
                warn!("Orphaned in Milvus (not in Neo4j): {}", milvus_id);

                if fix_inconsistencies {
                    milvus.delete_embedding(&milvus_id).await?;
                    inconsistencies_fixed += 1;
                    info!("Deleted orphaned embedding from Milvus: {}", milvus_id);
                }
            }
        }
    }

    let total_checked = items.len();
    let success_rate = if total_checked > 0 {
        matches as f64 / total_checked as f64
    } else {
        0.0
    };

    let avg_diff = if diff_count > 0 {
        sum_diff / diff_count as f64
    } else {
        0.0
    };

    let report = ValidationReport {
        sample_size: total_checked,
        full_scan,
        matches,
        mismatches,
        missing_in_milvus,
        missing_in_neo4j,
        success_rate,
        max_vector_diff: max_diff,
        avg_vector_diff: avg_diff,
        inconsistencies_fixed,
    };

    Ok(report)
}

pub async fn get_migration_progress() -> Result<MigrationProgress> {
    // In production, read from Redis or database
    // For now, mock implementation
    Ok(MigrationProgress {
        total_items: 150_000,
        migrated_items: 75_000,
        failed_items: 150,
        current_batch: 75,
        progress_percent: 0.5,
        elapsed_secs: 375,
        eta_secs: 375,
        throughput_per_sec: 200.0,
    })
}

pub async fn generate_migration_report() -> Result<MigrationReport> {
    let validation = validate_migration(1000, false, false).await?;

    Ok(MigrationReport {
        start_time: chrono::Utc::now().timestamp() - 750,
        end_time: chrono::Utc::now().timestamp(),
        duration_secs: 750,
        total_items: 150_000,
        successful: 149_850,
        failed: 150,
        validation,
        performance: PerformanceMetrics {
            avg_neo4j_latency_ms: 45.2,
            avg_milvus_latency_ms: 8.7,
            throughput_items_per_sec: 200.0,
            peak_memory_mb: 4096,
        },
    })
}

fn vectors_match(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    vector_l2_distance(a, b) < epsilon
}

fn vector_l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            (diff * diff) as f64
        })
        .sum::<f64>()
        .sqrt()
}

// Mock client functions (replace with actual implementations)

struct Neo4jValidationClient;
struct MilvusValidationClient;

async fn connect_neo4j() -> Result<Neo4jValidationClient> {
    Ok(Neo4jValidationClient)
}

async fn connect_milvus() -> Result<MilvusValidationClient> {
    Ok(MilvusValidationClient)
}

impl Neo4jValidationClient {
    async fn get_random_embeddings(&self, _limit: usize) -> Result<Vec<(String, Vec<f32>)>> {
        Ok(vec![])
    }

    async fn get_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>> {
        Ok(vec![])
    }
}

impl MilvusValidationClient {
    async fn get_embedding_by_id(&self, _id: &str) -> Result<Option<Vec<f32>>> {
        Ok(Some(vec![0.1; 768]))
    }

    async fn update_embedding(&self, _id: &str, _embedding: &[f32]) -> Result<()> {
        Ok(())
    }

    async fn insert_embedding(&self, _id: &str, _embedding: &[f32]) -> Result<()> {
        Ok(())
    }

    async fn delete_embedding(&self, _id: &str) -> Result<()> {
        Ok(())
    }

    async fn get_all_ids(&self) -> Result<Vec<String>> {
        Ok(vec![])
    }
}
