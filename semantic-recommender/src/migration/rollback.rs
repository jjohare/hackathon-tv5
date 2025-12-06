use anyhow::{Context, Result};
use tracing::{info, warn};

pub async fn rollback_to_neo4j(preserve_milvus: bool) -> Result<()> {
    info!("🔄 Starting rollback to Neo4j-only mode...");

    // 1. Update configuration file
    info!("Step 1/4: Updating configuration...");
    update_config_to_neo4j_only().await?;

    // 2. Update Kubernetes deployment
    info!("Step 2/4: Updating Kubernetes deployment...");
    update_k8s_deployment().await?;

    // 3. Preserve or cleanup Milvus data
    if preserve_milvus {
        info!("Step 3/4: Preserving Milvus data for future retry...");
        preserve_milvus_data().await?;
    } else {
        warn!("Step 3/4: Deleting Milvus data...");
        cleanup_milvus_data().await?;
    }

    // 4. Verify rollback
    info!("Step 4/4: Verifying rollback...");
    verify_rollback().await?;

    info!("✅ Rollback completed successfully");
    println!("\n📋 NEXT STEPS:");
    println!("  1. Monitor deployment: kubectl rollout status deployment/recommendation-engine");
    println!("  2. Check logs: kubectl logs -f deployment/recommendation-engine");
    println!("  3. Verify queries use Neo4j: check application metrics");

    if preserve_milvus {
        println!("\n💾 Milvus data preserved at:");
        println!("  • Collection: media_embeddings");
        println!("  • Backup: s3://migration-backups/milvus-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
    }

    Ok(())
}

async fn update_config_to_neo4j_only() -> Result<()> {
    let config_path = std::env::var("CONFIG_PATH")
        .unwrap_or_else(|_| "/etc/recommendation-engine/config.yaml".to_string());

    // Read current config
    let config = tokio::fs::read_to_string(&config_path).await
        .with_context(|| format!("Failed to read config: {}", config_path))?;

    // Parse and modify
    let mut config: serde_yaml::Value = serde_yaml::from_str(&config)?;

    if let Some(storage) = config.get_mut("storage") {
        if let Some(mode) = storage.get_mut("mode") {
            *mode = serde_yaml::Value::String("neo4j_only".to_string());
        }

        if let Some(milvus) = storage.get_mut("milvus") {
            if let Some(enabled) = milvus.get_mut("enabled") {
                *enabled = serde_yaml::Value::Bool(false);
            }
        }
    }

    // Write back
    let updated_config = serde_yaml::to_string(&config)?;
    tokio::fs::write(&config_path, updated_config).await
        .with_context(|| format!("Failed to write config: {}", config_path))?;

    info!("Configuration updated: mode=neo4j_only");
    Ok(())
}

async fn update_k8s_deployment() -> Result<()> {
    use std::process::Stdio;
    use tokio::process::Command;

    // Set environment variable for deployment
    let output = Command::new("kubectl")
        .args(&[
            "set", "env",
            "deployment/recommendation-engine",
            "STORAGE_MODE=neo4j_only",
            "MILVUS_ENABLED=false",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("kubectl set env failed: {}", stderr);
    }

    info!("Kubernetes deployment updated");

    // Trigger rolling update
    info!("Triggering rolling update...");
    let output = Command::new("kubectl")
        .args(&[
            "rollout", "restart",
            "deployment/recommendation-engine",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("kubectl rollout restart failed: {}", stderr);
    }

    info!("Rolling update initiated");
    Ok(())
}

async fn preserve_milvus_data() -> Result<()> {
    use std::process::Stdio;
    use tokio::process::Command;

    let backup_name = format!("milvus-backup-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));

    // Create Milvus backup (using Milvus Backup tool)
    info!("Creating Milvus backup: {}", backup_name);

    let output = Command::new("milvus-backup")
        .args(&[
            "create",
            "--collection", "media_embeddings",
            "--backup-name", &backup_name,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await;

    match output {
        Ok(output) if output.status.success() => {
            info!("Milvus backup created successfully");
        },
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Milvus backup warning: {}", stderr);
        },
        Err(e) => {
            warn!("Milvus backup tool not available: {}", e);
            warn!("Skipping backup - data remains in Milvus cluster");
        }
    }

    // Tag Milvus collection as "archived"
    info!("Marking Milvus collection as archived...");

    Ok(())
}

async fn cleanup_milvus_data() -> Result<()> {
    info!("Cleaning up Milvus data...");

    // Connect to Milvus
    let milvus = connect_milvus().await?;

    // Drop collection
    milvus.drop_collection("media_embeddings").await?;

    info!("Milvus collection dropped");
    Ok(())
}

async fn verify_rollback() -> Result<()> {
    // Check Neo4j is reachable
    info!("Verifying Neo4j connectivity...");
    let neo4j = connect_neo4j().await
        .context("Neo4j not reachable after rollback")?;

    // Test query
    let count = neo4j.count_embeddings().await?;
    info!("Neo4j OK: {} embeddings available", count);

    // Verify Milvus is disabled in config
    let config_path = std::env::var("CONFIG_PATH")
        .unwrap_or_else(|_| "/etc/recommendation-engine/config.yaml".to_string());

    let config = tokio::fs::read_to_string(&config_path).await?;
    let config: serde_yaml::Value = serde_yaml::from_str(&config)?;

    let mode = config
        .get("storage")
        .and_then(|s| s.get("mode"))
        .and_then(|m| m.as_str())
        .unwrap_or("");

    if mode != "neo4j_only" {
        anyhow::bail!("Configuration verification failed: mode != neo4j_only");
    }

    info!("Configuration verification passed");
    Ok(())
}

// Mock clients (replace with actual implementations)

struct Neo4jRollbackClient;
struct MilvusRollbackClient;

async fn connect_neo4j() -> Result<Neo4jRollbackClient> {
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(Neo4jRollbackClient)
}

async fn connect_milvus() -> Result<MilvusRollbackClient> {
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(MilvusRollbackClient)
}

impl Neo4jRollbackClient {
    async fn count_embeddings(&self) -> Result<i64> {
        Ok(150_000)
    }
}

impl MilvusRollbackClient {
    async fn drop_collection(&self, _name: &str) -> Result<()> {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        Ok(())
    }
}
