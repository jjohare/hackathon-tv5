use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber;

mod migration {
    pub mod preflight;
    pub mod embeddings;
    pub mod agentdb;
    pub mod validator;
    pub mod rollback;
}

#[derive(Parser)]
#[command(name = "migrate")]
#[command(about = "Hybrid architecture migration tool", version, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, global = true, default_value = "info")]
    log_level: Level,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate pre-migration checks
    Preflight {
        #[arg(long)]
        check_connectivity: bool,

        #[arg(long)]
        verbose: bool,
    },

    /// Migrate embeddings from Neo4j to Milvus
    MigrateEmbeddings {
        #[arg(long, default_value = "1000")]
        batch_size: usize,

        #[arg(long)]
        dry_run: bool,

        #[arg(long)]
        resume_from: Option<String>,
    },

    /// Create AgentDB schema and migrate RL state
    MigrateAgentDB {
        #[arg(long)]
        skip_history: bool,

        #[arg(long)]
        batch_size: Option<usize>,
    },

    /// Validate migration completeness
    Validate {
        #[arg(long, default_value = "100")]
        sample_size: usize,

        #[arg(long)]
        full_scan: bool,

        #[arg(long)]
        fix_inconsistencies: bool,
    },

    /// Rollback to Neo4j-only
    Rollback {
        #[arg(long)]
        confirm: bool,

        #[arg(long)]
        preserve_milvus: bool,
    },

    /// Monitor migration progress
    Monitor {
        #[arg(long, default_value = "5")]
        interval_secs: u64,
    },

    /// Generate migration report
    Report {
        #[arg(long)]
        output_file: Option<String>,

        #[arg(long)]
        format: Option<ReportFormat>,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ReportFormat {
    Json,
    Yaml,
    Markdown,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_max_level(cli.log_level)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("🚀 Migration tool starting...");

    let result = match cli.command {
        Commands::Preflight { check_connectivity, verbose } => {
            preflight_checks(check_connectivity, verbose).await
        },
        Commands::MigrateEmbeddings { batch_size, dry_run, resume_from } => {
            migrate_embeddings(batch_size, dry_run, resume_from).await
        },
        Commands::MigrateAgentDB { skip_history, batch_size } => {
            migrate_agentdb(skip_history, batch_size).await
        },
        Commands::Validate { sample_size, full_scan, fix_inconsistencies } => {
            validate_migration(sample_size, full_scan, fix_inconsistencies).await
        },
        Commands::Rollback { confirm, preserve_milvus } => {
            rollback_migration(confirm, preserve_milvus).await
        },
        Commands::Monitor { interval_secs } => {
            monitor_migration(interval_secs).await
        },
        Commands::Report { output_file, format } => {
            generate_report(output_file, format).await
        },
    };

    match result {
        Ok(_) => {
            info!("✅ Operation completed successfully");
            Ok(())
        },
        Err(e) => {
            tracing::error!("❌ Operation failed: {:#}", e);
            std::process::exit(1);
        }
    }
}

async fn preflight_checks(check_connectivity: bool, verbose: bool) -> Result<()> {
    use migration::preflight::run_preflight_checks;

    info!("Running preflight checks...");
    let report = run_preflight_checks(check_connectivity).await
        .context("Preflight checks failed")?;

    if verbose {
        println!("{:#?}", report);
    } else {
        report.print_summary();
    }

    if !report.is_ready() {
        anyhow::bail!("System not ready for migration. Please fix errors above.");
    }

    Ok(())
}

async fn migrate_embeddings(batch_size: usize, dry_run: bool, resume_from: Option<String>) -> Result<()> {
    use migration::embeddings::migrate_embeddings_batch;

    if dry_run {
        info!("🔍 DRY RUN MODE - No data will be modified");
    }

    info!("Migrating embeddings from Neo4j to Milvus (batch_size={})", batch_size);

    let stats = migrate_embeddings_batch(batch_size, dry_run, resume_from).await
        .context("Embedding migration failed")?;

    stats.print_summary();

    if !dry_run && stats.errors > 0 {
        anyhow::bail!("Migration completed with {} errors", stats.errors);
    }

    Ok(())
}

async fn migrate_agentdb(skip_history: bool, batch_size: Option<usize>) -> Result<()> {
    use migration::agentdb::migrate_rl_state;

    info!("Migrating AgentDB RL state to PostgreSQL");

    let stats = migrate_rl_state(skip_history, batch_size.unwrap_or(500)).await
        .context("AgentDB migration failed")?;

    stats.print_summary();
    Ok(())
}

async fn validate_migration(sample_size: usize, full_scan: bool, fix_inconsistencies: bool) -> Result<()> {
    use migration::validator::validate_migration as validate;

    info!("Validating migration completeness...");

    let report = validate(sample_size, full_scan, fix_inconsistencies).await
        .context("Validation failed")?;

    report.print_summary();

    if report.success_rate < 0.99 {
        anyhow::bail!("Validation failed: success rate {:.2}% < 99%", report.success_rate * 100.0);
    }

    Ok(())
}

async fn rollback_migration(confirm: bool, preserve_milvus: bool) -> Result<()> {
    use migration::rollback::rollback_to_neo4j;

    if !confirm {
        println!("⚠️  ROLLBACK PREVIEW");
        println!("This operation will:");
        println!("  1. Disable Milvus in hybrid coordinator");
        println!("  2. Route all queries back to Neo4j");
        println!("  3. Update Kubernetes deployment configuration");
        if preserve_milvus {
            println!("  4. Preserve Milvus data for future retry");
        } else {
            println!("  4. DELETE Milvus data (--preserve-milvus=false)");
        }
        println!("\nRe-run with --confirm to execute");
        return Ok(());
    }

    info!("Rolling back to Neo4j-only mode...");

    rollback_to_neo4j(preserve_milvus).await
        .context("Rollback failed")?;

    info!("✅ Rollback completed. Run: kubectl rollout status deployment/recommendation-engine");
    Ok(())
}

async fn monitor_migration(interval_secs: u64) -> Result<()> {
    use migration::validator::get_migration_progress;
    use std::time::Duration;
    use tokio::time;

    info!("📊 Monitoring migration progress (Ctrl+C to stop)...");

    let mut interval = time::interval(Duration::from_secs(interval_secs));

    loop {
        interval.tick().await;

        match get_migration_progress().await {
            Ok(progress) => {
                print!("\x1B[2J\x1B[1;1H"); // Clear screen
                progress.print_live();
            },
            Err(e) => {
                tracing::warn!("Failed to fetch progress: {}", e);
            }
        }
    }
}

async fn generate_report(output_file: Option<String>, format: Option<ReportFormat>) -> Result<()> {
    use migration::validator::generate_migration_report;

    info!("Generating migration report...");

    let report = generate_migration_report().await
        .context("Report generation failed")?;

    let format = format.unwrap_or(ReportFormat::Markdown);
    let content = match format {
        ReportFormat::Json => serde_json::to_string_pretty(&report)?,
        ReportFormat::Yaml => serde_yaml::to_string(&report)?,
        ReportFormat::Markdown => report.to_markdown(),
    };

    if let Some(file) = output_file {
        std::fs::write(&file, content)?;
        info!("Report written to: {}", file);
    } else {
        println!("{}", content);
    }

    Ok(())
}
