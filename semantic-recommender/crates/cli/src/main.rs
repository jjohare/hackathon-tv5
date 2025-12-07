// Semantic Recommender CLI - Complete Implementation
//
// GPU-accelerated semantic search with comprehensive benchmarking

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn, error};

mod commands;
mod dataset;
mod benchmark;
mod compare;

use commands::*;
use dataset::MovieDataset;

#[derive(Parser)]
#[command(name = "semantic-rec")]
#[command(author = "Media Gateway Team")]
#[command(version = "1.0.0")]
#[command(about = "GPU-accelerated semantic movie recommender", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Device to use for computation
    #[arg(short, long, default_value = "auto", global = true)]
    device: DeviceType,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Output format
    #[arg(short, long, default_value = "table", global = true)]
    output: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, ValueEnum, Debug)]
enum DeviceType {
    /// Use CPU only
    Cpu,
    /// Use CUDA GPU
    Cuda,
    /// Auto-detect (prefer CUDA)
    Auto,
}

#[derive(Clone, ValueEnum, Debug)]
enum OutputFormat {
    /// Pretty table output
    Table,
    /// JSON output
    Json,
    /// CSV output
    Csv,
}

#[derive(Subcommand)]
enum Commands {
    /// Run demo test query
    Test {
        /// Query text
        #[arg(short, long, default_value = "action thriller with car chases")]
        query: String,

        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,
    },

    /// Run comprehensive benchmarks
    Bench {
        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,

        /// Include comparison with Python baseline
        #[arg(long)]
        compare_python: bool,

        /// Output benchmark results to file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Execute single query
    Query {
        /// Search query
        text: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,

        /// Show detailed explanations
        #[arg(long)]
        explain: bool,
    },

    /// Load and index full dataset
    Load {
        /// Path to movie dataset CSV
        #[arg(long, default_value = "data/movies.csv")]
        dataset: PathBuf,

        /// Path to embeddings file (optional)
        #[arg(long)]
        embeddings: Option<PathBuf>,

        /// Force recompute embeddings
        #[arg(long)]
        force: bool,
    },

    /// Compare Rust vs Python implementations
    Compare {
        /// Number of test queries
        #[arg(short, long, default_value = "50")]
        queries: usize,

        /// Acceptable difference threshold
        #[arg(long, default_value = "0.01")]
        threshold: f64,
    },

    /// Interactive mode
    Interactive,

    /// Show system information
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose)?;

    // Print banner
    print_banner();

    // Execute command
    let start = Instant::now();
    let result = execute_command(&cli).await;
    let elapsed = start.elapsed();

    // Handle result
    match result {
        Ok(()) => {
            info!("✓ Command completed in {:.2}s", elapsed.as_secs_f64());
            Ok(())
        }
        Err(e) => {
            error!("✗ Command failed: {:#}", e);
            std::process::exit(1);
        }
    }
}

async fn execute_command(cli: &Cli) -> Result<()> {
    match &cli.command {
        Commands::Test { query, limit } => {
            commands::test::run_test(query, *limit, &cli.device, &cli.output).await
        }

        Commands::Bench { iterations, compare_python, output } => {
            commands::bench::run_benchmarks(*iterations, *compare_python, output.as_deref(), &cli.device).await
        }

        Commands::Query { text, limit, explain } => {
            commands::query::execute_query(text, *limit, *explain, &cli.device, &cli.output).await
        }

        Commands::Load { dataset, embeddings, force } => {
            commands::load::load_dataset(dataset, embeddings.as_deref(), *force, &cli.device).await
        }

        Commands::Compare { queries, threshold } => {
            commands::compare::compare_implementations(*queries, *threshold, &cli.device).await
        }

        Commands::Interactive => {
            commands::interactive::run_interactive(&cli.device, &cli.output).await
        }

        Commands::Info => {
            commands::info::show_system_info(&cli.device).await
        }
    }
}

fn init_logging(verbose: bool) -> Result<()> {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .init();

    Ok(())
}

fn print_banner() {
    println!("{}", "╔═══════════════════════════════════════════════════════════╗".cyan());
    println!("{}", "║   Semantic Movie Recommender - GPU Accelerated v1.0      ║".cyan());
    println!("{}", "║   62,423 Movies | A100 Optimized | <100ms Latency        ║".cyan());
    println!("{}", "╚═══════════════════════════════════════════════════════════╝".cyan());
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        // Test basic command parsing
        let cli = Cli::parse_from(["semantic-rec", "test"]);
        assert!(matches!(cli.command, Commands::Test { .. }));

        let cli = Cli::parse_from(["semantic-rec", "--device", "cuda", "query", "test query"]);
        assert!(matches!(cli.device, DeviceType::Cuda));
    }

    #[test]
    fn test_device_type() {
        let cli = Cli::parse_from(["semantic-rec", "--device", "auto", "info"]);
        assert!(matches!(cli.device, DeviceType::Auto));
    }

    #[test]
    fn test_output_format() {
        let cli = Cli::parse_from(["semantic-rec", "--output", "json", "test"]);
        assert!(matches!(cli.output, OutputFormat::Json));
    }
}
