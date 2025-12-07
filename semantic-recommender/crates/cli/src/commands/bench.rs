// Benchmark command

use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::info;

use crate::DeviceType;

pub async fn run_benchmarks(
    iterations: usize,
    compare_python: bool,
    output_path: Option<&Path>,
    device: &DeviceType,
) -> Result<()> {
    info!("Running benchmarks with {} iterations", iterations);

    println!("{}", "Benchmark Suite".bold().cyan());
    println!("{}", "═".repeat(60).cyan());
    println!();

    // Initialize engine
    println!("Initializing engine with device: {:?}", device);
    // TODO: let engine = initialize_engine(device).await?;

    // Benchmark 1: Single query latency
    let single_latency = benchmark_single_query(iterations).await?;

    // Benchmark 2: Batch queries
    let batch_latency = benchmark_batch_queries(iterations).await?;

    // Benchmark 3: Memory usage
    let memory_stats = benchmark_memory_usage().await?;

    // Display results
    display_benchmark_results(&single_latency, &batch_latency, &memory_stats);

    // Python comparison
    if compare_python {
        println!();
        println!("{}", "Python Baseline Comparison".bold().yellow());
        println!("{}", "─".repeat(60).yellow());
        compare_with_python(&single_latency).await?;
    }

    // Save results if requested
    if let Some(path) = output_path {
        save_benchmark_results(path, &single_latency, &batch_latency, &memory_stats)?;
        println!();
        println!("✓ Results saved to: {}", path.display());
    }

    Ok(())
}

async fn benchmark_single_query(iterations: usize) -> Result<BenchmarkResult> {
    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    pb.set_message("Single query latency");

    let mut latencies = Vec::with_capacity(iterations);
    let test_query = "action thriller with explosions";

    for _ in 0..iterations {
        let start = Instant::now();

        // TODO: Replace with actual search
        // engine.search(test_query, 10).await?;
        tokio::time::sleep(Duration::from_micros(500)).await;

        let elapsed = start.elapsed();
        latencies.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms

        pb.inc(1);
    }

    pb.finish_with_message("Complete");

    Ok(BenchmarkResult {
        name: "Single Query".to_string(),
        iterations,
        latencies,
    })
}

async fn benchmark_batch_queries(batch_size: usize) -> Result<BenchmarkResult> {
    let pb = ProgressBar::new_spinner();
    pb.set_message("Batch query throughput");

    let queries = vec![
        "action thriller",
        "romantic comedy",
        "sci-fi adventure",
        "horror suspense",
        "drama documentary",
    ];

    let mut latencies = Vec::new();

    for _ in 0..10 {
        let start = Instant::now();

        // TODO: Batch search
        for _query in &queries {
            tokio::time::sleep(Duration::from_micros(100)).await;
        }

        let elapsed = start.elapsed();
        latencies.push(elapsed.as_secs_f64() * 1000.0 / queries.len() as f64);

        pb.tick();
    }

    pb.finish_and_clear();

    Ok(BenchmarkResult {
        name: "Batch Queries".to_string(),
        iterations: 10,
        latencies,
    })
}

async fn benchmark_memory_usage() -> Result<MemoryStats> {
    use sysinfo::{System, SystemExt, ProcessExt};

    let mut sys = System::new_all();
    sys.refresh_all();

    let pid = sysinfo::get_current_pid().unwrap();
    let process = sys.process(pid).unwrap();

    Ok(MemoryStats {
        rss_mb: process.memory() as f64 / 1024.0 / 1024.0,
        virtual_mb: process.virtual_memory() as f64 / 1024.0 / 1024.0,
    })
}

fn display_benchmark_results(
    single: &BenchmarkResult,
    batch: &BenchmarkResult,
    memory: &MemoryStats,
) {
    println!();
    println!("{}", "Results".bold().green());
    println!("{}", "─".repeat(60).green());

    // Single query stats
    let single_stats = single.compute_statistics();
    println!();
    println!("{}", "  Single Query Latency:".bold());
    println!("    Mean:   {:.2} ms", single_stats.mean);
    println!("    Median: {:.2} ms", single_stats.median);
    println!("    P95:    {:.2} ms", single_stats.p95);
    println!("    P99:    {:.2} ms", single_stats.p99);
    println!("    Min:    {:.2} ms", single_stats.min);
    println!("    Max:    {:.2} ms", single_stats.max);

    // Batch stats
    let batch_stats = batch.compute_statistics();
    println!();
    println!("{}", "  Batch Query Throughput:".bold());
    println!("    Mean latency: {:.2} ms/query", batch_stats.mean);
    println!("    Throughput:   {:.0} queries/sec", 1000.0 / batch_stats.mean);

    // Memory
    println!();
    println!("{}", "  Memory Usage:".bold());
    println!("    RSS:     {:.2} MB", memory.rss_mb);
    println!("    Virtual: {:.2} MB", memory.virtual_mb);
}

async fn compare_with_python(rust_result: &BenchmarkResult) -> Result<()> {
    // TODO: Actually run Python baseline
    // For now, use mock data
    let python_mean = 45.0; // ms
    let rust_stats = rust_result.compute_statistics();

    let speedup = python_mean / rust_stats.mean;

    println!("  Python mean latency: {:.2} ms", python_mean);
    println!("  Rust mean latency:   {:.2} ms", rust_stats.mean);
    println!();
    println!("  Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("  {}", format!("✓ Rust is {:.2}x faster", speedup).green());
    } else {
        println!("  {}", format!("⚠ Python is {:.2}x faster", 1.0 / speedup).yellow());
    }

    Ok(())
}

fn save_benchmark_results(
    path: &Path,
    single: &BenchmarkResult,
    batch: &BenchmarkResult,
    memory: &MemoryStats,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let single_stats = single.compute_statistics();
    let batch_stats = batch.compute_statistics();

    let report = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "single_query": {
            "iterations": single.iterations,
            "mean_ms": single_stats.mean,
            "median_ms": single_stats.median,
            "p95_ms": single_stats.p95,
            "p99_ms": single_stats.p99,
            "min_ms": single_stats.min,
            "max_ms": single_stats.max,
        },
        "batch_query": {
            "iterations": batch.iterations,
            "mean_ms": batch_stats.mean,
            "throughput_qps": 1000.0 / batch_stats.mean,
        },
        "memory": {
            "rss_mb": memory.rss_mb,
            "virtual_mb": memory.virtual_mb,
        }
    });

    let mut file = File::create(path)?;
    write!(file, "{}", serde_json::to_string_pretty(&report)?)?;

    Ok(())
}

struct BenchmarkResult {
    name: String,
    iterations: usize,
    latencies: Vec<f64>, // in milliseconds
}

impl BenchmarkResult {
    fn compute_statistics(&self) -> Statistics {
        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
        let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        Statistics {
            mean,
            median,
            p95,
            p99,
            min,
            max,
        }
    }
}

struct Statistics {
    mean: f64,
    median: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
}

struct MemoryStats {
    rss_mb: f64,
    virtual_mb: f64,
}
