use clap::Parser;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;

#[derive(Parser, Debug)]
#[command(name = "load-generator")]
#[command(about = "Generate load for performance testing")]
struct Args {
    /// Target queries per second
    #[arg(long, default_value = "7000")]
    qps: usize,

    /// Test duration in seconds
    #[arg(long, default_value = "600")]
    duration: u64,

    /// Number of worker tasks
    #[arg(long, default_value = "100")]
    workers: usize,

    /// Output file for results
    #[arg(long, default_value = "results/load-test.json")]
    output: String,

    /// Query mix profile (simple, mixed, complex)
    #[arg(long, default_value = "mixed")]
    profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Measurement {
    worker_id: usize,
    timestamp: u64,
    latency_us: u64,
    success: bool,
    result_count: usize,
    query_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LoadTestReport {
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    duration_secs: f64,
    actual_qps: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
    p99_9_latency_ms: f64,
    max_latency_ms: f64,
    error_rate: f64,
    measurements: Vec<Measurement>,
}

// Mock coordinator - replace with actual implementation
struct HybridCoordinator {}

impl HybridCoordinator {
    async fn search_with_context(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, String> {
        // Simulate realistic latency distribution
        let base_latency = match query.query_type.as_str() {
            "simple" => 2000,  // 2ms
            "hybrid" => 5000,  // 5ms
            "complex" => 10000, // 10ms
            _ => 5000,
        };

        // Add jitter (±30%)
        let jitter = (rand::random::<f64>() - 0.5) * 0.6 * base_latency as f64;
        let latency_us = (base_latency as f64 + jitter) as u64;

        sleep(Duration::from_micros(latency_us)).await;

        // 99.5% success rate
        if rand::random::<f64>() < 0.995 {
            Ok(vec![SearchResult { id: 1, score: 0.95 }])
        } else {
            Err("Simulated failure".to_string())
        }
    }
}

#[derive(Clone)]
struct SearchQuery {
    vector: Vec<f32>,
    k: usize,
    query_type: String,
}

#[derive(Clone)]
struct SearchResult {
    id: u64,
    score: f32,
}

fn create_random_query(profile: &str) -> SearchQuery {
    let query_type = match profile {
        "simple" => "simple",
        "complex" => "complex",
        "mixed" => {
            let r = rand::random::<f64>();
            if r < 0.5 {
                "simple"
            } else if r < 0.8 {
                "hybrid"
            } else {
                "complex"
            }
        }
        _ => "hybrid",
    };

    SearchQuery {
        vector: (0..768).map(|_| rand::random::<f32>()).collect(),
        k: 10,
        query_type: query_type.to_string(),
    }
}

async fn setup_coordinator() -> Result<HybridCoordinator, Box<dyn std::error::Error>> {
    // Initialize hybrid coordinator
    Ok(HybridCoordinator {})
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== Load Test Configuration ===");
    println!("Target QPS: {}", args.qps);
    println!("Duration: {}s", args.duration);
    println!("Workers: {}", args.workers);
    println!("Profile: {}", args.profile);
    println!("================================\n");

    let coordinator = Arc::new(setup_coordinator().await?);
    let (tx, mut rx) = mpsc::channel::<Measurement>(100_000);

    let start = Instant::now();
    let end = start + Duration::from_secs(args.duration);

    // Spawn worker tasks
    let qps_per_worker = args.qps / args.workers;
    let delay_us = 1_000_000 / qps_per_worker;

    for worker_id in 0..args.workers {
        let coordinator = coordinator.clone();
        let tx = tx.clone();
        let profile = args.profile.clone();

        tokio::spawn(async move {
            let mut next_request = Instant::now();

            while Instant::now() < end {
                let query = create_random_query(&profile);
                let request_start = Instant::now();

                match coordinator.search_with_context(&query).await {
                    Ok(results) => {
                        let latency = request_start.elapsed();
                        tx.send(Measurement {
                            worker_id,
                            timestamp: request_start
                                .duration_since(start)
                                .as_micros() as u64,
                            latency_us: latency.as_micros() as u64,
                            success: true,
                            result_count: results.len(),
                            query_type: query.query_type,
                        })
                        .await
                        .ok();
                    }
                    Err(_e) => {
                        tx.send(Measurement {
                            worker_id,
                            timestamp: request_start
                                .duration_since(start)
                                .as_micros() as u64,
                            latency_us: request_start.elapsed().as_micros() as u64,
                            success: false,
                            result_count: 0,
                            query_type: query.query_type,
                        })
                        .await
                        .ok();
                    }
                }

                // Rate limiting with precise timing
                next_request += Duration::from_micros(delay_us as u64);
                let now = Instant::now();
                if next_request > now {
                    sleep(next_request - now).await;
                } else {
                    // Fell behind, reset
                    next_request = now;
                }
            }
        });
    }

    drop(tx);

    // Collect measurements with progress reporting
    let mut measurements = Vec::new();
    let mut last_report = Instant::now();

    while let Some(m) = rx.recv().await {
        measurements.push(m);

        // Print progress every second
        if last_report.elapsed() >= Duration::from_secs(1) {
            let elapsed = start.elapsed().as_secs_f64();
            let current_qps = measurements.len() as f64 / elapsed;
            let success_rate =
                measurements.iter().filter(|m| m.success).count() as f64 / measurements.len() as f64;

            println!(
                "[{:>3}s] Requests: {} | QPS: {:.0} | Success: {:.2}%",
                elapsed as u64,
                measurements.len(),
                current_qps,
                success_rate * 100.0
            );

            last_report = Instant::now();
        }
    }

    println!("\n=== Test Complete ===");
    println!("Analyzing results...\n");

    // Analyze and save
    let report = analyze_measurements(&measurements, start.elapsed());
    save_report(&args.output, &report)?;
    print_summary(&report);

    Ok(())
}

fn analyze_measurements(measurements: &[Measurement], total_duration: Duration) -> LoadTestReport {
    let mut latencies: Vec<u64> = measurements
        .iter()
        .filter(|m| m.success)
        .map(|m| m.latency_us)
        .collect();
    latencies.sort_unstable();

    let successful = latencies.len();
    let total = measurements.len();

    LoadTestReport {
        total_requests: total,
        successful_requests: successful,
        failed_requests: total - successful,
        duration_secs: total_duration.as_secs_f64(),
        actual_qps: total as f64 / total_duration.as_secs_f64(),
        p50_latency_ms: percentile(&latencies, 0.50) / 1000.0,
        p95_latency_ms: percentile(&latencies, 0.95) / 1000.0,
        p99_latency_ms: percentile(&latencies, 0.99) / 1000.0,
        p99_9_latency_ms: percentile(&latencies, 0.999) / 1000.0,
        max_latency_ms: (*latencies.last().unwrap_or(&0)) as f64 / 1000.0,
        error_rate: (total - successful) as f64 / total as f64,
        measurements: measurements.to_vec(),
    }
}

fn percentile(sorted_latencies: &[u64], p: f64) -> f64 {
    if sorted_latencies.is_empty() {
        return 0.0;
    }

    let idx = ((sorted_latencies.len() as f64 - 1.0) * p) as usize;
    sorted_latencies[idx] as f64
}

fn save_report(path: &str, report: &LoadTestReport) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)?;
    println!("Results saved to: {}", path);
    Ok(())
}

fn print_summary(report: &LoadTestReport) {
    println!("=== Load Test Results ===");
    println!("Total Requests:     {}", report.total_requests);
    println!("Successful:         {}", report.successful_requests);
    println!("Failed:             {}", report.failed_requests);
    println!(
        "Actual QPS:         {:.1}",
        report.actual_qps
    );
    println!();
    println!("Latency Distribution:");
    println!("  p50:  {:>8.2}ms", report.p50_latency_ms);
    println!("  p95:  {:>8.2}ms", report.p95_latency_ms);
    println!("  p99:  {:>8.2}ms", report.p99_latency_ms);
    println!("  p99.9:{:>8.2}ms", report.p99_9_latency_ms);
    println!("  max:  {:>8.2}ms", report.max_latency_ms);
    println!();
    println!("Error Rate:         {:.3}%", report.error_rate * 100.0);
    println!();

    // Validate against targets
    println!("=== Target Validation ===");

    let p99_pass = report.p99_latency_ms < 10.0;
    println!(
        "{} p99 latency: {:.2}ms (target: <10ms)",
        if p99_pass { "✓" } else { "✗" },
        report.p99_latency_ms
    );

    let qps_pass = report.actual_qps >= 7000.0;
    println!(
        "{} Sustained QPS: {:.0} (target: ≥7,000)",
        if qps_pass { "✓" } else { "✗" },
        report.actual_qps
    );

    let error_pass = report.error_rate < 0.01;
    println!(
        "{} Error rate: {:.2}% (target: <1%)",
        if error_pass { "✓" } else { "✗" },
        report.error_rate * 100.0
    );

    println!();
    if p99_pass && qps_pass && error_pass {
        println!("🎉 ALL TARGETS MET!");
    } else {
        println!("⚠️  Some targets not met. Review performance.");
    }
}
