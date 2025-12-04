use recommendation_engine::storage::HybridStorageCoordinator;
use tokio::sync::{mpsc, Arc};
use tokio::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use anyhow::Result;

mod fixtures;
use fixtures::{media_generator, query_generator};

#[derive(Debug)]
enum QueryResult {
    Success { latency: Duration, result_count: usize },
    Error(String),
}

struct LoadTestMetrics {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    total_latency_us: AtomicU64,
    latencies: tokio::sync::Mutex<Vec<Duration>>,
}

impl LoadTestMetrics {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            latencies: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    async fn record_success(&self, latency: Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us.fetch_add(latency.as_micros() as u64, Ordering::Relaxed);

        let mut latencies = self.latencies.lock().await;
        latencies.push(latency);
    }

    fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    async fn print_report(&self, duration: Duration) {
        let total = self.total_requests.load(Ordering::Relaxed);
        let successful = self.successful_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);

        let mut latencies = self.latencies.lock().await;
        latencies.sort();

        let actual_qps = total as f64 / duration.as_secs_f64();
        let success_rate = (successful as f64 / total as f64) * 100.0;

        let p50 = if !latencies.is_empty() {
            latencies[latencies.len() / 2]
        } else {
            Duration::from_secs(0)
        };

        let p90 = if !latencies.is_empty() {
            latencies[(latencies.len() as f64 * 0.90) as usize]
        } else {
            Duration::from_secs(0)
        };

        let p99 = if !latencies.is_empty() {
            latencies[(latencies.len() as f64 * 0.99) as usize]
        } else {
            Duration::from_secs(0)
        };

        let p999 = if !latencies.is_empty() {
            latencies[(latencies.len() as f64 * 0.999) as usize]
        } else {
            Duration::from_secs(0)
        };

        let max = latencies.last().copied().unwrap_or(Duration::from_secs(0));

        println!("\n=== Load Test Results ===");
        println!("Duration: {:.2}s", duration.as_secs_f64());
        println!("Total Requests: {}", total);
        println!("Successful: {}", successful);
        println!("Failed: {}", failed);
        println!("Success Rate: {:.2}%", success_rate);
        println!("Actual QPS: {:.0}", actual_qps);
        println!("\nLatency Percentiles:");
        println!("  p50:  {:?}", p50);
        println!("  p90:  {:?}", p90);
        println!("  p99:  {:?}", p99);
        println!("  p999: {:?}", p999);
        println!("  max:  {:?}", max);
    }
}

#[tokio::test]
#[ignore] // Run manually - requires significant resources
async fn test_sustained_load_7000_qps() {
    let env = setup_test_coordinator().await.unwrap();

    // Ingest large dataset
    println!("Ingesting 100,000 media items...");
    let ingest_start = Instant::now();
    insert_test_data(&env.coordinator, 100_000).await.unwrap();
    println!("Ingest completed in {:?}", ingest_start.elapsed());

    let duration = Duration::from_secs(60);
    let target_qps = 7000;
    let concurrency = 100;

    let coordinator = Arc::new(env.coordinator);
    let metrics = Arc::new(LoadTestMetrics::new());

    let start = Instant::now();
    let mut handles = vec![];

    println!("\nStarting load test: {} QPS for {:?}", target_qps, duration);
    println!("Concurrency: {} workers", concurrency);

    // Spawn worker threads
    for worker_id in 0..concurrency {
        let coordinator = coordinator.clone();
        let metrics = metrics.clone();
        let start = start.clone();

        let handle = tokio::spawn(async move {
            let queries_per_worker = target_qps / concurrency;
            let sleep_duration = Duration::from_micros(1_000_000 / queries_per_worker as u64);

            while start.elapsed() < duration {
                let query = query_generator::create_random_query();
                let query_start = Instant::now();

                match coordinator.search_with_context(&query).await {
                    Ok(results) => {
                        let latency = query_start.elapsed();
                        metrics.record_success(latency).await;
                    }
                    Err(e) => {
                        eprintln!("Worker {} error: {:?}", worker_id, e);
                        metrics.record_failure();
                    }
                }

                // Rate limiting
                tokio::time::sleep(sleep_duration).await;
            }
        });

        handles.push(handle);
    }

    // Progress monitoring
    let metrics_monitor = metrics.clone();
    let monitor_handle = tokio::spawn(async move {
        let mut last_count = 0u64;
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            interval.tick().await;

            let current_count = metrics_monitor.successful_requests.load(Ordering::Relaxed);
            let delta = current_count - last_count;
            let current_qps = delta / 5;

            println!(
                "[{:3}s] Requests: {} | Current QPS: {} | Failed: {}",
                start.elapsed().as_secs(),
                current_count,
                current_qps,
                metrics_monitor.failed_requests.load(Ordering::Relaxed)
            );

            last_count = current_count;

            if start.elapsed() >= duration {
                break;
            }
        }
    });

    // Wait for all workers
    for handle in handles {
        handle.await.unwrap();
    }

    monitor_handle.await.unwrap();

    // Generate report
    metrics.print_report(duration).await;

    // Validate SLA
    let total = metrics.total_requests.load(Ordering::Relaxed);
    let successful = metrics.successful_requests.load(Ordering::Relaxed);
    let failed = metrics.failed_requests.load(Ordering::Relaxed);

    let actual_qps = total as f64 / duration.as_secs_f64();
    let error_rate = failed as f64 / total as f64;

    let latencies = metrics.latencies.lock().await;
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    // Assertions
    assert!(
        actual_qps >= 6500.0,
        "QPS {:.0} below target (expected >= 6500)",
        actual_qps
    );

    assert!(
        p99 < Duration::from_millis(100),
        "p99 latency {:?} exceeds 100ms",
        p99
    );

    assert!(
        error_rate < 0.01,
        "Error rate {:.2}% exceeds 1%",
        error_rate * 100.0
    );

    println!("\n✅ Load test PASSED");
    println!("   QPS: {:.0} (target: 7000)", actual_qps);
    println!("   p99: {:?} (target: <100ms)", p99);
    println!("   Error rate: {:.2}% (target: <1%)", error_rate * 100.0);
}

#[tokio::test]
#[ignore]
async fn test_burst_traffic_handling() {
    let env = setup_test_coordinator().await.unwrap();
    insert_test_data(&env.coordinator, 10_000).await.unwrap();

    let coordinator = Arc::new(env.coordinator);
    let metrics = Arc::new(LoadTestMetrics::new());

    println!("Testing burst handling: 10,000 concurrent requests");

    let start = Instant::now();
    let mut handles = vec![];

    // Launch 10,000 concurrent requests (burst)
    for _ in 0..10_000 {
        let coordinator = coordinator.clone();
        let metrics = metrics.clone();

        let handle = tokio::spawn(async move {
            let query = query_generator::create_random_query();
            let query_start = Instant::now();

            match coordinator.search_with_context(&query).await {
                Ok(_) => {
                    metrics.record_success(query_start.elapsed()).await;
                }
                Err(_) => {
                    metrics.record_failure();
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all
    for handle in handles {
        handle.await.unwrap();
    }

    let duration = start.elapsed();
    metrics.print_report(duration).await;

    let successful = metrics.successful_requests.load(Ordering::Relaxed);
    let success_rate = successful as f64 / 10_000.0;

    // Should handle burst with >95% success
    assert!(
        success_rate > 0.95,
        "Burst success rate {:.2}% below 95%",
        success_rate * 100.0
    );

    println!("\n✅ Burst test PASSED");
}

#[tokio::test]
#[ignore]
async fn test_gradual_load_increase() {
    let env = setup_test_coordinator().await.unwrap();
    insert_test_data(&env.coordinator, 50_000).await.unwrap();

    let coordinator = Arc::new(env.coordinator);

    println!("Testing gradual load increase: 1000 -> 7000 QPS");

    for target_qps in [1000, 2000, 3000, 5000, 7000].iter() {
        println!("\nTesting at {} QPS...", target_qps);

        let metrics = Arc::new(LoadTestMetrics::new());
        let duration = Duration::from_secs(30);
        let concurrency = 50;

        let start = Instant::now();
        let mut handles = vec![];

        for _ in 0..concurrency {
            let coordinator = coordinator.clone();
            let metrics = metrics.clone();

            let handle = tokio::spawn(async move {
                let queries_per_worker = target_qps / concurrency;
                let sleep_duration = Duration::from_micros(1_000_000 / queries_per_worker as u64);

                while start.elapsed() < duration {
                    let query = query_generator::create_random_query();
                    let query_start = Instant::now();

                    match coordinator.search_with_context(&query).await {
                        Ok(_) => metrics.record_success(query_start.elapsed()).await,
                        Err(_) => metrics.record_failure(),
                    }

                    tokio::time::sleep(sleep_duration).await;
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        metrics.print_report(duration).await;

        let latencies = metrics.latencies.lock().await;
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

        assert!(
            p99 < Duration::from_millis(100),
            "p99 at {} QPS: {:?} exceeds limit",
            target_qps,
            p99
        );
    }

    println!("\n✅ Gradual load test PASSED");
}

#[tokio::test]
#[ignore]
async fn test_mixed_read_write_load() {
    let env = setup_test_coordinator().await.unwrap();
    insert_test_data(&env.coordinator, 10_000).await.unwrap();

    let coordinator = Arc::new(env.coordinator);
    let metrics = Arc::new(LoadTestMetrics::new());

    let duration = Duration::from_secs(60);
    let start = Instant::now();

    println!("Testing mixed load: 90% reads, 10% writes");

    let mut handles = vec![];

    // 90 reader threads
    for _ in 0..90 {
        let coordinator = coordinator.clone();
        let metrics = metrics.clone();

        handles.push(tokio::spawn(async move {
            while start.elapsed() < duration {
                let query = query_generator::create_random_query();
                let query_start = Instant::now();

                match coordinator.search_with_context(&query).await {
                    Ok(_) => metrics.record_success(query_start.elapsed()).await,
                    Err(_) => metrics.record_failure(),
                }

                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }));
    }

    // 10 writer threads
    for _ in 0..10 {
        let coordinator = coordinator.clone();

        handles.push(tokio::spawn(async move {
            let mut counter = 0;
            while start.elapsed() < duration {
                let media = media_generator::generate_single_media(&format!("write_{}", counter));
                let _ = coordinator.ingest_content(&media).await;
                counter += 1;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    metrics.print_report(duration).await;

    let success_rate = metrics.successful_requests.load(Ordering::Relaxed) as f64
        / metrics.total_requests.load(Ordering::Relaxed) as f64;

    assert!(success_rate > 0.99, "Mixed load success rate too low");

    println!("\n✅ Mixed load test PASSED");
}
