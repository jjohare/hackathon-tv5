/// Performance metrics collection and monitoring
///
/// Tracks API request metrics, latency, throughput, and system performance.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Metrics collector for API performance tracking
pub struct Metrics {
    /// Total number of API requests
    total_requests: AtomicU64,

    /// Total number of successful requests
    successful_requests: AtomicU64,

    /// Total number of failed requests
    failed_requests: AtomicU64,

    /// Total latency in microseconds
    total_latency_us: AtomicU64,

    /// Total number of cache hits
    cache_hits: AtomicU64,

    /// Total number of cache misses
    cache_misses: AtomicU64,
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Record a successful API request with latency
    pub fn record_success(&self, latency: Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us.fetch_add(
            latency.as_micros() as u64,
            Ordering::Relaxed,
        );
    }

    /// Record a failed API request
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record latency for any operation
    pub fn record_latency(&self, latency: Duration) {
        self.total_latency_us.fetch_add(
            latency.as_micros() as u64,
            Ordering::Relaxed,
        );
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let total = self.total_requests.load(Ordering::Relaxed);
        let successful = self.successful_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);
        let total_latency_us = self.total_latency_us.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);

        let avg_latency_ms = if total > 0 {
            (total_latency_us as f64 / total as f64) / 1000.0
        } else {
            0.0
        };

        let success_rate = if total > 0 {
            (successful as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            (cache_hits as f64 / (cache_hits + cache_misses) as f64) * 100.0
        } else {
            0.0
        };

        MetricsSnapshot {
            total_requests: total,
            successful_requests: successful,
            failed_requests: failed,
            avg_latency_ms,
            success_rate,
            cache_hits,
            cache_misses,
            cache_hit_rate,
        }
    }

    /// Reset all metrics to zero
    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.total_latency_us.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of current metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Total number of requests processed
    pub total_requests: u64,

    /// Number of successful requests
    pub successful_requests: u64,

    /// Number of failed requests
    pub failed_requests: u64,

    /// Average request latency in milliseconds
    pub avg_latency_ms: f64,

    /// Success rate as percentage
    pub success_rate: f64,

    /// Total cache hits
    pub cache_hits: u64,

    /// Total cache misses
    pub cache_misses: u64,

    /// Cache hit rate as percentage
    pub cache_hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = Metrics::new();

        metrics.record_success(Duration::from_millis(100));
        metrics.record_success(Duration::from_millis(200));
        metrics.record_failure();

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.total_requests, 3);
        assert_eq!(snapshot.successful_requests, 2);
        assert_eq!(snapshot.failed_requests, 1);
        assert!((snapshot.success_rate - 66.66).abs() < 0.1);
    }

    #[test]
    fn test_cache_metrics() {
        let metrics = Metrics::new();

        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.cache_hits, 2);
        assert_eq!(snapshot.cache_misses, 1);
        assert!((snapshot.cache_hit_rate - 66.66).abs() < 0.1);
    }
}
