use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,

    // Internal tracking
    latency_samples: Vec<f64>,
}

impl PerformanceMetrics {
    pub fn record_query(&mut self, latency_ms: f64, from_cache: bool) {
        self.total_queries += 1;

        if from_cache {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        // Update average
        self.avg_latency_ms = (self.avg_latency_ms * (self.total_queries - 1) as f64 + latency_ms)
            / self.total_queries as f64;

        // Store sample for percentile calculation
        self.latency_samples.push(latency_ms);

        // Keep last 1000 samples
        if self.latency_samples.len() > 1000 {
            self.latency_samples.remove(0);
        }

        // Compute percentiles
        let mut sorted = self.latency_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p95_idx = (sorted.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted.len() as f64 * 0.99) as usize;

        self.p95_latency_ms = sorted.get(p95_idx).copied().unwrap_or(0.0);
        self.p99_latency_ms = sorted.get(p99_idx).copied().unwrap_or(0.0);
    }

    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_queries as f64
        }
    }
}
