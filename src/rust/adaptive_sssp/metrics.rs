/// Performance metrics and monitoring for adaptive SSSP

use super::SsspMetrics;
use std::sync::Arc;
use std::sync::RwLock;

/// Global metrics collector
pub struct MetricsCollector {
    history: Arc<RwLock<Vec<SsspMetrics>>>,
    max_history: usize,
}

impl MetricsCollector {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Arc::new(RwLock::new(Vec::new())),
            max_history,
        }
    }

    pub fn record(&self, metrics: SsspMetrics) {
        let mut history = self.history.write().unwrap();
        history.push(metrics);

        // Limit history size
        if history.len() > self.max_history {
            history.drain(0..history.len() - self.max_history);
        }
    }

    pub fn get_statistics(&self) -> MetricsStatistics {
        let history = self.history.read().unwrap();

        if history.is_empty() {
            return MetricsStatistics::default();
        }

        let total = history.len();
        let avg_time = history.iter()
            .map(|m| m.total_time_ms)
            .sum::<f32>() / total as f32;

        let mut algorithm_counts = std::collections::HashMap::new();
        for metrics in history.iter() {
            *algorithm_counts.entry(metrics.algorithm_used.clone())
                .or_insert(0) += 1;
        }

        MetricsStatistics {
            total_queries: total,
            avg_time_ms: avg_time,
            algorithm_distribution: algorithm_counts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricsStatistics {
    pub total_queries: usize,
    pub avg_time_ms: f32,
    pub algorithm_distribution: std::collections::HashMap<String, usize>,
}

impl Default for MetricsStatistics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            avg_time_ms: 0.0,
            algorithm_distribution: std::collections::HashMap::new(),
        }
    }
}
