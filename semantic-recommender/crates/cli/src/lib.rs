// Semantic Recommender CLI Library
//
// Shared functionality for CLI commands

use anyhow::Result;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Formatting utilities for CLI output
pub mod format {
    use super::*;
    use comfy_table::{Table, presets::UTF8_FULL, modifiers::UTF8_ROUND_CORNERS};

    pub fn format_duration(duration: Duration) -> String {
        let micros = duration.as_micros();
        if micros < 1_000 {
            format!("{}μs", micros)
        } else if micros < 1_000_000 {
            format!("{:.2}ms", micros as f64 / 1_000.0)
        } else {
            format!("{:.2}s", duration.as_secs_f64())
        }
    }

    pub fn create_results_table() -> Table {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS);
        table
    }

    pub fn success(msg: &str) {
        println!("{} {}", "✓".green(), msg);
    }

    pub fn error(msg: &str) {
        eprintln!("{} {}", "✗".red(), msg);
    }

    pub fn warning(msg: &str) {
        println!("{} {}", "⚠".yellow(), msg);
    }
}

/// Progress tracking for long-running operations
pub mod progress {
    use indicatif::{ProgressBar, ProgressStyle};

    pub fn create_spinner(msg: &str) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        pb.set_message(msg.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb
    }

    pub fn create_progress_bar(len: u64) -> ProgressBar {
        let pb = ProgressBar::new(len);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        pb
    }
}

/// Statistics calculation for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub mean: f64,
    pub median: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

impl Statistics {
    pub fn from_samples(samples: &[f64]) -> Self {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted.len();
        let mean = sorted.iter().sum::<f64>() / len as f64;

        let variance = sorted.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / len as f64;
        let stddev = variance.sqrt();

        Self {
            mean,
            median: sorted[len / 2],
            p50: sorted[len / 2],
            p95: sorted[(len as f64 * 0.95) as usize],
            p99: sorted[(len as f64 * 0.99) as usize],
            min: sorted[0],
            max: sorted[len - 1],
            stddev,
        }
    }
}
