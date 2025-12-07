//! Attention Reranker Demo
//!
//! Demonstrates the attention-based reranking system for hyper-personalized recommendations.

use ndarray::{Array1, Array2};
use std::time::Instant;

// Since we can't directly import from src/rust in examples, we'll create a minimal demo
// In production, this would use: use recommendation_engine::attention_reranker::*;

fn main() {
    println!("=== Attention Reranker Demo ===\n");

    // Simulate a simple attention-based scoring system
    demo_basic_reranking();
    demo_performance_benchmark();
    demo_context_aware_reranking();

    println!("\n=== Demo Complete ===");
}

fn demo_basic_reranking() {
    println!("📊 Basic Reranking Demo");
    println!("------------------------");

    // Simulated query and candidates (384-dim embeddings)
    let query = Array1::from_elem(384, 0.1);
    let candidates = Array2::from_elem((10, 384), 0.05);

    let start = Instant::now();

    // Simplified attention: Q·K^T scoring
    let mut scores: Vec<f32> = Vec::with_capacity(10);
    for i in 0..10 {
        let candidate = candidates.row(i);
        let score = query.iter().zip(candidate.iter()).map(|(q, c)| q * c).sum::<f32>();
        scores.push(score);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("✓ Scored 10 candidates in {:.4}ms", elapsed);
    println!("  Score range: [{:.4}, {:.4}]",
        scores.iter().copied().fold(f32::INFINITY, f32::min),
        scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    println!();
}

fn demo_performance_benchmark() {
    println!("⚡ Performance Benchmark");
    println!("------------------------");

    for num_candidates in [10, 50, 100, 500, 1000] {
        let query = Array1::from_elem(384, 0.1);
        let candidates = Array2::from_elem((num_candidates, 384), 0.05);

        let start = Instant::now();

        // Batch scoring
        let _scores = candidates.dot(&query);

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        println!("  {:4} candidates: {:.4}ms", num_candidates, elapsed);

        if num_candidates == 100 && elapsed < 0.2 {
            println!("    ✓ Met <0.2ms target!");
        }
    }

    println!();
}

fn demo_context_aware_reranking() {
    println!("🎯 Context-Aware Reranking");
    println!("---------------------------");

    // Simulated context features
    let contexts = vec![
        ("Morning viewing", vec![1.0, 0.0, 0.0, 0.5, 0.3, 0.2, 0.8, 0.2]),
        ("Evening viewing", vec![0.0, 0.0, 1.0, 0.2, 0.3, 0.5, 0.3, 0.7]),
        ("Action fan", vec![0.33, 0.33, 0.34, 0.8, 0.1, 0.1, 0.5, 0.5]),
    ];

    for (name, context_vec) in contexts {
        let context = Array1::from_vec(context_vec);
        let query = Array1::from_elem(384, 0.1);

        // Add context to query (simplified)
        let context_weight = 0.3;
        let mut query_with_context = query.clone();
        for i in 0..8.min(384) {
            query_with_context[i] += context_weight * context[i];
        }

        let candidates = Array2::from_elem((10, 384), 0.05);
        let scores = candidates.dot(&query_with_context);

        let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;

        println!("  {}: avg_score = {:.4}", name, avg_score);
    }

    println!();
}
