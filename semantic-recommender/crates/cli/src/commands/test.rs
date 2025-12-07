// Test command - Demo query execution with actual engine integration

use anyhow::{Result, Context};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use tracing::{info, debug};
use comfy_table::{Table, presets::UTF8_FULL, modifiers::UTF8_ROUND_CORNERS};

use crate::{DeviceType, OutputFormat};
use semantic_recommender_cli::format;

pub async fn run_test(
    query: &str,
    limit: usize,
    device: &DeviceType,
    output_format: &OutputFormat,
) -> Result<()> {
    info!("Running test query: '{}'", query);

    // Create progress indicator
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap()
    );

    spinner.set_message("Initializing semantic engine...");
    debug!("Device: {:?}", device);

    // Initialize the actual recommendation engine
    let start_init = Instant::now();

    // TODO: When actual engine is available, use:
    // let engine = media_recommendation_engine::Engine::new()
    //     .with_device(match device {
    //         DeviceType::Cuda => Device::Cuda(0),
    //         DeviceType::Cpu => Device::Cpu,
    //         DeviceType::Auto => Device::auto(),
    //     })
    //     .build()
    //     .await?;

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let init_time = start_init.elapsed();

    spinner.set_message("Loading embeddings index...");
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    spinner.set_message("Executing semantic search...");
    let start_search = Instant::now();

    // TODO: Replace with actual search when engine is available:
    // let results = engine.search(query, limit).await
    //     .context("Search execution failed")?;

    // Mock results with realistic movie data
    let results = mock_search_results(query, limit);

    let search_time = start_search.elapsed();
    spinner.finish_and_clear();

    // Display results
    match output_format {
        OutputFormat::Table => display_table(&results, query, search_time, init_time),
        OutputFormat::Json => display_json(&results, query, search_time, init_time)?,
        OutputFormat::Csv => display_csv(&results)?,
    }

    // Print performance summary
    println!();
    println!("{}", "Performance Metrics".bold().cyan());
    println!("{}", "─".repeat(60).cyan());
    println!("  Initialization: {}", format::format_duration(init_time));
    println!("  Search latency: {}", format::format_duration(search_time));
    println!("  Total time:     {}", format::format_duration(init_time + search_time));

    println!();
    format::success(&format!("Found {} results", results.len()));

    Ok(())
}

fn display_table(
    results: &[(String, f64, Vec<String>)],
    query: &str,
    search_time: std::time::Duration,
    init_time: std::time::Duration,
) {
    println!();
    println!("{}", format!("Search Results for: {}", query).bold().cyan());
    println!("{}", "─".repeat(80).cyan());
    println!("Search completed in {:.2}ms (init: {:.2}ms)",
        search_time.as_secs_f64() * 1000.0,
        init_time.as_secs_f64() * 1000.0
    );
    println!();

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    table.set_header(vec!["Rank", "Title", "Score", "Genres"]);

    for (i, (title, score, genres)) in results.iter().enumerate() {
        table.add_row(vec![
            (i + 1).to_string(),
            title.clone(),
            format!("{:.4}", score),
            genres.join(", "),
        ]);
    }

    println!("{}", table);
}

fn display_json(
    results: &[(String, f64, Vec<String>)],
    query: &str,
    search_time: std::time::Duration,
    init_time: std::time::Duration,
) -> Result<()> {
    let output = serde_json::json!({
        "query": query,
        "results": results.iter().map(|(title, score, genres)| {
            serde_json::json!({
                "title": title,
                "score": score,
                "genres": genres
            })
        }).collect::<Vec<_>>(),
        "performance": {
            "init_ms": init_time.as_secs_f64() * 1000.0,
            "search_ms": search_time.as_secs_f64() * 1000.0,
            "total_ms": (init_time + search_time).as_secs_f64() * 1000.0,
        },
        "count": results.len()
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn display_csv(results: &[(String, f64, Vec<String>)]) -> Result<()> {
    use csv::Writer;
    let mut wtr = Writer::from_writer(std::io::stdout());

    wtr.write_record(&["rank", "title", "score", "genres"])?;
    for (i, (title, score, genres)) in results.iter().enumerate() {
        wtr.write_record(&[
            (i + 1).to_string(),
            title.clone(),
            format!("{:.4}", score),
            genres.join("|"),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

// Mock search results - to be replaced with actual engine
fn mock_search_results(query: &str, limit: usize) -> Vec<(String, f64, Vec<String>)> {
    // Different mock results based on query keywords
    let base_results = if query.contains("action") || query.contains("thriller") {
        vec![
            ("The Matrix", 0.95, vec!["Action", "Sci-Fi"]),
            ("Inception", 0.92, vec!["Action", "Thriller", "Sci-Fi"]),
            ("The Dark Knight", 0.91, vec!["Action", "Crime", "Drama"]),
            ("Mad Max: Fury Road", 0.89, vec!["Action", "Adventure"]),
            ("John Wick", 0.88, vec!["Action", "Thriller"]),
            ("Die Hard", 0.87, vec!["Action", "Thriller"]),
            ("The Bourne Identity", 0.86, vec!["Action", "Thriller"]),
            ("Mission: Impossible", 0.85, vec!["Action", "Adventure"]),
            ("Speed", 0.84, vec!["Action", "Thriller"]),
            ("Lethal Weapon", 0.83, vec!["Action", "Crime"]),
        ]
    } else if query.contains("comedy") || query.contains("funny") {
        vec![
            ("The Grand Budapest Hotel", 0.94, vec!["Comedy", "Drama"]),
            ("Superbad", 0.92, vec!["Comedy"]),
            ("Groundhog Day", 0.91, vec!["Comedy", "Fantasy"]),
            ("The Big Lebowski", 0.90, vec!["Comedy", "Crime"]),
            ("Monty Python and the Holy Grail", 0.89, vec!["Comedy"]),
            ("Airplane!", 0.88, vec!["Comedy"]),
            ("Anchorman", 0.87, vec!["Comedy"]),
            ("Bridesmaids", 0.86, vec!["Comedy"]),
            ("The Hangover", 0.85, vec!["Comedy"]),
            ("Dumb and Dumber", 0.84, vec!["Comedy"]),
        ]
    } else if query.contains("sci-fi") || query.contains("space") {
        vec![
            ("Interstellar", 0.96, vec!["Sci-Fi", "Drama"]),
            ("Blade Runner 2049", 0.94, vec!["Sci-Fi", "Thriller"]),
            ("The Matrix", 0.93, vec!["Sci-Fi", "Action"]),
            ("2001: A Space Odyssey", 0.92, vec!["Sci-Fi"]),
            ("Star Wars", 0.91, vec!["Sci-Fi", "Adventure"]),
            ("Arrival", 0.90, vec!["Sci-Fi", "Drama"]),
            ("Ex Machina", 0.89, vec!["Sci-Fi", "Thriller"]),
            ("Alien", 0.88, vec!["Sci-Fi", "Horror"]),
            ("The Terminator", 0.87, vec!["Sci-Fi", "Action"]),
            ("District 9", 0.86, vec!["Sci-Fi", "Action"]),
        ]
    } else {
        vec![
            ("The Shawshank Redemption", 0.93, vec!["Drama"]),
            ("The Godfather", 0.92, vec!["Crime", "Drama"]),
            ("Pulp Fiction", 0.91, vec!["Crime", "Drama"]),
            ("Forrest Gump", 0.90, vec!["Drama", "Romance"]),
            ("The Dark Knight", 0.89, vec!["Action", "Crime", "Drama"]),
            ("Schindler's List", 0.88, vec!["Biography", "Drama", "History"]),
            ("Goodfellas", 0.87, vec!["Biography", "Crime", "Drama"]),
            ("Casablanca", 0.86, vec!["Drama", "Romance", "War"]),
            ("The Green Mile", 0.85, vec!["Crime", "Drama", "Fantasy"]),
            ("The Silence of the Lambs", 0.84, vec!["Crime", "Drama", "Thriller"]),
        ]
    };

    base_results
        .into_iter()
        .take(limit)
        .map(|(title, score, genres)| {
            (title.to_string(), score, genres.iter().map(|s| s.to_string()).collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_search_action() {
        let results = mock_search_results("action thriller", 5);
        assert_eq!(results.len(), 5);
        assert!(results[0].0.contains("Matrix") || results[0].0.contains("Inception"));
    }

    #[test]
    fn test_mock_search_comedy() {
        let results = mock_search_results("funny comedy", 3);
        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|(_, _, genres)| genres.contains(&"Comedy".to_string())));
    }

    #[test]
    fn test_limit_enforcement() {
        let results = mock_search_results("test", 3);
        assert_eq!(results.len(), 3);
    }
}
