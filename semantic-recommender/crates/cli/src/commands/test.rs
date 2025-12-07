// Test command - Demo query execution

use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use tracing::info;

use crate::{DeviceType, OutputFormat};

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
    spinner.set_message("Initializing GPU engine...");

    // Initialize engine (placeholder - will use actual engine)
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    spinner.set_message("Loading embeddings...");
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    spinner.set_message("Executing search...");

    let start = Instant::now();

    // TODO: Replace with actual search
    // let engine = initialize_engine(device).await?;
    // let results = engine.search(query, limit).await?;

    // Mock results for now
    let results = vec![
        ("The Matrix", 0.95),
        ("Inception", 0.92),
        ("Blade Runner", 0.89),
        ("The Terminator", 0.87),
        ("Total Recall", 0.85),
    ];

    let elapsed = start.elapsed();
    spinner.finish_and_clear();

    // Display results
    match output_format {
        OutputFormat::Table => display_table(&results, elapsed),
        OutputFormat::Json => display_json(&results, elapsed)?,
        OutputFormat::Csv => display_csv(&results)?,
    }

    // Print summary
    println!();
    println!("{}", format!("✓ Found {} results in {:.2}ms",
        results.len(),
        elapsed.as_secs_f64() * 1000.0
    ).green());

    Ok(())
}

fn display_table(results: &[(&str, f64)], elapsed: std::time::Duration) {
    use tabled::{Table, settings::Style};

    println!();
    println!("{}", "Search Results".bold().cyan());
    println!("{}", "─".repeat(60).cyan());
    println!("Query execution time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    let data: Vec<(usize, &str, String)> = results
        .iter()
        .enumerate()
        .map(|(i, (title, score))| {
            (i + 1, *title, format!("{:.4}", score))
        })
        .collect();

    let table = Table::new(data)
        .with(Style::modern())
        .to_string();

    println!("{}", table);
}

fn display_json(results: &[(&str, f64)], elapsed: std::time::Duration) -> Result<()> {
    let output = serde_json::json!({
        "query": "test query",
        "results": results.iter().map(|(title, score)| {
            serde_json::json!({
                "title": title,
                "score": score
            })
        }).collect::<Vec<_>>(),
        "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
        "count": results.len()
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn display_csv(results: &[(&str, f64)]) -> Result<()> {
    use csv::Writer;
    let mut wtr = Writer::from_writer(std::io::stdout());

    wtr.write_record(&["rank", "title", "score"])?;
    for (i, (title, score)) in results.iter().enumerate() {
        wtr.write_record(&[
            (i + 1).to_string(),
            title.to_string(),
            format!("{:.4}", score),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}
