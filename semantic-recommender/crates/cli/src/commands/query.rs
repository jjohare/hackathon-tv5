// Query command - execute single search

use anyhow::Result;
use colored::Colorize;

use crate::{DeviceType, OutputFormat};

pub async fn execute_query(
    text: &str,
    limit: usize,
    explain: bool,
    device: &DeviceType,
    output_format: &OutputFormat,
) -> Result<()> {
    // Delegate to test command for now
    crate::commands::test::run_test(text, limit, device, output_format).await?;

    if explain {
        println!();
        println!("{}", "Explanation:".bold().yellow());
        println!("  Search used semantic embeddings to find similar movies");
        println!("  Results ranked by cosine similarity score");
        println!("  Ontology reasoning applied for contextual relevance");
    }

    Ok(())
}
