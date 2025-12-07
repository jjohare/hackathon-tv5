// Interactive mode

use anyhow::Result;
use colored::Colorize;
use std::io::{self, Write};

use crate::{DeviceType, OutputFormat};

pub async fn run_interactive(
    device: &DeviceType,
    output_format: &OutputFormat,
) -> Result<()> {
    println!("{}", "Interactive Mode".bold().cyan());
    println!("{}", "─".repeat(60).cyan());
    println!("Type your query or 'quit' to exit");
    println!();

    loop {
        print!("{} ", "query>".green().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let query = input.trim();

        if query.is_empty() {
            continue;
        }

        if query == "quit" || query == "exit" {
            println!("Goodbye!");
            break;
        }

        if query == "help" {
            show_help();
            continue;
        }

        // Execute query
        match crate::commands::test::run_test(query, 10, device, output_format).await {
            Ok(_) => {}
            Err(e) => {
                eprintln!("{} {}", "Error:".red(), e);
            }
        }

        println!();
    }

    Ok(())
}

fn show_help() {
    println!();
    println!("{}", "Available commands:".bold());
    println!("  <query>  - Search for movies");
    println!("  help     - Show this help");
    println!("  quit     - Exit interactive mode");
    println!();
}
