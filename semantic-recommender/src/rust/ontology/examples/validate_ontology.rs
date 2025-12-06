//! Command-line example for ontology validation
//!
//! Usage:
//!   cargo run --example validate_ontology -- path/to/ontology.ttl
//!   cargo run --example validate_ontology -- path/to/ontology.ttl --json report.json

use ontology_validator::{OntologyValidator, Result};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <ontology.ttl> [--json output.json]", args[0]);
        std::process::exit(1);
    }

    let ontology_path = PathBuf::from(&args[1]);

    if !ontology_path.exists() {
        eprintln!("Error: File not found: {:?}", ontology_path);
        std::process::exit(1);
    }

    let json_output = if args.len() > 2 && args[2] == "--json" {
        args.get(3).map(PathBuf::from)
    } else {
        None
    };

    println!("Loading ontology: {:?}", ontology_path);

    let mut validator = OntologyValidator::new();

    match validator.load_from_file(&ontology_path) {
        Ok(_) => {
            validator.run_all_validations();
            let report = validator.get_report();

            report.print_report();

            if let Some(json_path) = json_output {
                let json = report.to_json()?;
                fs::write(&json_path, json)?;
                println!("\nReport exported to: {:?}", json_path);
            }

            if !report.passed {
                std::process::exit(1);
            }

            Ok(())
        }
        Err(e) => {
            eprintln!("Failed to load ontology: {}", e);
            std::process::exit(1);
        }
    }
}
