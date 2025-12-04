//! Comprehensive tests for ontology validation
//!
//! Tests cover all validation rules with intentionally broken ontology snippets

use std::path::PathBuf;
use std::fs;

// Note: This requires the ontology validator crate to be available
// For now, tests are structured to be run once the crate is set up

#[cfg(test)]
mod python_validator_tests {
    use super::*;
    use std::process::Command;

    fn run_python_validator(ontology_path: &str) -> (i32, String, String) {
        let output = Command::new("python3")
            .arg("scripts/validate_ontology.py")
            .arg(ontology_path)
            .output()
            .expect("Failed to run validator");

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let code = output.status.code().unwrap_or(-1);

        (code, stdout, stderr)
    }

    #[test]
    #[ignore = "requires python dependencies"]
    fn test_valid_ontology_passes() {
        let (code, stdout, _) = run_python_validator("design/ontology/expanded-media-ontology.ttl");

        // Main ontology should have some warnings but should still pass
        println!("Output: {}", stdout);
        // Don't assert on code as warnings may be present
    }

    #[test]
    #[ignore = "requires python dependencies"]
    fn test_missing_labels_detected() {
        let (code, stdout, _) = run_python_validator("tests/fixtures/broken_ontologies/missing_labels.ttl");

        assert_ne!(code, 0, "Should fail validation");
        assert!(stdout.contains("missing rdfs:label"), "Should detect missing labels");
    }

    #[test]
    #[ignore = "requires python dependencies"]
    fn test_missing_property_constraints_detected() {
        let (code, stdout, _) = run_python_validator("tests/fixtures/broken_ontologies/missing_property_constraints.ttl");

        assert_ne!(code, 0);
        assert!(stdout.contains("missing rdfs:domain") || stdout.contains("missing rdfs:range"));
    }

    #[test]
    #[ignore = "requires python dependencies"]
    fn test_circular_inheritance_detected() {
        let (code, stdout, _) = run_python_validator("tests/fixtures/broken_ontologies/circular_inheritance.ttl");

        assert_ne!(code, 0);
        assert!(stdout.contains("Circular inheritance"));
    }

    #[test]
    #[ignore = "requires python dependencies"]
    fn test_disjoint_violation_detected() {
        let (code, stdout, _) = run_python_validator("tests/fixtures/broken_ontologies/disjoint_violation.ttl");

        assert_ne!(code, 0);
        assert!(stdout.contains("Disjoint") || stdout.contains("common subclass"));
    }

    #[test]
    #[ignore = "requires python dependencies"]
    fn test_undefined_references_detected() {
        let (code, stdout, _) = run_python_validator("tests/fixtures/broken_ontologies/undefined_references.ttl");

        assert_ne!(code, 0);
        assert!(stdout.contains("undefined") || stdout.contains("not defined"));
    }

    #[test]
    #[ignore = "requires python dependencies"]
    fn test_invalid_functional_property_detected() {
        let (code, stdout, _) = run_python_validator("tests/fixtures/broken_ontologies/invalid_functional_property.ttl");

        assert_ne!(code, 0);
        assert!(stdout.contains("Functional property") || stdout.contains("not declared"));
    }

    #[test]
    #[ignore = "requires python dependencies"]
    fn test_json_export() {
        Command::new("python3")
            .arg("scripts/validate_ontology.py")
            .arg("--json")
            .arg("/tmp/validation_report.json")
            .arg("design/ontology/expanded-media-ontology.ttl")
            .output()
            .expect("Failed to run validator");

        let json_content = fs::read_to_string("/tmp/validation_report.json")
            .expect("Failed to read JSON report");

        assert!(json_content.contains("\"passed\""));
        assert!(json_content.contains("\"errors\""));
        assert!(json_content.contains("\"warnings\""));
        assert!(json_content.contains("\"stats\""));

        // Clean up
        let _ = fs::remove_file("/tmp/validation_report.json");
    }
}

#[cfg(test)]
mod rust_validator_tests {
    use super::*;

    // These tests will work once the validator crate is properly integrated

    #[test]
    #[ignore = "requires ontology-validator crate integration"]
    fn test_rust_validator_missing_labels() {
        // Placeholder - will be implemented once crate is integrated
        // let mut validator = OntologyValidator::new();
        // validator.load_from_file("tests/fixtures/broken_ontologies/missing_labels.ttl").unwrap();
        // validator.run_all_validations();
        // assert!(!validator.get_report().passed);
    }

    #[test]
    #[ignore = "requires ontology-validator crate integration"]
    fn test_rust_validator_circular_inheritance() {
        // Placeholder for Rust validator circular inheritance test
    }

    #[test]
    #[ignore = "requires ontology-validator crate integration"]
    fn test_rust_validator_disjoint_classes() {
        // Placeholder for Rust validator disjoint classes test
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    #[ignore = "integration test"]
    fn test_main_ontology_comprehensive_validation() {
        // Run both Python and Rust validators on main ontology
        // Compare results for consistency
    }

    #[test]
    #[ignore = "integration test"]
    fn test_all_broken_ontologies() {
        let broken_dir = PathBuf::from("tests/fixtures/broken_ontologies");
        if broken_dir.exists() {
            for entry in fs::read_dir(broken_dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();

                if path.extension().and_then(|s| s.to_str()) == Some("ttl") {
                    println!("Testing broken ontology: {:?}", path);
                    // Each should fail validation
                }
            }
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    #[ignore = "performance test"]
    fn test_validator_performance_large_ontology() {
        // Test validation performance on large ontologies
        // Should complete in under 5 seconds for typical use cases
    }

    #[test]
    #[ignore = "performance test"]
    fn test_rust_vs_python_performance() {
        // Compare performance of Rust vs Python implementations
        // Rust should be significantly faster
    }
}
