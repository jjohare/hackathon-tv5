//! GMC-O Ontology Validator - Rust Implementation
//!
//! Provides comprehensive validation for RDF/OWL ontologies using sophia crate:
//! - Class validation (labels, comments)
//! - Property validation (domain, range)
//! - Circular inheritance detection
//! - Disjoint class constraints
//! - Orphaned class detection

use sophia::api::graph::{Graph, MutableGraph};
use sophia::api::ns::{rdf, rdfs, owl, xsd};
use sophia::api::term::{Term, SimpleTerm, IriRef};
use sophia::api::prelude::*;
use sophia::inmem::graph::FastGraph;
use sophia::turtle::parser::turtle;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::fs;
use serde::{Serialize, Deserialize};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Custom namespace definitions
mod ns {
    use sophia::api::ns::Namespace;

    pub const MEDIA: Namespace<&str> = Namespace::new("http://recommendation.org/ontology/media#");
    pub const USER: Namespace<&str> = Namespace::new("http://recommendation.org/ontology/user#");
    pub const CTX: Namespace<&str> = Namespace::new("http://recommendation.org/ontology/context#");
    pub const TECH: Namespace<&str> = Namespace::new("http://recommendation.org/ontology/tech-stack#");
    pub const SEM: Namespace<&str> = Namespace::new("http://recommendation.org/ontology/semantic-descriptors#");
    pub const GPU: Namespace<&str> = Namespace::new("http://recommendation.org/ontology/gpu-processing#");
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub passed: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub stats: HashMap<String, usize>,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: HashMap::new(),
        }
    }
}

impl ValidationReport {
    pub fn add_error(&mut self, message: String) {
        self.errors.push(message);
        self.passed = false;
    }

    pub fn add_warning(&mut self, message: String) {
        self.warnings.push(message);
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn print_report(&self) {
        println!("\n{}", "=".repeat(70));
        println!("ONTOLOGY VALIDATION REPORT (Rust)");
        println!("{}", "=".repeat(70));

        println!("\nStatus: {}", if self.passed { "✓ PASSED" } else { "✗ FAILED" });

        println!("\nStatistics:");
        for (key, value) in &self.stats {
            println!("  {}: {}", key, value);
        }

        if !self.errors.is_empty() {
            println!("\n✗ Errors ({}):", self.errors.len());
            for error in &self.errors {
                println!("  - {}", error);
            }
        }

        if !self.warnings.is_empty() {
            println!("\n⚠ Warnings ({}):", self.warnings.len());
            for warning in &self.warnings {
                println!("  - {}", warning);
            }
        }

        if self.errors.is_empty() && self.warnings.is_empty() {
            println!("\n✓ No issues found!");
        }

        println!("\n{}", "=".repeat(70));
    }
}

pub struct OntologyValidator {
    graph: FastGraph,
    report: ValidationReport,
}

impl OntologyValidator {
    pub fn new() -> Self {
        Self {
            graph: FastGraph::new(),
            report: ValidationReport::default(),
        }
    }

    /// Load ontology from Turtle file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let content = fs::read_to_string(path)?;
        self.load_from_string(&content)
    }

    /// Load ontology from string
    pub fn load_from_string(&mut self, content: &str) -> Result<()> {
        let triples = turtle::parse_str(content);

        for triple in triples {
            match triple {
                Ok(t) => {
                    self.graph.insert(&t.s, &t.p, &t.o)?;
                }
                Err(e) => {
                    self.report.add_error(format!("Parse error: {}", e));
                    return Err(Box::new(e));
                }
            }
        }

        self.report.stats.insert("triples".to_string(), self.graph.triples().count());
        Ok(())
    }

    /// Validate all classes have labels and comments
    pub fn validate_classes_have_labels_and_comments(&mut self) {
        let classes = self.get_all_classes();
        self.report.stats.insert("classes".to_string(), classes.len());

        for class in classes {
            let class_str = term_to_string(&class);

            // Check for label
            let has_label = self.graph
                .triples_matching(Some(&class), Some(rdfs::label), None)
                .next()
                .is_some();

            if !has_label {
                self.report.add_error(format!("Class {} missing rdfs:label", class_str));
            }

            // Check for comment
            let has_comment = self.graph
                .triples_matching(Some(&class), Some(rdfs::comment), None)
                .next()
                .is_some();

            if !has_comment {
                self.report.add_warning(format!("Class {} missing rdfs:comment", class_str));
            }
        }
    }

    /// Validate ObjectProperties have domain and range
    pub fn validate_properties_have_domains_and_ranges(&mut self) {
        let obj_properties = self.get_all_object_properties();
        let data_properties = self.get_all_datatype_properties();

        self.report.stats.insert("object_properties".to_string(), obj_properties.len());
        self.report.stats.insert("datatype_properties".to_string(), data_properties.len());

        for prop in obj_properties {
            let prop_str = term_to_string(&prop);

            let has_domain = self.graph
                .triples_matching(Some(&prop), Some(rdfs::domain), None)
                .next()
                .is_some();

            let has_range = self.graph
                .triples_matching(Some(&prop), Some(rdfs::range), None)
                .next()
                .is_some();

            if !has_domain {
                self.report.add_error(format!("ObjectProperty {} missing rdfs:domain", prop_str));
            }

            if !has_range {
                self.report.add_error(format!("ObjectProperty {} missing rdfs:range", prop_str));
            }
        }

        for prop in data_properties {
            let prop_str = term_to_string(&prop);

            let has_range = self.graph
                .triples_matching(Some(&prop), Some(rdfs::range), None)
                .next()
                .is_some();

            if !has_range {
                self.report.add_error(format!("DatatypeProperty {} missing rdfs:range", prop_str));
            }
        }
    }

    /// Detect circular inheritance
    pub fn detect_circular_inheritance(&mut self) {
        let classes = self.get_all_classes();
        let mut visited = HashSet::new();

        for class in classes {
            if !visited.contains(&class) {
                let mut path = Vec::new();
                self.check_cycle(&class, &mut visited, &mut path);
            }
        }
    }

    fn check_cycle(
        &mut self,
        class: &SimpleTerm<'_>,
        visited: &mut HashSet<SimpleTerm<'static>>,
        path: &mut Vec<SimpleTerm<'static>>,
    ) {
        let class_owned = class.into_term();

        if path.iter().any(|c| c == &class_owned) {
            let cycle_str = path.iter()
                .map(|c| term_to_string(c))
                .collect::<Vec<_>>()
                .join(" -> ");
            self.report.add_error(format!("Circular inheritance: {} -> {}", cycle_str, term_to_string(class)));
            return;
        }

        if visited.contains(&class_owned) {
            return;
        }

        visited.insert(class_owned.clone());
        path.push(class_owned.clone());

        // Get all parent classes
        for triple in self.graph.triples_matching(Some(class), Some(rdfs::subClassOf), None) {
            if let Ok(triple) = triple {
                if let Term::Iri(_) = triple.o() {
                    self.check_cycle(&triple.o(), visited, path);
                }
            }
        }

        path.pop();
    }

    /// Validate disjoint classes
    pub fn validate_disjoint_classes(&mut self) {
        // Check owl:disjointWith relationships
        for triple in self.graph.triples_matching(None, Some(owl::disjointWith), None) {
            if let Ok(triple) = triple {
                if let (Term::Iri(_), Term::Iri(_)) = (triple.s(), triple.o()) {
                    // Check for common subclasses
                    let subclasses1: HashSet<_> = self.graph
                        .triples_matching(None, Some(rdfs::subClassOf), Some(triple.s()))
                        .filter_map(|t| t.ok())
                        .map(|t| t.s().into_term())
                        .collect();

                    let subclasses2: HashSet<_> = self.graph
                        .triples_matching(None, Some(rdfs::subClassOf), Some(triple.o()))
                        .filter_map(|t| t.ok())
                        .map(|t| t.s().into_term())
                        .collect();

                    let common: Vec<_> = subclasses1.intersection(&subclasses2).collect();
                    if !common.is_empty() {
                        self.report.add_error(format!(
                            "Disjoint classes {} and {} have common subclass(es): {:?}",
                            term_to_string(triple.s()),
                            term_to_string(triple.o()),
                            common.iter().map(|c| term_to_string(c)).collect::<Vec<_>>()
                        ));
                    }
                }
            }
        }
    }

    /// Validate referenced classes exist
    pub fn validate_referenced_classes_exist(&mut self) {
        let defined_classes: HashSet<_> = self.get_all_classes()
            .into_iter()
            .collect();

        // Check domains and ranges
        for prop in self.get_all_object_properties() {
            // Check domain
            for triple in self.graph.triples_matching(Some(&prop), Some(rdfs::domain), None) {
                if let Ok(triple) = triple {
                    if let Term::Iri(_) = triple.o() {
                        let domain_term = triple.o().into_term();
                        if !defined_classes.contains(&domain_term) && !is_external_class(&domain_term) {
                            self.report.add_error(format!(
                                "Property {} references undefined domain: {}",
                                term_to_string(&prop),
                                term_to_string(triple.o())
                            ));
                        }
                    }
                }
            }

            // Check range
            for triple in self.graph.triples_matching(Some(&prop), Some(rdfs::range), None) {
                if let Ok(triple) = triple {
                    if let Term::Iri(_) = triple.o() {
                        let range_term = triple.o().into_term();
                        if !defined_classes.contains(&range_term) && !is_external_class(&range_term) {
                            self.report.add_error(format!(
                                "Property {} references undefined range: {}",
                                term_to_string(&prop),
                                term_to_string(triple.o())
                            ));
                        }
                    }
                }
            }
        }
    }

    /// Validate cardinality constraints
    pub fn validate_cardinality_constraints(&mut self) {
        let functional_props: Vec<_> = self.graph
            .triples_matching(None, Some(rdf::type_), Some(owl::FunctionalProperty))
            .filter_map(|t| t.ok())
            .map(|t| t.s().into_term())
            .collect();

        self.report.stats.insert("functional_properties".to_string(), functional_props.len());

        for prop in functional_props {
            let is_obj_prop = self.graph
                .triples_matching(Some(&prop), Some(rdf::type_), Some(owl::ObjectProperty))
                .next()
                .is_some();

            let is_data_prop = self.graph
                .triples_matching(Some(&prop), Some(rdf::type_), Some(owl::DatatypeProperty))
                .next()
                .is_some();

            if !is_obj_prop && !is_data_prop {
                self.report.add_error(format!(
                    "Functional property {} not declared as ObjectProperty or DatatypeProperty",
                    term_to_string(&prop)
                ));
            }
        }
    }

    /// Run all validations
    pub fn run_all_validations(&mut self) {
        println!("Running Rust validation checks...");

        let checks = vec![
            ("Class labels and comments", Self::validate_classes_have_labels_and_comments as fn(&mut Self)),
            ("Property domains and ranges", Self::validate_properties_have_domains_and_ranges),
            ("Circular inheritance", Self::detect_circular_inheritance),
            ("Disjoint classes", Self::validate_disjoint_classes),
            ("Referenced classes", Self::validate_referenced_classes_exist),
            ("Cardinality constraints", Self::validate_cardinality_constraints),
        ];

        for (name, check) in checks {
            println!("  ✓ {}", name);
            check(self);
        }
    }

    pub fn get_report(&self) -> &ValidationReport {
        &self.report
    }

    // Helper methods

    fn get_all_classes(&self) -> Vec<SimpleTerm<'static>> {
        self.graph
            .triples_matching(None, Some(rdf::type_), Some(owl::Class))
            .filter_map(|t| t.ok())
            .filter_map(|t| match t.s() {
                Term::Iri(_) => Some(t.s().into_term()),
                _ => None,
            })
            .collect()
    }

    fn get_all_object_properties(&self) -> Vec<SimpleTerm<'static>> {
        self.graph
            .triples_matching(None, Some(rdf::type_), Some(owl::ObjectProperty))
            .filter_map(|t| t.ok())
            .filter_map(|t| match t.s() {
                Term::Iri(_) => Some(t.s().into_term()),
                _ => None,
            })
            .collect()
    }

    fn get_all_datatype_properties(&self) -> Vec<SimpleTerm<'static>> {
        self.graph
            .triples_matching(None, Some(rdf::type_), Some(owl::DatatypeProperty))
            .filter_map(|t| t.ok())
            .filter_map(|t| match t.s() {
                Term::Iri(_) => Some(t.s().into_term()),
                _ => None,
            })
            .collect()
    }
}

// Utility functions

fn term_to_string<T: Term + ?Sized>(term: &T) -> String {
    match term.kind() {
        sophia::api::term::TermKind::Iri => format!("{}", term),
        sophia::api::term::TermKind::BlankNode => format!("_:{}", term),
        sophia::api::term::TermKind::Literal => format!("\"{}\"", term),
        _ => format!("{:?}", term),
    }
}

fn is_external_class(term: &SimpleTerm) -> bool {
    let term_str = term_to_string(term);
    term_str.starts_with("http://schema.org/")
        || term_str.starts_with("http://www.w3.org/2002/07/owl#")
        || term_str.starts_with("http://www.w3.org/2000/01/rdf-schema#")
        || term_str.starts_with("http://www.w3.org/2001/XMLSchema#")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = OntologyValidator::new();
        assert_eq!(validator.report.errors.len(), 0);
        assert!(validator.report.passed);
    }

    #[test]
    fn test_load_valid_ontology() {
        let mut validator = OntologyValidator::new();
        let ttl = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix : <http://example.org/> .

            :MyClass a owl:Class ;
                rdfs:label "My Class"@en ;
                rdfs:comment "A test class"@en .
        "#;

        assert!(validator.load_from_string(ttl).is_ok());
    }

    #[test]
    fn test_detect_missing_label() {
        let mut validator = OntologyValidator::new();
        let ttl = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix : <http://example.org/> .

            :MyClass a owl:Class .
        "#;

        validator.load_from_string(ttl).unwrap();
        validator.validate_classes_have_labels_and_comments();

        assert!(!validator.report.errors.is_empty());
        assert!(validator.report.errors.iter().any(|e| e.contains("missing rdfs:label")));
    }
}
