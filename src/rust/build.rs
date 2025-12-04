/// Build script for generating Rust types from OWL ontology
///
/// Parses expanded-media-ontology.ttl at compile time and generates:
/// - Rust enums for Genre, VisualAesthetic, NarrativeStructure, Mood, Pacing
/// - Bidirectional mapping (Rust ↔ OWL URI)
/// - Serde serialization support
/// - Type-safe compile-time validation

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=../../design/ontology/expanded-media-ontology.ttl");

    let ontology_path = "../../design/ontology/expanded-media-ontology.ttl";
    let output_path = "models/generated.rs";

    // Parse ontology
    let ontology = parse_ontology(ontology_path);

    // Generate Rust code
    let generated_code = generate_rust_types(&ontology);

    // Write to file
    let mut file = File::create(output_path).expect("Failed to create generated.rs");
    file.write_all(generated_code.as_bytes())
        .expect("Failed to write generated.rs");

    println!("cargo:warning=Generated Rust types from ontology: {} classes",
             ontology.classes.len());
}

#[derive(Debug, Default)]
struct Ontology {
    classes: HashMap<String, ClassDef>,
    prefixes: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct ClassDef {
    uri: String,
    local_name: String,
    label: Option<String>,
    comment: Option<String>,
    parent: Option<String>,
}

fn parse_ontology(path: &str) -> Ontology {
    let file = File::open(path).expect("Failed to open ontology file");
    let reader = BufReader::new(file);

    let mut ontology = Ontology::default();
    let mut current_subject: Option<String> = None;

    for line in reader.lines().flatten() {
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse prefix declarations
        if line.starts_with("@prefix") {
            if let Some((prefix, uri)) = parse_prefix(line) {
                ontology.prefixes.insert(prefix, uri);
            }
            continue;
        }

        // Parse class definitions
        if line.contains("a owl:Class") {
            if let Some(subject) = extract_subject(line) {
                let expanded = expand_uri(&subject, &ontology.prefixes);
                current_subject = Some(subject.clone());

                ontology.classes.insert(
                    subject.clone(),
                    ClassDef {
                        uri: expanded.clone(),
                        local_name: extract_local_name(&expanded),
                        label: None,
                        comment: None,
                        parent: None,
                    },
                );
            }
        }

        // Parse rdfs:label
        if line.contains("rdfs:label") {
            if let Some(ref subject) = current_subject {
                if let Some(label) = extract_literal(line) {
                    if let Some(class) = ontology.classes.get_mut(subject) {
                        class.label = Some(label);
                    }
                }
            }
        }

        // Parse rdfs:comment
        if line.contains("rdfs:comment") {
            if let Some(ref subject) = current_subject {
                if let Some(comment) = extract_literal(line) {
                    if let Some(class) = ontology.classes.get_mut(subject) {
                        class.comment = Some(comment);
                    }
                }
            }
        }

        // Parse rdfs:subClassOf
        if line.contains("rdfs:subClassOf") {
            if let Some(ref subject) = current_subject {
                if let Some(parent) = extract_object(line) {
                    if let Some(class) = ontology.classes.get_mut(subject) {
                        class.parent = Some(parent);
                    }
                }
            }
        }

        // Reset subject on statement end
        if line.ends_with('.') && !line.starts_with("@prefix") {
            current_subject = None;
        }
    }

    ontology
}

fn parse_prefix(line: &str) -> Option<(String, String)> {
    // @prefix media: <http://recommendation.org/ontology/media#> .
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 3 {
        let prefix = parts[1].trim_end_matches(':').to_string();
        let uri = parts[2]
            .trim_start_matches('<')
            .trim_end_matches('>')
            .trim_end_matches('.')
            .to_string();
        return Some((prefix, uri));
    }
    None
}

fn extract_subject(line: &str) -> Option<String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if !parts.is_empty() {
        return Some(parts[0].to_string());
    }
    None
}

fn extract_object(line: &str) -> Option<String> {
    // Extract object after predicate
    if let Some(pos) = line.find("rdfs:subClassOf") {
        let after = &line[pos + "rdfs:subClassOf".len()..];
        let obj = after
            .trim()
            .trim_end_matches(';')
            .trim_end_matches('.')
            .trim();
        if !obj.is_empty() {
            return Some(obj.to_string());
        }
    }
    None
}

fn extract_literal(line: &str) -> Option<String> {
    // Extract string literal from quotes
    if let Some(start) = line.find('"') {
        if let Some(end) = line[start + 1..].find('"') {
            return Some(line[start + 1..start + 1 + end].to_string());
        }
    }
    None
}

fn expand_uri(prefixed: &str, prefixes: &HashMap<String, String>) -> String {
    if let Some(colon_pos) = prefixed.find(':') {
        let prefix = &prefixed[..colon_pos];
        let local = &prefixed[colon_pos + 1..];

        if let Some(base_uri) = prefixes.get(prefix) {
            return format!("{}{}", base_uri, local);
        }
    }
    prefixed.to_string()
}

fn extract_local_name(uri: &str) -> String {
    uri.split(&['#', '/'][..])
        .last()
        .unwrap_or(uri)
        .to_string()
}

fn generate_rust_types(ontology: &Ontology) -> String {
    let mut output = String::new();

    // File header
    output.push_str("// AUTO-GENERATED - DO NOT EDIT\n");
    output.push_str("// Generated from: design/ontology/expanded-media-ontology.ttl\n");
    output.push_str("// Build script: build.rs\n");
    output.push_str("//\n");
    output.push_str("// This file provides compile-time type safety for ontology-defined enums.\n");
    output.push_str("// Any changes to the ontology will automatically update these types.\n\n");

    output.push_str("#![allow(dead_code)]\n\n");
    output.push_str("use serde::{Deserialize, Serialize};\n");
    output.push_str("use std::fmt;\n\n");

    // Group classes by parent
    let grouped = group_by_parent(ontology);

    // Generate enums for main categories
    generate_enum(&mut output, ontology, &grouped, "media:Genre", "Genre");
    generate_enum(&mut output, ontology, &grouped, "media:VisualAesthetic", "VisualAesthetic");
    generate_enum(&mut output, ontology, &grouped, "media:NarrativeStructure", "NarrativeStructure");
    generate_enum(&mut output, ontology, &grouped, "media:Mood", "Mood");
    generate_enum(&mut output, ontology, &grouped, "media:Pacing", "Pacing");

    // Generate URI mapping traits
    generate_uri_mapping(&mut output);

    output
}

fn group_by_parent(ontology: &Ontology) -> HashMap<String, Vec<String>> {
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();

    for (name, class_def) in &ontology.classes {
        if let Some(parent) = &class_def.parent {
            groups
                .entry(parent.clone())
                .or_insert_with(Vec::new)
                .push(name.clone());
        }
    }

    groups
}

fn generate_enum(
    output: &mut String,
    ontology: &Ontology,
    grouped: &HashMap<String, Vec<String>>,
    parent_class: &str,
    enum_name: &str,
) {
    // Get all subclasses
    let subclasses = match grouped.get(parent_class) {
        Some(classes) => classes,
        None => return,
    };

    if subclasses.is_empty() {
        return;
    }

    // Generate enum documentation
    if let Some(parent) = ontology.classes.get(parent_class) {
        if let Some(comment) = &parent.comment {
            output.push_str(&format!("/// {}\n", comment));
        }
    }
    output.push_str("///\n");
    output.push_str(&format!("/// Generated from ontology class: {}\n", parent_class));
    output.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]\n");
    output.push_str(&format!("pub enum {} {{\n", enum_name));

    // Generate variants
    for subclass_name in subclasses {
        if let Some(class_def) = ontology.classes.get(subclass_name) {
            if let Some(comment) = &class_def.comment {
                output.push_str(&format!("    /// {}\n", comment));
            }
            if let Some(label) = &class_def.label {
                output.push_str(&format!("    #[serde(rename = \"{}\")]\n", label));
            }
            let variant_name = sanitize_variant_name(&class_def.local_name);
            output.push_str(&format!("    {},\n", variant_name));
        }
    }

    output.push_str("}\n\n");

    // Generate Display implementation
    output.push_str(&format!("impl fmt::Display for {} {{\n", enum_name));
    output.push_str("    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {\n");
    output.push_str("        match self {\n");

    for subclass_name in subclasses {
        if let Some(class_def) = ontology.classes.get(subclass_name) {
            let variant_name = sanitize_variant_name(&class_def.local_name);
            let display_name = class_def.label.as_deref().unwrap_or(&class_def.local_name);
            output.push_str(&format!(
                "            Self::{} => write!(f, \"{}\"),\n",
                variant_name, display_name
            ));
        }
    }

    output.push_str("        }\n");
    output.push_str("    }\n");
    output.push_str("}\n\n");

    // Generate OWL URI mapping
    output.push_str(&format!("impl {} {{\n", enum_name));
    output.push_str("    /// Convert to OWL ontology URI\n");
    output.push_str("    pub fn to_owl_uri(&self) -> &'static str {\n");
    output.push_str("        match self {\n");

    for subclass_name in subclasses {
        if let Some(class_def) = ontology.classes.get(subclass_name) {
            let variant_name = sanitize_variant_name(&class_def.local_name);
            output.push_str(&format!(
                "            Self::{} => \"{}\",\n",
                variant_name, class_def.uri
            ));
        }
    }

    output.push_str("        }\n");
    output.push_str("    }\n\n");

    output.push_str("    /// Parse from OWL ontology URI\n");
    output.push_str("    pub fn from_owl_uri(uri: &str) -> Option<Self> {\n");
    output.push_str("        match uri {\n");

    for subclass_name in subclasses {
        if let Some(class_def) = ontology.classes.get(subclass_name) {
            let variant_name = sanitize_variant_name(&class_def.local_name);
            output.push_str(&format!(
                "            \"{}\" => Some(Self::{}),\n",
                class_def.uri, variant_name
            ));
        }
    }

    output.push_str("            _ => None,\n");
    output.push_str("        }\n");
    output.push_str("    }\n");
    output.push_str("}\n\n");
}

fn sanitize_variant_name(name: &str) -> String {
    // Handle special cases
    let sanitized = match name {
        "SciFi" => "SciFi",
        "NoirAesthetic" => "Noir",
        "NeonAesthetic" => "Neon",
        "PastelAesthetic" => "Pastel",
        "NaturalisticAesthetic" => "Naturalistic",
        "LinearNarrative" => "Linear",
        "NonLinearNarrative" => "NonLinear",
        "HerosJourney" => "HerosJourney",
        "EnsembleCast" => "EnsembleCast",
        "FastPaced" => "Fast",
        "ModeratePaced" => "Moderate",
        "SlowPaced" => "Slow",
        _ => name,
    };

    sanitized.to_string()
}

fn generate_uri_mapping(output: &mut String) {
    output.push_str("/// Trait for types that map to OWL ontology URIs\n");
    output.push_str("pub trait OntologyMappable {\n");
    output.push_str("    /// Get the OWL URI for this value\n");
    output.push_str("    fn to_owl_uri(&self) -> &'static str;\n");
    output.push_str("    \n");
    output.push_str("    /// Parse from OWL URI\n");
    output.push_str("    fn from_owl_uri(uri: &str) -> Option<Self> where Self: Sized;\n");
    output.push_str("}\n\n");

    // Implement for all generated enums
    for enum_name in &["Genre", "VisualAesthetic", "NarrativeStructure", "Mood", "Pacing"] {
        output.push_str(&format!("impl OntologyMappable for {} {{\n", enum_name));
        output.push_str("    fn to_owl_uri(&self) -> &'static str {\n");
        output.push_str("        self.to_owl_uri()\n");
        output.push_str("    }\n\n");
        output.push_str("    fn from_owl_uri(uri: &str) -> Option<Self> {\n");
        output.push_str("        Self::from_owl_uri(uri)\n");
        output.push_str("    }\n");
        output.push_str("}\n\n");
    }
}
