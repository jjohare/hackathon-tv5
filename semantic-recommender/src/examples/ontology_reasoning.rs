/// Example: OWL Ontology Reasoning
///
/// Demonstrates:
/// - Building GMC-O ontology
/// - Creating semantic triples
/// - Applying reasoning rules
/// - Constraint satisfaction

use hackathon_tv5::models::{
    OntologyClass, OntologyProperty, PropertyType, SemanticTriple,
    TripleObject, LiteralValue, OWLAxiom, AxiomType,
    SemanticConstraint, ConstraintCondition, ConstraintAction,
    GMCOntologyBuilder,
};

fn main() {
    println!("=== OWL Ontology Reasoning Example ===\n");

    // Build GMC-O ontology
    let mut ontology = build_gmc_ontology();

    println!("1. Ontology Structure:");
    println!("  Classes: {}", ontology.classes.len());
    println!("  Properties: {}", ontology.properties.len());
    println!("  Axioms: {}\n", ontology.axioms.len());

    // Create semantic triples for Inception
    let film_uri = "film:inception";
    let triples = create_film_triples(film_uri);

    println!("2. Semantic Triples for 'Inception':");
    for triple in &triples {
        print_triple(&triple);
    }

    // Apply reasoning rules
    println!("\n3. Reasoning Results:");
    let inferred = apply_reasoning(&ontology, &triples);

    println!("  Original triples: {}", triples.len());
    println!("  Inferred triples: {}", inferred.len());
    println!("\n  Inferred facts:");
    for triple in &inferred {
        print_triple(&triple);
    }

    // Demonstrate constraint satisfaction
    println!("\n4. Semantic Constraints:");
    let constraints = create_constraints();

    for constraint in &constraints {
        println!("  Constraint '{}': {:?} → {:?}",
                 constraint.id,
                 constraint.condition,
                 constraint.action);
    }

    // Check triple validity
    println!("\n5. Triple Validation:");
    for triple in &triples {
        if triple.meets_threshold(0.75) {
            println!("  ✓ High confidence: {} {} (confidence: {:.2})",
                     triple.subject,
                     triple.predicate,
                     triple.confidence);
        } else {
            println!("  ⚠ Low confidence: {} {} (confidence: {:.2})",
                     triple.subject,
                     triple.predicate,
                     triple.confidence);
        }
    }

    // Export ontology to Turtle
    println!("\n6. Ontology Export (Turtle format):");
    let turtle = ontology.to_turtle();
    println!("{}", &turtle[..200.min(turtle.len())]); // Print first 200 chars
    println!("  ... (truncated)");
}

/// Build GMC-O ontology with core classes
fn build_gmc_ontology() -> GMCOntologyBuilder {
    let mut builder = GMCOntologyBuilder::new();

    // Define core classes
    let creative_work = OntologyClass::new(
        "media:CreativeWork",
        "Creative Work",
    ).with_comment("Top-level class for any film, episode, or series");

    let narrative = OntologyClass::new(
        "media:NarrativeStructure",
        "Narrative Structure",
    ).with_comment("The architectural flow of the story");

    let visual = OntologyClass::new(
        "media:VisualAesthetic",
        "Visual Aesthetic",
    ).with_comment("GPU-derived classification of visual style");

    let psychographic = OntologyClass::new(
        "user:PsychographicState",
        "Psychographic State",
    ).with_comment("Current psychological inclination");

    builder.add_class(creative_work);
    builder.add_class(narrative);
    builder.add_class(visual);
    builder.add_class(psychographic);

    // Define object properties
    let has_narrative = OntologyProperty::new_object_property(
        "sem:hasNarrativeArc",
        "has narrative arc",
        "media:CreativeWork",
        "media:NarrativeStructure",
    );

    let induces_state = OntologyProperty::new_object_property(
        "sem:inducesPsychographicState",
        "induces psychographic state",
        "media:CreativeWork",
        "user:PsychographicState",
    );

    builder.add_property(has_narrative);
    builder.add_property(induces_state);

    // Define axioms
    builder.add_axiom(OWLAxiom::transitive_property("sem:similarTo"));
    builder.add_axiom(OWLAxiom::subclass_of(
        "media:Film",
        "media:CreativeWork"
    ));

    builder
}

/// Create semantic triples for a film
fn create_film_triples(film_uri: &str) -> Vec<SemanticTriple> {
    vec![
        // Type assertion
        SemanticTriple::new_resource(
            film_uri,
            "rdf:type",
            "media:Film",
        ).with_confidence(1.0),

        // Genre
        SemanticTriple::new_resource(
            film_uri,
            "media:hasGenre",
            "genre:SciFi",
        ).with_confidence(0.95),

        // Narrative structure
        SemanticTriple::new_resource(
            film_uri,
            "sem:hasNarrativeArc",
            "narrative:NonLinear",
        ).with_confidence(0.92),

        // Visual aesthetic
        SemanticTriple::new_resource(
            film_uri,
            "media:hasVisualTone",
            "aesthetic:Noir",
        ).with_confidence(0.88),

        // Psychographic induction
        SemanticTriple::new_resource(
            film_uri,
            "sem:inducesPsychographicState",
            "state:Contemplative",
        ).with_confidence(0.85),

        // Datatype property: duration
        SemanticTriple::new_literal(
            film_uri,
            "media:durationMinutes",
            LiteralValue::Integer(148),
        ).with_confidence(1.0),

        // Complexity score
        SemanticTriple::new_literal(
            film_uri,
            "media:complexityScore",
            LiteralValue::Float(8.5),
        ).with_confidence(0.90),
    ]
}

/// Apply reasoning rules to infer new triples
fn apply_reasoning(
    _ontology: &GMCOntologyBuilder,
    triples: &[SemanticTriple],
) -> Vec<SemanticTriple> {
    let mut inferred = Vec::new();

    // Rule 1: IF hasGenre(SciFi) AND hasVisualTone(Noir) THEN hasAesthetic(NeoNoir)
    if triples.iter().any(|t| {
        t.predicate == "media:hasGenre" &&
        matches!(&t.object, TripleObject::Resource(r) if r == "genre:SciFi")
    }) && triples.iter().any(|t| {
        t.predicate == "media:hasVisualTone" &&
        matches!(&t.object, TripleObject::Resource(r) if r == "aesthetic:Noir")
    }) {
        let subject = triples.first().map(|t| t.subject.clone()).unwrap_or_default();
        inferred.push(
            SemanticTriple::new_resource(
                subject,
                "media:hasAesthetic",
                "aesthetic:NeoNoir",
            ).with_confidence(0.85)
        );
    }

    // Rule 2: IF complexityScore > 8.0 THEN targetAudience(Intellectual)
    for triple in triples {
        if triple.predicate == "media:complexityScore" {
            if let TripleObject::Literal(LiteralValue::Float(score)) = &triple.object {
                if *score > 8.0 {
                    inferred.push(
                        SemanticTriple::new_resource(
                            triple.subject.clone(),
                            "media:targetAudience",
                            "audience:Intellectual",
                        ).with_confidence(0.80)
                    );
                }
            }
        }
    }

    // Rule 3: Transitive closure - similar films
    for triple in triples {
        if triple.predicate == "sem:similarTo" {
            // In a full implementation, this would transitively infer
            // additional similarity relationships
            inferred.push(triple.clone());
        }
    }

    inferred
}

/// Create semantic constraints for embedding adjustment
fn create_constraints() -> Vec<SemanticConstraint> {
    vec![
        // Constraint 1: Sci-Fi films should boost futuristic concepts
        SemanticConstraint::new(
            "scifi_boost",
            ConstraintCondition::HasProperty {
                property: "media:hasGenre".to_string(),
                value: "genre:SciFi".to_string(),
            },
            ConstraintAction::AdjustEmbedding {
                direction: vec![0.2; 1024], // Simplified direction vector
                magnitude: 0.1,
            },
        ).with_weight(0.8),

        // Constraint 2: Non-linear narratives indicate complexity
        SemanticConstraint::new(
            "nonlinear_complexity",
            ConstraintCondition::HasProperty {
                property: "sem:hasNarrativeArc".to_string(),
                value: "narrative:NonLinear".to_string(),
            },
            ConstraintAction::AddTriple(
                SemanticTriple::new_resource(
                    "film:unknown",
                    "media:requiresAttention",
                    "level:High",
                ).with_confidence(0.85)
            ),
        ).with_weight(0.7),

        // Constraint 3: High similarity should boost recommendations
        SemanticConstraint::new(
            "similarity_boost",
            ConstraintCondition::SimilarTo {
                target: "film:reference".to_string(),
                threshold: 0.8,
            },
            ConstraintAction::ModifyScore {
                delta: 0.15,
            },
        ).with_weight(0.9),
    ]
}

/// Helper: Print triple in readable format
fn print_triple(triple: &SemanticTriple) {
    let object_str = match &triple.object {
        TripleObject::Resource(r) => r.clone(),
        TripleObject::Literal(l) => format!("{:?}", l),
        TripleObject::BlankNode(b) => format!("_:{}", b),
    };

    println!("  {} -- {} --> {} (confidence: {:.2})",
             triple.subject,
             triple.predicate,
             object_str,
             triple.confidence);
}

// To run: cargo run --example ontology_reasoning
