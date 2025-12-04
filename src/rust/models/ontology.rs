/// OWL ontology types and semantic reasoning structures
///
/// Implements GMC-O (Global Media & Context Ontology) with:
/// - OWL classes and properties
/// - RDF triples for knowledge graph
/// - Semantic axioms and constraints
/// - Reasoning rules

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OWL Class representation (GMC-O ontology)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OntologyClass {
    /// Class URI (e.g., "media:CreativeWork")
    pub uri: String,

    /// Human-readable label
    pub label: String,

    /// Class description
    pub comment: Option<String>,

    /// Parent class (subClassOf)
    pub parent: Option<Box<OntologyClass>>,

    /// Equivalent classes
    pub equivalent_to: Vec<String>,
}

impl OntologyClass {
    /// Create new ontology class
    pub fn new(uri: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            label: label.into(),
            comment: None,
            parent: None,
            equivalent_to: Vec::new(),
        }
    }

    /// Set parent class (subClassOf relationship)
    pub fn with_parent(mut self, parent: OntologyClass) -> Self {
        self.parent = Some(Box::new(parent));
        self
    }

    /// Set comment/description
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }
}

/// OWL Object Property (relationships between classes)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OntologyProperty {
    /// Property URI (e.g., "sem:hasNarrativeArc")
    pub uri: String,

    /// Human-readable label
    pub label: String,

    /// Domain (subject class)
    pub domain: String,

    /// Range (object class)
    pub range: String,

    /// Property type
    pub property_type: PropertyType,

    /// Comment/description
    pub comment: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyType {
    /// Object property (relates two resources)
    ObjectProperty,
    /// Datatype property (relates resource to literal)
    DatatypeProperty,
    /// Annotation property (metadata)
    AnnotationProperty,
}

impl OntologyProperty {
    /// Create new object property
    pub fn new_object_property(
        uri: impl Into<String>,
        label: impl Into<String>,
        domain: impl Into<String>,
        range: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            label: label.into(),
            domain: domain.into(),
            range: range.into(),
            property_type: PropertyType::ObjectProperty,
            comment: None,
        }
    }

    /// Create new datatype property
    pub fn new_datatype_property(
        uri: impl Into<String>,
        label: impl Into<String>,
        domain: impl Into<String>,
        range: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            label: label.into(),
            domain: domain.into(),
            range: range.into(),
            property_type: PropertyType::DatatypeProperty,
            comment: None,
        }
    }
}

/// RDF Triple (subject-predicate-object)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticTriple {
    /// Subject (resource URI or blank node)
    pub subject: String,

    /// Predicate (property URI)
    pub predicate: String,

    /// Object (resource URI, blank node, or literal)
    pub object: TripleObject,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TripleObject {
    /// Resource URI
    Resource(String),
    /// Literal value
    Literal(LiteralValue),
    /// Blank node
    BlankNode(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    DateTime(chrono::DateTime<chrono::Utc>),
}

impl SemanticTriple {
    /// Create new triple with resource object
    pub fn new_resource(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: TripleObject::Resource(object.into()),
            confidence: 1.0,
        }
    }

    /// Create new triple with literal object
    pub fn new_literal(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        literal: LiteralValue,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: TripleObject::Literal(literal),
            confidence: 1.0,
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Check if confidence meets threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// OWL Axiom (semantic constraints and rules)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OWLAxiom {
    /// Axiom type
    pub axiom_type: AxiomType,

    /// Axiom components (varies by type)
    pub components: Vec<String>,

    /// Human-readable description
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxiomType {
    /// SubClassOf(A, B) - A is subclass of B
    SubClassOf,
    /// EquivalentClasses(A, B) - A and B are equivalent
    EquivalentClasses,
    /// DisjointClasses(A, B) - A and B are disjoint
    DisjointClasses,
    /// SubPropertyOf(P, Q) - P is subproperty of Q
    SubPropertyOf,
    /// TransitiveProperty(P) - P is transitive
    TransitiveProperty,
    /// SymmetricProperty(P) - P is symmetric
    SymmetricProperty,
    /// InverseProperties(P, Q) - P and Q are inverses
    InverseProperties,
}

impl OWLAxiom {
    /// Create SubClassOf axiom
    pub fn subclass_of(subclass: impl Into<String>, superclass: impl Into<String>) -> Self {
        let subclass_str = subclass.into();
        let superclass_str = superclass.into();
        Self {
            axiom_type: AxiomType::SubClassOf,
            components: vec![subclass_str.clone(), superclass_str.clone()],
            description: format!("{} is a subclass of {}", subclass_str, superclass_str),
        }
    }

    /// Create TransitiveProperty axiom
    pub fn transitive_property(property: impl Into<String>) -> Self {
        let prop = property.into();
        Self {
            axiom_type: AxiomType::TransitiveProperty,
            components: vec![prop.clone()],
            description: format!("{} is a transitive property", prop),
        }
    }
}

/// Ontology reasoner result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    /// Inferred triples
    pub inferred_triples: Vec<SemanticTriple>,

    /// Applied rules
    pub applied_rules: Vec<String>,

    /// Reasoning time (milliseconds)
    pub reasoning_time_ms: f32,
}

/// Semantic constraint (IF-THEN rule for embeddings)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConstraint {
    /// Constraint ID
    pub id: String,

    /// Condition (IF part)
    pub condition: ConstraintCondition,

    /// Action (THEN part)
    pub action: ConstraintAction,

    /// Weight/importance (0.0-1.0)
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintCondition {
    /// Has property with value
    HasProperty { property: String, value: String },
    /// Semantic similarity above threshold
    SimilarTo { target: String, threshold: f32 },
    /// Combination of conditions (AND)
    And(Vec<ConstraintCondition>),
    /// Alternative conditions (OR)
    Or(Vec<ConstraintCondition>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintAction {
    /// Adjust embedding in direction of target
    AdjustEmbedding { direction: Vec<f32>, magnitude: f32 },
    /// Add semantic triple
    AddTriple(SemanticTriple),
    /// Boost/penalize similarity score
    ModifyScore { delta: f32 },
}

impl SemanticConstraint {
    /// Create new constraint
    pub fn new(
        id: impl Into<String>,
        condition: ConstraintCondition,
        action: ConstraintAction,
    ) -> Self {
        Self {
            id: id.into(),
            condition,
            action,
            weight: 1.0,
        }
    }

    /// Set constraint weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }
}

/// GMC-O Ontology builder helper
pub struct GMCOntologyBuilder {
    classes: HashMap<String, OntologyClass>,
    properties: HashMap<String, OntologyProperty>,
    axioms: Vec<OWLAxiom>,
}

impl GMCOntologyBuilder {
    /// Create new ontology builder
    pub fn new() -> Self {
        Self {
            classes: HashMap::new(),
            properties: HashMap::new(),
            axioms: Vec::new(),
        }
    }

    /// Add class to ontology
    pub fn add_class(&mut self, class: OntologyClass) {
        self.classes.insert(class.uri.clone(), class);
    }

    /// Add property to ontology
    pub fn add_property(&mut self, property: OntologyProperty) {
        self.properties.insert(property.uri.clone(), property);
    }

    /// Add axiom to ontology
    pub fn add_axiom(&mut self, axiom: OWLAxiom) {
        self.axioms.push(axiom);
    }

    /// Get class by URI
    pub fn get_class(&self, uri: &str) -> Option<&OntologyClass> {
        self.classes.get(uri)
    }

    /// Export to Turtle format (simplified)
    pub fn to_turtle(&self) -> String {
        let mut turtle = String::from("@prefix media: <http://recommendation.org/ontology/media#> .\n");
        turtle.push_str("@prefix user: <http://recommendation.org/ontology/user#> .\n");
        turtle.push_str("@prefix ctx: <http://recommendation.org/ontology/context#> .\n\n");

        // Export classes
        for class in self.classes.values() {
            turtle.push_str(&format!("{} a owl:Class ;\n", class.uri));
            turtle.push_str(&format!("  rdfs:label \"{}\"@en ;\n", class.label));
            if let Some(comment) = &class.comment {
                turtle.push_str(&format!("  rdfs:comment \"{}\"@en .\n\n", comment));
            } else {
                turtle.push_str(" .\n\n");
            }
        }

        turtle
    }
}

impl Default for GMCOntologyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ontology_class_creation() {
        let class = OntologyClass::new("media:Film", "Film")
            .with_comment("A feature film");

        assert_eq!(class.uri, "media:Film");
        assert_eq!(class.label, "Film");
        assert!(class.comment.is_some());
    }

    #[test]
    fn test_semantic_triple() {
        let triple = SemanticTriple::new_resource(
            "film:Inception",
            "media:hasGenre",
            "genre:SciFi"
        ).with_confidence(0.95);

        assert_eq!(triple.confidence, 0.95);
        assert!(triple.meets_threshold(0.9));
    }

    #[test]
    fn test_owl_axiom() {
        let axiom = OWLAxiom::subclass_of("media:Film", "media:CreativeWork");
        assert_eq!(axiom.axiom_type, AxiomType::SubClassOf);
        assert_eq!(axiom.components.len(), 2);
    }
}
