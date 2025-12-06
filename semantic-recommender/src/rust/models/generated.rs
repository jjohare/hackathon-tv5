// AUTO-GENERATED - DO NOT EDIT
// Generated from: design/ontology/expanded-media-ontology.ttl
// Build script: build.rs
//
// This file provides compile-time type safety for ontology-defined enums.
// Any changes to the ontology will automatically update these types.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

///
/// Generated from ontology class: media:Genre
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Genre {
    #[serde(rename = "Science Fiction")]
    SciFi,
    Thriller,
    Action,
    Comedy,
    Documentary,
    Drama,
    Horror,
    Romance,
}

impl fmt::Display for Genre {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SciFi => write!(f, "Science Fiction"),
            Self::Thriller => write!(f, "Thriller"),
            Self::Action => write!(f, "Action"),
            Self::Comedy => write!(f, "Comedy"),
            Self::Documentary => write!(f, "Documentary"),
            Self::Drama => write!(f, "Drama"),
            Self::Horror => write!(f, "Horror"),
            Self::Romance => write!(f, "Romance"),
        }
    }
}

impl Genre {
    /// Convert to OWL ontology URI
    pub fn to_owl_uri(&self) -> &'static str {
        match self {
            Self::SciFi => "http://recommendation.org/ontology/media#SciFi",
            Self::Thriller => "http://recommendation.org/ontology/media#Thriller",
            Self::Action => "http://recommendation.org/ontology/media#Action",
            Self::Comedy => "http://recommendation.org/ontology/media#Comedy",
            Self::Documentary => "http://recommendation.org/ontology/media#Documentary",
            Self::Drama => "http://recommendation.org/ontology/media#Drama",
            Self::Horror => "http://recommendation.org/ontology/media#Horror",
            Self::Romance => "http://recommendation.org/ontology/media#Romance",
        }
    }

    /// Parse from OWL ontology URI
    pub fn from_owl_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://recommendation.org/ontology/media#SciFi" => Some(Self::SciFi),
            "http://recommendation.org/ontology/media#Thriller" => Some(Self::Thriller),
            "http://recommendation.org/ontology/media#Action" => Some(Self::Action),
            "http://recommendation.org/ontology/media#Comedy" => Some(Self::Comedy),
            "http://recommendation.org/ontology/media#Documentary" => Some(Self::Documentary),
            "http://recommendation.org/ontology/media#Drama" => Some(Self::Drama),
            "http://recommendation.org/ontology/media#Horror" => Some(Self::Horror),
            "http://recommendation.org/ontology/media#Romance" => Some(Self::Romance),
            _ => None,
        }
    }
}

/// GPU-derived color grading and lighting classification
///
/// Generated from ontology class: media:VisualAesthetic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisualAesthetic {
    /// Natural lighting, realistic color grading
    #[serde(rename = "Naturalistic Aesthetic")]
    Naturalistic,
    /// Vibrant neon colors, futuristic urban environments
    #[serde(rename = "Neon/Cyberpunk Aesthetic")]
    Neon,
    /// Soft, desaturated colors, dreamy quality
    #[serde(rename = "Pastel Aesthetic")]
    Pastel,
    /// High contrast, low key lighting, chiaroscuro
    #[serde(rename = "Film Noir Aesthetic")]
    Noir,
}

impl fmt::Display for VisualAesthetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Naturalistic => write!(f, "Naturalistic Aesthetic"),
            Self::Neon => write!(f, "Neon/Cyberpunk Aesthetic"),
            Self::Pastel => write!(f, "Pastel Aesthetic"),
            Self::Noir => write!(f, "Film Noir Aesthetic"),
        }
    }
}

impl VisualAesthetic {
    /// Convert to OWL ontology URI
    pub fn to_owl_uri(&self) -> &'static str {
        match self {
            Self::Naturalistic => "http://recommendation.org/ontology/media#NaturalisticAesthetic",
            Self::Neon => "http://recommendation.org/ontology/media#NeonAesthetic",
            Self::Pastel => "http://recommendation.org/ontology/media#PastelAesthetic",
            Self::Noir => "http://recommendation.org/ontology/media#NoirAesthetic",
        }
    }

    /// Parse from OWL ontology URI
    pub fn from_owl_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://recommendation.org/ontology/media#NaturalisticAesthetic" => Some(Self::Naturalistic),
            "http://recommendation.org/ontology/media#NeonAesthetic" => Some(Self::Neon),
            "http://recommendation.org/ontology/media#PastelAesthetic" => Some(Self::Pastel),
            "http://recommendation.org/ontology/media#NoirAesthetic" => Some(Self::Noir),
            _ => None,
        }
    }
}

/// Architectural flow of story (Linear, Non-linear, Hero's Journey, Ensemble)
///
/// Generated from ontology class: media:NarrativeStructure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NarrativeStructure {
    #[serde(rename = "Linear Narrative")]
    Linear,
    #[serde(rename = "Non-Linear Narrative")]
    NonLinear,
    #[serde(rename = "Hero's Journey")]
    HerosJourney,
    #[serde(rename = "Ensemble Cast Structure")]
    EnsembleCast,
}

impl fmt::Display for NarrativeStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => write!(f, "Linear Narrative"),
            Self::NonLinear => write!(f, "Non-Linear Narrative"),
            Self::HerosJourney => write!(f, "Hero's Journey"),
            Self::EnsembleCast => write!(f, "Ensemble Cast Structure"),
        }
    }
}

impl NarrativeStructure {
    /// Convert to OWL ontology URI
    pub fn to_owl_uri(&self) -> &'static str {
        match self {
            Self::Linear => "http://recommendation.org/ontology/media#LinearNarrative",
            Self::NonLinear => "http://recommendation.org/ontology/media#NonLinearNarrative",
            Self::HerosJourney => "http://recommendation.org/ontology/media#HerosJourney",
            Self::EnsembleCast => "http://recommendation.org/ontology/media#EnsembleCast",
        }
    }

    /// Parse from OWL ontology URI
    pub fn from_owl_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://recommendation.org/ontology/media#LinearNarrative" => Some(Self::Linear),
            "http://recommendation.org/ontology/media#NonLinearNarrative" => Some(Self::NonLinear),
            "http://recommendation.org/ontology/media#HerosJourney" => Some(Self::HerosJourney),
            "http://recommendation.org/ontology/media#EnsembleCast" => Some(Self::EnsembleCast),
            _ => None,
        }
    }
}

/// Emotional tone derived from audio-visual analysis
///
/// Generated from ontology class: media:Mood
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mood {
    #[serde(rename = "Tense/Suspenseful")]
    Tense,
    #[serde(rename = "Melancholic/Sad")]
    Melancholic,
    #[serde(rename = "Uplifting/Joyful")]
    Uplifting,
    #[serde(rename = "Intense/Action-Packed")]
    Intense,
}

impl fmt::Display for Mood {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tense => write!(f, "Tense/Suspenseful"),
            Self::Melancholic => write!(f, "Melancholic/Sad"),
            Self::Uplifting => write!(f, "Uplifting/Joyful"),
            Self::Intense => write!(f, "Intense/Action-Packed"),
        }
    }
}

impl Mood {
    /// Convert to OWL ontology URI
    pub fn to_owl_uri(&self) -> &'static str {
        match self {
            Self::Tense => "http://recommendation.org/ontology/media#Tense",
            Self::Melancholic => "http://recommendation.org/ontology/media#Melancholic",
            Self::Uplifting => "http://recommendation.org/ontology/media#Uplifting",
            Self::Intense => "http://recommendation.org/ontology/media#Intense",
        }
    }

    /// Parse from OWL ontology URI
    pub fn from_owl_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://recommendation.org/ontology/media#Tense" => Some(Self::Tense),
            "http://recommendation.org/ontology/media#Melancholic" => Some(Self::Melancholic),
            "http://recommendation.org/ontology/media#Uplifting" => Some(Self::Uplifting),
            "http://recommendation.org/ontology/media#Intense" => Some(Self::Intense),
            _ => None,
        }
    }
}

/// Quantified scene cuts per minute and dialogue density
///
/// Generated from ontology class: media:Pacing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pacing {
    /// High cut frequency (>30 cuts/min), intense action
    #[serde(rename = "Fast Paced")]
    Fast,
    /// Balanced pacing (10-30 cuts/min)
    #[serde(rename = "Moderate Paced")]
    Moderate,
    /// Contemplative, long takes (<10 cuts/min)
    #[serde(rename = "Slow Paced")]
    Slow,
}

impl fmt::Display for Pacing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fast => write!(f, "Fast Paced"),
            Self::Moderate => write!(f, "Moderate Paced"),
            Self::Slow => write!(f, "Slow Paced"),
        }
    }
}

impl Pacing {
    /// Convert to OWL ontology URI
    pub fn to_owl_uri(&self) -> &'static str {
        match self {
            Self::Fast => "http://recommendation.org/ontology/media#FastPaced",
            Self::Moderate => "http://recommendation.org/ontology/media#ModeratePaced",
            Self::Slow => "http://recommendation.org/ontology/media#SlowPaced",
        }
    }

    /// Parse from OWL ontology URI
    pub fn from_owl_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://recommendation.org/ontology/media#FastPaced" => Some(Self::Fast),
            "http://recommendation.org/ontology/media#ModeratePaced" => Some(Self::Moderate),
            "http://recommendation.org/ontology/media#SlowPaced" => Some(Self::Slow),
            _ => None,
        }
    }
}

/// Trait for types that map to OWL ontology URIs
pub trait OntologyMappable {
    /// Get the OWL URI for this value
    fn to_owl_uri(&self) -> &'static str;
    
    /// Parse from OWL URI
    fn from_owl_uri(uri: &str) -> Option<Self> where Self: Sized;
}

impl OntologyMappable for Genre {
    fn to_owl_uri(&self) -> &'static str {
        self.to_owl_uri()
    }

    fn from_owl_uri(uri: &str) -> Option<Self> {
        Self::from_owl_uri(uri)
    }
}

impl OntologyMappable for VisualAesthetic {
    fn to_owl_uri(&self) -> &'static str {
        self.to_owl_uri()
    }

    fn from_owl_uri(uri: &str) -> Option<Self> {
        Self::from_owl_uri(uri)
    }
}

impl OntologyMappable for NarrativeStructure {
    fn to_owl_uri(&self) -> &'static str {
        self.to_owl_uri()
    }

    fn from_owl_uri(uri: &str) -> Option<Self> {
        Self::from_owl_uri(uri)
    }
}

impl OntologyMappable for Mood {
    fn to_owl_uri(&self) -> &'static str {
        self.to_owl_uri()
    }

    fn from_owl_uri(uri: &str) -> Option<Self> {
        Self::from_owl_uri(uri)
    }
}

impl OntologyMappable for Pacing {
    fn to_owl_uri(&self) -> &'static str {
        self.to_owl_uri()
    }

    fn from_owl_uri(uri: &str) -> Option<Self> {
        Self::from_owl_uri(uri)
    }
}

