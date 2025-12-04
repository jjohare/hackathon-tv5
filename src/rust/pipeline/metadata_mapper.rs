use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Confidence threshold for assertions (0.0 - 1.0)
const MIN_CONFIDENCE: f64 = 0.7;

/// Visual analysis input structure
#[derive(Debug, Deserialize, Clone)]
pub struct VisualAnalysis {
    pub color_palette: Vec<String>,
    pub contrast: f64,
    pub composition_style: Option<String>,
    pub motion_vectors: Option<Vec<MotionVector>>,
    pub lighting: Option<LightingAnalysis>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MotionVector {
    pub magnitude: f64,
    pub direction: String,
    pub frequency: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LightingAnalysis {
    pub key_light_ratio: f64,
    pub shadow_depth: f64,
    pub color_temperature: i32,
}

/// Audio analysis input structure
#[derive(Debug, Deserialize, Clone)]
pub struct AudioAnalysis {
    pub tempo: Option<f64>,
    pub key: Option<String>,
    pub dialogue_complexity: Option<f64>,
    pub sound_design_score: Option<f64>,
    pub dynamic_range: Option<f64>,
}

/// Text analysis input structure
#[derive(Debug, Deserialize, Clone)]
pub struct TextAnalysis {
    pub themes: Vec<ThemeDetection>,
    pub tropes: Vec<TropeDetection>,
    pub emotional_arc: Option<EmotionalArc>,
    pub narrative_structure: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ThemeDetection {
    pub name: String,
    pub confidence: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TropeDetection {
    pub name: String,
    pub confidence: f64,
    pub timestamp: Option<f64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmotionalArc {
    pub start_valence: f64,
    pub peak_valence: f64,
    pub end_valence: f64,
    pub tension_points: Vec<f64>,
}

/// RDF Triple representation
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source: String,
}

/// Combined analysis input
#[derive(Debug, Deserialize)]
pub struct AnalysisInput {
    pub media_id: String,
    pub visual: Option<VisualAnalysis>,
    pub audio: Option<AudioAnalysis>,
    pub text: Option<TextAnalysis>,
}

/// Metadata mapper with conflict resolution
pub struct MetadataMapper {
    base_uri: String,
    ontology_prefix: HashMap<String, String>,
}

impl MetadataMapper {
    pub fn new(base_uri: String) -> Self {
        let mut ontology_prefix = HashMap::new();
        ontology_prefix.insert("media".to_string(), "http://schema.tv5.ai/media#".to_string());
        ontology_prefix.insert("ctx".to_string(), "http://schema.tv5.ai/context#".to_string());
        ontology_prefix.insert("aesthetic".to_string(), "http://schema.tv5.ai/aesthetic#".to_string());

        Self {
            base_uri,
            ontology_prefix,
        }
    }

    /// Convert analysis input to RDF triples
    pub fn map_to_rdf(&self, input: &AnalysisInput) -> Vec<RdfTriple> {
        let mut triples = Vec::new();
        let subject = format!("{}:{}", self.base_uri, input.media_id);

        if let Some(visual) = &input.visual {
            triples.extend(self.map_visual_analysis(&subject, visual));
        }

        if let Some(audio) = &input.audio {
            triples.extend(self.map_audio_analysis(&subject, audio));
        }

        if let Some(text) = &input.text {
            triples.extend(self.map_text_analysis(&subject, text));
        }

        self.resolve_conflicts(triples)
    }

    /// Map visual analysis to RDF triples
    fn map_visual_analysis(&self, subject: &str, visual: &VisualAnalysis) -> Vec<RdfTriple> {
        let mut triples = Vec::new();

        // Color palette analysis
        let aesthetic = self.analyze_color_palette(&visual.color_palette);
        if let Some((aesthetic_class, confidence)) = aesthetic {
            if confidence >= MIN_CONFIDENCE {
                triples.push(RdfTriple {
                    subject: subject.to_string(),
                    predicate: "media:hasVisualAesthetic".to_string(),
                    object: aesthetic_class,
                    confidence,
                    source: "visual_analysis".to_string(),
                });
            }
        }

        // Contrast ratio
        triples.push(RdfTriple {
            subject: subject.to_string(),
            predicate: "media:visualContrast".to_string(),
            object: format!("\"{}\"^^xsd:decimal", visual.contrast),
            confidence: 1.0,
            source: "visual_analysis".to_string(),
        });

        // Composition style
        if let Some(style) = &visual.composition_style {
            triples.push(RdfTriple {
                subject: subject.to_string(),
                predicate: "aesthetic:compositionStyle".to_string(),
                object: format!("aesthetic:{}", self.normalize_label(style)),
                confidence: 0.85,
                source: "visual_analysis".to_string(),
            });
        }

        // Motion vectors to genre hints
        if let Some(motion) = &visual.motion_vectors {
            let genre_hints = self.motion_to_genre(motion);
            for (genre, confidence) in genre_hints {
                if confidence >= MIN_CONFIDENCE {
                    triples.push(RdfTriple {
                        subject: subject.to_string(),
                        predicate: "media:suggestedGenre".to_string(),
                        object: format!("media:{}", genre),
                        confidence,
                        source: "motion_analysis".to_string(),
                    });
                }
            }
        }

        // Lighting analysis
        if let Some(lighting) = &visual.lighting {
            let mood = self.analyze_lighting(lighting);
            if let Some((mood_class, confidence)) = mood {
                if confidence >= MIN_CONFIDENCE {
                    triples.push(RdfTriple {
                        subject: subject.to_string(),
                        predicate: "aesthetic:moodLighting".to_string(),
                        object: format!("aesthetic:{}", mood_class),
                        confidence,
                        source: "lighting_analysis".to_string(),
                    });
                }
            }
        }

        triples
    }

    /// Map audio analysis to RDF triples
    fn map_audio_analysis(&self, subject: &str, audio: &AudioAnalysis) -> Vec<RdfTriple> {
        let mut triples = Vec::new();

        // Tempo and key
        if let (Some(tempo), Some(key)) = (audio.tempo, &audio.key) {
            triples.push(RdfTriple {
                subject: subject.to_string(),
                predicate: "media:hasMusicFeatures".to_string(),
                object: format!("_:musicFeatures_{}", self.hash_string(&format!("{}{}", tempo, key))),
                confidence: 0.95,
                source: "audio_analysis".to_string(),
            });

            let music_node = format!("_:musicFeatures_{}", self.hash_string(&format!("{}{}", tempo, key)));
            triples.push(RdfTriple {
                subject: music_node.clone(),
                predicate: "media:tempo".to_string(),
                object: format!("\"{}\"^^xsd:decimal", tempo),
                confidence: 1.0,
                source: "audio_analysis".to_string(),
            });

            triples.push(RdfTriple {
                subject: music_node,
                predicate: "media:musicalKey".to_string(),
                object: format!("\"{}\"", key),
                confidence: 1.0,
                source: "audio_analysis".to_string(),
            });
        }

        // Dialogue complexity to audience level
        if let Some(complexity) = audio.dialogue_complexity {
            let audience_level = self.complexity_to_audience_level(complexity);
            triples.push(RdfTriple {
                subject: subject.to_string(),
                predicate: "ctx:audienceLevel".to_string(),
                object: format!("ctx:{}", audience_level),
                confidence: self.calculate_complexity_confidence(complexity),
                source: "dialogue_analysis".to_string(),
            });
        }

        // Sound design to aesthetic markers
        if let Some(sound_design) = audio.sound_design_score {
            let aesthetic_markers = self.sound_design_to_aesthetic(sound_design);
            for (marker, confidence) in aesthetic_markers {
                if confidence >= MIN_CONFIDENCE {
                    triples.push(RdfTriple {
                        subject: subject.to_string(),
                        predicate: "aesthetic:soundDesign".to_string(),
                        object: format!("aesthetic:{}", marker),
                        confidence,
                        source: "sound_design_analysis".to_string(),
                    });
                }
            }
        }

        triples
    }

    /// Map text analysis to RDF triples
    fn map_text_analysis(&self, subject: &str, text: &TextAnalysis) -> Vec<RdfTriple> {
        let mut triples = Vec::new();

        // Themes
        for theme in &text.themes {
            if theme.confidence >= MIN_CONFIDENCE {
                triples.push(RdfTriple {
                    subject: subject.to_string(),
                    predicate: "media:hasTheme".to_string(),
                    object: format!("media:{}", self.normalize_label(&theme.name)),
                    confidence: theme.confidence,
                    source: "theme_analysis".to_string(),
                });
            }
        }

        // Tropes
        for trope in &text.tropes {
            if trope.confidence >= MIN_CONFIDENCE {
                triples.push(RdfTriple {
                    subject: subject.to_string(),
                    predicate: "media:containsTrope".to_string(),
                    object: format!("media:{}", self.normalize_label(&trope.name)),
                    confidence: trope.confidence,
                    source: "trope_analysis".to_string(),
                });
            }
        }

        // Emotional arc to narrative structure
        if let Some(arc) = &text.emotional_arc {
            let narrative_type = self.emotional_arc_to_narrative(arc);
            triples.push(RdfTriple {
                subject: subject.to_string(),
                predicate: "media:narrativeStructure".to_string(),
                object: format!("media:{}", narrative_type),
                confidence: self.calculate_arc_confidence(arc),
                source: "emotional_arc_analysis".to_string(),
            });
        }

        triples
    }

    /// Analyze color palette for aesthetic classification
    fn analyze_color_palette(&self, palette: &[String]) -> Option<(String, f64)> {
        let mut noir_score = 0.0;
        let mut vibrant_score = 0.0;
        let mut pastel_score = 0.0;

        for color_hex in palette {
            let rgb = self.hex_to_rgb(color_hex)?;
            let (r, g, b) = rgb;

            // Calculate luminance
            let luminance = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;

            // Noir detection (dark colors, low luminance)
            if luminance < 50.0 {
                noir_score += 1.0 / palette.len() as f64;
            }

            // Vibrant detection (high saturation)
            let max = r.max(g).max(b) as f64;
            let min = r.min(g).min(b) as f64;
            let saturation = if max > 0.0 { (max - min) / max } else { 0.0 };

            if saturation > 0.6 {
                vibrant_score += 1.0 / palette.len() as f64;
            }

            // Pastel detection (high luminance, low saturation)
            if luminance > 180.0 && saturation < 0.3 {
                pastel_score += 1.0 / palette.len() as f64;
            }
        }

        // Determine dominant aesthetic
        if noir_score > vibrant_score && noir_score > pastel_score && noir_score > 0.6 {
            Some(("media:Noir".to_string(), noir_score))
        } else if vibrant_score > noir_score && vibrant_score > pastel_score && vibrant_score > 0.5 {
            Some(("aesthetic:Vibrant".to_string(), vibrant_score))
        } else if pastel_score > 0.5 {
            Some(("aesthetic:Pastel".to_string(), pastel_score))
        } else {
            None
        }
    }

    /// Convert motion vectors to genre hints
    fn motion_to_genre(&self, motion: &[MotionVector]) -> Vec<(String, f64)> {
        let mut genres = Vec::new();

        let avg_magnitude: f64 = motion.iter().map(|m| m.magnitude).sum::<f64>() / motion.len() as f64;
        let avg_frequency: f64 = motion.iter().map(|m| m.frequency).sum::<f64>() / motion.len() as f64;

        // High motion + high frequency = Action
        if avg_magnitude > 0.7 && avg_frequency > 0.6 {
            genres.push(("Action".to_string(), 0.85));
        }

        // Low motion + low frequency = Drama
        if avg_magnitude < 0.3 && avg_frequency < 0.3 {
            genres.push(("Drama".to_string(), 0.80));
        }

        // Medium motion + varied frequency = Thriller
        if avg_magnitude > 0.4 && avg_magnitude < 0.7 {
            let variance = self.calculate_variance(&motion.iter().map(|m| m.frequency).collect::<Vec<_>>());
            if variance > 0.2 {
                genres.push(("Thriller".to_string(), 0.75));
            }
        }

        genres
    }

    /// Analyze lighting for mood classification
    fn analyze_lighting(&self, lighting: &LightingAnalysis) -> Option<(String, f64)> {
        // High key lighting (comedy, romance)
        if lighting.key_light_ratio > 0.8 && lighting.shadow_depth < 0.3 {
            return Some(("HighKey".to_string(), 0.85));
        }

        // Low key lighting (noir, thriller, horror)
        if lighting.key_light_ratio < 0.4 && lighting.shadow_depth > 0.7 {
            return Some(("LowKey".to_string(), 0.90));
        }

        // Warm lighting (romance, comfort)
        if lighting.color_temperature > 3500 {
            return Some(("WarmLighting".to_string(), 0.75));
        }

        // Cool lighting (sci-fi, clinical)
        if lighting.color_temperature < 3000 {
            return Some(("CoolLighting".to_string(), 0.75));
        }

        None
    }

    /// Map dialogue complexity to audience level
    fn complexity_to_audience_level(&self, complexity: f64) -> String {
        match complexity {
            x if x < 0.3 => "Children",
            x if x < 0.5 => "Family",
            x if x < 0.7 => "General",
            x if x < 0.85 => "Mature",
            _ => "Academic",
        }.to_string()
    }

    /// Calculate confidence for complexity mapping
    fn calculate_complexity_confidence(&self, complexity: f64) -> f64 {
        // Higher confidence for extreme values, lower for boundary cases
        let boundary_distances = vec![
            (complexity - 0.3).abs(),
            (complexity - 0.5).abs(),
            (complexity - 0.7).abs(),
            (complexity - 0.85).abs(),
        ];

        let min_distance = boundary_distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Map distance to confidence (further from boundaries = higher confidence)
        0.7 + (min_distance * 0.6).min(0.25)
    }

    /// Map sound design score to aesthetic markers
    fn sound_design_to_aesthetic(&self, score: f64) -> Vec<(String, f64)> {
        let mut markers = Vec::new();

        if score > 0.8 {
            markers.push(("Cinematic".to_string(), score));
            markers.push(("Immersive".to_string(), score * 0.9));
        } else if score > 0.6 {
            markers.push(("Professional".to_string(), score));
        } else if score < 0.3 {
            markers.push(("Minimalist".to_string(), 1.0 - score));
        }

        markers
    }

    /// Map emotional arc to narrative structure
    fn emotional_arc_to_narrative(&self, arc: &EmotionalArc) -> String {
        let start = arc.start_valence;
        let peak = arc.peak_valence;
        let end = arc.end_valence;

        // Rising arc (journey, triumph)
        if end > start && peak > start && (end - start) > 0.3 {
            return "HeroJourney".to_string();
        }

        // Falling arc (tragedy)
        if end < start && (start - end) > 0.3 {
            return "Tragedy".to_string();
        }

        // Roller coaster (thriller, complex drama)
        if arc.tension_points.len() > 3 {
            let variance = self.calculate_variance(&arc.tension_points);
            if variance > 0.3 {
                return "ComplexNarrative".to_string();
            }
        }

        // Flat/static (experimental, slice-of-life)
        if (end - start).abs() < 0.2 && peak - start < 0.3 {
            return "SliceOfLife".to_string();
        }

        "Standard".to_string()
    }

    /// Calculate confidence for emotional arc mapping
    fn calculate_arc_confidence(&self, arc: &EmotionalArc) -> f64 {
        let valence_range = (arc.peak_valence - arc.start_valence).abs() + (arc.peak_valence - arc.end_valence).abs();
        let tension_consistency = if arc.tension_points.len() > 1 {
            1.0 - self.calculate_variance(&arc.tension_points)
        } else {
            0.5
        };

        // Higher confidence for clear patterns
        (0.6 + valence_range * 0.2 + tension_consistency * 0.2).min(0.95)
    }

    /// Resolve conflicting triples
    fn resolve_conflicts(&self, mut triples: Vec<RdfTriple>) -> Vec<RdfTriple> {
        let mut resolved = Vec::new();
        let mut predicate_groups: HashMap<(String, String), Vec<RdfTriple>> = HashMap::new();

        // Group by subject + predicate
        for triple in triples.drain(..) {
            let key = (triple.subject.clone(), triple.predicate.clone());
            predicate_groups.entry(key).or_insert_with(Vec::new).push(triple);
        }

        // Resolve conflicts within each group
        for (_key, group) in predicate_groups {
            if group.len() == 1 {
                resolved.extend(group);
            } else {
                // Keep highest confidence assertion
                let best = group.iter().max_by(|a, b| {
                    a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal)
                }).cloned();

                if let Some(triple) = best {
                    resolved.push(triple);
                }

                // Add conflicting assertions as alternative properties
                for triple in group.iter().filter(|t| t.confidence < best.as_ref().unwrap().confidence) {
                    if triple.confidence >= MIN_CONFIDENCE {
                        let mut alt_triple = triple.clone();
                        alt_triple.predicate = format!("{}:alternative", alt_triple.predicate);
                        resolved.push(alt_triple);
                    }
                }
            }
        }

        resolved
    }

    /// Serialize triples to Turtle format
    pub fn to_turtle(&self, triples: &[RdfTriple]) -> String {
        let mut output = String::new();

        // Add prefixes
        for (prefix, uri) in &self.ontology_prefix {
            output.push_str(&format!("@prefix {}: <{}>.\n", prefix, uri));
        }
        output.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.\n\n");

        // Group by subject
        let mut by_subject: HashMap<String, Vec<&RdfTriple>> = HashMap::new();
        for triple in triples {
            by_subject.entry(triple.subject.clone()).or_insert_with(Vec::new).push(triple);
        }

        // Format triples
        for (subject, subject_triples) in by_subject {
            output.push_str(&format!("{}\n", subject));
            for (i, triple) in subject_triples.iter().enumerate() {
                let separator = if i == subject_triples.len() - 1 { "." } else { ";" };
                output.push_str(&format!("  {} {} {}\n", triple.predicate, triple.object, separator));
            }
            output.push('\n');
        }

        output
    }

    // Helper functions
    fn hex_to_rgb(&self, hex: &str) -> Option<(u8, u8, u8)> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }

        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;

        Some((r, g, b))
    }

    fn normalize_label(&self, label: &str) -> String {
        label.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn hash_string(&self, s: &str) -> String {
        format!("{:x}", s.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64)))
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noir_color_detection() {
        let mapper = MetadataMapper::new("movie".to_string());
        let palette = vec!["#000000".to_string(), "#1a1a1a".to_string(), "#0d0d0d".to_string()];

        let result = mapper.analyze_color_palette(&palette);
        assert!(result.is_some());

        let (aesthetic, confidence) = result.unwrap();
        assert_eq!(aesthetic, "media:Noir");
        assert!(confidence > 0.8);
    }

    #[test]
    fn test_complexity_to_audience() {
        let mapper = MetadataMapper::new("movie".to_string());

        assert_eq!(mapper.complexity_to_audience_level(0.2), "Children");
        assert_eq!(mapper.complexity_to_audience_level(0.4), "Family");
        assert_eq!(mapper.complexity_to_audience_level(0.6), "General");
        assert_eq!(mapper.complexity_to_audience_level(0.8), "Mature");
        assert_eq!(mapper.complexity_to_audience_level(0.95), "Academic");
    }

    #[test]
    fn test_conflict_resolution() {
        let mapper = MetadataMapper::new("movie".to_string());

        let triples = vec![
            RdfTriple {
                subject: "movie:123".to_string(),
                predicate: "media:hasGenre".to_string(),
                object: "media:Action".to_string(),
                confidence: 0.9,
                source: "motion".to_string(),
            },
            RdfTriple {
                subject: "movie:123".to_string(),
                predicate: "media:hasGenre".to_string(),
                object: "media:Drama".to_string(),
                confidence: 0.7,
                source: "emotional".to_string(),
            },
        ];

        let resolved = mapper.resolve_conflicts(triples);

        // Should keep highest confidence as primary
        assert!(resolved.iter().any(|t|
            t.object == "media:Action" && t.confidence == 0.9
        ));

        // Lower confidence should become alternative
        assert!(resolved.iter().any(|t|
            t.predicate.contains("alternative") && t.object == "media:Drama"
        ));
    }
}
