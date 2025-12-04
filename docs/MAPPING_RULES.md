# ML Analysis to RDF Mapping Rules

## Overview

This document specifies the transformation rules for converting machine learning analysis outputs (visual, audio, text) into RDF triples using the TV5 Media Ontology.

## Confidence Scoring

**Minimum Confidence Threshold**: 0.7

All assertions must meet this threshold to be included in the output. Lower confidence assertions may be included as `alternative` properties.

### Confidence Calculation Methods

1. **Direct Analysis Confidence**: Use ML model output confidence directly
2. **Boundary Distance Confidence**: For categorical mappings, calculate distance from decision boundaries
3. **Pattern Strength Confidence**: Based on statistical measures (variance, magnitude, etc.)
4. **Composite Confidence**: Weighted average of multiple signals

## Visual Analysis Mappings

### Color Palette → Visual Aesthetic

**Rule**: Analyze color palette luminance and saturation to determine aesthetic classification.

**Formulas**:
- Luminance: `0.299*R + 0.587*G + 0.114*B`
- Saturation: `(max(R,G,B) - min(R,G,B)) / max(R,G,B)`

**Classifications**:

| Pattern | Threshold | Aesthetic Class | Confidence Calculation |
|---------|-----------|-----------------|------------------------|
| Dark colors (luminance < 50) | >60% of palette | `media:Noir` | `noir_color_ratio` |
| High saturation (>0.6) | >50% of palette | `aesthetic:Vibrant` | `vibrant_color_ratio` |
| Light + low saturation | >50% of palette | `aesthetic:Pastel` | `pastel_color_ratio` |

**Example**:
```json
Input: {"color_palette": ["#000000", "#1a1a1a", "#0d0d0d"], "contrast": 0.95}

Output:
:movie_123 media:hasVisualAesthetic media:Noir .
:movie_123 media:visualContrast "0.95"^^xsd:decimal .

Confidence: 0.95 (based on 100% dark colors)
```

### Composition Style → Aesthetic Properties

**Rule**: Map recognized composition patterns to aesthetic classifications.

**Mappings**:
- `symmetrical` → `aesthetic:SymmetricalComposition`
- `rule_of_thirds` → `aesthetic:BalancedComposition`
- `center_dominant` → `aesthetic:CentralFocus`
- `diagonal` → `aesthetic:DynamicComposition`

**Confidence**: Fixed at 0.85 (composition detection reliability)

### Motion Vectors → Genre Hints

**Rule**: Analyze average magnitude and frequency of motion to suggest genres.

**Formulas**:
```
avg_magnitude = sum(motion.magnitude) / len(motion)
avg_frequency = sum(motion.frequency) / len(motion)
variance = sqrt(sum((x - mean)^2) / n)
```

**Genre Detection**:

| Conditions | Genre | Confidence |
|------------|-------|------------|
| `avg_magnitude > 0.7 AND avg_frequency > 0.6` | `media:Action` | 0.85 |
| `avg_magnitude < 0.3 AND avg_frequency < 0.3` | `media:Drama` | 0.80 |
| `avg_magnitude ∈ [0.4,0.7] AND variance > 0.2` | `media:Thriller` | 0.75 |

**Example**:
```json
Input: {"motion_vectors": [
  {"magnitude": 0.9, "direction": "forward", "frequency": 0.8},
  {"magnitude": 0.85, "direction": "pan", "frequency": 0.7}
]}

Output:
:movie_123 media:suggestedGenre media:Action .

Confidence: 0.85
```

### Lighting Analysis → Mood Classification

**Rule**: Key light ratio and shadow depth determine lighting mood.

**Classifications**:

| Key Light Ratio | Shadow Depth | Color Temp | Mood Class | Confidence |
|-----------------|--------------|------------|------------|------------|
| >0.8 | <0.3 | Any | `aesthetic:HighKey` | 0.85 |
| <0.4 | >0.7 | Any | `aesthetic:LowKey` | 0.90 |
| Any | Any | >3500K | `aesthetic:WarmLighting` | 0.75 |
| Any | Any | <3000K | `aesthetic:CoolLighting` | 0.75 |

**Example**:
```json
Input: {
  "lighting": {
    "key_light_ratio": 0.3,
    "shadow_depth": 0.85,
    "color_temperature": 2800
  }
}

Output:
:movie_123 aesthetic:moodLighting aesthetic:LowKey .

Confidence: 0.90
```

## Audio Analysis Mappings

### Tempo & Key → Music Features

**Rule**: Create blank node for music features composite.

**Structure**:
```turtle
:movie_123 media:hasMusicFeatures _:musicFeatures_<hash> .
_:musicFeatures_<hash> media:tempo "120.0"^^xsd:decimal .
_:musicFeatures_<hash> media:musicalKey "C major" .
```

**Confidence**: 0.95 (high reliability of audio analysis)

### Dialogue Complexity → Audience Level

**Rule**: Map complexity score to target audience classification.

**Thresholds**:

| Complexity Range | Audience Level | Example Content |
|------------------|----------------|-----------------|
| [0.0, 0.3) | `ctx:Children` | Simple vocabulary, short sentences |
| [0.3, 0.5) | `ctx:Family` | Moderate vocabulary, mixed complexity |
| [0.5, 0.7) | `ctx:General` | Standard adult vocabulary |
| [0.7, 0.85) | `ctx:Mature` | Complex themes, advanced vocabulary |
| [0.85, 1.0] | `ctx:Academic` | Technical jargon, specialized concepts |

**Confidence Calculation**:
```
min_distance = min(|complexity - 0.3|, |complexity - 0.5|, |complexity - 0.7|, |complexity - 0.85|)
confidence = 0.7 + min(min_distance * 0.6, 0.25)
```

Higher confidence when complexity is far from boundaries, lower near decision points.

**Example**:
```json
Input: {"dialogue_complexity": 0.85}

Output:
:movie_123 ctx:audienceLevel ctx:Mature .

Confidence: 0.70 (near boundary between Mature and Academic)
```

### Sound Design Score → Aesthetic Markers

**Rule**: High-quality sound design correlates with production aesthetic.

**Mappings**:

| Score Range | Aesthetic Markers | Confidence |
|-------------|-------------------|------------|
| >0.8 | `aesthetic:Cinematic`, `aesthetic:Immersive` | `score`, `score * 0.9` |
| >0.6 | `aesthetic:Professional` | `score` |
| <0.3 | `aesthetic:Minimalist` | `1.0 - score` |

**Example**:
```json
Input: {"sound_design_score": 0.9}

Output:
:movie_123 aesthetic:soundDesign aesthetic:Cinematic .
:movie_123 aesthetic:soundDesign aesthetic:Immersive .

Confidence: 0.9, 0.81
```

## Text Analysis Mappings

### Themes → Theme Assertions

**Rule**: Direct mapping with confidence passthrough.

**Format**:
```turtle
:movie_123 media:hasTheme media:<NormalizedThemeName> .
```

**Normalization**: Remove special characters, capitalize each word, remove spaces.

**Example**:
```json
Input: {"themes": [{"name": "coming of age", "confidence": 0.92}]}

Output:
:movie_123 media:hasTheme media:ComingOfAge .

Confidence: 0.92 (from ML model)
```

### Tropes → Trope Assertions

**Rule**: Same as themes, with optional timestamp preservation.

**Format**:
```turtle
:movie_123 media:containsTrope media:<NormalizedTropeName> .
```

**Example**:
```json
Input: {"tropes": [{"name": "hero's journey", "confidence": 0.88, "timestamp": 120.0}]}

Output:
:movie_123 media:containsTrope media:HerosJourney .

Confidence: 0.88
```

### Emotional Arc → Narrative Structure

**Rule**: Analyze valence trajectory and tension points to classify narrative.

**Classification Logic**:

```python
start, peak, end = arc.start_valence, arc.peak_valence, arc.end_valence

if end > start and peak > start and (end - start) > 0.3:
    return "HeroJourney"
elif end < start and (start - end) > 0.3:
    return "Tragedy"
elif len(tension_points) > 3 and variance(tension_points) > 0.3:
    return "ComplexNarrative"
elif abs(end - start) < 0.2 and peak - start < 0.3:
    return "SliceOfLife"
else:
    return "Standard"
```

**Confidence Calculation**:
```
valence_range = |peak - start| + |peak - end|
tension_consistency = 1.0 - variance(tension_points)
confidence = min(0.6 + valence_range * 0.2 + tension_consistency * 0.2, 0.95)
```

**Narrative Patterns**:

| Pattern | Visual Representation | Example Films |
|---------|----------------------|---------------|
| HeroJourney | Low → High → High | Star Wars, Rocky |
| Tragedy | High → High → Low | Hamlet, Romeo and Juliet |
| ComplexNarrative | Variable with multiple peaks | Inception, Pulp Fiction |
| SliceOfLife | Flat trajectory | Lost in Translation, Paterson |

**Example**:
```json
Input: {
  "emotional_arc": {
    "start_valence": 0.3,
    "peak_valence": 0.9,
    "end_valence": 0.8,
    "tension_points": [0.4, 0.7, 0.9, 0.6]
  }
}

Output:
:movie_123 media:narrativeStructure media:HeroJourney .

Confidence: 0.82
```

## Conflict Resolution

### Strategy: Highest Confidence Wins

When multiple analyses produce conflicting assertions for the same predicate:

1. **Primary Assertion**: Keep the triple with highest confidence
2. **Alternative Assertions**: Demote lower confidence triples to `<predicate>:alternative`
3. **Threshold Filter**: Only alternatives with confidence ≥ 0.7 are preserved

**Example**:
```turtle
# Input conflicts:
:movie_123 media:suggestedGenre media:Action .  # confidence: 0.85 (motion)
:movie_123 media:suggestedGenre media:Drama .   # confidence: 0.75 (emotional)

# Output resolution:
:movie_123 media:suggestedGenre media:Action .                    # Primary
:movie_123 media:suggestedGenre:alternative media:Drama .         # Alternative
```

### Multi-Modal Agreement Boosting

When multiple modalities agree on the same assertion, boost confidence:

```
boosted_confidence = 1.0 - (1.0 - conf1) * (1.0 - conf2) * (1.0 - conf3)
```

**Example**:
- Visual suggests `media:Noir` (0.85)
- Audio suggests `media:Noir` (0.80) via low-key sound design
- Text suggests `media:Noir` (0.75) via dark themes

Boosted confidence: `1.0 - (0.15 * 0.20 * 0.25) = 0.9925` → capped at 0.99

## Serialization Format

### Turtle Output Structure

```turtle
@prefix media: <http://schema.tv5.ai/media#>.
@prefix ctx: <http://schema.tv5.ai/context#>.
@prefix aesthetic: <http://schema.tv5.ai/aesthetic#>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

:movie_123
  media:hasVisualAesthetic media:Noir ;
  media:visualContrast "0.95"^^xsd:decimal ;
  aesthetic:moodLighting aesthetic:LowKey ;
  media:hasMusicFeatures _:musicFeatures_abc123 ;
  ctx:audienceLevel ctx:Mature ;
  media:hasTheme media:Redemption ;
  media:containsTrope media:HerosJourney ;
  media:narrativeStructure media:HeroJourney .

_:musicFeatures_abc123
  media:tempo "100.0"^^xsd:decimal ;
  media:musicalKey "D minor" .
```

## Implementation Notes

### Performance Considerations

1. **Batch Processing**: Process multiple media items in parallel
2. **Caching**: Cache color analysis results for palette patterns
3. **Lazy Evaluation**: Only compute complex metrics when needed
4. **Index Structures**: Use hash maps for conflict detection

### Error Handling

1. **Invalid Input**: Return empty triple set with error log
2. **Missing Fields**: Skip optional analysis, continue with available data
3. **Confidence Failures**: Log low-confidence assertions separately
4. **Parsing Errors**: Validate hex colors, numeric ranges before processing

### Validation

All outputs must pass:
- RDF syntax validation (N-Triples parser)
- Confidence range check (0.0 ≤ confidence ≤ 1.0)
- URI format validation
- Ontology term existence check

## Testing Strategy

### Unit Tests

- Color analysis accuracy (noir, vibrant, pastel detection)
- Audience level boundary cases
- Emotional arc classification edge cases
- Conflict resolution correctness

### Integration Tests

- Full pipeline JSON → RDF conversion
- Multi-modal agreement boosting
- Turtle serialization correctness
- Large batch processing performance

### Test Data Sets

1. **Noir Films**: Dark Knight, Blade Runner, Sin City
2. **Vibrant Films**: Amélie, Wes Anderson films, Pixar movies
3. **Complex Narratives**: Inception, Pulp Fiction, Memento
4. **Action Films**: Mad Max, John Wick, Mission Impossible

## Version History

- **v1.0** (2025-12-04): Initial mapping rules specification
- Confidence thresholds: 0.7 minimum
- Conflict resolution: highest confidence wins
- Multi-modal boosting implemented

## References

- TV5 Media Ontology: `/home/devuser/workspace/hackathon-tv5/docs/ONTOLOGY.md`
- RDF Turtle Specification: https://www.w3.org/TR/turtle/
- Color Theory: Luminance and saturation formulas (ITU-R BT.709)
- Narrative Structure Theory: Joseph Campbell's Hero's Journey
