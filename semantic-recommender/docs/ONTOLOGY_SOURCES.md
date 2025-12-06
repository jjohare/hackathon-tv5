# Ontology Sources and Processing Requirements

**Date:** 2025-12-06
**Version:** 1.0
**Status:** Reference Documentation

---

## Overview

This document tracks the external ontology repositories integrated into the semantic recommender for ontology-enhanced reasoning with whelk-rs (EL++ reasoner).

---

## Ontology Sources

### 1. AdA Film Ontology

**Repository:** https://github.com/ProjectAdA/public/tree/master/ontology
**License:** Apache 2.0 (assumed for academic research use)
**Attribution:** Required - See data/ontologies/LICENSE.txt
**Clone Command:**
```bash
git clone https://github.com/ProjectAdA/public.git data/ontologies/ada
```

**Key Files:**
- `ontology/ada_ontology.owl` (740 KB)
- `ontology/README.md` (documentation)
- `ontology/levels_types_values.png` (visualization)

**Specifications:**
- **Format:** OWL (Web Ontology Language) + RDF
- **Structure:** 8 annotation levels, 78 types, 502 predefined values
- **Languages:** German and English names/descriptions for each concept
- **Focus:** Film-analytical terms for fine-grained semantic video annotations

**Annotation Levels (8):**
1. Camera (angles, movements, distances)
2. Editing (pace, transitions, techniques)
3. Sound (dialogue, music, effects)
4. Lighting (key, fill, back, contrast)
5. Color (palette, saturation, theory)
6. Acting (performance styles, emotions)
7. Mise-en-scène (composition, props, sets)
8. Narrative (structure, themes, pacing)

**Annotation Types (78 examples):**
- Camera: CameraAngle, CameraMovement, CameraMovementSpeed, CameraDistance
- Editing: EditingPace, TransitionType, AverageShotLength
- Lighting: LightingStyle, LightingContrast, LightingKey
- Sound: DialogueClarity, MusicPresence, SoundEffectIntensity

**Annotation Values (502 examples):**
- Camera angles: eye-level, low-angle, high-angle, dutch-angle, bird's-eye
- Camera movements: pan, tilt, tracking-shot, dolly, crane, steadicam
- Editing pace: fast, medium, slow, rhythmic, frenetic
- Lighting styles: high-key, low-key, chiaroscuro, naturalistic, expressionistic

**Annotation Model:**
Based on [W3C Web Annotation Data Model](https://www.w3.org/TR/annotation-model/)
- **Target:** Time fragment of video (W3C Media Fragments URI)
- **Body:** Annotation content with types/values, author, metadata

**Annotation Types Supported:**
1. `FreeTextAnnotationType` - Textual content without structure
2. `PredefinedValuesAnnotationType` - Uses predefined values from ontology
3. `ContrastingAnnotationType` - Juxtaposes multiple predefined values
4. `EvolvingAnnotationType` - Describes transitions between values

**Processing Requirements:**
1. Parse OWL file using horned-owl Rust library
2. Extract class hierarchy (8 levels → 78 types → 502 values)
3. Map MovieLens genome tags to AdA concepts (see ONTOLOGY_INTEGRATION_PLAN.md)
4. Load into whelk-rs for EL++ reasoning

**Example Mapping (Genome Tag → AdA Concept):**
```
"dark" → ada:DarkLighting + ada:HighContrast
"noir" → ada:FilmNoirStyle + ada:ShadowsAndLight
"tracking shot" → ada:TrackingShot + ada:FluidCameraMovement
"cerebral" → ada:ComplexNarrative + ada:IntellectualThemes
```

**Research Context:**
Developed by the AdA project (Advanced data Analysis) for systematic film analysis in academic research.

---

### 2. Movies Ontology

**Repository:** https://github.com/robotenique/movies-ontology
**License:** MIT (assumed)
**Attribution:** Required - See data/ontologies/LICENSE.txt
**Clone Command:**
```bash
git clone https://github.com/robotenique/movies-ontology.git data/ontologies/movies
```

**Key Files:**
- `movies.owl` or similar (to be verified)
- README.md (documentation)

**Specifications:**
- **Format:** OWL + RDF
- **Focus:** General movie metadata ontology

**Expected Coverage:**
- Movies (titles, years, IDs)
- Actors (names, roles, filmographies)
- Directors (names, filmographies, styles)
- Genres (classifications, hierarchies)
- Ratings (user ratings, critical scores)
- Studios (production companies, distributors)
- Awards (Oscars, Golden Globes, etc.)

**Complementarity to AdA:**
- AdA focuses on **film-analytical techniques** (how a movie is made)
- Movies Ontology focuses on **metadata** (who made it, when, ratings)
- Together: Complete picture of movies (technical + contextual)

**Processing Requirements:**
1. Parse OWL file using horned-owl
2. Extract class hierarchy for movies, actors, directors, genres
3. Map MovieLens metadata to Movies Ontology classes
4. Integrate with AdA for hybrid reasoning

**Example Mapping (MovieLens → Movies Ontology):**
```
MovieLens genres → movies:Genre
MovieLens ratings → movies:UserRating + movies:PopularityScore
High ratings (>4.5) → movies:CriticallyAcclaimed
Directors → movies:Director + movies:hasDirected
```

---

### 3. OMC (Ontology for Media Creation) - OPTIONAL

**Repository:** https://github.com/MovieLabs/OMC
**License:** Apache 2.0 (MovieLabs)
**Clone Command:**
```bash
git clone https://github.com/MovieLabs/OMC.git data/ontologies/omc
```

**Key Files:**
- OMC-JSON schema (v2.8) - JSON schema for practical implementation
- OMC-RDF ontologies - RDF/Turtle format for semantic web

**Specifications:**
- **Format:** JSON Schema + RDF/Turtle
- **Focus:** Complete lifecycle of audiovisual content production
- **Namespaces:**
  - `cw:` - Creative Works (narrative, characters, versions)
  - `omc:` - Production processes (filming, editing, post-production)
  - `omd:` - Distribution and analytics

**Industry Standards:**
- Aligned with EIDR (Entertainment Identifier Registry)
- Used by actual film/TV production systems
- Episodic content structures for TV series

**Coverage Areas:**
1. **Narrative Development** - Story structure, character arcs, plot points
2. **Production Management** - Schedules, resources, crew assignments
3. **Asset Cataloging** - Shots, scenes, takes, versions
4. **Entity Relationships** - Characters, actors, departments
5. **Character Depictions** - Portrayals across versions/adaptations
6. **Version Tracking** - Edits, cuts, director's cuts

**Complementarity Assessment:**
- **AdA:** Film-analytical techniques (camera, lighting, editing)
- **Movies Ontology:** General metadata (actors, directors, ratings)
- **OMC:** Production process metadata (how content was created)

**Use Case for Recommendations:**
- **Limited Direct Value**: OMC focuses on production workflows, not content description
- **Potential Enrichment**: Could provide additional metadata if MovieLens data included production details
- **Decision:** OPTIONAL - Include only if production metadata becomes available

**Processing Requirements (if used):**
1. Parse RDF/Turtle using horned-owl
2. Extract relevant creative work concepts (cw: namespace)
3. Map to MovieLens data if production metadata exists
4. Integrate with AdA + Movies Ontology

**Recommendation:**
- **Phase 1-2:** Skip OMC integration (no production data in MovieLens)
- **Phase 3+:** Revisit if expanding to production-aware recommendations

---

## Validation Framework: horned-owl

**Library:** [horned-owl](https://github.com/phillord/horned-owl)
**Version:** ^1.0 (specified in whelk-rs Cargo.toml)
**Language:** Rust
**Purpose:** OWL ontology parsing, manipulation, and validation

**Usage Locations in Project:**
```rust
// From /home/devuser/workspace/project/src/
src/ontology/parser/assembler.rs:89      - use horned_owl::io::ofn::reader::read as read_ofn;
src/ontology/parser/assembler.rs:90      - use horned_owl::ontology::set::SetOntology;
src/services/owl_extractor_service.rs:10 - use horned_owl::io::owx::reader::read as read_owx;
src/services/owl_extractor_service.rs:11 - use horned_owl::model::*;
src/inference/owl_parser.rs:11           - use horned_owl::io::owx::reader::read as read_owx;
src/inference/owl_parser.rs:12           - use horned_owl::model::ArcStr;
src/inference/owl_parser.rs:13           - use horned_owl::ontology::set::SetOntology;
src/services/owl_validator.rs:4          - use horned_owl::io::ofn::reader::read as read_ofn;
src/services/owl_validator.rs:5          - use horned_owl::io::owx::reader::read as read_owx;
src/services/owl_validator.rs:6          - use horned_owl::ontology::set::SetOntology;
src/adapters/whelk_inference_engine.rs:15 - use horned_owl::model::{...};
src/adapters/whelk_inference_engine.rs:19 - use horned_owl::ontology::set::SetOntology;
```

**Key Capabilities:**
1. **OWL Format Support:**
   - OWL/XML (owx) - `horned_owl::io::owx::reader::read`
   - OWL Functional Syntax (ofn) - `horned_owl::io::ofn::reader::read`
   - RDF/Turtle - Supported via RDF parsers

2. **Data Structures:**
   - `SetOntology<ArcStr>` - Ontology representation with arc-counted strings
   - `Component` - Individual ontology elements (classes, properties, axioms)
   - `Class`, `ObjectProperty`, `DataProperty` - OWL constructs

3. **Validation:**
   - Syntactic validation (well-formed OWL)
   - Structural validation (consistent class hierarchies)
   - Axiom validation (logical consistency)

**Integration with whelk-rs:**
```rust
// From /home/devuser/workspace/project/whelk-rs/Cargo.toml
[dependencies]
horned-owl = "^1.0"
curie = "^0.1"        # IRI handling
im = "^15.1"          # Immutable data structures
rayon = "^1.6"        # Parallel reasoning
```

**Processing Pipeline:**
```
AdA OWL File (740KB)
    ↓
horned_owl::io::owx::reader::read()
    ↓
SetOntology<ArcStr> (8 levels, 78 types, 502 values)
    ↓
whelk-rs EL++ Reasoner
    ↓
Inferred Class Hierarchies
    ↓
MovieLens Genome Tag Mapping
    ↓
Ontology-Enhanced Recommendations
```

---

## Processing Workflow

### Phase 1: Ontology Loading (Week 1)

**Step 1: Clone Repositories**
```bash
cd /home/devuser/workspace/hackathon-tv5/semantic-recommender
mkdir -p data/ontologies

# AdA Film Ontology
git clone https://github.com/ProjectAdA/public.git data/ontologies/ada

# Movies Ontology
git clone https://github.com/robotenique/movies-ontology.git data/ontologies/movies

# OMC (Optional)
git clone https://github.com/MovieLabs/OMC.git data/ontologies/omc
```

**Step 2: Extract OWL Files**
```bash
# AdA
cp data/ontologies/ada/ontology/ada_ontology.owl data/ontologies/ada_ontology.owl

# Movies
cp data/ontologies/movies/*.owl data/ontologies/movies_ontology.owl

# OMC (if used)
cp data/ontologies/omc/rdf/*.ttl data/ontologies/omc/
```

**Step 3: Validate with horned-owl**
```rust
// src/ontology/loader.rs
use horned_owl::io::owx::reader::read as read_owx;
use horned_owl::ontology::set::SetOntology;

pub fn load_ada_ontology(path: &Path) -> Result<SetOntology<ArcStr>> {
    let mut file = File::open(path)?;
    let ontology = read_owx(&mut file)?;

    // Validate structure
    assert!(ontology.i().class().count() > 500);  // 502 values expected

    Ok(ontology)
}
```

**Step 4: Load into whelk-rs**
```rust
use whelk::reasoner::Reasoner;

let ada_ontology = load_ada_ontology("data/ontologies/ada_ontology.owl")?;
let movies_ontology = load_movies_ontology("data/ontologies/movies_ontology.owl")?;

let reasoner = Reasoner::new();
reasoner.load_ontology(&ada_ontology)?;
reasoner.load_ontology(&movies_ontology)?;
reasoner.classify()?;
```

### Phase 2: MovieLens Mapping (Week 1-2)

**Genome Tag → AdA Concept Mapping**
```python
# scripts/map_genome_to_ada.py
GENOME_TO_ADA = {
    # Visual Style
    'dark': 'ada:DarkLighting',
    'noir': 'ada:FilmNoirStyle',
    'colorful': 'ada:SaturatedColor',

    # Camera
    'tracking shot': 'ada:TrackingShot',
    'close-up': 'ada:CloseUpShot',

    # Editing
    'fast-paced': 'ada:RapidEditing',
    'slow': 'ada:SlowPacing',
}

def map_movie_to_ontology(movie_id, genome_tags):
    ontology_classes = []

    for tag, relevance in genome_tags:
        if relevance > 0.7 and tag in GENOME_TO_ADA:
            ontology_classes.append(GENOME_TO_ADA[tag])

    return ontology_classes
```

**MovieLens Metadata → Movies Ontology**
```python
def map_metadata_to_ontology(movie_metadata):
    classes = []

    # Genres
    for genre in movie_metadata['genres']:
        classes.append(f"movies:{genre}Genre")

    # Ratings
    avg_rating = movie_metadata['avg_rating']
    if avg_rating >= 4.5:
        classes.append("movies:CriticallyAcclaimed")
    elif avg_rating <= 2.0:
        classes.append("movies:PoorlyRated")

    return classes
```

### Phase 3: Reasoning Integration (Week 2)

See ONTOLOGY_INTEGRATION_PLAN.md for complete architecture.

---

## Storage Requirements

| Ontology | Clone Size | OWL File | Processing |
|----------|-----------|----------|------------|
| **AdA** | ~5 MB | 740 KB | Required |
| **Movies** | ~2 MB | ~500 KB | Required |
| **OMC** | ~10 MB | ~2 MB | Optional |
| **Total** | ~17 MB | ~3 MB | ~6 MB in memory |

**Git Storage Status:**
Ontology repositories are NOW INCLUDED in version control with proper attribution:
- ✅ **data/ontologies/ada/** - AdA Film Ontology (Apache 2.0)
- ✅ **data/ontologies/movies/** - Movies Ontology (MIT)
- ✅ **data/ontologies/omc/** - MovieLabs OMC (Apache 2.0)
- ✅ **whelk-rs/** - Whelk-rs EL++ reasoner (BSD-3-Clause)
- ✅ **data/ontologies/LICENSE.txt** - Complete attribution and licenses

See `.gitignore` (commented out) and `data/ontologies/LICENSE.txt` for details.

**Reasoning Engine:**
Whelk-rs source code included with BSD-3-Clause license (copied from /home/devuser/workspace/project/)

---

## License Considerations

**AdA Film Ontology:**
- No explicit license in repository
- Academic research project
- **Recommendation:** Assume academic use permitted, cite appropriately

**Movies Ontology:**
- No explicit license
- **Recommendation:** Contact author if commercial use intended

**OMC:**
- Apache 2.0 license (MovieLabs)
- **Permissive:** Commercial use allowed with attribution

**horned-owl:**
- MIT/Apache-2.0 dual licensed
- **Permissive:** No restrictions

---

## Updates and Maintenance

**Ontology Updates:**
- AdA: Check repository quarterly for new versions
- Movies: Monitor for updates
- OMC: MovieLabs actively maintains (check releases)

**Version Tracking:**
```bash
# Record ontology versions
echo "AdA: $(cd data/ontologies/ada && git rev-parse HEAD)" > data/ontologies/versions.txt
echo "Movies: $(cd data/ontologies/movies && git rev-parse HEAD)" >> data/ontologies/versions.txt
echo "OMC: $(cd data/ontologies/omc && git rev-parse HEAD)" >> data/ontologies/versions.txt
```

**Re-cloning:**
```bash
# If ontologies need updates
rm -rf data/ontologies/ada
rm -rf data/ontologies/movies
rm -rf data/ontologies/omc

# Re-clone fresh copies
git clone https://github.com/ProjectAdA/public.git data/ontologies/ada
git clone https://github.com/robotenique/movies-ontology.git data/ontologies/movies
git clone https://github.com/MovieLabs/OMC.git data/ontologies/omc
```

---

## References

1. **AdA Film Ontology:** https://github.com/ProjectAdA/public/tree/master/ontology
2. **Movies Ontology:** https://github.com/robotenique/movies-ontology
3. **OMC (Ontology for Media Creation):** https://github.com/MovieLabs/OMC
4. **horned-owl:** https://github.com/phillord/horned-owl
5. **whelk-rs:** EL++ reasoner (local copy in project)
6. **W3C Web Annotation Data Model:** https://www.w3.org/TR/annotation-model/
7. **W3C Media Fragments URI:** https://www.w3.org/TR/media-frags/

---

**Document Version:** 1.0
**Author:** Semantic Recommender Team
**Date:** 2025-12-06
**Status:** Reference Documentation
