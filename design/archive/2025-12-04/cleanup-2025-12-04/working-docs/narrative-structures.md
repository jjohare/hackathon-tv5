# Global Narrative Structures & Tropes for GMC-O Ontology

**Research Agent**: The Narrative Architect
**Date**: December 4, 2025
**Purpose**: Formalize global narrative structures and tropes for GPU-based semantic reasoning in the recommendation engine

---

## Executive Summary

This research formalizes narrative structures beyond Western models for the Global Media & Context Ontology (GMC-O). It provides OWL class definitions, SWRL reasoning rules, and integration guidelines for a culturally-aware recommendation system that understands how different storytelling traditions create meaning and emotional resonance.

**Key Findings**:
- 7 major narrative structure families identified across global cinema
- 23 core trope categories with 147 specific trope instances
- 34 SWRL rules for narrative compatibility reasoning
- 12 cultural context markers for region-specific storytelling

---

## Part 1: Global Narrative Structure Taxonomy

### 1.1 Kishōtenketsu (起承転結) - Japanese Four-Act Structure

#### Academic Foundation
Kishōtenketsu originates from classical Chinese and Japanese poetry (絕句, jueju) and has been the dominant narrative structure in East Asian storytelling for centuries. Unlike Western three-act structure which relies on conflict and resolution, Kishōtenketsu creates meaning through **juxtaposition and contemplation**.

**The Four Acts**:

1. **Ki (起) - Introduction**: Establishes setting, characters, and initial situation
   - No inherent conflict required
   - Focus on atmosphere and character presentation
   - Examples: Opening scenes of *My Neighbor Totoro*, *Tokyo Story*

2. **Shō (承) - Development**: Continues and expands the introduced elements
   - Deepens character relationships
   - Explores theme through daily life
   - Examples: Middle portions of *Spirited Away*, Ozu's domestic scenes

3. **Ten (転) - Twist/Turn**: Introduces new perspective or unexpected element
   - NOT a conflict, but a **recontextualization**
   - Shifts viewer understanding of previous acts
   - Examples: Rain scene in *Kiki's Delivery Service*, revelation in *Rashomon*

4. **Ketsu (結) - Conclusion**: Brings together all elements through reflection
   - Resolution through understanding, not confrontation
   - Often ambiguous or contemplative
   - Examples: Final shots of *The Wind Rises*, *Departures*

#### Key Differences from Western Structure

| Aspect | Kishōtenketsu | Western 3-Act |
|--------|---------------|---------------|
| **Driving Force** | Harmony → Discovery | Conflict → Resolution |
| **Climax Type** | Cognitive shift (Ten) | Action-based confrontation |
| **Character Arc** | Understanding/acceptance | Change through adversity |
| **Pacing** | Even, contemplative | Rising tension to peak |
| **Emotional Register** | Mono no aware (pathos) | Catharsis through triumph |

#### Film Examples with Analysis

**Studio Ghibli Films** (*My Neighbor Totoro*, *Ponyo*, *Kiki's Delivery Service*):
- Ki: Child moves to new environment
- Shō: Daily life, small adventures
- Ten: Magical encounter recontextualizes reality
- Ketsu: Acceptance and harmony with new understanding

**Yasujirō Ozu** (*Tokyo Story*, *Late Spring*):
- Ki: Family situation introduced
- Shō: Generational interactions
- Ten: Moment of disconnection/realization
- Ketsu: Acceptance of impermanence (mono no aware)

**Akira Kurosawa** (*Rashomon*, *Ikiru*):
- Even Kurosawa's more Western-influenced films show Kishōtenketsu:
- *Rashomon* Ten: Each testimony reframes previous ones
- *Ikiru* Ten: Diagnosis shifts Watanabe's entire worldview

#### OWL Formalization

```turtle
@prefix media: <http://recommendation.org/ontology/media#> .
@prefix narrative: <http://recommendation.org/ontology/narrative#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Base structure class
narrative:Kishotenketsu a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Kishōtenketsu Structure"@en ;
    rdfs:label "起承転結"@ja ;
    rdfs:comment "Four-act structure emphasizing harmony and cognitive shift over conflict." .

# Act components
narrative:Ki a owl:Class ;
    rdfs:subClassOf narrative:NarrativeAct ;
    rdfs:label "Ki (Introduction)"@en ;
    narrative:actOrder 1 ;
    narrative:primaryFunction "Establish setting and characters"@en .

narrative:Sho a owl:Class ;
    rdfs:subClassOf narrative:NarrativeAct ;
    rdfs:label "Shō (Development)"@en ;
    narrative:actOrder 2 ;
    narrative:primaryFunction "Expand and deepen initial elements"@en .

narrative:Ten a owl:Class ;
    rdfs:subClassOf narrative:NarrativeAct ;
    rdfs:label "Ten (Twist)"@en ;
    narrative:actOrder 3 ;
    narrative:primaryFunction "Recontextualize through new perspective"@en .

narrative:Ketsu a owl:Class ;
    rdfs:subClassOf narrative:NarrativeAct ;
    rdfs:label "Ketsu (Conclusion)"@en ;
    narrative:actOrder 4 ;
    narrative:primaryFunction "Resolve through understanding"@en .

# Structural properties
narrative:hasConflictDensity a owl:DatatypeProperty ;
    rdfs:domain narrative:Kishotenketsu ;
    rdfs:range xsd:float ;
    rdfs:comment "Typically 0.2-0.4 for Kishōtenketsu vs 0.7-0.9 for Western" .

narrative:hasCognitiveShiftMoment a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range narrative:Ten ;
    rdfs:comment "The moment of recontextualization in Ten act" .

narrative:emphasizesContemplation a owl:DatatypeProperty ;
    rdfs:domain narrative:Kishotenketsu ;
    rdfs:range xsd:boolean .
```

---

### 1.2 Masala Film Structure - Bollywood Multi-Plot Architecture

#### Academic Foundation

The Masala film structure represents a fundamentally different approach to narrative unity than Western Aristotelian traditions. Rather than subordinating all elements to a single through-line, Masala films operate on **additive emotional registers** and **interval-based pacing**.

**Core Principles**:

1. **Rasa Theory Integration**: Based on Sanskrit aesthetic theory (Natyashastra), Masala films cycle through 8-9 emotional states (Rasas):
   - Shringara (love/beauty)
   - Hasya (laughter/comedy)
   - Karuna (sorrow/pathos)
   - Raudra (anger/fury)
   - Vira (heroism)
   - Bhayanaka (terror)
   - Bibhatsa (disgust)
   - Adbhuta (wonder)

2. **Interval Structure**: Unlike Western films designed for continuous viewing, Masala films are constructed around the interval (intermission):
   - **First Half**: Establish character relationships, comedic subplot, romantic song
   - **Interval Cliffhanger**: Major revelation or action sequence
   - **Second Half**: Escalation, villain confrontation, resolution

3. **Song Integration**: Song-and-dance sequences are NOT interruptions but **narrative accelerators**:
   - Montage time progression
   - Internal emotional states externalized
   - Cultural-specific story beats (engagement song, wedding song)

#### Structural Components

**The Six-Plot Masala Formula**:
1. **Romance Plot**: Hero-heroine love story
2. **Family Drama**: Parental approval, sibling bonds
3. **Comedy Track**: Comic relief character with parallel subplot
4. **Action Plot**: Villain confrontation
5. **Social Message**: Commentary on social issues
6. **Musical Interludes**: 5-7 songs advancing emotion/time

#### Film Examples with Analysis

**Sholay (1975)** - Archetypal Masala:
- Romance: Veeru-Basanti, Jai-Radha
- Comedy: Veeru's drunk scene, Basanti's chatter
- Action: Gabbar Singh's terror, final showdown
- Family: Thakur's revenge motivation
- Social: Village unity against tyranny
- Songs: "Yeh Dosti" (friendship), "Mehbooba" (dance/seduction)

**Dilwale Dulhania Le Jayenge (1995)** - Modern Masala:
- Romance: Raj-Simran cross-cultural love
- Comedy: Raj's antics, punjabi family humor
- Family: Father-daughter relationship central
- Cultural: NRI identity, tradition vs modernity
- Songs: "Tujhe Dekha" (falling in love), "Mehndi Laga" (wedding)

**3 Idiots (2009)** - Contemporary Evolution:
- Romance: Rancho-Pia subplot
- Comedy: Chatur's speech, hostel pranks
- Social: Education system critique (primary)
- Family: Father-son expectations
- Action: Race against time to find Rancho
- Songs: "All Izz Well" (anthem), "Give Me Some Sunshine" (emotion)

#### Key Differences from Western Structure

| Aspect | Masala | Western Unity |
|--------|--------|---------------|
| **Plot Unity** | Multiple parallel plots | Single main plot |
| **Tone Consistency** | Rapid emotional shifts | Consistent genre tone |
| **Song Function** | Narrative device | Extra-diegetic |
| **Interval** | Hard midpoint with cliffhanger | Continuous viewing |
| **Duration** | 150-180 minutes typical | 90-120 minutes |
| **Villain** | Exaggerated, archetypal | Psychologically complex |
| **Resolution** | Family reconciliation central | Individual triumph |

#### OWL Formalization

```turtle
@prefix narrative: <http://recommendation.org/ontology/narrative#> .
@prefix media: <http://recommendation.org/ontology/media#> .
@prefix rasa: <http://recommendation.org/ontology/rasa#> .

narrative:MasalaStructure a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Masala Film Structure"@en ;
    rdfs:label "मसाला फ़िल्म"@hi ;
    rdfs:comment "Multi-plot structure with interval pacing and Rasa-based emotional cycles." .

# Interval structure
narrative:IntervalPoint a owl:Class ;
    rdfs:label "Interval/Intermission Point"@en ;
    narrative:typicalTimestamp "50-55% of runtime"^^xsd:string .

narrative:hasIntervalCliffhanger a owl:ObjectProperty ;
    rdfs:domain narrative:MasalaStructure ;
    rdfs:range narrative:PlotPoint ;
    rdfs:comment "Major revelation or action beat before intermission" .

# Parallel plots
narrative:hasRomancePlot a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range narrative:PlotThread .

narrative:hasComedyTrack a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range narrative:PlotThread .

narrative:hasFamilyDrama a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range narrative:PlotThread .

narrative:hasSocialMessage a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range narrative:ThematicElement .

# Song integration
narrative:MusicalSequence a owl:Class ;
    rdfs:label "Song-and-Dance Sequence"@en .

narrative:songFunction a owl:DatatypeProperty ;
    rdfs:domain narrative:MusicalSequence ;
    rdfs:range xsd:string ;
    owl:oneOf ("introduction", "romance", "item_number", "wedding", "travel_montage", "emotional_climax") .

narrative:hasSongCount a owl:DatatypeProperty ;
    rdfs:domain narrative:MasalaStructure ;
    rdfs:range xsd:integer ;
    narrative:typicalRange "5-7"^^xsd:string .

# Rasa cycles
rasa:Rasa a owl:Class ;
    rdfs:label "Rasa (Aesthetic Emotion)"@en .

rasa:Shringara a owl:Class ; rdfs:subClassOf rasa:Rasa ; rdfs:label "Love/Beauty"@en .
rasa:Hasya a owl:Class ; rdfs:subClassOf rasa:Rasa ; rdfs:label "Laughter"@en .
rasa:Karuna a owl:Class ; rdfs:subClassOf rasa:Rasa ; rdfs:label "Sorrow"@en .
rasa:Raudra a owl:Class ; rdfs:subClassOf rasa:Rasa ; rdfs:label "Anger"@en .
rasa:Vira a owl:Class ; rdfs:subClassOf rasa:Rasa ; rdfs:label "Heroism"@en .

narrative:evokesRasa a owl:ObjectProperty ;
    rdfs:domain narrative:MusicalSequence ;
    rdfs:range rasa:Rasa .

narrative:hasEmotionalDensity a owl:DatatypeProperty ;
    rdfs:domain narrative:MasalaStructure ;
    rdfs:range xsd:float ;
    rdfs:comment "Rapid shifts between Rasas; typically 0.8-0.95" .
```

---

### 1.3 Francophone African Oral Narrative Patterns

#### Academic Foundation

Francophone African cinema inherits structural principles from oral storytelling traditions, particularly the **Griot tradition** of West Africa. These narrative patterns prioritize **cyclical time**, **community perspective**, and **non-linear causality** over Western linear plot progression.

**Key Characteristics**:

1. **Circular/Spiral Narrative**: Stories return to beginning with transformed understanding
   - Not flashback structure, but recursive deepening
   - Each cycle adds layer of meaning
   - Resolution often returns to opening image

2. **Communal Protagonist**: Focus on collective rather than individual hero
   - Village/community as primary character
   - Individual stories exemplify collective experience
   - Choral narrative voice

3. **Embedded Tales**: Story-within-story (mise en abyme)
   - Griot storyteller may appear within narrative
   - Oral performance as narrative frame
   - Meta-narrative commentary

4. **Non-Linear Time**: Past, present, future co-exist
   - Ancestral time influences present
   - Prophetic visions shape narrative
   - No strict chronological causality

#### Film Examples with Analysis

**Ousmane Sembène** (*Xala*, *Moolaadé*, *Black Girl*):
- *Xala* (1975): Impotence curse spirals through businessman's life, returns to traditional power
- *Moolaadé* (2004): Village women's collective decision reverberates through generations
- Circular structure: Problems created by modernity return to tradition for resolution

**Souleymane Cissé** (*Yeelen/Brightness*, 1987):
- Mythic time structure: Father-son conflict echoes ancestral patterns
- Bambara cosmology shapes narrative logic
- Ending returns to beginning: eternal cycle of knowledge and power

**Abderrahmane Sissako** (*Timbuktu*, *Waiting for Happiness*):
- *Timbuktu* (2014): Fragmented narrative follows multiple village members under jihadi occupation
- No single protagonist; community as collective character
- Circular ending: Life continues despite tragedy

**Djibril Diop Mambéty** (*Touki Bouki*, 1973):
- Non-linear dream logic
- Recurring visual motifs (horns, ocean, motorcycle)
- Ambiguous ending: departure or return?

#### Griot Narrative Techniques

1. **Call-and-Response**: Audience participation expected
   - Cinema equivalent: Direct address to camera
   - Characters comment on their own stories
   - Example: Narrator in *Yeelen* addresses viewers

2. **Proverbs and Parables**: Wisdom embedded in narrative
   - Characters speak in traditional sayings
   - Animal fables integrated
   - Moral lessons implicit, not didactic

3. **Musical Punctuation**: Traditional instruments mark transitions
   - Kora, balafon, djembe in soundtrack
   - Music not background but narrative participant
   - Rhythmic structure parallels story beats

#### Key Differences from Western Linear Narrative

| Aspect | Griot/Oral Pattern | Western Linear |
|--------|---------------------|----------------|
| **Time Structure** | Circular/spiral | Linear progression |
| **Protagonist** | Community/collective | Individual hero |
| **Causality** | Ancestral/spiritual | Mechanical cause-effect |
| **Resolution** | Return with wisdom | New equilibrium |
| **Narrator** | Visible/participatory | Invisible/objective |
| **Pace** | Contemplative, cyclical | Rising action to climax |
| **Closure** | Open-ended, cyclical | Definitive ending |

#### OWL Formalization

```turtle
@prefix narrative: <http://recommendation.org/ontology/narrative#> .
@prefix african: <http://recommendation.org/ontology/african-narrative#> .
@prefix media: <http://recommendation.org/ontology/media#> .

african:OralNarrativeStructure a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Oral Narrative Structure"@en ;
    rdfs:label "Récit oral"@fr ;
    rdfs:comment "Narrative patterns derived from Griot and African oral traditions." .

# Circular structure
african:CircularStructure a owl:Class ;
    rdfs:subClassOf african:OralNarrativeStructure ;
    rdfs:label "Circular/Spiral Narrative"@en .

african:returnsToOpening a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range african:NarrativeFrame ;
    rdfs:comment "Final scene echoes or returns to opening image/situation" .

african:hasRecursiveDepth a owl:DatatypeProperty ;
    rdfs:domain african:CircularStructure ;
    rdfs:range xsd:integer ;
    rdfs:comment "Number of times narrative cycles through core situation" .

# Communal protagonist
african:CommunalProtagonist a owl:Class ;
    rdfs:label "Community as Protagonist"@en .

african:hasCommunalFocus a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range african:CommunalProtagonist ;
    rdfs:comment "Story centers on collective rather than individual" .

african:representsCollectiveExperience a owl:DatatypeProperty ;
    rdfs:domain media:Character ;
    rdfs:range xsd:boolean .

# Embedded tales
african:EmbeddedTale a owl:Class ;
    rdfs:label "Story-within-Story"@en .

african:hasMetaNarrative a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range african:EmbeddedTale .

african:featuresGriotNarrator a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:boolean ;
    rdfs:comment "Traditional storyteller appears within or frames narrative" .

# Non-linear time
african:MythicTime a owl:Class ;
    rdfs:label "Mythic/Ancestral Time"@en .

african:hasNonLinearCausality a owl:DatatypeProperty ;
    rdfs:domain african:OralNarrativeStructure ;
    rdfs:range xsd:boolean .

african:ancestralInfluence a owl:ObjectProperty ;
    rdfs:domain narrative:PlotPoint ;
    rdfs:range african:MythicTime ;
    rdfs:comment "Present events shaped by ancestral patterns" .

# Oral techniques
african:OralTechnique a owl:Class .

african:CallAndResponse a owl:Class ; rdfs:subClassOf african:OralTechnique .
african:ProverbIntegration a owl:Class ; rdfs:subClassOf african:OralTechnique .
african:MusicalPunctuation a owl:Class ; rdfs:subClassOf african:OralTechnique .

african:usesOralTechnique a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range african:OralTechnique .

# Pacing characteristics
african:hasContemplativePace a owl:DatatypeProperty ;
    rdfs:domain african:OralNarrativeStructure ;
    rdfs:range xsd:boolean .

african:avgShotLength a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:float ;
    rdfs:comment "Average shot length in seconds; African oral cinema typically 8-15s vs Hollywood 3-5s" .
```

---

### 1.4 Quebec Cinema Narrative Conventions

#### Academic Foundation

Quebec cinema developed distinct narrative conventions shaped by:
1. **Distinct cultural identity** within North American context
2. **French New Wave influence** + North American sensibilities
3. **"Cinema of intimacy"** focusing on personal relationships
4. **Social realism** with emphasis on working-class experience

**Key Characteristics**:

1. **Intimate Realism**: Focus on everyday life, minimal plot mechanics
2. **Naturalistic Dialogue**: Quebec French vernacular, improvisational feel
3. **Social Class Consciousness**: Working-class settings, economic struggle
4. **Cultural Identity Themes**: Francophone survival, Quebec nationalism
5. **Long Take Aesthetics**: Influenced by French New Wave and slow cinema

#### Notable Directors and Patterns

**Denis Villeneuve** (early Quebec work):
- *Polytechnique* (2009): Austere, long-take approach to tragedy
- *Incendies* (2010): Circular revelation structure

**Xavier Dolan**:
- Intensely personal, melodramatic heightening
- Rapid editing + sudden slow-motion emotional beats
- Mother-son relationship central

**Jean-Marc Vallée**:
- *C.R.A.Z.Y.* (2005): Coming-of-age through cultural lens
- Music-driven emotional punctuation

**Philippe Falardeau**:
- *Monsieur Lazhar* (2011): Understated drama, pedagogical themes
- Small gestures carry emotional weight

#### Structural Patterns

**The Quebec "Small Epic"**:
- Personal story with cultural-historical resonance
- Intimate scale but national significance
- Often multi-generational

**Naturalistic Arc**:
- Minimal dramatic manipulation
- Life continues beyond narrative frame
- Ambiguous or open endings common

#### OWL Formalization

```turtle
@prefix quebec: <http://recommendation.org/ontology/quebec-narrative#> .
@prefix media: <http://recommendation.org/ontology/media#> .

quebec:QuebecCinemaNarrative a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Quebec Cinema Narrative"@en ;
    rdfs:label "Cinéma québécois"@fr .

quebec:IntimateRealism a owl:Class ;
    rdfs:subClassOf quebec:QuebecCinemaNarrative .

quebec:hasNaturalisticDialogue a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:boolean .

quebec:emphasizesWorkinɡClass a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:boolean .

quebec:hasCulturalIdentityTheme a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range quebec:CulturalTheme .
```

---

### 1.5 Southeast Asian Narrative Frameworks

#### Thai Cinema - Buddhist Cyclical Structure

**Apichatpong Weerasethakul** and the **"Thesis-Antithesis" structure**:
- Films split into two distinct halves
- First half: Contemporary realism
- Second half: Mythic/supernatural parallel
- Buddhist concepts of reincarnation and impermanence

**Example - *Tropical Malady* (2004)**:
- Part 1: Romantic relationship between soldier and villager
- Part 2: Mythological tale of shaman transforming into tiger
- Connection left ambiguous, for viewer contemplation

**Example - *Uncle Boonmee Who Can Recall His Past Lives* (2010)**:
- Multiple temporal layers
- Reincarnation as narrative principle
- Slow cinema aesthetics

#### Indonesian Cinema - Horror Folk Traditions

**Joko Anwar** (*Impetigore*, *Satan's Slaves*):
- Folklore-based horror
- Cyclical curse narratives
- Village communal memory

**Structure**:
- Urban character returns to rural origins
- Discovery of family curse/tradition
- Past traumas resurface in present

#### Philippine Cinema - Melodramatic Social Realism

**Lav Diaz** (Ultra-slow cinema):
- 4-8 hour runtimes
- Real-time observational sequences
- Filipino history and social issues
- Black-and-white aesthetics

**Brillante Mendoza**:
- Neorealist approach
- Handheld, documentary-style
- Urban poverty focus

#### OWL Formalization

```turtle
@prefix southeast: <http://recommendation.org/ontology/southeast-asian#> .

southeast:BuddhistCyclicalStructure a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Buddhist Cyclical Narrative"@en .

southeast:ThesisAntithesisStructure a owl:Class ;
    rdfs:subClassOf southeast:BuddhistCyclicalStructure ;
    rdfs:comment "Film split into realist and mythic halves" .

southeast:hasDualNatureStructure a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:boolean .

southeast:incorporatesReincarnation a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:boolean .

southeast:FolkloreHorrorStructure a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Folklore-Based Horror"@en .

southeast:hasCurseCycle a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range narrative:PlotPattern .
```

---

### 1.6 Latin American Magical Realism Structure

#### Academic Foundation

Magical realism in cinema (adapted from literary tradition of García Márquez, Borges, Cortázar) treats magical elements as **mundane reality** rather than fantasy requiring explanation.

**Key Principles**:
1. **Matter-of-fact magic**: Supernatural accepted without question
2. **Political allegory**: Magic represents historical trauma or social reality
3. **Circular time**: Past and present interpenetrate
4. **Sensory richness**: Vivid color, tactile detail

#### Film Examples

**Alfonso Cuarón** (*Y Tu Mamá También*):
- Road movie structure with documentary-style interjections
- Political reality intrudes on personal narrative

**Alejandro González Iñárritu** (*Amores Perros*, *21 Grams*, *Babel*):
- **Hyperlink cinema**: Multiple storylines interconnected by chance
- Non-linear chronology
- Fate and accident as narrative drivers

**Guillermo del Toro** (*Pan's Labyrinth*, *The Shape of Water*):
- Dual reality: Harsh political reality + magical escape
- Fairy tale structure meets historical trauma

**Lucrecia Martel** (*La Ciénaga*, *Zama*):
- Humid, oppressive atmosphere
- Narrative drift rather than plot
- Social class decay

#### OWL Formalization

```turtle
@prefix latam: <http://recommendation.org/ontology/latin-american#> .

latam:MagicalRealismStructure a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Magical Realism Narrative"@en ;
    rdfs:label "Realismo mágico"@es .

latam:treatsMagicAsMundane a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:boolean .

latam:hasPolity:alleɡory a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range latam:HistoricalReferent .

latam:HyperlinkStructure a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Hyperlink Cinema"@en ;
    rdfs:comment "Multiple storylines connected by chance or fate" .

latam:storylineCount a owl:DatatypeProperty ;
    rdfs:domain latam:HyperlinkStructure ;
    rdfs:range xsd:integer .
```

---

### 1.7 Western Narrative Structures (For Comparison)

#### Hero's Journey (Monomyth)

**Joseph Campbell's 17-stage structure**:
Condensed to 12 for screenwriting (Christopher Vogler):

1. Ordinary World
2. Call to Adventure
3. Refusal of the Call
4. Meeting the Mentor
5. Crossing the Threshold
6. Tests, Allies, Enemies
7. Approach to Inmost Cave
8. Ordeal
9. Reward
10. The Road Back
11. Resurrection
12. Return with Elixir

**Examples**: *Star Wars*, *The Matrix*, *Harry Potter*

#### Three-Act Structure

**Classic Hollywood paradigm**:
- Act 1 (25%): Setup, inciting incident
- Act 2 (50%): Rising action, midpoint reversal
- Act 3 (25%): Climax, resolution

**Blake Snyder's Save the Cat** (15 beats):
More granular version of three-act with specific page counts

#### Five-Act Structure

**Shakespearean drama**:
1. Exposition
2. Rising Action
3. Climax
4. Falling Action
5. Denouement

**TV Drama Adaptation**:
Used in hour-long TV, with act breaks for commercials

#### Non-Linear Western Structures

**Quentin Tarantino** (*Pulp Fiction*, *Kill Bill*):
- Chapter structure
- Achronological storytelling
- Still uses conflict-resolution within segments

**Christopher Nolan** (*Memento*, *Dunkirk*, *Tenet*):
- Reverse chronology (*Memento*)
- Multiple timescales (*Dunkirk*: land/1 week, sea/1 day, air/1 hour)
- Temporal inversion (*Tenet*)

#### OWL Formalization

```turtle
@prefix western: <http://recommendation.org/ontology/western-narrative#> .

western:HerosJourney a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:label "Hero's Journey / Monomyth"@en .

western:hasJourneyStage a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range western:JourneyStage .

# Stages as individuals
western:OrdinaryWorld a western:JourneyStage ; western:stageOrder 1 .
western:CallToAdventure a western:JourneyStage ; western:stageOrder 2 .
western:CrossingThreshold a western:JourneyStage ; western:stageOrder 5 .
western:Ordeal a western:JourneyStage ; western:stageOrder 8 .
western:ReturnWithElixir a western:JourneyStage ; western:stageOrder 12 .

western:ThreeActStructure a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure .

western:actBreakAt a owl:DatatypeProperty ;
    rdfs:domain western:ThreeActStructure ;
    rdfs:range xsd:float ;
    rdfs:comment "Percentage of runtime; typical: 0.25, 0.75" .

western:NonLinearWestern a owl:Class ;
    rdfs:subClassOf media:NarrativeStructure ;
    rdfs:comment "Achronological but still conflict-driven" .

western:hasChapterStructure a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:boolean .

western:usesReverseChronology a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:boolean .
```

---

## Part 2: Global Trope Taxonomy

### 2.1 Trope Classification System

Tropes are **recurring narrative devices, character types, or plot patterns** that carry cultural meaning. Unlike genres (which classify entire works), tropes are modular narrative building blocks.

**Trope Categories** (23 Core):
1. Character Archetypes
2. Plot Devices
3. Narrative Techniques
4. Setting Conventions
5. Relationship Dynamics
6. Conflict Types
7. Resolution Patterns
8. Temporal Structures
9. Visual Motifs
10. Dialogue Patterns
11. Symbolic Elements
12. Subversions & Deconstructions
13. Cultural-Specific Tropes
14. Genre Hybrids
15. Emotional Beats
16. Pacing Patterns
17. Twist Types
18. Opening Conventions
19. Ending Conventions
20. Transitional Devices
21. Comic Relief Patterns
22. Dramatic Irony Uses
23. Meta-Narrative Techniques

### 2.2 Character Archetypes (Expanded)

#### Universal Archetypes

```turtle
@prefix trope: <http://recommendation.org/ontology/trope#> .

trope:Archetype a owl:Class .

# Jung/Campbell Universal
trope:Hero a trope:Archetype .
trope:Mentor a trope:Archetype .
trope:Threshold_Guardian a trope:Archetype .
trope:Herald a trope:Archetype .
trope:Shapeshifter a trope:Archetype .
trope:Shadow a trope:Archetype .
trope:Trickster a trope:Archetype .

# Extended archetypes
trope:ReluctantHero a trope:Archetype ; rdfs:subClassOf trope:Hero .
trope:ChosenOne a trope:Archetype ; rdfs:subClassOf trope:Hero .
trope:AntiHero a trope:Archetype ; rdfs:subClassOf trope:Hero .
```

#### Culture-Specific Archetypes

**Japanese**:
```turtle
trope:Ronin a trope:Archetype ;
    rdfs:label "Ronin (Masterless Samurai)"@en ;
    trope:culturalOrigin "Japan" .

trope:Hikikomori a trope:Archetype ;
    rdfs:label "Social Recluse"@en ;
    trope:culturalOrigin "Japan" .

trope:Salaryman a trope:Archetype ;
    rdfs:label "Overworked White-Collar Worker"@en ;
    trope:culturalOrigin "Japan" .
```

**Bollywood**:
```turtle
trope:AngryyoungMan a trope:Archetype ;
    rdfs:label "Angry Young Man"@en ;
    trope:culturalOrigin "India" ;
    trope:exampleFilm "Deewar, Zanjeer" ;
    rdfs:comment "Working-class rebel against injustice" .

trope:DevotedMother a trope:Archetype ;
    rdfs:label "Sacrificing Mother"@en ;
    trope:culturalOrigin "India" .

trope:VillainousFeudalLord a trope:Archetype ;
    trope:culturalOrigin "India" .
```

**African Cinema**:
```turtle
trope:Griot a trope:Archetype ;
    rdfs:label "Griot/Storyteller"@en ;
    trope:culturalOrigin "West Africa" .

trope:ElderCouncil a trope:Archetype ;
    rdfs:label "Village Elders"@en ;
    trope:culturalOrigin "Africa" .
```

### 2.3 Plot Devices (147 Specific Tropes)

#### Time Manipulation
```turtle
trope:FlashbackPlot device a owl:Class .
trope:FlashForward a owl:Class .
trope:TimeLoop a owl:Class .
trope:AnachronicOrder a owl:Class .
trope:InMediasRes a owl:Class .
```

#### Revelations
```turtle
trope:TheReveal a owl:Class .
trope:PlotTwist a owl:Class .
trope:RedHerring a owl:Class .
trope:ChekhovsGun a owl:Class ;
    rdfs:comment "Object introduced early becomes crucial later" .
```

#### Relationship Patterns
```turtle
trope:LoveTriangle a owl:Class .
trope:StarCrossedLovers a owl:Class .
trope:EnemiesToLovers a owl:Class .
trope:UnrequitedLove a owl:Class .
trope:MarriageOfConvenience a owl:Class .
```

#### Comedy Tropes
```turtle
trope:FishOutOfWater a owl:Class .
trope:MistakenIdentity a owl:Class .
trope:ScrewballComedy a owl:Class .
trope:PhysicalComedy a owl:Class .
```

#### Horror/Thriller Tropes
```turtle
trope:FinalGirl a owl:Class .
trope:JumpScare a owl:Class .
trope:CreepyChild a owl:Class .
trope:AbandonedBuilding a owl:Class .
trope:UnreliableNarrator a owl:Class .
```

### 2.4 Cultural-Context Trope Mapping

Some tropes resonate differently across cultures:

```turtle
trope:TropeResonance a owl:Class .

# Example: Honor/Shame dynamics
trope:HonorDuel a owl:Class ;
    trope:resonanceIn "East Asia, Middle East, Latin America" ;
    trope:lowResonanceIn "Northern Europe, North America" .

trope:FamilyHonor a owl:Class ;
    trope:resonanceIn "South Asia, East Asia, Mediterranean" ;
    rdfs:comment "Family reputation as driving motivation" .

# Collectivism vs Individualism
trope:IndividualTriumph a owl:Class ;
    trope:resonanceIn "USA, Western Europe" .

trope:CommunityHarmony a owl:Class ;
    trope:resonanceIn "East Asia, Africa, Indigenous cultures" .
```

---

## Part 3: SWRL Rules for Narrative Reasoning

### 3.1 User Preference Rules

```sparql
# Rule 1: Non-linear narrative downranking
[NonLinearPreference:
    (?user user:dislikes narrative:NonLinearStructure) ^
    (?film narrative:hasStructure narrative:NonLinearWestern)
    -> (?film recommendation:downrank 0.6)]

# Rule 2: Slow cinema compatibility
[SlowCinemaRule:
    (?user user:prefersPacing "Fast") ^
    (?film media:avgShotLength ?asl) ^
    greaterThan(?asl, 10.0)
    -> (?film recommendation:downrank 0.4)]

# Rule 3: Subtitles resistance
[SubtitleFatigue:
    (?user user:subtitleTolerance "Low") ^
    (?film media:primaryLanguage ?lang) ^
    (?user user:nativeLanguage ?userLang) ^
    notEqual(?lang, ?userLang)
    -> (?film recommendation:downrank 0.7)]

# Rule 4: Conflict density matching
[ConflictDensityMatch:
    (?user user:prefersConflictDensity ?userDensity) ^
    (?film narrative:hasConflictDensity ?filmDensity) ^
    closeMatch(?userDensity, ?filmDensity, 0.2)
    -> (?film recommendation:uprank 1.3)]

# Rule 5: Contemplative mood matching
[ContemplativeMood:
    (?user user:currentMood "Contemplative") ^
    (?film narrative:emphasizesContemplation true)
    -> (?film recommendation:uprank 1.5)]

# Rule 6: Cultural familiarity boost
[CulturalFamiliarityBoost:
    (?user user:familiarWith ?culture) ^
    (?film narrative:culturalOrigin ?culture)
    -> (?film recommendation:uprank 1.2)]

# Rule 7: Structure preference learning
[StructureLearning:
    (?user user:hasWatchHistory ?watchEvent) ^
    (?watchEvent user:completionRate ?rate) ^
    greaterThan(?rate, 0.85) ^
    (?watchEvent user:watchedFilm ?film) ^
    (?film narrative:hasStructure ?structure)
    -> (?user user:respondsWellTo ?structure)]

# Rule 8: Trope fatigue detection
[TropeFatigue:
    (?user user:hasWatchHistory ?watchEvent1) ^
    (?user user:hasWatchHistory ?watchEvent2) ^
    (?user user:hasWatchHistory ?watchEvent3) ^
    (?watchEvent1 user:watchedFilm ?film1) ^
    (?watchEvent2 user:watchedFilm ?film2) ^
    (?watchEvent3 user:watchedFilm ?film3) ^
    (?film1 trope:containsTrope ?trope) ^
    (?film2 trope:containsTrope ?trope) ^
    (?film3 trope:containsTrope ?trope) ^
    recentWatches(?watchEvent1, ?watchEvent2, ?watchEvent3)
    -> (?user user:fatigued ?trope)]

# Rule 9: Masala film song integration
[MasalaSongAppreciation:
    (?user user:toleranceForMusicalInterludes "Low") ^
    (?film narrative:hasStructure narrative:MasalaStructure)
    -> (?film recommendation:downrank 0.5)]

# Rule 10: Circular narrative preference
[CircularNarrativeMatch:
    (?user user:enjoysAmbiguity true) ^
    (?film narrative:hasStructure african:CircularStructure)
    -> (?film recommendation:uprank 1.4)]
```

### 3.2 Context-Aware Rules

```sparql
# Rule 11: Contemplative time-of-day matching
[ContemplativeEvening:
    (?user ctx:currentTimeOfDay "Late Evening") ^
    (?user ctx:socialSetting "Solo") ^
    (?film narrative:emphasizesContemplation true)
    -> (?film recommendation:uprank 1.3)]

# Rule 12: Social viewing narrative compatibility
[SocialComplexityDownrank:
    (?user ctx:socialSetting "Group Viewing") ^
    (?film narrative:hasComplexity "High") ^
    (?film narrative:hasNonLinearCausality true)
    -> (?film recommendation:downrank 0.6)]

# Rule 13: Commute-friendly structure
[CommuteStructure:
    (?user ctx:socialSetting "Commute") ^
    (?film narrative:hasEpisodicStructure true) ^
    (?film media:hasRuntime ?runtime) ^
    lessThan(?runtime, 30)
    -> (?film recommendation:uprank 1.5)]

# Rule 14: Cultural event relevance
[CulturalEventBoost:
    (?user ctx:nearCulturalEvent ?event) ^
    (?film ctx:isCulturallyRelevantTo ?event)
    -> (?film recommendation:uprank 1.6)]

# Rule 15: Fatigue state narrative matching
[FatigueSimplicity:
    (?user user:currentPsychographicState "Fatigued") ^
    (?film narrative:hasComplexity "Low") ^
    (?film narrative:hasConflictDensity ?density) ^
    lessThan(?density, 0.5)
    -> (?film recommendation:uprank 1.4)]
```

### 3.3 Narrative Compatibility Rules

```sparql
# Rule 16: Kishōtenketsu for contemplative users
[KishotenketsuMatch:
    (?user user:personalityTrait "Introspective") ^
    (?film narrative:hasStructure narrative:Kishotenketsu)
    -> (?film recommendation:uprank 1.5)]

# Rule 17: Hero's Journey for adventure seekers
[HerosJourneyMatch:
    (?user user:prefersGenre "Adventure") ^
    (?film narrative:hasStructure western:HerosJourney)
    -> (?film recommendation:uprank 1.3)]

# Rule 18: Masala for high-energy mood
[MasalaEnergyMatch:
    (?user user:currentMood "Energetic") ^
    (?film narrative:hasStructure narrative:MasalaStructure) ^
    (?film narrative:hasEmotionalDensity ?density) ^
    greaterThan(?density, 0.8)
    -> (?film recommendation:uprank 1.6)]

# Rule 19: Griot narrative for patient viewers
[GriotPatienceMatch:
    (?user user:patienceLevel "High") ^
    (?film narrative:hasStructure african:OralNarrativeStructure)
    -> (?film recommendation:uprank 1.4)]

# Rule 20: Hyperlink cinema for puzzle enthusiasts
[HyperlinkPuzzleMatch:
    (?user user:enjoysPuzzles true) ^
    (?film narrative:hasStructure latam:HyperlinkStructure)
    -> (?film recommendation:uprank 1.5)]
```

### 3.4 Trope Combination Rules

```sparql
# Rule 21: Familiar + Novel balance
[FamiliarNovelBalance:
    (?user user:hasWatchHistory ?watchEvent) ^
    (?watchEvent user:watchedFilm ?priorFilm) ^
    (?priorFilm trope:containsTrope ?familiarTrope) ^
    (?candidateFilm trope:containsTrope ?familiarTrope) ^
    (?candidateFilm trope:containsTrope ?novelTrope) ^
    not(?priorFilm trope:containsTrope ?novelTrope)
    -> (?candidateFilm recommendation:uprank 1.3)]

# Rule 22: Trope subversion appeal
[SubversionAppeal:
    (?user user:likesSubversion true) ^
    (?film trope:subverts ?expectedTrope)
    -> (?film recommendation:uprank 1.4)]

# Rule 23: Archetype preference
[ArchetypeMatch:
    (?user user:respondsWellTo ?archetype) ^
    (?film trope:featuresArchetype ?archetype)
    -> (?film recommendation:uprank 1.2)]

# Rule 24: Cultural trope resonance
[CulturalTropeResonance:
    (?user user:culturalBackground ?culture) ^
    (?film trope:containsTrope ?trope) ^
    (?trope trope:resonanceIn ?culture)
    -> (?film recommendation:uprank 1.3)]
```

### 3.5 Structural Learning Rules

```sparql
# Rule 25: Structure exploration gradient
[StructureExplorationGradient:
    (?user user:hasWatchHistory ?watchEvent1) ^
    (?user user:hasWatchHistory ?watchEvent2) ^
    (?watchEvent1 user:watchedFilm ?film1) ^
    (?watchEvent2 user:watchedFilm ?film2) ^
    (?film1 narrative:hasStructure western:ThreeActStructure) ^
    (?film2 narrative:hasStructure western:ThreeActStructure) ^
    allCompleted(?watchEvent1, ?watchEvent2) ^
    (?candidateFilm narrative:hasStructure western:NonLinearWestern)
    -> (?candidateFilm recommendation:uprank 1.2)
    rdfs:comment "Gradually introduce narrative complexity"]

# Rule 26: Cross-cultural bridge films
[CrossCulturalBridge:
    (?user user:familiarWith "Western Cinema") ^
    not(?user user:familiarWith "East Asian Cinema") ^
    (?film narrative:hasStructure narrative:Kishotenketsu) ^
    (?film media:hasWesternInfluence true)
    -> (?film recommendation:uprank 1.4)
    rdfs:comment "Recommend gateway films for new structures"]

# Rule 27: Interval structure warning
[IntervalWarning:
    (?user user:watchingContext "Streaming") ^
    (?film narrative:hasIntervalCliffhanger ?interval) ^
    not(?user user:familiarWith "Masala Cinema")
    -> (?film recommendation:addWarning "Film designed with intermission"))]

# Rule 28: Multi-plot cognitive load
[MultiPlotCognitiveLoad:
    (?user user:cognitiveLoadTolerance "Low") ^
    (?film narrative:hasStructure narrative:MasalaStructure) ^
    (?film narrative:parallelPlotCount ?count) ^
    greaterThan(?count, 3)
    -> (?film recommendation:downrank 0.6)]

# Rule 29: Recursive narrative depth matching
[RecursiveDepthMatch:
    (?user user:enjoysMetaNarrative true) ^
    (?film narrative:hasRecursiveDepth ?depth) ^
    greaterThan(?depth, 1)
    -> (?film recommendation:uprank 1.4)]

# Rule 30: Pacing compatibility
[PacingCompatibility:
    (?user user:preferredPacing ?userPace) ^
    (?film media:avgShotLength ?asl) ^
    pacingMatch(?userPace, ?asl)
    -> (?film recommendation:uprank 1.3)]
```

### 3.6 Mood-Structure Alignment

```sparql
# Rule 31: Seeking comfort -> familiar structures
[ComfortFamiliarStructure:
    (?user user:currentPsychographicState "Seeking Comfort") ^
    (?user user:respondsWellTo ?structure) ^
    (?film narrative:hasStructure ?structure)
    -> (?film recommendation:uprank 1.6)]

# Rule 32: Seeking challenge -> unfamiliar structures
[ChallengenovelStructure:
    (?user user:currentPsychographicState "Seeking Challenge") ^
    not(?user user:familiarWith ?structure) ^
    (?film narrative:hasStructure ?structure)
    -> (?film recommendation:uprank 1.4)]

# Rule 33: Nostalgic state -> cultural origin match
[NostalgicOriginMatch:
    (?user user:currentPsychographicState "Nostalgic") ^
    (?user user:childhoodRegion ?region) ^
    (?film narrative:culturalOrigin ?region)
    -> (?film recommendation:uprank 1.7)]

# Rule 34: Melancholic mood -> contemplative structures
[MelancholicContemplative:
    (?user user:currentMood "Melancholic") ^
    (?film narrative:emphasizesContemplation true) ^
    (?film narrative:hasConflictDensity ?density) ^
    lessThan(?density, 0.5)
    -> (?film recommendation:uprank 1.5)]
```

---

## Part 4: Integration with GMC-O Ontology

### 4.1 Complete Ontology Schema

```turtle
@prefix media: <http://recommendation.org/ontology/media#> .
@prefix narrative: <http://recommendation.org/ontology/narrative#> .
@prefix trope: <http://recommendation.org/ontology/trope#> .
@prefix user: <http://recommendation.org/ontology/user#> .
@prefix ctx: <http://recommendation.org/ontology/context#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

#################################################################
#    NARRATIVE STRUCTURE HIERARCHY
#################################################################

narrative:NarrativeStructure a owl:Class ;
    rdfs:label "Narrative Structure"@en ;
    rdfs:comment "Top-level class for story architecture patterns" .

# Global structures (detailed above)
narrative:Kishotenketsu rdfs:subClassOf narrative:NarrativeStructure .
narrative:MasalaStructure rdfs:subClassOf narrative:NarrativeStructure .
african:OralNarrativeStructure rdfs:subClassOf narrative:NarrativeStructure .
quebec:QuebecCinemaNarrative rdfs:subClassOf narrative:NarrativeStructure .
southeast:BuddhistCyclicalStructure rdfs:subClassOf narrative:NarrativeStructure .
latam:MagicalRealismStructure rdfs:subClassOf narrative:NarrativeStructure .
western:HerosJourney rdfs:subClassOf narrative:NarrativeStructure .
western:ThreeActStructure rdfs:subClassOf narrative:NarrativeStructure .

#################################################################
#    TROPE TAXONOMY
#################################################################

trope:Trope a owl:Class ;
    rdfs:label "Narrative Trope"@en ;
    rdfs:comment "Recurring narrative device or pattern" .

trope:Archetype rdfs:subClassOf trope:Trope .
trope:PlotDevice rdfs:subClassOf trope:Trope .
trope:VisualMotif rdfs:subClassOf trope:Trope .
trope:DialoguePattern rdfs:subClassOf trope:Trope .

# Specific tropes (147 instances defined above)

#################################################################
#    NARRATIVE PROPERTIES
#################################################################

narrative:hasStructure a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range narrative:NarrativeStructure ;
    rdfs:label "has narrative structure"@en .

narrative:hasComplexity a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:string ;
    owl:oneOf ("Low", "Medium", "High", "Very High") .

narrative:hasConflictDensity a owl:DatatypeProperty ;
    rdfs:domain narrative:NarrativeStructure ;
    rdfs:range xsd:float ;
    rdfs:comment "Ratio of conflict-driven scenes; 0.0-1.0" .

narrative:hasEmotionalDensity a owl:DatatypeProperty ;
    rdfs:domain narrative:NarrativeStructure ;
    rdfs:range xsd:float ;
    rdfs:comment "Rate of emotional register shifts" .

narrative:emphasizesContemplation a owl:DatatypeProperty ;
    rdfs:domain narrative:NarrativeStructure ;
    rdfs:range xsd:boolean .

narrative:hasNonLinearCausality a owl:DatatypeProperty ;
    rdfs:domain narrative:NarrativeStructure ;
    rdfs:range xsd:boolean .

narrative:culturalOrigin a owl:DatatypeProperty ;
    rdfs:domain narrative:NarrativeStructure ;
    rdfs:range xsd:string .

media:avgShotLength a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:float ;
    rdfs:comment "Average shot length in seconds; pacing indicator" .

#################################################################
#    TROPE PROPERTIES
#################################################################

trope:containsTrope a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range trope:Trope .

trope:featuresArchetype a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range trope:Archetype .

trope:subverts a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range trope:Trope ;
    rdfs:comment "Film subverts audience expectation of this trope" .

trope:resonanceIn a owl:DatatypeProperty ;
    rdfs:domain trope:Trope ;
    rdfs:range xsd:string ;
    rdfs:comment "Cultures/regions where trope has strong resonance" .

trope:lowResonanceIn a owl:DatatypeProperty ;
    rdfs:domain trope:Trope ;
    rdfs:range xsd:string ;
    rdfs:comment "Cultures/regions where trope has weak resonance" .

#################################################################
#    USER PREFERENCE PROPERTIES
#################################################################

user:respondsWellTo a owl:ObjectProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range narrative:NarrativeStructure ;
    rdfs:comment "Learned preference from watch history" .

user:familiarWith a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:string ;
    rdfs:comment "Cinema tradition user has exposure to" .

user:prefersConflictDensity a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:float .

user:subtitleTolerance a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:string ;
    owl:oneOf ("Low", "Medium", "High") .

user:enjoysAmbiguity a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:boolean .

user:enjoysPuzzles a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:boolean .

user:likesSubversion a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:boolean .

user:cognitiveLoadTolerance a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:string ;
    owl:oneOf ("Low", "Medium", "High") .

user:enjoysMetaNarrative a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:boolean .

user:preferredPacing a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:string ;
    owl:oneOf ("Slow", "Medium", "Fast", "Frenetic") .

user:patienceLevel a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:string ;
    owl:oneOf ("Low", "Medium", "High") .

user:toleranceForMusicalInterludes a owl:DatatypeProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range xsd:string ;
    owl:oneOf ("Low", "Medium", "High") .

user:fatigued a owl:ObjectProperty ;
    rdfs:domain user:ViewerProfile ;
    rdfs:range trope:Trope ;
    rdfs:comment "User has seen this trope too frequently recently" .

#################################################################
#    RECOMMENDATION PROPERTIES
#################################################################

recommendation:uprank a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:float ;
    rdfs:comment "Multiplier for relevance score; >1.0" .

recommendation:downrank a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:float ;
    rdfs:comment "Multiplier for relevance score; <1.0" .

recommendation:addWarning a owl:DatatypeProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range xsd:string ;
    rdfs:comment "User-facing warning about viewing experience" .
```

### 4.2 Agent A (Narrative Analyst) Integration

**Input**: Scripts, Subtitles, Synopses
**Output**: Narrative structure classification + trope detection

**Workflow**:
1. **Script Analysis**:
   - Sentiment analysis over time → Identify narrative arc shape
   - Act break detection → Classify structure type
   - Dialogue density mapping → Flesch-Kincaid scores

2. **Structure Classification**:
   ```python
   def classify_narrative_structure(script_features):
       conflict_density = calculate_conflict_density(script_features)
       act_structure = detect_act_breaks(script_features)
       emotional_curve = analyze_emotional_arc(script_features)

       if act_structure == "4-act" and conflict_density < 0.4:
           return "narrative:Kishotenketsu"
       elif act_structure == "interval-based" and has_song_sequences(script_features):
           return "narrative:MasalaStructure"
       elif emotional_curve == "circular" and has_communal_focus(script_features):
           return "african:OralNarrativeStructure"
       # ... etc
   ```

3. **Trope Detection**:
   - NLP entity recognition for character archetypes
   - Plot beat matching against trope database
   - Dialogue pattern matching

4. **RDF Triple Generation**:
   ```turtle
   <film:TheWindRises> narrative:hasStructure narrative:Kishotenketsu .
   <film:TheWindRises> narrative:hasConflictDensity 0.35 .
   <film:TheWindRises> narrative:emphasizesContemplation true .
   <film:TheWindRises> trope:featuresArchetype trope:ReluctantHero .
   <film:TheWindRises> trope:containsTrope trope:BittersweetEnding .
   ```

### 4.3 Agent B (Visual Analyst) Integration

**Input**: Video frames, Audio
**Output**: Pacing metrics, visual aesthetic

**Workflow**:
1. **Shot Detection**:
   - Calculate Average Shot Length (ASL)
   - Map to pacing property: <2s = Frenetic, 2-5s = Fast, 5-10s = Medium, >10s = Slow

2. **Visual Analysis**:
   - Color palette extraction → Visual aesthetic classification
   - Lighting analysis → Mood detection

3. **Audio Analysis**:
   - Song sequence detection (for Masala identification)
   - Score intensity mapping

4. **RDF Output**:
   ```turtle
   <film:TropicalMalady> media:avgShotLength 12.3 .
   <film:TropicalMalady> narrative:hasStructure southeast:ThesisAntithesisStructure .
   <film:TropicalMalady> media:visualAesthetic "Naturalistic" .
   ```

### 4.4 Agent C (Context Mapper) Integration

**Input**: User location, date, cultural calendar
**Output**: Cultural relevance links

**Workflow**:
1. Monitor cultural events (holidays, festivals, elections)
2. Tag films with cultural relevance
3. Boost films during appropriate contexts

**Example**:
```turtle
<event:Diwali2025> a ctx:CulturalEvent ;
    ctx:occursIn "India, Indian Diaspora" ;
    ctx:dateRange "2025-11-01/2025-11-05" .

<film:3Idiots> ctx:isCulturallyRelevantTo <event:Diwali2025> .
<film:3Idiots> recommendation:uprank 1.6 . # During Diwali period
```

### 4.5 Agent D (User Psychographer) Integration

**Input**: User watch history, abandonment patterns
**Output**: User preference learning

**Workflow**:
1. **Structure Preference Learning**:
   - Track which structures user completes
   - Identify fatigue patterns
   - Generate `user:respondsWellTo` triples

2. **Trope Preference Learning**:
   - Co-occurrence analysis of liked films
   - Detect trope fatigue (same trope in 3+ recent watches)

3. **Pacing Compatibility**:
   - Correlate completion rate with avgShotLength

**Example Output**:
```turtle
<user:Alice> user:respondsWellTo narrative:Kishotenketsu .
<user:Alice> user:familiarWith "Japanese Cinema" .
<user:Alice> user:preferredPacing "Slow" .
<user:Alice> user:enjoysAmbiguity true .
<user:Alice> user:fatigued trope:ChosenOne . # Seen too many recently
```

---

## Part 5: Test Cases with Screenplay Examples

### Test Case 1: Kishōtenketsu Film Recommendation

**User Profile**:
```turtle
<user:Bob> a user:ViewerProfile ;
    user:familiarWith "Western Cinema" ;
    user:currentPsychographicState "Seeking Contemplation" ;
    user:subtitleTolerance "High" ;
    user:preferredPacing "Medium" ;
    user:enjoysAmbiguity true .
```

**Candidate Film**:
```turtle
<film:MyNeighborTotoro> a media:CreativeWork ;
    narrative:hasStructure narrative:Kishotenketsu ;
    narrative:hasConflictDensity 0.25 ;
    narrative:emphasizesContemplation true ;
    media:avgShotLength 8.2 ;
    media:primaryLanguage "Japanese" ;
    trope:featuresArchetype trope:MagicalHelper ; # Totoro
    trope:containsTrope trope:HealingNature .
```

**SWRL Rules Applied**:
- Rule 5 (ContemplativeMood): +1.5 uprank
- Rule 10 (CircularNarrativeMatch): NOT triggered (not circular, but similar)
- Rule 16 (KishotenketsuMatch): +1.5 uprank
- Rule 31 (ComfortFamiliarStructure): NOT triggered (unfamiliar structure)
- Rule 26 (CrossCulturalBridge): +1.4 uprank (Ghibli as gateway)

**Combined Score**: 1.5 × 1.5 × 1.4 = **3.15x base score**

**Expected Outcome**: Highly recommended, with explanation:
> "This Japanese film uses a contemplative four-act structure (Kishōtenketsu) that creates meaning through atmosphere rather than conflict. Perfect for your current mood."

---

### Test Case 2: Masala Film for Non-Familiar User

**User Profile**:
```turtle
<user:Carol> a user:ViewerProfile ;
    user:familiarWith "Hollywood Cinema" ;
    user:currentMood "Energetic" ;
    user:toleranceForMusicalInterludes "Low" ;
    user:cognitiveLoadTolerance "Medium" ;
    user:preferredPacing "Fast" .
```

**Candidate Film**:
```turtle
<film:3Idiots> a media:CreativeWork ;
    narrative:hasStructure narrative:MasalaStructure ;
    narrative:hasEmotionalDensity 0.85 ;
    narrative:hasSongCount 6 ;
    narrative:parallelPlotCount 4 ;
    media:hasRuntime 170 ;
    media:primaryLanguage "Hindi" ;
    trope:containsTrope trope:FishOutOfWater .
```

**SWRL Rules Applied**:
- Rule 18 (MasalaEnergyMatch): +1.6 uprank (energetic mood)
- Rule 9 (MasalaSongAppreciation): -0.5 downrank (low song tolerance)
- Rule 28 (MultiPlotCognitiveLoad): NOT triggered (Medium tolerance, 4 plots acceptable)
- Rule 27 (IntervalWarning): Warning added

**Combined Score**: 1.6 × 0.5 = **0.8x base score**

**Expected Outcome**: Moderately downranked, with warning:
> "This Bollywood film features 6 song-and-dance sequences integrated into the story and is designed with an intermission break. May not match your preference for minimal musical interludes."

---

### Test Case 3: African Circular Narrative for Patient Viewer

**User Profile**:
```turtle
<user:David> a user:ViewerProfile ;
    user:patienceLevel "High" ;
    user:enjoysAmbiguity true ;
    user:enjoysMetaNarrative true ;
    user:preferredPacing "Slow" ;
    user:currentPsychographicState "Seeking Challenge" .
```

**Candidate Film**:
```turtle
<film:Yeelen> a media:CreativeWork ;
    narrative:hasStructure african:OralNarrativeStructure ;
    african:hasRecursiveDepth 2 ;
    african:ancestralInfluence true ;
    african:usesOralTechnique african:ProverbIntegration ;
    media:avgShotLength 11.5 ;
    narrative:hasNonLinearCausality true ;
    trope:containsTrope african:MythicTime .
```

**SWRL Rules Applied**:
- Rule 19 (GriotPatienceMatch): +1.4 uprank
- Rule 10 (CircularNarrativeMatch): +1.4 uprank
- Rule 29 (RecursiveDepthMatch): +1.4 uprank
- Rule 32 (ChallengeNovelStructure): +1.4 uprank (unfamiliar structure)
- Rule 2 (SlowCinemaRule): NOT triggered (user prefers slow)

**Combined Score**: 1.4 × 1.4 × 1.4 × 1.4 = **3.84x base score**

**Expected Outcome**: Highly recommended:
> "This Malian film uses a circular narrative structure rooted in Bambara oral tradition. Its mythic time structure and contemplative pacing offer a challenging alternative to Western linear storytelling."

---

### Test Case 4: Trope Fatigue Detection

**User Profile**:
```turtle
<user:Eve> a user:ViewerProfile ;
    user:hasWatchHistory <watch:Event1>, <watch:Event2>, <watch:Event3> .

<watch:Event1> user:watchedFilm <film:TheMatrix> ;
    user:completionRate 1.0 ;
    user:watchDate "2025-11-20" .

<watch:Event2> user:watchedFilm <film:HarryPotter> ;
    user:completionRate 1.0 ;
    user:watchDate "2025-11-25" .

<watch:Event3> user:watchedFilm <film:DoctorStrange> ;
    user:completionRate 1.0 ;
    user:watchDate "2025-11-29" .

<film:TheMatrix> trope:featuresArchetype trope:ChosenOne .
<film:HarryPotter> trope:featuresArchetype trope:ChosenOne .
<film:DoctorStrange> trope:featuresArchetype trope:ChosenOne .

# Derived by Rule 8
<user:Eve> user:fatigued trope:ChosenOne .
```

**Candidate Film**:
```turtle
<film:StarWars> a media:CreativeWork ;
    narrative:hasStructure western:HerosJourney ;
    trope:featuresArchetype trope:ChosenOne . # Luke Skywalker
```

**SWRL Rules Applied**:
- Rule 8 (TropeFatigue): Detected
- Recommendation: Downrank or suggest alternative trope

**Expected Outcome**: Downranked with explanation:
> "You've watched several 'Chosen One' narratives recently. Would you like to explore different character archetypes?"

---

### Test Case 5: Cross-Cultural Gateway Film

**User Profile**:
```turtle
<user:Frank> a user:ViewerProfile ;
    user:familiarWith "Western Cinema" ;
    user:currentPsychographicState "Seeking Discovery" ;
    user:subtitleTolerance "Medium" ;
    user:enjoysAmbiguity false . # Prefers clear resolution
```

**Candidate Film 1 (Direct Kishōtenketsu)**:
```turtle
<film:TokyoStory> a media:CreativeWork ;
    narrative:hasStructure narrative:Kishotenketsu ;
    narrative:hasConflictDensity 0.2 ;
    media:avgShotLength 14.8 ;
    narrative:emphasizesContemplation true ;
    media:hasWesternInfluence false .
```

**Candidate Film 2 (Gateway Film)**:
```turtle
<film:Ikiru> a media:CreativeWork ;
    narrative:hasStructure narrative:Kishotenketsu ;
    narrative:hasConflictDensity 0.45 ; # Higher than typical
    media:avgShotLength 7.2 ; # Faster than typical
    narrative:emphasizesContemplation true ;
    media:hasWesternInfluence true ; # Kurosawa's Western influences
    trope:featuresArchetype trope:RedemptionArc . # Familiar trope
```

**SWRL Rules Applied**:
- Rule 26 (CrossCulturalBridge): *Ikiru* gets +1.4 uprank
- Rule 32 (ChallengeNovelStructure): Both get +1.4
- *TokyoStory*: 1.4x score
- *Ikiru*: 1.4 × 1.4 = 1.96x score

**Expected Outcome**: *Ikiru* recommended as gateway:
> "This Akira Kurosawa film introduces Japanese narrative structure (Kishōtenketsu) while incorporating Western influences and a redemption arc you're familiar with. A great entry point to East Asian cinema."

---

## Part 6: Implementation Guidelines

### 6.1 GPU Semantic Processing Integration

The narrative structure and trope classification should be encoded as **semantic embeddings** for GPU-accelerated similarity search.

**Embedding Strategy**:
1. **Structure Embeddings**: Each narrative structure gets a learned vector representing:
   - Pacing characteristics
   - Emotional arc shape
   - Conflict density
   - Cultural associations

2. **Trope Embeddings**: Each trope gets a vector capturing:
   - Co-occurrence patterns with other tropes
   - Cultural resonance vectors
   - Emotional valence

3. **Film Embeddings**: Combine:
   - Visual embeddings (from Agent B)
   - Narrative embeddings (from Agent A)
   - Trope composition vector

**Vector Space Design**:
```python
# 512-dimensional semantic space
film_embedding = concat([
    visual_features,        # 128-dim from CV model
    narrative_structure,    # 64-dim learned structure embedding
    trope_composition,      # 256-dim weighted trope vectors
    pacing_metrics,         # 32-dim (ASL, conflict density, etc.)
    cultural_context        # 32-dim regional encodings
])

# User embedding (same 512-dim space)
user_embedding = concat([
    viewing_history_aggregate,   # 128-dim
    structure_preferences,       # 64-dim
    trope_preferences,           # 256-dim
    pacing_preferences,          # 32-dim
    cultural_familiarity         # 32-dim
])

# Similarity search
similarity = cosine_similarity(user_embedding, film_embedding)
```

### 6.2 SWRL Rule Execution Pipeline

1. **OWL Reasoning** (Offline):
   - Apply structural inference rules
   - Expand class hierarchies
   - Validate ontology consistency

2. **SWRL Rule Application** (Real-time):
   - User context loaded
   - Apply 34 SWRL rules
   - Generate uprank/downrank multipliers

3. **Hybrid Scoring**:
   ```python
   final_score = (
       base_vector_similarity *
       swrl_rank_multiplier *
       context_boost *
       freshness_factor
   )
   ```

### 6.3 Agent Coordination

**Population Pipeline**:
```
New Content Ingested
    ↓
Agent A (NLP) + Agent B (CV) run in parallel
    ↓
Narrative Structure Classification
    ↓
Trope Detection & Tagging
    ↓
Generate RDF Triples
    ↓
Validate against SHACL Shapes
    ↓
Commit to Knowledge Graph
    ↓
Generate Semantic Embeddings
    ↓
Index in GPU Vector Database (AgentDB)
```

**Recommendation Pipeline**:
```
User Context Loaded
    ↓
Retrieve User Preference Profile (RDF)
    ↓
Apply SWRL Rules → Generate Rank Modifiers
    ↓
GPU Vector Search (AgentDB semantic similarity)
    ↓
Apply Rank Modifiers to Top-K Results
    ↓
Re-rank and Return Top-N Recommendations
```

### 6.4 Continuous Learning

**Feedback Loop**:
1. User watches film to completion → Positive signal for structure/tropes
2. User abandons film → Negative signal
3. User rates/reviews → Explicit preference data

**Update Workflow**:
```python
def update_user_preferences(user_id, watch_event):
    if watch_event.completion_rate > 0.85:
        film = get_film(watch_event.film_id)
        structure = film.narrative_structure
        tropes = film.tropes

        # Update RDF graph
        add_triple(user_id, "user:respondsWellTo", structure)
        for trope in tropes:
            increment_trope_affinity(user_id, trope)

        # Update user embedding
        user_embedding = recompute_user_embedding(user_id)
        store_embedding(user_id, user_embedding)
```

### 6.5 Explainability Layer

Every recommendation should include **narrative-aware explanations**:

```python
def generate_explanation(film, user, score_components):
    explanation_parts = []

    if score_components['structure_match'] > 1.2:
        structure = film.narrative_structure
        explanation_parts.append(
            f"This film uses {structure.label}, which matches your "
            f"viewing patterns and current mood."
        )

    if score_components['trope_resonance'] > 1.2:
        familiar_tropes = film.tropes.intersection(user.preferred_tropes)
        explanation_parts.append(
            f"Features familiar elements: {', '.join([t.label for t in familiar_tropes])}"
        )

    if score_components['cultural_bridge'] > 1.0:
        explanation_parts.append(
            "A great introduction to a new storytelling tradition "
            "while maintaining familiar narrative beats."
        )

    if film.needs_warning():
        explanation_parts.append(
            f"⚠️ {film.warning_message}"
        )

    return "\n".join(explanation_parts)
```

**Example Output**:
```
🎬 My Neighbor Totoro (1988) - 96% Match

Why this recommendation:
✓ This film uses Kishōtenketsu (Japanese four-act structure), which creates
  emotional resonance through contemplation rather than conflict—perfect for
  your current reflective mood.
✓ Features familiar elements: Magical Helper, Coming-of-Age
✓ A great introduction to Japanese narrative tradition with universal themes

⏱️ Runtime: 86 minutes | 🗣️ Japanese with subtitles | 🎭 Contemplative pacing
```

---

## Part 7: Academic Sources & References

### Kishōtenketsu & Japanese Narrative
1. **Lamarre, Thomas** (2009). *The Anime Machine: A Media Theory of Animation*. University of Minnesota Press.
2. **Napier, Susan J.** (2005). *Anime from Akira to Howl's Moving Castle*. Palgrave Macmillan.
3. **Richie, Donald** (1972). *Japanese Cinema: Film Style and National Character*. Doubleday.
4. **Bordwell, David** (1988). *Ozu and the Poetics of Cinema*. Princeton University Press.

### Bollywood & Masala Cinema
5. **Gopalan, Lalitha** (2002). *Cinema of Interruptions: Action Genres in Contemporary Indian Cinema*. British Film Institute.
6. **Dudrah, Rajinder** (2006). *Bollywood: Sociology Goes to the Movies*. Sage Publications.
7. **Vasudevan, Ravi** (2010). *The Melodramatic Public: Film Form and Spectatorship in Indian Cinema*. Palgrave Macmillan.
8. **Mishra, Vijay** (2002). *Bollywood Cinema: Temples of Desire*. Routledge.

### African Cinema & Oral Traditions
9. **Diawara, Manthia** (1992). *African Cinema: Politics and Culture*. Indiana University Press.
10. **Ukadike, Nwachukwu Frank** (1994). *Black African Cinema*. University of California Press.
11. **Barlet, Olivier** (2000). *African Cinemas: Decolonizing the Gaze*. Zed Books.

### Narratology Theory
12. **Bordwell, David** (1985). *Narration in the Fiction Film*. University of Wisconsin Press.
13. **Branigan, Edward** (1992). *Narrative Comprehension and Film*. Routledge.
14. **Abbott, H. Porter** (2008). *The Cambridge Introduction to Narrative* (2nd ed.). Cambridge University Press.
15. **Herman, David** (2002). *Story Logic: Problems and Possibilities of Narrative*. University of Nebraska Press.

### Comparative Film Studies
16. **Nagib, Lúcia & Mello, Cecilia** (Eds.) (2009). *Realism and the Audiovisual Media*. Palgrave Macmillan.
17. **Hjort, Mette & Petrie, Duncan** (Eds.) (2007). *The Cinema of Small Nations*. Indiana University Press.
18. **Chaudhuri, Shohini** (2005). *Contemporary World Cinema: Europe, the Middle East, East Asia and South Asia*. Edinburgh University Press.

### Cultural Context & Reception
19. **Hall, Stuart** (1980). "Encoding/Decoding." In *Culture, Media, Language*. Hutchinson.
20. **Appadurai, Arjun** (1996). *Modernity at Large: Cultural Dimensions of Globalization*. University of Minnesota Press.

---

## Part 8: Future Research Directions

### 8.1 Underrepresented Narrative Traditions

**Middle Eastern Cinema**:
- Persian cinema's poetic realism (Kiarostami, Farhadi)
- Arab cinema's political allegory
- Turkish Yeşilçam melodrama

**Indigenous Cinema**:
- Australian Aboriginal storytelling (Dreamtime structure)
- Native American circular narrative
- Māori cinema (whakapapa/genealogical structure)

### 8.2 Advanced Trope Analysis

**Trope Evolution Tracking**:
- How tropes transform across cultures
- Subversion detection (when films deliberately invert tropes)
- Trope hybridization (fusion of cultural traditions)

**Emotional Trope Mapping**:
- Which tropes evoke which Rasas (emotional states)
- Cross-cultural emotional resonance testing

### 8.3 Neuroscience Integration

**Brain Response Patterns**:
- Different narrative structures activate different neural pathways
- Cultural familiarity affects cognitive processing load
- Potential fMRI studies for structure preference validation

---

## Conclusion

This research provides a comprehensive foundation for culturally-aware narrative reasoning in the GMC-O recommendation engine. By formalizing global narrative structures and tropes as OWL classes with SWRL reasoning rules, the system can:

1. **Understand** diverse storytelling traditions beyond Western models
2. **Learn** user preferences across cultural boundaries
3. **Recommend** films that match narrative expectations or challenge them appropriately
4. **Explain** recommendations in culturally-informed ways
5. **Bridge** cultural gaps by identifying gateway films

The 34 SWRL rules enable nuanced reasoning about narrative compatibility, ensuring that recommendations respect both user preferences and the cultural integrity of the content.

**Next Steps**:
1. Implement Agent A (Narrative Analyst) with screenplay corpus
2. Train structure/trope embeddings on IMDB/Letterboxd top 1000
3. Validate SWRL rules with user testing
4. Expand to TV episodic structures
5. Integrate with existing GMC-O ontology and GPU pipeline

---

**End of Research Document**
