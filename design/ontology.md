This is a comprehensive request involving two distinct but connected deliverables:
1.  **The New Ontology:** A semantic framework designed for high-level reasoning about content, users, and context (optimized for GPU-based semantic indexing).
2.  **The PRD (Product Requirement Document):** A blueprint for building the AI Agents responsible for ingesting raw data and mapping it into this ontology.

---

### Part 1: The Global Media & Context Ontology (GMC-O)

This ontology moves beyond simple metadata (Year, Director) into **semantic concepts** (Mood, Narrative Structure, Cultural Context) to leverage your "Advanced GPU Semantic Intelligence."

```turtle
@prefix media: <http://recommendation.org/ontology/media#> .
@prefix user: <http://recommendation.org/ontology/user#> .
@prefix ctx: <http://recommendation.org/ontology/context#> .
@prefix tech: <http://recommendation.org/ontology/tech-stack#> .
@prefix sem: <http://recommendation.org/ontology/semantic-descriptors#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

#################################################################
#    DOMAIN: MEDIA & NARRATIVE (The "What")
#################################################################

media:CreativeWork a owl:Class ;
    rdfs:label "Creative Work"@en ;
    rdfs:comment "Top-level class for any film, episode, or series." .

media:NarrativeStructure a owl:Class ;
    rdfs:label "Narrative Structure"@en ;
    rdfs:comment "The architectural flow of the story (e.g., Non-linear, Hero's Journey, Ensemble Cast)." .

media:Pacing a owl:Class ;
    rdfs:label "Pacing Metrics"@en ;
    rdfs:comment "Quantifiable measure of scene cuts per minute and dialogue density." .

media:VisualAesthetic a owl:Class ;
    rdfs:label "Visual Aesthetic"@en ;
    rdfs:comment "GPU-derived classification of color grading, lighting styles (e.g., Noir, Neon-Noir, Pastel)." .

media:SemanticEmbeddingVector a owl:Class ;
    rdfs:label "Semantic Embedding Vector"@en ;
    rdfs:comment "The high-dimensional vector address representing the content in latent space." .

#################################################################
#    DOMAIN: USER & PSYCHOGRAPHICS (The "Who")
#################################################################

user:ViewerProfile a owl:Class ;
    rdfs:label "Viewer Profile"@en ;
    rdfs:subClassOf user:Agent .

user:PsychographicState a owl:Class ;
    rdfs:label "Psychographic State"@en ;
    rdfs:comment "Current psychological inclination (e.g., 'Seeking Comfort', 'Seeking Challenge', 'Nostalgic')." .

user:TasteCluster a owl:Class ;
    rdfs:label "Taste Cluster"@en ;
    rdfs:comment "Dynamic grouping of users based on vector similarity of consumption history." .

user:ToleranceLevel a owl:Class ;
    rdfs:label "Tolerance Level"@en ;
    rdfs:comment "User specific thresholds for violence, complexity, subtitles, or slow pacing." .

#################################################################
#    DOMAIN: LOCAL CONTEXT (The "Where/When")
#################################################################

ctx:CulturalContext a owl:Class ;
    rdfs:label "Cultural Context"@en ;
    rdfs:comment "The socio-cultural framework of the viewer (e.g., Regional holidays, Political climate, Taboos)." .

ctx:SocialSetting a owl:Class ;
    rdfs:label "Social Setting"@en ;
    rdfs:comment "The immediate social environment (e.g., 'Date Night', 'Family Gathering', 'Solo Commute')." .

ctx:EnvironmentalFactors a owl:Class ;
    rdfs:label "Environmental Factors"@en ;
    rdfs:comment "Physical conditions (Lighting level, Ambient noise, Time of day)." .

#################################################################
#    DOMAIN: TECH STACK (The "How")
#################################################################

tech:DeliveryConstraint a owl:Class ;
    rdfs:label "Delivery Constraint"@en ;
    rdfs:comment "Limitations imposed by hardware or network (e.g., HDR support, Mobile Data Saver mode)." .

tech:LatencyRequirement a owl:Class ;
    rdfs:label "Latency Requirement"@en ;
    rdfs:comment "Required buffer speeds for current content bitrate." .

#################################################################
#    OBJECT PROPERTIES (The "Connectors")
#################################################################

sem:hasNarrativeArc a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range media:NarrativeStructure .

sem:inducesPsychographicState a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range user:PsychographicState ;
    rdfs:comment "Predictive property: This film is likely to induce State X." .

ctx:isCulturallyRelevantTo a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range ctx:CulturalContext ;
    rdfs:comment "Dynamic link between content and local cultural events/norms." .

tech:requiresCapability a owl:ObjectProperty ;
    rdfs:domain media:CreativeWork ;
    rdfs:range tech:DeliveryConstraint .
```

***

### Part 2: Product Requirement Document (PRD) for Population Agents

**Project Name:** Semantic Knowledge Graph Population (SKGP)
**Version:** 1.0
**Target System:** Global Film/TV Recommendation Engine

#### 1. Executive Summary
We are building a fleet of **Autonomous Semantic Agents**. Their purpose is not just to scrape metadata (IMDb style), but to utilize GPU-based inference (LLMs and Computer Vision) to "watch" and "read" content, extracting deep semantic features to populate the Ontology defined above.

#### 2. Architecture Overview
The system operates on a **Extract-Transform-Load-Embed (ETLE)** architecture.
1.  **Ingestion:** Agents accept video files, scripts, and subtitles.
2.  **Inference:** Agents run GPU models to extract ontology classes.
3.  **Mapping:** Agents map inference outputs to RDF triples.
4.  **Injection:** Agents write to the Graph Database (Neo4j/GraphDB).

---

#### 3. Agent Definitions & Responsibilities

We require four specific classes of agents to populate the ontology.

##### **Agent A: The Narrative Analyst (NLP-Based)**
*   **Input:** Scripts, Subtitles, Synopses, User Reviews.
*   **Ontology Target:** `media:NarrativeStructure`, `media:Theme`, `sem:hasNarrativeArc`.
*   **Core Task:**
    *   Ingest the script.
    *   Perform sentiment analysis over time (Narrative Arc).
    *   Classify dialogue complexity (Flesch-Kincaid score mapped to ontology).
    *   **Output:** RDF Triple $\to$ `<MovieX> <hasDialogueComplexity> "High" .`

##### **Agent B: The Visual & Auditory Analyst (CV/Audio-Based)**
*   **Input:** Video Frames (sampled), Audio Tracks.
*   **Ontology Target:** `media:VisualAesthetic`, `media:Pacing`, `media:Mood`.
*   **Core Task:**
    *   **Shot Detection:** Calculate average shot length (ASL) $\to$ Map to `media:Pacing`.
    *   **Color Analysis:** Extract dominant color palettes $\to$ Map to `media:VisualAesthetic` (e.g., "Dark", "High Contrast").
    *   **Audio Event Detection:** Identify score intensity vs. silence.
    *   **Output:** RDF Triple $\to$ `<MovieX> <hasVisualTone> "Noir" .`

##### **Agent C: The Context & Culture Mapper (Real-time Data)**
*   **Input:** Global Calendar, Geolocation APIs, News Feeds, Regulatory APIs.
*   **Ontology Target:** `ctx:CulturalContext`, `ctx:EnvironmentalFactors`.
*   **Core Task:**
    *   Monitor real-world events (e.g., Halloween, Elections, Holidays).
    *   Tag content relevant to current local context (e.g., Horror movies during October in the US; Romance movies on Valentine's Day in Brazil).
    *   **Output:** RDF Triple $\to$ `<MovieX> <isCulturallyRelevantTo> <RegionY_EventZ> .`

##### **Agent D: The User Psychographer (Behavioral Analysis)**
*   **Input:** User Interaction Logs (Clickstream), Watch Time, Abandonment Rates.
*   **Ontology Target:** `user:PsychographicState`, `user:TasteCluster`.
*   **Core Task:**
    *   Analyze viewing sessions to determine the user's *current* state (e.g., user skips heavy dialogue $\to$ map to `PsychographicState:Fatigued`).
    *   Generate embeddings for user history.
    *   **Output:** RDF Triple $\to$ `<User123> <hasCurrentState> <SeekingComfort> .`

---

#### 4. Functional Requirements for Agents

**4.1. Vector Alignment**
*   All Agents must output a `media:SemanticEmbeddingVector`.
*   The system must use a shared embedding space (e.g., OpenAI text-embedding-3 or a custom fine-tuned multimodal model) so that a "User Vector" and a "Movie Vector" can be compared mathematically.

**4.2. Confidence Scoring**
*   Agents must attach a confidence score to every triple generated.
*   *Example:* `<MovieX> <hasGenre> <SciFi> (Confidence: 0.98)` vs `<MovieX> <hasSubtext> <PoliticalSatire> (Confidence: 0.65)`.
*   **Rule:** Only triples with confidence > 0.75 are written to the Production Ontology.

**4.3. Entity Resolution**
*   Agents must resolve ambiguities (e.g., distinguishing between the 1990 vs 2017 version of "IT").
*   Agents must utilize Unique Universal Identifiers (UUIDs) derived from EIDR (Entertainment ID Registry) standards.

---

#### 5. Data Flow Pipeline

1.  **Trigger:** New content asset arrives in the CMS.
2.  **Orchestration:** The "Director Agent" spins up Agents A and B.
3.  **Processing:**
    *   Agent A reads the subtitles.
    *   Agent B samples video frames at 1fps.
4.  **Synthesis:** Agents generate a JSON-LD file containing the semantic metadata.
5.  **Validation:** The JSON-LD is validated against the SHACL shapes (constraints) of the Ontology.
6.  **Commit:** Data is committed to the Knowledge Graph.

#### 6. Success Metrics (KPIs)

*   **Semantic Density:** Average number of semantic triples per content asset (Target: >50 meaningful data points per movie, excluding basic cast/crew).
*   **Contextual Accuracy:** % of recommendations rejected by users due to "Wrong Vibe" (Goal: < 5%).
*   **Cold Start Latency:** Time taken from content ingestion to full semantic mapping (Target: < 15 minutes for a 2-hour feature).


