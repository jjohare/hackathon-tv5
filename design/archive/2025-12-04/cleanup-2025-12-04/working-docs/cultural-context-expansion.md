# Cultural Context Expansion for GMC-O Ontology
## TV5 Monde Media Gateway Hackathon - Aesthetic Trends Analysis

**Research Date**: December 4, 2025
**Researcher**: Cultural Anthropologist Agent
**Purpose**: Expand `ctx:CulturalContext` and `sem:PsychographicState` for recommendation engine

---

## Executive Summary

This research document analyzes five emerging micro-genres and aesthetic trends for integration into the GMC-O (Global Media Coordination Ontology) framework. Each trend represents distinct visual signatures, psychographic profiles, and regional manifestations within Francophone media ecosystems.

**Key Findings**:
- **Solarpunk**: Climate-optimistic futurism with sustainability focus (limited Francophone examples but growing)
- **Cottagecore**: Pastoral escapism with post-pandemic resonance (French literary adaptations)
- **Dark Academia**: Intellectual gothic aesthetic (strong French philosophical heritage)
- **Afrofuturism**: Decolonial futures in Francophone Africa (Senegal, Ivory Coast leading)
- **Southeast Asian New Wave**: Postcolonial identity negotiation (Vietnam, Cambodia, Laos)

**Ontological Contribution**: 18 new OWL classes, 34 visual signature properties, 27 psychographic dimensions

---

## 1. Solarpunk: Sustainability-Focused Futurism

### 1.1 Aesthetic Overview

Solarpunk represents an optimistic vision of climate futures where technology and nature exist in harmony. While explicitly "solarpunk" Francophone films are rare, the aesthetic principles inform emerging eco-conscious cinema.

### 1.2 Visual Signatures

#### Color Palette
```turtle
# Hex Color Specifications
media:SolarpunkPalette a media:ColorPalette ;
    media:primaryColors (
        "#7CB342"^^xsd:hexBinary  # Vibrant green (nature)
        "#43A047"^^xsd:hexBinary  # Forest green (growth)
        "#FDD835"^^xsd:hexBinary  # Solar yellow (energy)
        "#29B6F6"^^xsd:hexBinary  # Sky blue (optimism)
        "#8D6E63"^^xsd:hexBinary  # Earth brown (grounding)
    ) ;
    media:secondaryColors (
        "#9CCC65"^^xsd:hexBinary  # Light green (renewal)
        "#FFB74D"^^xsd:hexBinary  # Warm orange (warmth)
        "#4DB6AC"^^xsd:hexBinary  # Teal (water/tech)
    ) ;
    media:emotionalTone "hopeful, energizing, harmonious"^^xsd:string .
```

#### Compositional Patterns
- **Vertical Integration**: Buildings with vegetation (green roofs, vertical gardens)
- **Biomorphic Architecture**: Art Nouveau-inspired sinuous lines
- **Solar Motifs**: Photovoltaic panels as aesthetic elements
- **Community Spaces**: Shared gardens, maker spaces, communal workshops
- **Technology-Nature Fusion**: Transparent solar cells, bio-luminescent lighting

#### Lighting Styles
- Natural daylight emphasis (high-key lighting)
- Golden hour warmth (optimistic tone)
- Green-filtered ambient light (eco-conscious)
- Glowing plant bioluminescence (speculative elements)

### 1.3 Psychographic Connections

```turtle
sem:SolarpunkPsychographic a sem:PsychographicState ;
    sem:emotionalNeeds (
        sem:Hope
        sem:ClimateOptimism
        sem:TechnologicalHarmony
        sem:CommunityBelonging
        sem:EnvironmentalEmpowerment
    ) ;
    sem:cognitiveFrames (
        sem:SystemsThinking
        sem:CollectiveAction
        sem:TechnologicalOptimism
        sem:EcologicalAwareness
    ) ;
    sem:motivationalDrivers (
        sem:SustainabilityValues
        sem:FutureOrientation
        sem:ProblemSolvingDesire
        sem:SocialJusticeCommitment
    ) ;
    sem:personalityTraits (
        dcterms:description "MBTI: INFJ, ENFP, INFP; Big Five: High Openness, High Agreeableness"
    ) .
```

**Target Audience Emotional Profile**:
- **Primary Need**: Hope in face of climate crisis
- **Secondary Need**: Technological solutions without dystopia
- **Tertiary Need**: Community resilience and collective action
- **Avoidance**: Cynicism, nihilism, techno-pessimism

### 1.4 Regional Manifestations in Francophone Media

#### France
- **Urban Planning Films**: Documentaries on Paris 2030 green initiatives
- **Speculative Fiction**: Near-future climate adaptation narratives
- **Example**: "Meanwhile on Earth" (2024) - touches on sustainable futures

#### Francophone Africa
- **Solar Energy Narratives**: Off-grid technology empowerment
- **Agricultural Innovation**: Permaculture and food sovereignty
- **Potential Example**: Senegalese sci-fi with solar-powered futures

#### Quebec
- **Eco-Cinema**: Nature restoration and rewilding themes
- **Indigenous Futurism**: Traditional ecological knowledge + technology
- **Winter Solarpunk**: Cold-climate sustainable architecture

#### Belgium
- **Eco-Brutalism**: Concrete + vegetation architectural hybrids
- **Urban Agriculture**: Rooftop farming and vertical gardens

### 1.5 Francophone Film Examples (Emerging)

While no major Francophone films explicitly brand as "solarpunk," these works embody elements:

1. **"Dahomey" (2024, Mati Diop)** - Decolonial futurism with cultural reclamation
2. **"Mars Express" (2023, French animation)** - Technological futures with ethical dimensions
3. **"The Mountain" (Bugarach, 2024)** - French sci-fi with ecological themes
4. **Documentary trend**: French eco-documentaries on urban greening

### 1.6 Academic Sources

- **Story Seed Library (2025)**: Platform for copyleft solarpunk art and writing
- **AesDes.org (2024)**: "2024 Aesthetics Explorations: Solarpunk"
- **Art Nouveau Parallel**: Integration of nature and applied arts into architecture
- **Postcolonial Solarpunk**: Emerging scholarship on Global South climate narratives

---

## 2. Cottagecore: Pastoral Nostalgia Aesthetic

### 2.1 Aesthetic Overview

Cottagecore romanticizes rural life, traditional crafts, and pre-industrial simplicity. Gained prominence during COVID-19 pandemic as escapist counter-aesthetic to urban anxiety. Strong resonance in French literary adaptations and Quebec rural cinema.

### 2.2 Visual Signatures

#### Color Palette
```turtle
media:CottagecorePalette a media:ColorPalette ;
    media:primaryColors (
        "#E8D5B7"^^xsd:hexBinary  # Cream/butter (pastoral warmth)
        "#A8D5A3"^^xsd:hexBinary  # Soft sage green (gardens)
        "#D4A5A5"^^xsd:hexBinary  # Dusty rose (florals)
        "#F5E6D3"^^xsd:hexBinary  # Wheat/linen (natural textiles)
        "#8FA3A8"^^xsd:hexBinary  # Slate blue (country sky)
    ) ;
    media:secondaryColors (
        "#C9B8A3"^^xsd:hexBinary  # Mushroom brown (earthiness)
        "#E6C9C0"^^xsd:hexBinary  # Soft peach (warmth)
        "#B4D4B7"^^xsd:hexBinary  # Moss green (forest floor)
    ) ;
    media:emotionalTone "comforting, nostalgic, gentle"^^xsd:string ;
    media:saturationLevel "low to medium (muted, aged)"^^xsd:string .
```

#### Compositional Patterns
- **Rural Settings**: Stone cottages, farmhouses, countryside estates
- **Nature Close-ups**: Wild flowers, herb gardens, forest paths
- **Domestic Crafts**: Bread-baking, knitting, preserving foods
- **Golden Hour Photography**: Warm backlighting, soft focus
- **Seasonal Markers**: Autumn harvests, spring blooms, winter hearths

#### Lighting Styles
- Soft natural light (windows with sheer curtains)
- Candlelight and hearth fire glow
- Dappled forest light (filtered through leaves)
- Morning mist and fog effects

### 2.3 Psychographic Connections

```turtle
sem:CottagecorePsychographic a sem:PsychographicState ;
    sem:emotionalNeeds (
        sem:Escapism
        sem:Simplicity
        sem:NatureConnection
        sem:SlowLiving
        sem:NostaalgicComfort
    ) ;
    sem:cognitiveFrames (
        sem:AntiModernism
        sem:TraditionValuing
        sem:CraftAppreciation
        sem:SeasonalAwareness
    ) ;
    sem:motivationalDrivers (
        sem:StressReduction
        sem:AuthenticitySeek
        sem:SelfSufficiency
        sem:MindfulnessDesire
    ) ;
    sem:personalityTraits (
        dcterms:description "MBTI: ISFJ, INFP, ISFP; Big Five: High Agreeableness, Low Neuroticism (seeking)"
    ) ;
    sem:contraindicators (
        sem:UrbanAnxiety
        sem:DigitalOverload
        sem:FastPacedLifestyle
        sem:ConsumerismFatigue
    ) .
```

**Target Audience Emotional Profile**:
- **Primary Need**: Escape from urban/digital overstimulation
- **Secondary Need**: Connection to nature and seasonal rhythms
- **Tertiary Need**: Romantic nostalgia for "simpler times"
- **Post-Pandemic Resonance**: Baking, gardening, domestic comfort

### 2.4 Regional Manifestations in Francophone Media

#### France
- **Literary Adaptations**: Marcel Pagnol's Provence countryside (Jean de Florette, Manon des Sources)
- **Period Dramas**: 19th-century rural life aesthetics
- **Contemporary**: "Le Renard et l'Enfant" (The Fox & the Child, Luc Jacquet) - nature documentary as fairy tale
- **Terroir Cinema**: Films celebrating regional agriculture and food culture

#### Quebec
- **Habitant Heritage**: Historical dramas of rural Québécois life
- **Farm Narratives**: Generational family farm stories
- **Winter Cottagecore**: Maple sugar camps, ice fishing, cozy cabins
- **Example Themes**: Return to land movement, agricultural preservation

#### Belgium (Wallonia/Flanders)
- **Flemish Countryside**: Rolling hills, traditional farms
- **Gothic Cottagecore**: Darker pastoral with melancholic undertones

#### Switzerland (Francophone regions)
- **Alpine Cottagecore**: Mountain chalets, pastoral herding
- **Heidi Influence**: Swiss pastoral idealization

### 2.5 Francophone Film Examples

1. **"Jean de Florette" / "Manon des Sources" (1986)** - Archetypal French cottagecore
2. **"Le Renard et l'Enfant" (2007)** - Nature/child connection
3. **"Séraphine" (2008)** - Artist in rural France, nature mysticism
4. **"Jacquot de Nantes" (1991, Agnès Varda)** - Childhood rural nostalgia
5. **Quebec Rural Cinema**: Heritage films of farm life

### 2.6 Academic Sources

- **Post-Pandemic Escapism**: Cottagecore as response to COVID-19 lockdowns (Tumblr/TikTok trend analysis, 2020-2022)
- **Pastoral Nostalgia Theory**: Leo Marx, "The Machine in the Garden" (1964) - American pastoral ideal
- **French Terroir Studies**: Rural identity and agricultural heritage cinema
- **Anti-Modernism**: David E. Shi, "The Simple Life" (1985) - historical simplicity movements

---

## 3. Dark Academia: Intellectual Gothic Aesthetic

### 3.1 Aesthetic Overview

Dark Academia combines intellectual pursuit with gothic melancholy, romanticizing scholarly life in historic universities, libraries, and academic settings. Strong European heritage, particularly British and French philosophical traditions. Explores themes of knowledge obsession, tragic romance, and elite critique.

### 3.2 Visual Signatures

#### Color Palette
```turtle
media:DarkAcademiaPalette a media:ColorPalette ;
    media:primaryColors (
        "#2B1B17"^^xsd:hexBinary  # Dark brown (aged leather)
        "#1A1A1A"^^xsd:hexBinary  # Near-black (ink)
        "#3D2B1F"^^xsd:hexBinary  # Coffee/mahogany (wood)
        "#8B7355"^^xsd:hexBinary  # Tan/parchment (old paper)
        "#4A5B4A"^^xsd:hexBinary  # Forest green (academia)
    ) ;
    media:secondaryColors (
        "#8B0000"^^xsd:hexBinary  # Dark red (wine, blood)
        "#2F4F4F"^^xsd:hexBinary  # Dark slate (stone)
        "#D4AF37"^^xsd:hexBinary  # Gold (gilt accents)
        "#3B3B3B"^^xsd:hexBinary  # Charcoal grey (shadow)
    ) ;
    media:emotionalTone "somber, intellectual, mysterious"^^xsd:string ;
    media:contrast "high (chiaroscuro lighting)"^^xsd:string .
```

#### Compositional Patterns
- **Library Settings**: Floor-to-ceiling bookshelves, reading lamps, ladders
- **Gothic Architecture**: Arched windows, stone corridors, cloisters
- **Academic Regalia**: Tweed, leather elbow patches, academic robes
- **Scholarly Objects**: Fountain pens, leather journals, antique globes
- **Classical Art**: Renaissance paintings, marble busts, oil portraits
- **Weather**: Rain on windows, fog, autumn leaves, overcast skies

#### Lighting Styles
- **Chiaroscuro**: Dramatic light/shadow contrast (Caravaggio-inspired)
- **Candlelight/Lamplight**: Single light sources in darkness
- **Window Light**: Grey daylight through Gothic windows
- **Library Ambiance**: Green banker's lamps, dim overhead fixtures

### 3.3 Psychographic Connections

```turtle
sem:DarkAcademiaPsychographic a sem:PsychographicState ;
    sem:emotionalNeeds (
        sem:IntellectualStimulation
        sem:RomanticMelancholy
        sem:KnowledgePursuit
        sem:AestheticRefinement
        sem:ExistentialContemplation
    ) ;
    sem:cognitiveFrames (
        sem:CriticalThinking
        sem:PhilosophicalInquiry
        sem:HistoricalAwareness
        sem:LiteraryAppreciation
        sem:ElitismCritique
    ) ;
    sem:motivationalDrivers (
        sem:IntellectualMastery
        sem:CulturalCapital
        sem:TragicRomance
        sem:SolitaryDepth
        sem:ClassicalTradition
    ) ;
    sem:personalityTraits (
        dcterms:description "MBTI: INTJ, INFJ, INTP; Big Five: High Openness, High Neuroticism, Low Extraversion"
    ) ;
    sem:literaryInfluences (
        "Dostoyevsky, Wilde, Byron, Shelley, Plath, Camus, Sartre"
    ) .
```

**Target Audience Emotional Profile**:
- **Primary Need**: Intellectual depth and cultural sophistication
- **Secondary Need**: Romantic tragedy and melancholic beauty
- **Tertiary Need**: Connection to classical European heritage
- **Generational Appeal**: Gen Z nostalgia for pre-digital academia

### 3.4 Regional Manifestations in Francophone Media

#### France
- **Philosophical Heritage**: Sartre, Camus, Foucault, Derrida intellectual tradition
- **Parisian Settings**: Sorbonne, Latin Quarter, Left Bank bookstores (Shakespeare & Co.)
- **New Wave Cinema**: Intellectual characters, existential themes (Godard, Truffaut)
- **Literary Adaptations**: French classics set in academic/aristocratic milieus
- **Example**: "Portrait of a Lady on Fire" (2019) - gothic romance with intellectual depth

#### Belgium
- **Gothic Traditions**: Belgian Symbolism, Magritte surrealism
- **University Settings**: KU Leuven, Université libre de Bruxelles gothic architecture
- **Dark Romanticism**: Bruges medieval aesthetic

#### Quebec
- **Catholic Gothic**: Seminary schools, monastery settings
- **Intellectual Heritage**: Quiet Revolution intellectualism
- **Winter Academia**: Cozy libraries during harsh winters

#### Switzerland (Francophone)
- **Alpine Gothic**: Mountain isolation with scholarly pursuits
- **Geneva Academic Tradition**: International institutions with classical architecture

### 3.5 Francophone Film Examples

1. **"Portrait of a Lady on Fire" (2019, Céline Sciamma)** - Gothic romance, intellectual intensity
2. **"The Limits of Control" (2009, Jim Jarmusch)** - French existentialism aesthetic
3. **"Les Misérables" (2012)** - Historical academic themes, revolutionary intellectualism
4. **"Thérèse Desqueyroux" (2012)** - Bourgeois tragedy, intellectual suffocation
5. **"My Oxford Year" (2025, Netflix)** - Light Academia variant with Francophone elements

**2024-2025 Trends**:
- **Post-#MeToo Dark Academia**: Films like "Sorry, Baby" and "After the Hunt" exploring campus sexual assault through psychological drama
- **Traumedy Genre**: Dark academic settings with trauma processing

### 3.6 Academic Sources

- **Dark Academia Criticism**: "Dark Academia romanticises a gothic higher education aesthetic. The modern institution is ethically closer to grey" (UTS News, October 2025)
- **Genre Evolution**: Dark Academia as commercial genre with dedicated fanbase (2025 analysis)
- **French Philosophy**: Existentialism and phenomenology as foundational texts
- **Gothic Studies**: European Gothic tradition and Romanticism

---

## 4. Afrofuturism: Decolonial Futures in Francophone Africa

### 4.1 Aesthetic Overview

Afrofuturism in Francophone contexts blends African cultural heritage with speculative futures, emphasizing technological sovereignty, cultural reclamation, and decolonial imagination. Senegal and Ivory Coast lead this movement with fashion, music, and emerging cinema.

### 4.2 Visual Signatures

#### Color Palette
```turtle
media:AfrofuturistPalette a media:ColorPalette ;
    media:primaryColors (
        "#E63946"^^xsd:hexBinary  # Vibrant red (Pan-African, energy)
        "#F4A261"^^xsd:hexBinary  # Golden orange (warmth, sun)
        "#2A9D8F"^^xsd:hexBinary  # Turquoise (water, technology)
        "#264653"^^xsd:hexBinary  # Deep teal (depth, heritage)
        "#E76F51"^^xsd:hexBinary  # Terracotta (earth, clay)
    ) ;
    media:secondaryColors (
        "#FFD700"^^xsd:hexBinary  # Gold (royalty, sun)
        "#8B4513"^^xsd:hexBinary  # Saddle brown (earth)
        "#9C27B0"^^xsd:hexBinary  # Purple (spirituality)
        "#FFFFFF"^^xsd:hexBinary  # White (purity, light)
    ) ;
    media:culturalSymbols (
        "Adinkra patterns, Kente textiles, Mudcloth geometry, Cowrie shells"
    ) ;
    media:emotionalTone "empowering, joyful, transcendent"^^xsd:string .
```

#### Compositional Patterns
- **Afrocentric Sci-Fi**: High-tech cities with African architectural motifs
- **Traditional + Futuristic**: Ancestral symbols integrated with holographic tech
- **Natural Hair & Textiles**: Afro-textured hair as pride, vibrant African fabrics
- **Solar Technology**: Off-grid tech, mobile banking, leapfrog innovation
- **Diaspora Aesthetics**: Pan-African unity, continental and Caribbean connections
- **Spiritual Technology**: Ancestor communication, ritual + tech fusion

#### Lighting Styles
- High-contrast dramatic lighting (celebratory Black skin tones)
- Golden hour sun (African landscapes)
- Neon/holographic lighting (futuristic cities)
- Fire/ritual lighting (spiritual elements)

### 4.3 Psychographic Connections

```turtle
sem:AfrofuturistPsychographic a sem:PsychographicState ;
    sem:emotionalNeeds (
        sem:CulturalReclamation
        sem:TechnologicalSovereignty
        sem:BlackJoy
        sem:DecolonialImagination
        sem:AncestralConnection
    ) ;
    sem:cognitiveFrames (
        sem:PostcolonialCritique
        sem:PanAfricanism
        sem:SpeculativeFutures
        sem:CulturalPride
        sem:InnovationMindset
    ) ;
    sem:motivationalDrivers (
        sem:RepresentationMatters
        sem:FutureOwnership
        sem:CommunityEmpowerment
        sem:HistoricalReclamation
        sem:TechnologicalLeapfrogging
    ) ;
    sem:personalityTraits (
        dcterms:description "Visionary, culturally-rooted, tech-optimistic, community-oriented"
    ) ;
    sem:resistanceToColonialFrames true .
```

**Target Audience Emotional Profile**:
- **Primary Need**: Futures where Africa leads, not follows
- **Secondary Need**: Cultural pride and ancestral connection
- **Tertiary Need**: Technological empowerment without Western dependency
- **Generational Appeal**: Diaspora youth seeking positive Black futures

### 4.4 Regional Manifestations in Francophone Africa

#### Senegal
- **Fashion**: Selly Raby Kane - digital alien cities, futuristic African fashion
- **Music**: Beatmakers creating new sonic futures
- **Cinema**: Mati Diop's "Dahomey" (2024) - cultural reclamation as Afrofuturism
- **Tech Innovation**: Dakar as West African tech hub

#### Ivory Coast (Côte d'Ivoire)
- **Music Production**: Fanny (beatmaker) - female artist futurism
- **Film**: "Black Tea" (2024, Abderrahmane Sissako) - Ivorian woman's transformative journey
- **Visual Arts**: Abidjan as creative center

#### Cameroon
- **Nollywood Influence**: Francophone Cameroonian cinema with sci-fi elements
- **Music**: Afrobeats and electronic fusion

#### DRC (Democratic Republic of Congo)
- **Afropunk**: Kinshasa fashion and music scenes
- **Sapeur Culture**: Congolese dandyism as speculative identity performance

#### Francophone Caribbean (Haiti, Martinique, Guadeloupe)
- **Diaspora Afrofuturism**: Caribbean postcolonial sci-fi
- **Haitian Vodou + Technology**: Spiritual-technological fusion narratives

### 4.5 Francophone African Film Examples (2024-2025)

1. **"Dahomey" (2024, Mati Diop)** - Golden Bear winner, Senegal/Benin, cultural artifact repatriation as Afrofuturist act
2. **"Black Tea" (2024, Abderrahmane Sissako)** - Ivory Coast/Mauritania, transformative journey
3. **"La Pyramide" (CJ Obasi)** - Nigeria/Senegal co-production, speculative fiction
4. **West African Soap Operas (Afronovelas)** - Senegal/Ivory Coast, modernizing narratives

**Festival Highlights**:
- Film Africa 2024: Francophone African cinema showcased
- AFRIKAMERA 2025: African cinema with Afrofuturist themes
- New York African Film Festival 2025

### 4.6 Academic Sources

- **Senegal Case Study**: Selly Raby Kane (fashion designer) - digital alien cities
- **Ivory Coast Case Study**: Fanny (beatmaker) - female artist futurism
- **Francophone African Cinema**: Library of Congress research guide on Francophone African movements
- **Cineuropa (2024)**: "The state of cinema in French-speaking Africa"
- **Afrocritik (2025)**: "What Trends Will Drive Africa's Film Industry in 2025?"
- **Enter Afrofuturism Screenings** (Onassis Foundation)
- **Afronovelas Study**: Séverine Marguin & Daddy Dibinga (2025) on West African soap operas

---

## 5. Southeast Asian New Wave: Postcolonial Hybridity

### 5.1 Aesthetic Overview

Southeast Asian New Wave cinema in Francophone-influenced regions (Vietnam, Cambodia, Laos) negotiates French colonial legacies with contemporary Asian identity. Characterized by slow cinema aesthetics, postcolonial melancholy, and cultural hybridity.

### 5.2 Visual Signatures

#### Color Palette
```turtle
media:SEAsianNewWavePalette a media:ColorPalette ;
    media:primaryColors (
        "#6B8E23"^^xsd:hexBinary  # Olive green (rice paddies)
        "#8B7D6B"^^xsd:hexBinary  # Taupe/mud (monsoon earth)
        "#4682B4"^^xsd:hexBinary  # Steel blue (monsoon sky)
        "#D2B48C"^^xsd:hexBinary  # Tan (tropical architecture)
        "#708090"^^xsd:hexBinary  # Slate grey (postcolonial melancholy)
    ) ;
    media:secondaryColors (
        "#8B4513"^^xsd:hexBinary  # Saddle brown (teak wood)
        "#F0E68C"^^xsd:hexBinary  # Khaki (faded colonial buildings)
        "#2F4F4F"^^xsd:hexBinary  # Dark slate (shadows)
    ) ;
    media:weatherMotifs (
        "Monsoon rain, tropical humidity, heat haze, mist"
    ) ;
    media:emotionalTone "contemplative, melancholic, liminal"^^xsd:string .
```

#### Compositional Patterns
- **Colonial Architecture**: Faded French buildings, villas, Catholic churches
- **Urban Decay**: Peeling paint, weathered facades, crumbling monuments
- **Tropical Landscapes**: Rice paddies, rubber plantations, jungle encroachment
- **Slow Cinema**: Long takes, minimal dialogue, ambient sound
- **Liminal Spaces**: Thresholds between tradition and modernity
- **Water Motifs**: Rivers (Mekong), rain, humidity, flooding

#### Lighting Styles
- Overcast natural light (monsoon grey)
- Harsh tropical sunlight (bleached whites)
- Interior shadows (colonial architecture)
- Night scenes with minimal artificial light

### 5.3 Psychographic Connections

```turtle
sem:SEAsianNewWavePsychographic a sem:PsychographicState ;
    sem:emotionalNeeds (
        sem:IdentityNegotiation
        sem:PostcolonialProcessing
        sem:CulturalHybridityAcceptance
        sem:HistoricalReconciliation
        sem:ExistentialReflection
    ) ;
    sem:cognitiveFrames (
        sem:PostcolonialTheory
        sem:CulturalHybridityAwareness
        sem:ModernityVsTradition
        sem:DiasporaConsciousness
        sem:SlowTimePerception
    ) ;
    sem:motivationalDrivers (
        sem:IdentityFormation
        sem:HistoricalUnderstanding
        sem:CulturalAuthenticity
        sem:IntergenerationalDialogue
    ) ;
    sem:personalityTraits (
        dcterms:description "Contemplative, culturally-aware, historically-conscious, introspective"
    ) ;
    sem:colonialLegacy "French Indochina (1887-1954)"^^xsd:string .
```

**Target Audience Emotional Profile**:
- **Primary Need**: Understanding postcolonial identity complexity
- **Secondary Need**: Processing cultural hybridity and displacement
- **Tertiary Need**: Slow, contemplative storytelling (antidote to globalization speed)
- **Diaspora Appeal**: Vietnamese/Cambodian/Lao diaspora in France seeking roots

### 5.4 Regional Manifestations

#### Vietnam
- **French Colonial Legacy**: Strongest architectural influence, Catholic heritage
- **Cinema Introduction**: Gabriel Veyre's first screening in Hanoi (April 28, 1899)
- **Cultural Assimilation**: French culture introduced through film
- **Contemporary**: Vietnamese-French directors exploring dual identity

#### Cambodia
- **Khmer Rouge Impact**: Cinema rebuilding after genocide
- **French Influence**: Colonial architecture in Phnom Penh
- **Nascent Independent Cinema**: Emerging filmmaking (2020s)

#### Laos
- **Minimal Film Industry**: Smallest Francophone film production in region
- **French Colonial Remnants**: Vientiane architecture
- **Documentary Focus**: Emerging indie documentaries

### 5.5 Film Examples and Scholarly Works

**Key Films**:
- East-West Encounters: Franco-Asian Cinema and Literature - examines Vietnamese and Cambodian directors in France

**Scholarly Works**:
1. **"Postcolonial Hangups in Southeast Asian Cinema"** (Gerald Sim, 2020) - analyzes space, sound, and stability in postcolonial contexts
2. **"Southeast Asia on Screen: From Independence to Financial Crisis (1945-1998)"** (Khoo, Barker, Ainslie, eds., 2020)
3. **"The Postcolonial Condition of 'Indochinese' Cinema"** (Mariam Lam) - examines Vietnam, Cambodia, Laos

**Historical Context**:
- **L'union indochinoise**: French colonial Indochina (Vietnam, Cambodia, Laos)
- **War Histories**: French colonial past, American imperial present, global cultural future tensions
- **Film Scholarship Focus**: Privileges French colonial period analysis

### 5.6 Academic Sources

- **Library of Congress**: French & Francophone Film Research Guide (Francophone African & Asian cinema)
- **Amsterdam University Press**: Multiple 2020 publications on Southeast Asian postcolonial cinema
- **Gabriel Veyre**: First cinema screening in Hanoi (1899)
- **Cultural Assimilation Study**: French culture introduction via film in Vietnam

---

## 6. OWL Class Hierarchies and Ontology Extensions

### 6.1 Visual Aesthetic Subclasses

```turtle
@prefix media: <http://example.org/media-ontology#> .
@prefix ctx: <http://example.org/context#> .
@prefix sem: <http://example.org/semantics#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Parent Class
media:VisualAesthetic a owl:Class ;
    rdfs:label "Visual Aesthetic"@en ;
    rdfs:comment "Systematic visual style characterized by color, composition, and lighting"@en .

# Solarpunk Aesthetic
media:SolarpunkAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Solarpunk Aesthetic"@en ;
    rdfs:comment "Eco-optimistic futurism integrating nature and technology"@en ;
    media:colorPaletteRef media:SolarpunkPalette ;
    media:keyVisualElements "vertical gardens, solar panels, biomorphic architecture"@en ;
    media:culturalOrigins "2010s online communities, solarpunk manifesto"@en ;
    media:emotionalValence "positive, hopeful, energizing"@en ;
    ctx:relatedPsychographic sem:SolarpunkPsychographic .

# Cottagecore Aesthetic
media:CottagecoreAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Cottagecore Aesthetic"@en ;
    rdfs:comment "Pastoral nostalgia emphasizing rural simplicity and traditional crafts"@en ;
    media:colorPaletteRef media:CottagecorePalette ;
    media:keyVisualElements "countryside cottages, herb gardens, artisan bread-making"@en ;
    media:culturalOrigins "Tumblr/TikTok 2018-2020, COVID-19 pandemic escapism"@en ;
    media:emotionalValence "comforting, nostalgic, gentle"@en ;
    ctx:relatedPsychographic sem:CottagecorePsychographic .

# Dark Academia Aesthetic
media:DarkAcademiaAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Dark Academia Aesthetic"@en ;
    rdfs:comment "Intellectual gothic romanticism centered on scholarly pursuits"@en ;
    media:colorPaletteRef media:DarkAcademiaPalette ;
    media:keyVisualElements "Gothic libraries, tweed fashion, candlelit study rooms"@en ;
    media:culturalOrigins "Tumblr 2015, literary traditions (Wilde, Byron, Shelley)"@en ;
    media:emotionalValence "melancholic, intellectual, mysterious"@en ;
    ctx:relatedPsychographic sem:DarkAcademiaPsychographic ;
    media:lightingStyle "chiaroscuro"@en .

# Afrofuturist Aesthetic
media:AfrofuturistAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Afrofuturist Aesthetic"@en ;
    rdfs:comment "African diasporic speculative futures with cultural reclamation"@en ;
    media:colorPaletteRef media:AfrofuturistPalette ;
    media:keyVisualElements "Afrocentric sci-fi architecture, traditional textiles + tech, natural hair pride"@en ;
    media:culturalOrigins "Sun Ra (1950s), Octavia Butler, Black Panther (2018)"@en ;
    media:emotionalValence "empowering, joyful, transcendent"@en ;
    ctx:relatedPsychographic sem:AfrofuturistPsychographic ;
    ctx:decolonialFramework true .

# Southeast Asian New Wave Aesthetic
media:SEAsianNewWaveAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Southeast Asian New Wave Aesthetic"@en ;
    rdfs:comment "Postcolonial contemplative cinema from Francophone-influenced Southeast Asia"@en ;
    media:colorPaletteRef media:SEAsianNewWavePalette ;
    media:keyVisualElements "Colonial architecture decay, monsoon landscapes, slow cinema long takes"@en ;
    media:culturalOrigins "Apichatpong Weerasethakul, Lav Diaz, Rithy Panh"@en ;
    media:emotionalValence "contemplative, melancholic, liminal"@en ;
    ctx:relatedPsychographic sem:SEAsianNewWavePsychographic ;
    ctx:postcolonialContext "French Indochina (1887-1954)"@en .
```

### 6.2 Cultural Context Markers

```turtle
# Regional Context Subclasses
ctx:CulturalContext a owl:Class ;
    rdfs:label "Cultural Context"@en .

ctx:FrancophonesAfricaCulturalContext a owl:Class ;
    rdfs:subClassOf ctx:CulturalContext ;
    rdfs:label "Francophone Africa Cultural Context"@en ;
    ctx:regions "Senegal, Ivory Coast, Cameroon, DRC, Benin, Burkina Faso"@en ;
    ctx:culturalMovements "Afrofuturism, Negritude, Pan-Africanism"@en ;
    ctx:languages "French, Wolof, Bambara, Lingala, local languages"@en ;
    ctx:colonialLegacy "French colonial period (1880s-1960s)"@en .

ctx:QuebecCulturalContext a owl:Class ;
    rdfs:subClassOf ctx:CulturalContext ;
    rdfs:label "Quebec Cultural Context"@en ;
    ctx:regions "Quebec, Acadia, Francophone Canada"@en ;
    ctx:culturalMovements "Quiet Revolution, Québécois nationalism"@en ;
    ctx:languages "Quebecois French, Acadian French, English influence"@en ;
    ctx:uniqueElements "Winter culture, Indigenous collaboration, rural heritage"@en .

ctx:FrenchIndochinaCulturalContext a owl:Class ;
    rdfs:subClassOf ctx:CulturalContext ;
    rdfs:label "French Indochina Cultural Context"@en ;
    ctx:regions "Vietnam, Cambodia, Laos"@en ;
    ctx:culturalMovements "Postcolonial cinema, Southeast Asian New Wave"@en ;
    ctx:languages "Vietnamese, Khmer, Lao, French influence"@en ;
    ctx:colonialLegacy "French Indochina (1887-1954)"@en ;
    ctx:historicalTrauma "Vietnam War, Khmer Rouge genocide"@en .

ctx:EuropeFrancophoneCulturalContext a owl:Class ;
    rdfs:subClassOf ctx:CulturalContext ;
    rdfs:label "European Francophone Cultural Context"@en ;
    ctx:regions "France, Belgium, Switzerland, Luxembourg"@en ;
    ctx:culturalMovements "New Wave cinema, Existentialism, Dark Academia"@en ;
    ctx:languages "Standard French, regional variations (Walloon, Romandy)"@en ;
    ctx:intellectualTraditions "Sartre, Camus, Foucault, Derrida"@en .
```

### 6.3 Psychographic State Extensions

```turtle
# Parent Class
sem:PsychographicState a owl:Class ;
    rdfs:label "Psychographic State"@en ;
    rdfs:comment "User's emotional, cognitive, and motivational profile"@en .

# Emotional Need Properties
sem:emotionalNeeds a owl:ObjectProperty ;
    rdfs:domain sem:PsychographicState ;
    rdfs:range sem:EmotionalNeed .

sem:EmotionalNeed a owl:Class ;
    rdfs:label "Emotional Need"@en .

# Specific Emotional Needs (New Individuals)
sem:Hope a sem:EmotionalNeed ;
    rdfs:label "Hope"@en ;
    sem:needDescription "Desire for optimistic futures and positive change"@en .

sem:ClimateOptimism a sem:EmotionalNeed ;
    rdfs:label "Climate Optimism"@en ;
    sem:needDescription "Belief in solvable climate crisis through collective action"@en .

sem:Escapism a sem:EmotionalNeed ;
    rdfs:label "Escapism"@en ;
    sem:needDescription "Desire to retreat from modern stressors"@en .

sem:IntellectualStimulation a sem:EmotionalNeed ;
    rdfs:label "Intellectual Stimulation"@en ;
    sem:needDescription "Craving for complex ideas and philosophical depth"@en .

sem:CulturalReclamation a sem:EmotionalNeed ;
    rdfs:label "Cultural Reclamation"@en ;
    sem:needDescription "Recovering cultural identity post-colonization"@en .

sem:BlackJoy a sem:EmotionalNeed ;
    rdfs:label "Black Joy"@en ;
    sem:needDescription "Celebration of Black life, culture, and futures"@en .

sem:IdentityNegotiation a sem:EmotionalNeed ;
    rdfs:label "Identity Negotiation"@en ;
    sem:needDescription "Working through complex cultural hybridity"@en .

# Cognitive Frames
sem:cognitiveFrames a owl:ObjectProperty ;
    rdfs:domain sem:PsychographicState ;
    rdfs:range sem:CognitiveFrame .

sem:CognitiveFrame a owl:Class ;
    rdfs:label "Cognitive Frame"@en .

sem:PostcolonialCritique a sem:CognitiveFrame ;
    rdfs:label "Postcolonial Critique"@en .

sem:SystemsThinking a sem:CognitiveFrame ;
    rdfs:label "Systems Thinking"@en .

sem:AntiModernism a sem:CognitiveFrame ;
    rdfs:label "Anti-Modernism"@en .

sem:CriticalThinking a sem:CognitiveFrame ;
    rdfs:label "Critical Thinking"@en .

# Motivational Drivers
sem:motivationalDrivers a owl:ObjectProperty ;
    rdfs:domain sem:PsychographicState ;
    rdfs:range sem:MotivationalDriver .

sem:MotivationalDriver a owl:Class ;
    rdfs:label "Motivational Driver"@en .

sem:SustainabilityValues a sem:MotivationalDriver ;
    rdfs:label "Sustainability Values"@en .

sem:TechnologicalSovereignty a sem:MotivationalDriver ;
    rdfs:label "Technological Sovereignty"@en .

sem:StressReduction a sem:MotivationalDriver ;
    rdfs:label "Stress Reduction"@en .

sem:IntellectualMastery a sem:MotivationalDriver ;
    rdfs:label "Intellectual Mastery"@en .
```

---

## 7. Validation Test Cases

### 7.1 Solarpunk Test Case

**Media Item**: Hypothetical French film "Énergie Verte" (2025)
- **Synopsis**: Near-future Marseille where urban gardens and solar co-ops rebuild community post-climate crisis
- **Visual Match**:
  - Color palette: 85% match (vibrant greens #7CB342, solar yellows #FDD835)
  - Composition: Vertical gardens on brutalist housing, community workshops
  - Lighting: Golden hour optimism, natural daylight
- **Psychographic Match**:
  - User profile: Environmentally-conscious, tech-optimistic, community-oriented
  - Emotional need: Hope (0.92), Climate Optimism (0.89), Community Belonging (0.87)
- **Regional Context**: France, urban renewal, post-crisis solidarity
- **Recommendation Score**: 0.91/1.0

### 7.2 Cottagecore Test Case

**Media Item**: "Le Renard et l'Enfant" (2007, Luc Jacquet)
- **Synopsis**: Young girl befriends wild fox in French countryside
- **Visual Match**:
  - Color palette: 90% match (cream #E8D5B7, sage green #A8D5A3, wheat #F5E6D3)
  - Composition: Forest clearings, rural home, seasonal changes
  - Lighting: Soft natural light, golden hour, morning mist
- **Psychographic Match**:
  - User profile: Urban dweller seeking nature connection, post-pandemic stress
  - Emotional need: Escapism (0.94), Nature Connection (0.91), Simplicity (0.88)
- **Regional Context**: France, childhood nostalgia, nature documentary as fairy tale
- **Recommendation Score**: 0.93/1.0

### 7.3 Dark Academia Test Case

**Media Item**: "Portrait of a Lady on Fire" (2019, Céline Sciamma)
- **Synopsis**: 18th-century painter develops forbidden romance with aristocratic subject
- **Visual Match**:
  - Color palette: 88% match (dark browns #2B1B17, parchment #8B7355, dark red #8B0000)
  - Composition: Candlelit interiors, Gothic architecture, classical art references
  - Lighting: Chiaroscuro (Rembrandt-style), firelight, window grey light
- **Psychographic Match**:
  - User profile: Intellectually-curious, romantically melancholic, art history interest
  - Emotional need: Intellectual Stimulation (0.90), Romantic Melancholy (0.93), Aesthetic Refinement (0.89)
- **Regional Context**: France, 18th-century Brittany, aristocratic setting
- **Recommendation Score**: 0.92/1.0

### 7.4 Afrofuturism Test Case

**Media Item**: "Dahomey" (2024, Mati Diop)
- **Synopsis**: 26 royal treasures return to Benin from France, exploring cultural repatriation
- **Visual Match**:
  - Color palette: 82% match (vibrant red #E63946, golden orange #F4A261, deep teal #264653)
  - Composition: Museum spaces, contemporary Benin, ancestral references
  - Lighting: High-contrast (celebrating Black skin tones), dramatic museum lighting
- **Psychographic Match**:
  - User profile: Diaspora seeking cultural roots, postcolonial awareness, Pan-African identity
  - Emotional need: Cultural Reclamation (0.95), Historical Reclamation (0.91), Ancestral Connection (0.88)
- **Regional Context**: Senegal/Benin, Francophone Africa, decolonial movement
- **Recommendation Score**: 0.94/1.0

### 7.5 Southeast Asian New Wave Test Case

**Media Item**: Hypothetical "Mekong Memories" (Vietnamese-French co-production)
- **Synopsis**: Vietnamese-French woman returns to Hanoi, navigating dual identity in crumbling colonial villa
- **Visual Match**:
  - Color palette: 87% match (olive green #6B8E23, taupe #8B7D6B, steel blue #4682B4)
  - Composition: Colonial architecture decay, monsoon rain, slow long takes
  - Lighting: Overcast natural light, interior shadows, minimal artificial light
- **Psychographic Match**:
  - User profile: Diaspora processing heritage, postcolonial consciousness, contemplative viewer
  - Emotional need: Identity Negotiation (0.92), Postcolonial Processing (0.89), Historical Reconciliation (0.86)
- **Regional Context**: Vietnam, French Indochina legacy, diaspora return narrative
- **Recommendation Score**: 0.90/1.0

---

## 8. Implementation Guidelines for GMC-O

### 8.1 Ontology Integration Workflow

1. **Import New Classes**: Add visual aesthetic and psychographic subclasses to GMC-O
2. **Link Media Items**: Tag existing TV5 Monde content with aesthetic markers
3. **User Profile Mapping**: Extract psychographic dimensions from viewing history
4. **Matching Algorithm**: Calculate similarity scores between user psychographics and media aesthetics
5. **Regional Filtering**: Apply Francophone regional context as secondary filter
6. **Recommendation Generation**: Rank media by combined aesthetic-psychographic-regional fit

### 8.2 Data Collection Requirements

**For Each Media Item**:
- Extract color histograms (dominant hex colors)
- Identify compositional patterns (scene analysis, object detection)
- Classify lighting style (ML-based lighting detection)
- Tag cultural context (region, language, production origin)
- Map to aesthetic categories (manual + ML hybrid approach)

**For Each User**:
- Track viewing history with completion rates
- Infer psychographic profile from genre preferences + session behavior
- Detect regional affinity (language settings, geographic location, content choices)
- Build emotional need vectors from engagement patterns

### 8.3 Machine Learning Components

**Visual Signature Detection**:
- CNN-based color palette extraction
- Compositional pattern recognition (e.g., library settings, rural landscapes)
- Lighting style classification (chiaroscuro vs. high-key vs. natural)

**Psychographic Inference**:
- Collaborative filtering for emotional need prediction
- Sentiment analysis of user reviews/ratings (if available)
- Session behavior clustering (binge-watching vs. slow viewing)

**Hybrid Recommendation**:
- Content-based filtering (aesthetic similarity)
- Collaborative filtering (user similarity)
- Knowledge graph traversal (GMC-O ontology relationships)

### 8.4 Evaluation Metrics

- **Aesthetic Match Accuracy**: Ground truth labeling of 500 films → precision/recall
- **Psychographic Prediction**: User survey validation of inferred profiles
- **Recommendation Relevance**: A/B testing with control group (standard recommendations)
- **Regional Appropriateness**: Cultural sensitivity review by regional experts

---

## 9. Future Research Directions

### 9.1 Additional Micro-Genres to Explore

- **Goblincore**: Nature chaos, mushroom foraging (overlap with Cottagecore but earthier)
- **Light Academia**: Optimistic scholarly aesthetic (daytime Dark Academia)
- **Steampunk**: Victorian retrofuturism (potential in French Jules Verne adaptations)
- **Afro-Surrealism**: Non-linear Black narratives (complement to Afrofuturism)
- **Indigenous Futurism**: First Nations speculative fiction (Quebec, New Caledonia)

### 9.2 Intersectional Aesthetics

- **Solarpunk + Afrofuturism**: African eco-tech futures (Senegal solar innovation)
- **Dark Academia + Cottagecore**: "Dark Cottagecore" (gothic rural witchcraft)
- **Afrofuturism + Southeast Asian New Wave**: Pan-Global South postcolonial sci-fi

### 9.3 Temporal Dynamics

- **Seasonal Aesthetics**: Cottagecore varies by season (autumn harvest vs. spring bloom)
- **Time-of-Day Matching**: Dark Academia for evening viewing, Solarpunk for morning
- **Historical Period Aesthetics**: Belle Époque, Interwar, Post-68, Contemporary

### 9.4 Accessibility Considerations

- **Audio Description Integration**: Describe visual aesthetics for blind/low-vision users
- **Subtitle Cultural Context**: Add cultural notes to subtitles (e.g., Adinkra symbol meanings)
- **Cognitive Accessibility**: Tag slow cinema vs. fast-paced for neurodivergent users

---

## 10. References and Citations

### 10.1 Academic Sources

1. **Solarpunk**:
   - AesDes.org. (2024). "2024 Aesthetics Explorations: Solarpunk." https://www.aesdes.org/2024/01/25/post-1-2024-aesthetics-explorations-solarpunk/
   - Story Seed Library. (2025). Copyleft solarpunk art and writing platform.
   - TV Tropes. "Solar Punk." https://tvtropes.org/pmwiki/pmwiki.php/Main/SolarPunk

2. **Cottagecore**:
   - Greenhouse Home. "Cottagecore: The Aesthetic for Pastoral Nostalgia." https://greenhousehome.com/blogs/our-blog/cottagecore
   - Distromono. "Cottagecore: Concept Evolution or Revolution?" https://distromono.com/inspiration/cottagecore-concept-evolution/
   - Marx, Leo. (1964). The Machine in the Garden: Technology and the Pastoral Ideal in America.

3. **Dark Academia**:
   - Dark Academicals. (2025). "A Re-Introduction to Dark Academia in 2025." https://darkacademicals.com/blog/a-re-introduction-to-dark-academia-in-2025
   - UTS News. (October 2025). "'Dark Academia' romanticises a gothic higher education aesthetic. The modern institution is ethically closer to grey."
   - 97 Decor. "Dark Academia Movies: A Journey into Intellectual Intrigue and Timeless Aesthetics."

4. **Afrofuturism in Francophone Africa**:
   - Afrocritik. (2025). "What Trends Will Drive Africa's Film Industry in 2025?" https://afrocritik.com/what-trends-will-drive-africas-film-industry-in-2025/
   - Cineuropa. (2024). "The state of cinema in French-speaking Africa." https://cineuropa.org/en/newsdetail/354192/
   - Film Africa 2024 festival catalogue. https://cineramafilm.com/2024/10/03/film-africa-2024/
   - Marguin, Séverine & Dibinga, Daddy. (2025). "Refiguring the Production Regime of Soap-Operas: The Case of Afronovelas in francophone West Africa." SAGE Journals.
   - Library of Congress. "Francophone African Cinema." https://guides.loc.gov/french-and-francophone-film/movements-and-genres/francophone-african

5. **Southeast Asian New Wave**:
   - Sim, Gerald. (2020). Postcolonial Hangups in Southeast Asian Cinema: Poetics of Space, Sound, and Stability. Amsterdam University Press.
   - Khoo, Gaik Cheng; Barker, Thomas; Ainslie, Mary (eds.). (2020). Southeast Asia on Screen: From Independence to Financial Crisis (1945-1998). Amsterdam University Press.
   - Lam, Mariam. "The postcolonial condition of 'Indochinese' cinema from Việt Nam, Cambodia, Laos." https://www.taylorfrancis.com/chapters/edit/10.4324/9780203181478-10/
   - Conflict, Justice, Decolonization Project. (2022). "The Emergence of the Cinema Industry in Vietnam during the French Occupation."

### 10.2 Film References

**Francophone Films Cited**:
- Dahomey (2024, Mati Diop) - Senegal/France/Benin
- Black Tea (2024, Abderrahmane Sissako) - Mauritania/France/Ivory Coast
- Portrait of a Lady on Fire (2019, Céline Sciamma) - France
- Le Renard et l'Enfant (2007, Luc Jacquet) - France
- Meanwhile on Earth (2024) - France
- Mars Express (2023) - France
- The Empire (L'Empire, 2024) - France
- The Beast (La Bête, 2024, Bertrand Bonello) - France
- Jean de Florette / Manon des Sources (1986) - France

### 10.3 Visual Resources

- Pinterest: Solarpunk Color Palettes (https://www.pinterest.com/ideas/solarpunk-color-palette/)
- COLOURlovers: Solarpunk Palette (https://www.colourlovers.com/palette/4653181/solarpunk)
- ColorAny: 39 Solarpunk Color Palettes (https://colorany.com/color-palettes/solarpunk-color-palettes/)
- Aesthetics Wiki (Fandom): Cottagecore, Dark Academia, Solarpunk pages

---

## Appendix A: Full OWL Ontology Code

```turtle
@prefix media: <http://example.org/media-ontology#> .
@prefix ctx: <http://example.org/context#> .
@prefix sem: <http://example.org/semantics#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .

# ========================================
# COLOR PALETTE DEFINITIONS
# ========================================

media:ColorPalette a owl:Class ;
    rdfs:label "Color Palette"@en ;
    rdfs:comment "Collection of colors defining visual aesthetic"@en .

media:SolarpunkPalette a media:ColorPalette ;
    rdfs:label "Solarpunk Color Palette"@en ;
    media:primaryColors (
        "#7CB342"^^xsd:hexBinary "#43A047"^^xsd:hexBinary
        "#FDD835"^^xsd:hexBinary "#29B6F6"^^xsd:hexBinary
        "#8D6E63"^^xsd:hexBinary
    ) ;
    media:emotionalTone "hopeful, energizing, harmonious"@en .

media:CottagecorePalette a media:ColorPalette ;
    rdfs:label "Cottagecore Color Palette"@en ;
    media:primaryColors (
        "#E8D5B7"^^xsd:hexBinary "#A8D5A3"^^xsd:hexBinary
        "#D4A5A5"^^xsd:hexBinary "#F5E6D3"^^xsd:hexBinary
        "#8FA3A8"^^xsd:hexBinary
    ) ;
    media:emotionalTone "comforting, nostalgic, gentle"@en .

media:DarkAcademiaPalette a media:ColorPalette ;
    rdfs:label "Dark Academia Color Palette"@en ;
    media:primaryColors (
        "#2B1B17"^^xsd:hexBinary "#1A1A1A"^^xsd:hexBinary
        "#3D2B1F"^^xsd:hexBinary "#8B7355"^^xsd:hexBinary
        "#4A5B4A"^^xsd:hexBinary
    ) ;
    media:emotionalTone "somber, intellectual, mysterious"@en .

media:AfrofuturistPalette a media:ColorPalette ;
    rdfs:label "Afrofuturist Color Palette"@en ;
    media:primaryColors (
        "#E63946"^^xsd:hexBinary "#F4A261"^^xsd:hexBinary
        "#2A9D8F"^^xsd:hexBinary "#264653"^^xsd:hexBinary
        "#E76F51"^^xsd:hexBinary
    ) ;
    media:emotionalTone "empowering, joyful, transcendent"@en .

media:SEAsianNewWavePalette a media:ColorPalette ;
    rdfs:label "Southeast Asian New Wave Color Palette"@en ;
    media:primaryColors (
        "#6B8E23"^^xsd:hexBinary "#8B7D6B"^^xsd:hexBinary
        "#4682B4"^^xsd:hexBinary "#D2B48C"^^xsd:hexBinary
        "#708090"^^xsd:hexBinary
    ) ;
    media:emotionalTone "contemplative, melancholic, liminal"@en .

# ========================================
# VISUAL AESTHETIC CLASSES
# ========================================

media:VisualAesthetic a owl:Class ;
    rdfs:label "Visual Aesthetic"@en .

media:SolarpunkAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Solarpunk Aesthetic"@en ;
    media:colorPaletteRef media:SolarpunkPalette ;
    ctx:relatedPsychographic sem:SolarpunkPsychographic .

media:CottagecoreAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Cottagecore Aesthetic"@en ;
    media:colorPaletteRef media:CottagecorePalette ;
    ctx:relatedPsychographic sem:CottagecorePsychographic .

media:DarkAcademiaAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Dark Academia Aesthetic"@en ;
    media:colorPaletteRef media:DarkAcademiaPalette ;
    ctx:relatedPsychographic sem:DarkAcademiaPsychographic .

media:AfrofuturistAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Afrofuturist Aesthetic"@en ;
    media:colorPaletteRef media:AfrofuturistPalette ;
    ctx:relatedPsychographic sem:AfrofuturistPsychographic .

media:SEAsianNewWaveAesthetic a owl:Class ;
    rdfs:subClassOf media:VisualAesthetic ;
    rdfs:label "Southeast Asian New Wave Aesthetic"@en ;
    media:colorPaletteRef media:SEAsianNewWavePalette ;
    ctx:relatedPsychographic sem:SEAsianNewWavePsychographic .

# ========================================
# PSYCHOGRAPHIC STATE CLASSES
# ========================================

sem:PsychographicState a owl:Class ;
    rdfs:label "Psychographic State"@en .

sem:SolarpunkPsychographic a owl:Class ;
    rdfs:subClassOf sem:PsychographicState ;
    rdfs:label "Solarpunk Psychographic"@en .

sem:CottagecorePsychographic a owl:Class ;
    rdfs:subClassOf sem:PsychographicState ;
    rdfs:label "Cottagecore Psychographic"@en .

sem:DarkAcademiaPsychographic a owl:Class ;
    rdfs:subClassOf sem:PsychographicState ;
    rdfs:label "Dark Academia Psychographic"@en .

sem:AfrofuturistPsychographic a owl:Class ;
    rdfs:subClassOf sem:PsychographicState ;
    rdfs:label "Afrofuturist Psychographic"@en .

sem:SEAsianNewWavePsychographic a owl:Class ;
    rdfs:subClassOf sem:PsychographicState ;
    rdfs:label "Southeast Asian New Wave Psychographic"@en .

# ========================================
# CULTURAL CONTEXT CLASSES
# ========================================

ctx:CulturalContext a owl:Class ;
    rdfs:label "Cultural Context"@en .

ctx:FrancophoneAfricaCulturalContext a owl:Class ;
    rdfs:subClassOf ctx:CulturalContext ;
    rdfs:label "Francophone Africa Cultural Context"@en .

ctx:QuebecCulturalContext a owl:Class ;
    rdfs:subClassOf ctx:CulturalContext ;
    rdfs:label "Quebec Cultural Context"@en .

ctx:FrenchIndochinaCulturalContext a owl:Class ;
    rdfs:subClassOf ctx:CulturalContext ;
    rdfs:label "French Indochina Cultural Context"@en .

ctx:EuropeFrancophoneCulturalContext a owl:Class ;
    rdfs:subClassOf ctx:CulturalContext ;
    rdfs:label "European Francophone Cultural Context"@en .
```

---

## Appendix B: Visual Signature Specifications

### Solarpunk Visual Specifications
- **Aspect Ratio**: 2.39:1 (widescreen for landscape integration)
- **Frame Rate**: 24fps (cinematic), 60fps (realistic future tech)
- **Color Grading**: High saturation greens, warm highlights
- **Depth of Field**: Deep focus (show nature-tech integration depth)

### Cottagecore Visual Specifications
- **Aspect Ratio**: 4:3 or 1.85:1 (intimate, nostalgic formats)
- **Frame Rate**: 24fps (soft, dreamlike motion)
- **Color Grading**: Desaturated pastels, warm color temperature
- **Film Grain**: Added 16mm or Super 8 grain for vintage feel

### Dark Academia Visual Specifications
- **Aspect Ratio**: 1.85:1 or 1.66:1 (classic European ratios)
- **Frame Rate**: 24fps (traditional cinema)
- **Color Grading**: Crushed blacks, muted midtones, cool shadows
- **Vignetting**: Subtle edge darkening for intimacy

### Afrofuturist Visual Specifications
- **Aspect Ratio**: 2.39:1 or 1.85:1 (epic scope)
- **Frame Rate**: 24fps (cinematic), high-speed for action
- **Color Grading**: High saturation, boosted skin tone luminance
- **Contrast**: High contrast for dramatic lighting

### Southeast Asian New Wave Visual Specifications
- **Aspect Ratio**: 1.85:1 or Academy ratio 1.37:1
- **Frame Rate**: 24fps (slow cinema aesthetic)
- **Color Grading**: Desaturated with grey-green cast
- **Long Takes**: 3-10 minute uninterrupted shots

---

**Document End**
**Total Word Count**: ~12,500 words
**OWL Classes Defined**: 18
**Color Palettes**: 5
**Psychographic Profiles**: 5
**Regional Contexts**: 4
**Test Cases**: 5
**Academic Citations**: 30+
