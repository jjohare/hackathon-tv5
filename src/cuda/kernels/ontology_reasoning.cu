// CUDA Kernels for Media Ontology Reasoning - GMC-O (Global Media & Context Ontology)
// GPU-accelerated constraint enforcement for media content relationships and semantic reasoning
// Target: ~2ms per frame for 10K media nodes with complex relationships
//
// Use Cases:
// - Genre hierarchy enforcement (SciFi ⊑ Genre)
// - Disjoint content classes (Action ⊥ Documentary)
// - Content equivalence (SameAs for remakes, adaptations)
// - Mood/aesthetic consistency constraints
// - Cultural context alignment
// - Viewer preference relationship reasoning
//
// Performance: Optimized for real-time media discovery and recommendation systems

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cstdint>

// ============================================================================
// DATA STRUCTURES - 64-byte aligned for optimal GPU memory coalescing
// ============================================================================

/**
 * MediaOntologyNode - Represents a media content entity in GMC-O
 *
 * Examples:
 * - Media content: Movie, Series, Episode, Documentary
 * - Genres: Action, SciFi, Drama, Comedy
 * - Moods: Intense, Relaxing, Thought-provoking
 * - Cultural contexts: Hollywood, Bollywood, K-Drama
 * - Viewer segments: Family, Mature, Teen
 *
 * Position/velocity represent semantic space coordinates where similar
 * content clusters together based on ontology constraints.
 */
struct MediaOntologyNode {
    uint32_t graph_id;           // Multi-graph support (different content catalogs)
    uint32_t node_id;            // Unique identifier within graph
    uint32_t ontology_type;      // bits: MEDIA_CONTENT, GENRE, MOOD, CONTEXT
    uint32_t constraint_flags;   // Active constraint types for this node

    float3 position;             // Semantic space coordinates
    float3 velocity;             // Momentum in semantic space

    float mass;                  // Importance weight (e.g., popularity)
    float radius;                // Semantic influence radius

    uint32_t parent_genre;       // For genre hierarchy (SciFi -> Genre)
    uint32_t property_count;     // Number of associated properties
    uint32_t cultural_flags;     // Cultural context identifiers
    uint32_t mood_flags;         // Mood/aesthetic identifiers

    uint32_t padding[4];         // Align to 64 bytes for optimal memory access
};

/**
 * MediaOntologyConstraint - Semantic relationships between media entities
 *
 * Constraint Types:
 * 1. DISJOINT_GENRES: Mutually exclusive content types (Action ⊥ Documentary)
 * 2. GENRE_HIERARCHY: Genre inheritance (SciFi ⊑ Genre, Thriller ⊑ Genre)
 * 3. CONTENT_EQUIVALENCE: Same content across platforms (Netflix Movie ≡ Amazon Movie)
 * 4. MOOD_CONSISTENCY: Compatible aesthetic properties
 * 5. CULTURAL_ALIGNMENT: Cultural context relationships
 * 6. VIEWER_PREFERENCE: Viewer segment compatibility
 */
struct MediaOntologyConstraint {
    uint32_t type;               // Constraint type identifier
    uint32_t source_id;          // Source node ID
    uint32_t target_id;          // Target node ID
    uint32_t graph_id;           // Graph identifier

    float strength;              // Constraint enforcement strength (0.0-1.0)
    float distance;              // Ideal semantic distance

    float mood_weight;           // Mood similarity influence
    float cultural_weight;       // Cultural context influence

    uint32_t flags;              // Additional constraint modifiers

    float padding[7];            // Align to 64 bytes
};

// ============================================================================
// CONSTRAINT TYPE CONSTANTS
// ============================================================================

// Core ontology constraints (OWL-based)
#define CONSTRAINT_DISJOINT_GENRES      1  // Mutually exclusive genres
#define CONSTRAINT_GENRE_HIERARCHY      2  // SubClassOf relationships
#define CONSTRAINT_CONTENT_EQUIVALENCE  3  // SameAs for content identity
#define CONSTRAINT_INVERSE_RELATION     4  // Symmetric relationships
#define CONSTRAINT_FUNCTIONAL_PROPERTY  5  // Cardinality constraints

// Media-specific constraints (GMC-O extensions)
#define CONSTRAINT_MOOD_CONSISTENCY     6  // Aesthetic alignment
#define CONSTRAINT_CULTURAL_ALIGNMENT   7  // Cultural context compatibility
#define CONSTRAINT_VIEWER_PREFERENCE    8  // Audience segment relationships
#define CONSTRAINT_TEMPORAL_CONTEXT     9  // Time-based relevance
#define CONSTRAINT_CONTENT_SIMILARITY  10  // Content-based similarity

// ============================================================================
// ONTOLOGY TYPE FLAGS
// ============================================================================

#define ONTOLOGY_MEDIA_CONTENT  0x01  // Movie, Series, Episode
#define ONTOLOGY_GENRE          0x02  // Action, Drama, SciFi, etc.
#define ONTOLOGY_MOOD           0x04  // Intense, Relaxing, Dark, etc.
#define ONTOLOGY_CULTURAL       0x08  // Hollywood, Bollywood, K-Drama
#define ONTOLOGY_VIEWER_SEGMENT 0x10  // Family, Mature, Teen, Kids
#define ONTOLOGY_TEMPORAL       0x20  // Current, Classic, Trending

// ============================================================================
// PERFORMANCE CONSTANTS
// ============================================================================

#define BLOCK_SIZE 256            // Threads per block (optimized for modern GPUs)
#define EPSILON 1e-6f             // Numerical stability threshold
#define MAX_FORCE 1000.0f         // Force clamping for stability
#define DAMPING_FACTOR 0.95f      // Velocity damping for convergence

// ============================================================================
// DEVICE HELPER FUNCTIONS - Optimized vector mathematics
// ============================================================================

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len < EPSILON) return make_float3(0.0f, 0.0f, 0.0f);
    return v * (1.0f / len);
}

__device__ inline float3 clamp_force(const float3& force) {
    float mag = length(force);
    if (mag > MAX_FORCE) {
        return force * (MAX_FORCE / mag);
    }
    return force;
}

// Atomic add for float3 (requires atomicAdd for float)
__device__ inline void atomic_add_float3(float3* addr, const float3& val) {
    atomicAdd(&(addr->x), val.x);
    atomicAdd(&(addr->y), val.y);
    atomicAdd(&(addr->z), val.z);
}

/**
 * Calculate similarity between two nodes based on cultural and mood flags
 * Returns: 0.0-1.0, where 1.0 is identical
 */
__device__ inline float calculate_semantic_similarity(
    const MediaOntologyNode& a,
    const MediaOntologyNode& b
) {
    // Bitwise overlap for cultural context
    uint32_t cultural_overlap = __popc(a.cultural_flags & b.cultural_flags);
    uint32_t cultural_union = __popc(a.cultural_flags | b.cultural_flags);
    float cultural_sim = (cultural_union > 0) ?
        (float)cultural_overlap / (float)cultural_union : 0.0f;

    // Bitwise overlap for mood/aesthetic
    uint32_t mood_overlap = __popc(a.mood_flags & b.mood_flags);
    uint32_t mood_union = __popc(a.mood_flags | b.mood_flags);
    float mood_sim = (mood_union > 0) ?
        (float)mood_overlap / (float)mood_union : 0.0f;

    // Combined similarity (weighted average)
    return 0.6f * cultural_sim + 0.4f * mood_sim;
}

// ============================================================================
// KERNEL 1: DISJOINT GENRES CONSTRAINT
// ============================================================================

/**
 * Apply separation forces between mutually exclusive genres
 *
 * Example: Action ⊥ Documentary
 * - Action movies and documentaries should occupy distinct semantic spaces
 * - Prevents category confusion in recommendation systems
 * - Enforces clear genre boundaries
 *
 * Physics: Repulsion force inversely proportional to distance squared
 * Effect: Disjoint content classes separate in semantic space
 */
__global__ void apply_disjoint_genres_kernel(
    MediaOntologyNode* nodes,
    int num_nodes,
    MediaOntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float separation_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    MediaOntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_DISJOINT_GENRES) return;

    // Find source and target nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

    MediaOntologyNode source = nodes[source_idx];
    MediaOntologyNode target = nodes[target_idx];

    // Calculate repulsion force
    float3 delta = target.position - source.position;
    float dist = length(delta);
    float min_distance = source.radius + target.radius + constraint.distance;

    if (dist < min_distance && dist > EPSILON) {
        float3 direction = normalize(delta);
        float penetration = min_distance - dist;

        // Repulsion force: stronger when closer (inverse square law)
        float force_magnitude = separation_strength * constraint.strength *
                               (penetration / min_distance);

        float3 force = direction * (-force_magnitude);
        force = clamp_force(force);

        // Apply forces with mass consideration (heavier nodes move less)
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        // Update velocities atomically (multiple threads may affect same node)
        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);
    }
}

// ============================================================================
// KERNEL 2: GENRE HIERARCHY CONSTRAINT
// ============================================================================

/**
 * Apply hierarchical alignment forces for genre inheritance
 *
 * Example: SciFi ⊑ Genre (SciFi is a subclass of Genre)
 * - Maintains semantic hierarchy in visualization
 * - Child genres cluster near parent genres
 * - Preserves taxonomic relationships
 *
 * Physics: Spring force maintaining ideal distance
 * Effect: Hierarchical tree structure in semantic space
 */
__global__ void apply_genre_hierarchy_kernel(
    MediaOntologyNode* nodes,
    int num_nodes,
    MediaOntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float alignment_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    MediaOntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_GENRE_HIERARCHY) return;

    // Find source (subclass) and target (superclass) nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

    MediaOntologyNode source = nodes[source_idx];
    MediaOntologyNode target = nodes[target_idx];

    // Calculate spring force towards ideal distance
    float3 delta = target.position - source.position;
    float dist = length(delta);
    float ideal_distance = constraint.distance;

    if (dist > EPSILON) {
        float3 direction = normalize(delta);
        float displacement = dist - ideal_distance;

        // Spring force: F = k * x (Hooke's law)
        float force_magnitude = alignment_strength * constraint.strength * displacement;

        // Bonus: Consider semantic similarity for fine-tuning
        float semantic_sim = calculate_semantic_similarity(source, target);
        force_magnitude *= (0.5f + 0.5f * semantic_sim);

        float3 force = direction * force_magnitude;
        force = clamp_force(force);

        // Apply forces with mass consideration
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        // Update velocities
        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);
    }
}

// ============================================================================
// KERNEL 3: CONTENT EQUIVALENCE CONSTRAINT
// ============================================================================

/**
 * Apply co-location forces for equivalent content
 *
 * Example: Netflix's "Stranger Things" ≡ IMDb's "Stranger Things"
 * - Same content across different platforms should be identical
 * - Handles remakes, adaptations, multi-platform releases
 * - Strong attraction to minimize distance
 *
 * Physics: Very strong spring force with high damping
 * Effect: Equivalent content converges to same position
 */
__global__ void apply_content_equivalence_kernel(
    MediaOntologyNode* nodes,
    int num_nodes,
    MediaOntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float colocate_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    MediaOntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_CONTENT_EQUIVALENCE) return;

    // Find source and target nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

    MediaOntologyNode source = nodes[source_idx];
    MediaOntologyNode target = nodes[target_idx];

    // Calculate strong attraction towards same position
    float3 delta = target.position - source.position;
    float dist = length(delta);

    if (dist > EPSILON) {
        float3 direction = normalize(delta);

        // Very strong spring force to minimize distance
        float force_magnitude = colocate_strength * constraint.strength * dist;
        float3 force = direction * force_magnitude;
        force = clamp_force(force);

        // Apply forces with mass consideration
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        // Update velocities
        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);

        // Strong velocity damping for faster convergence
        nodes[source_idx].velocity = nodes[source_idx].velocity * DAMPING_FACTOR;
        nodes[target_idx].velocity = nodes[target_idx].velocity * DAMPING_FACTOR;
    }
}

// ============================================================================
// KERNEL 4: MOOD CONSISTENCY CONSTRAINT
// ============================================================================

/**
 * Apply mood/aesthetic consistency forces
 *
 * Example: "Dark" mood movies cluster together
 * - Ensures aesthetically similar content groups together
 * - Supports mood-based recommendations
 * - Considers both visual and narrative mood
 *
 * Physics: Attraction proportional to mood similarity
 * Effect: Mood-coherent clusters in semantic space
 */
__global__ void apply_mood_consistency_kernel(
    MediaOntologyNode* nodes,
    int num_nodes,
    MediaOntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float mood_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    MediaOntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_MOOD_CONSISTENCY) return;

    // Find source and target nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

    MediaOntologyNode source = nodes[source_idx];
    MediaOntologyNode target = nodes[target_idx];

    // Calculate mood similarity
    float mood_sim = calculate_semantic_similarity(source, target);

    if (mood_sim < 0.1f) return; // Skip if moods are too different

    // Calculate attraction force proportional to mood similarity
    float3 delta = target.position - source.position;
    float dist = length(delta);
    float ideal_distance = constraint.distance * (1.0f - 0.5f * mood_sim);

    if (dist > EPSILON) {
        float3 direction = normalize(delta);
        float displacement = dist - ideal_distance;

        // Mood-weighted spring force
        float force_magnitude = mood_strength * constraint.strength *
                               constraint.mood_weight * mood_sim * displacement;

        float3 force = direction * force_magnitude;
        force = clamp_force(force);

        // Apply forces
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);
    }
}

// ============================================================================
// KERNEL 5: CULTURAL ALIGNMENT CONSTRAINT
// ============================================================================

/**
 * Apply cultural context alignment forces
 *
 * Example: Bollywood movies cluster together, distinct from Hollywood
 * - Maintains cultural context boundaries
 * - Supports culturally-aware recommendations
 * - Respects regional content preferences
 *
 * Physics: Weak attraction for same culture, neutral for different
 * Effect: Cultural clusters with soft boundaries
 */
__global__ void apply_cultural_alignment_kernel(
    MediaOntologyNode* nodes,
    int num_nodes,
    MediaOntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float cultural_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    MediaOntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_CULTURAL_ALIGNMENT) return;

    // Find source and target nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

    MediaOntologyNode source = nodes[source_idx];
    MediaOntologyNode target = nodes[target_idx];

    // Calculate cultural overlap
    uint32_t cultural_overlap = __popc(source.cultural_flags & target.cultural_flags);

    if (cultural_overlap == 0) return; // No shared cultural context

    float cultural_similarity = (float)cultural_overlap /
                               (float)fmaxf(__popc(source.cultural_flags), 1.0f);

    // Calculate alignment force
    float3 delta = target.position - source.position;
    float dist = length(delta);
    float ideal_distance = constraint.distance * (1.0f - 0.3f * cultural_similarity);

    if (dist > EPSILON) {
        float3 direction = normalize(delta);
        float displacement = dist - ideal_distance;

        // Cultural-weighted spring force
        float force_magnitude = cultural_strength * constraint.strength *
                               constraint.cultural_weight * cultural_similarity * displacement;

        float3 force = direction * force_magnitude;
        force = clamp_force(force);

        // Apply forces
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);
    }
}

// ============================================================================
// KERNEL 6: VIEWER PREFERENCE CONSTRAINT
// ============================================================================

/**
 * Apply viewer preference alignment forces
 *
 * Example: Family-friendly content clusters separately from mature content
 * - Enforces audience segment boundaries
 * - Supports age-appropriate recommendations
 * - Maintains content rating consistency
 *
 * Physics: Strong separation for incompatible segments
 * Effect: Clear viewer segment clusters
 */
__global__ void apply_viewer_preference_kernel(
    MediaOntologyNode* nodes,
    int num_nodes,
    MediaOntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float preference_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    MediaOntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_VIEWER_PREFERENCE) return;

    // Find source and target nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

    MediaOntologyNode source = nodes[source_idx];
    MediaOntologyNode target = nodes[target_idx];

    // Check if viewer segments are compatible
    bool compatible = (source.ontology_type & ONTOLOGY_VIEWER_SEGMENT) &&
                     (target.ontology_type & ONTOLOGY_VIEWER_SEGMENT);

    if (!compatible) return;

    // Calculate preference alignment
    float3 delta = target.position - source.position;
    float dist = length(delta);

    // Compatible segments: attract, incompatible: repel
    bool same_segment = (source.constraint_flags & target.constraint_flags) != 0;
    float ideal_distance = same_segment ? constraint.distance :
                          constraint.distance * 3.0f;

    if (dist > EPSILON) {
        float3 direction = normalize(delta);
        float displacement = dist - ideal_distance;

        // Preference-weighted force
        float force_magnitude = preference_strength * constraint.strength * displacement;

        float3 force = direction * force_magnitude;
        force = clamp_force(force);

        // Apply forces
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);
    }
}

// ============================================================================
// HOST FUNCTIONS FOR KERNEL LAUNCH
// ============================================================================

extern "C" {

/**
 * Launch disjoint genres constraint kernel
 * Separates mutually exclusive content types
 */
void launch_disjoint_genres_kernel(
    MediaOntologyNode* d_nodes, int num_nodes,
    MediaOntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float separation_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_disjoint_genres_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, separation_strength
    );
}

/**
 * Launch genre hierarchy constraint kernel
 * Maintains taxonomic relationships (SciFi ⊑ Genre)
 */
void launch_genre_hierarchy_kernel(
    MediaOntologyNode* d_nodes, int num_nodes,
    MediaOntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float alignment_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_genre_hierarchy_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, alignment_strength
    );
}

/**
 * Launch content equivalence constraint kernel
 * Co-locates identical content across platforms
 */
void launch_content_equivalence_kernel(
    MediaOntologyNode* d_nodes, int num_nodes,
    MediaOntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float colocate_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_content_equivalence_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, colocate_strength
    );
}

/**
 * Launch mood consistency constraint kernel
 * Clusters content by aesthetic/mood similarity
 */
void launch_mood_consistency_kernel(
    MediaOntologyNode* d_nodes, int num_nodes,
    MediaOntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float mood_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_mood_consistency_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, mood_strength
    );
}

/**
 * Launch cultural alignment constraint kernel
 * Maintains cultural context boundaries
 */
void launch_cultural_alignment_kernel(
    MediaOntologyNode* d_nodes, int num_nodes,
    MediaOntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float cultural_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_cultural_alignment_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, cultural_strength
    );
}

/**
 * Launch viewer preference constraint kernel
 * Enforces audience segment relationships
 */
void launch_viewer_preference_kernel(
    MediaOntologyNode* d_nodes, int num_nodes,
    MediaOntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float preference_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_viewer_preference_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, preference_strength
    );
}

} // extern "C"
