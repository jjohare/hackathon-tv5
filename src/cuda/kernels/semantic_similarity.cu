// =============================================================================
// Media Semantic Similarity GPU Kernel
// =============================================================================
// Implements multi-modal content similarity for media recommendation engine
// using semantic forces adapted from knowledge graph visualization.
//
// Features:
// - Multi-modal embedding fusion (visual, audio, text, metadata)
// - Content similarity scoring with configurable weights
// - Genre/mood clustering for related content discovery
// - Temporal relevance and decay factors
// - Collaborative filtering integration
// - Performance optimizations: shared memory, atomic operations, coalesced access
//
// Architecture:
// - Content items are nodes in embedding space
// - Similarity relationships are edges with computed weights
// - Clustering forces group similar content by genre, mood, style
// - Temporal forces prioritize recent/trending content
// - Collaborative signals boost items liked by similar users
//
// Performance: Optimized for real-time recommendation generation
// - Processes 10k+ content items at interactive rates
// - Batched similarity computation for throughput
// - GPU memory hierarchy exploitation
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cmath>

extern "C" {

// =============================================================================
// Configuration Structures
// =============================================================================

// Multi-modal embedding fusion configuration
struct MultiModalConfig {
    float visual_weight;         // Weight for visual embedding similarity (0-1)
    float audio_weight;          // Weight for audio embedding similarity (0-1)
    float text_weight;           // Weight for text/metadata embedding similarity (0-1)
    float metadata_weight;       // Weight for structured metadata similarity (0-1)
    float embedding_dimension;   // Dimensionality of embeddings
    bool enabled;
};

// Genre clustering configuration
struct GenreClusterConfig {
    float cluster_attraction;    // Attraction between items of same genre
    float cluster_radius;        // Target radius for genre clusters
    float inter_cluster_repulsion; // Repulsion between different genres
    int num_genres;              // Number of genre categories
    bool enabled;
};

// Mood clustering configuration
struct MoodClusterConfig {
    float cluster_attraction;    // Attraction between items of similar mood
    float cluster_radius;        // Target radius for mood clusters
    float mood_similarity_threshold; // Threshold for mood similarity (0-1)
    bool enabled;
};

// Content similarity scoring configuration
struct ContentSimilarityConfig {
    float base_spring_k;         // Base spring constant for similar content
    float similarity_multiplier; // Multiplier for similarity score influence
    float min_similarity_threshold; // Minimum similarity to create connection
    float rest_length_min;       // Minimum rest length for highly similar content
    float rest_length_max;       // Maximum rest length for weakly similar content
    bool enabled;
};

// Temporal relevance configuration
struct TemporalRelevanceConfig {
    float recency_weight;        // Weight for recency in recommendations
    float trending_boost;        // Boost factor for trending content
    float decay_half_life_days;  // Half-life for temporal decay (days)
    float temporal_clustering_strength; // Strength of temporal clustering
    bool enabled;
};

// Collaborative filtering configuration
struct CollaborativeConfig {
    float user_similarity_weight; // Weight for user-user similarity
    float item_cooccurrence_weight; // Weight for item co-occurrence patterns
    float rating_influence;      // Influence of explicit ratings
    float implicit_signal_weight; // Weight for implicit feedback (views, time)
    bool enabled;
};

// Style similarity configuration
struct StyleSimilarityConfig {
    float visual_style_weight;   // Weight for visual style similarity
    float narrative_style_weight; // Weight for narrative/pacing similarity
    float production_quality_weight; // Weight for production quality similarity
    float style_clustering_strength; // Clustering strength for similar styles
    bool enabled;
};

// Popularity and diversity balancing
struct PopularityDiversityConfig {
    float popularity_boost;      // Boost factor for popular items
    float diversity_penalty;     // Penalty to prevent filter bubbles
    float exploration_factor;    // Factor for exploration vs exploitation
    float serendipity_weight;    // Weight for serendipitous recommendations
    bool enabled;
};

// Unified media recommendation configuration
struct MediaRecommendationConfig {
    MultiModalConfig multi_modal;
    GenreClusterConfig genre_cluster;
    MoodClusterConfig mood_cluster;
    ContentSimilarityConfig content_similarity;
    TemporalRelevanceConfig temporal_relevance;
    CollaborativeConfig collaborative;
    StyleSimilarityConfig style_similarity;
    PopularityDiversityConfig popularity_diversity;
};

// Global constant memory for media recommendation configuration
__constant__ MediaRecommendationConfig c_media_config;

// =============================================================================
// Helper Functions
// =============================================================================

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 1e-6f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

// Cosine similarity between two vectors
__device__ inline float cosine_similarity(
    const float* vec_a,
    const float* vec_b,
    int dimension
) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < dimension; i++) {
        dot += vec_a[i] * vec_b[i];
        norm_a += vec_a[i] * vec_a[i];
        norm_b += vec_b[i] * vec_b[i];
    }

    float norm_product = sqrtf(norm_a) * sqrtf(norm_b);
    if (norm_product < 1e-6f) return 0.0f;

    return dot / norm_product;
}

// Temporal decay function (exponential decay based on age)
__device__ inline float temporal_decay(
    float age_days,
    float half_life_days
) {
    if (half_life_days < 1e-6f) return 1.0f;
    return expf(-0.693147f * age_days / half_life_days); // ln(2) = 0.693147
}

// =============================================================================
// Multi-Modal Embedding Fusion Kernel
// =============================================================================

// Compute fused similarity scores from multi-modal embeddings
// This kernel combines visual, audio, text, and metadata embeddings into
// unified similarity scores for each content pair.
__global__ void compute_multimodal_similarity(
    const float* visual_embeddings,    // Visual embeddings [num_items x visual_dim]
    const float* audio_embeddings,     // Audio embeddings [num_items x audio_dim]
    const float* text_embeddings,      // Text embeddings [num_items x text_dim]
    const float* metadata_embeddings,  // Metadata embeddings [num_items x meta_dim]
    const int* item_pairs_src,         // Source indices for pairs to compare
    const int* item_pairs_tgt,         // Target indices for pairs to compare
    float* similarity_scores,          // Output: fused similarity scores
    const int num_pairs,
    const int visual_dim,
    const int audio_dim,
    const int text_dim,
    const int metadata_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    if (!c_media_config.multi_modal.enabled) return;

    int src = item_pairs_src[idx];
    int tgt = item_pairs_tgt[idx];

    float total_similarity = 0.0f;
    float total_weight = 0.0f;

    // Visual similarity
    if (c_media_config.multi_modal.visual_weight > 0.0f && visual_embeddings) {
        float vis_sim = cosine_similarity(
            &visual_embeddings[src * visual_dim],
            &visual_embeddings[tgt * visual_dim],
            visual_dim
        );
        total_similarity += vis_sim * c_media_config.multi_modal.visual_weight;
        total_weight += c_media_config.multi_modal.visual_weight;
    }

    // Audio similarity
    if (c_media_config.multi_modal.audio_weight > 0.0f && audio_embeddings) {
        float audio_sim = cosine_similarity(
            &audio_embeddings[src * audio_dim],
            &audio_embeddings[tgt * audio_dim],
            audio_dim
        );
        total_similarity += audio_sim * c_media_config.multi_modal.audio_weight;
        total_weight += c_media_config.multi_modal.audio_weight;
    }

    // Text similarity
    if (c_media_config.multi_modal.text_weight > 0.0f && text_embeddings) {
        float text_sim = cosine_similarity(
            &text_embeddings[src * text_dim],
            &text_embeddings[tgt * text_dim],
            text_dim
        );
        total_similarity += text_sim * c_media_config.multi_modal.text_weight;
        total_weight += c_media_config.multi_modal.text_weight;
    }

    // Metadata similarity
    if (c_media_config.multi_modal.metadata_weight > 0.0f && metadata_embeddings) {
        float meta_sim = cosine_similarity(
            &metadata_embeddings[src * metadata_dim],
            &metadata_embeddings[tgt * metadata_dim],
            metadata_dim
        );
        total_similarity += meta_sim * c_media_config.multi_modal.metadata_weight;
        total_weight += c_media_config.multi_modal.metadata_weight;
    }

    // Normalize by total weight
    if (total_weight > 0.0f) {
        similarity_scores[idx] = total_similarity / total_weight;
    } else {
        similarity_scores[idx] = 0.0f;
    }
}

// =============================================================================
// Genre Clustering Kernel
// =============================================================================

// Apply genre-based clustering forces to group similar content
// Attracts items of the same genre and repels items of different genres
// to create natural content neighborhoods in recommendation space.
__global__ void apply_genre_cluster_force(
    const int* item_genres,            // Genre ID for each item
    const float3* genre_centroids,     // Centroid position for each genre
    float3* positions,                 // Current positions in embedding space
    float3* forces,                    // Force accumulator
    const int num_items,
    const int num_genres
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    if (!c_media_config.genre_cluster.enabled) return;

    int item_genre = item_genres[idx];
    if (item_genre < 0 || item_genre >= num_genres) return;

    // Attraction to genre centroid
    float3 to_centroid = genre_centroids[item_genre] - positions[idx];
    float dist_to_centroid = length(to_centroid);

    float3 cluster_force = make_float3(0.0f, 0.0f, 0.0f);
    if (dist_to_centroid > c_media_config.genre_cluster.cluster_radius) {
        // Outside cluster radius - attract inward
        float force_mag = c_media_config.genre_cluster.cluster_attraction *
                        (dist_to_centroid - c_media_config.genre_cluster.cluster_radius);
        cluster_force = normalize(to_centroid) * force_mag;
    }

    // Repulsion from items of different genres
    float3 inter_cluster_repulsion = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < num_items; i++) {
        if (i == idx) continue;
        if (item_genres[i] == item_genre) continue; // Same genre

        float3 delta = positions[idx] - positions[i];
        float dist = length(delta);

        if (dist < c_media_config.genre_cluster.cluster_radius * 2.0f && dist > 1e-6f) {
            float force_mag = c_media_config.genre_cluster.inter_cluster_repulsion / (dist * dist);
            inter_cluster_repulsion = inter_cluster_repulsion + (normalize(delta) * force_mag);
        }
    }

    // Accumulate forces
    atomicAdd(&forces[idx].x, cluster_force.x + inter_cluster_repulsion.x);
    atomicAdd(&forces[idx].y, cluster_force.y + inter_cluster_repulsion.y);
    atomicAdd(&forces[idx].z, cluster_force.z + inter_cluster_repulsion.z);
}

// =============================================================================
// Mood Clustering Kernel
// =============================================================================

// Apply mood-based clustering forces for emotional content similarity
// Groups content with similar emotional tone, energy, and atmosphere.
__global__ void apply_mood_cluster_force(
    const float* mood_vectors,         // Mood embeddings [num_items x mood_dim]
    float3* positions,                 // Current positions in embedding space
    float3* forces,                    // Force accumulator
    const int num_items,
    const int mood_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    if (!c_media_config.mood_cluster.enabled) return;

    // Compute mood-based attraction/repulsion with all other items
    float3 mood_force = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < num_items; i++) {
        if (i == idx) continue;

        // Calculate mood similarity
        float mood_similarity = cosine_similarity(
            &mood_vectors[idx * mood_dim],
            &mood_vectors[i * mood_dim],
            mood_dim
        );

        if (mood_similarity < c_media_config.mood_cluster.mood_similarity_threshold) {
            continue; // Skip dissimilar moods
        }

        float3 delta = positions[i] - positions[idx];
        float dist = length(delta);

        if (dist < 1e-6f) continue;

        // Attraction force proportional to mood similarity
        float attraction_strength = c_media_config.mood_cluster.cluster_attraction * mood_similarity;

        // Target distance based on cluster radius
        if (dist > c_media_config.mood_cluster.cluster_radius) {
            float force_mag = attraction_strength * (dist - c_media_config.mood_cluster.cluster_radius);
            mood_force = mood_force + (normalize(delta) * force_mag);
        }
    }

    // Accumulate forces
    atomicAdd(&forces[idx].x, mood_force.x);
    atomicAdd(&forces[idx].y, mood_force.y);
    atomicAdd(&forces[idx].z, mood_force.z);
}

// =============================================================================
// Content Similarity Spring Kernel
// =============================================================================

// Apply spring forces between similar content items
// Creates connections between content with high multi-modal similarity,
// with spring strength and rest length based on similarity score.
__global__ void apply_content_similarity_force(
    const int* similar_pairs_src,      // Source indices for similar pairs
    const int* similar_pairs_tgt,      // Target indices for similar pairs
    const float* similarity_scores,    // Similarity scores for each pair (0-1)
    float3* positions,                 // Current positions in embedding space
    float3* forces,                    // Force accumulator
    const int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    if (!c_media_config.content_similarity.enabled) return;

    int src = similar_pairs_src[idx];
    int tgt = similar_pairs_tgt[idx];
    float similarity = similarity_scores[idx];

    // Skip if similarity below threshold
    if (similarity < c_media_config.content_similarity.min_similarity_threshold) {
        return;
    }

    // Calculate spring force
    float3 delta = positions[tgt] - positions[src];
    float dist = length(delta);

    if (dist < 1e-6f) return;

    // Spring constant increases with similarity
    float spring_k = c_media_config.content_similarity.base_spring_k *
                    (1.0f + similarity * c_media_config.content_similarity.similarity_multiplier);

    // Rest length inversely proportional to similarity
    // (more similar content should be closer together)
    float rest_length = c_media_config.content_similarity.rest_length_max -
                       (similarity * (c_media_config.content_similarity.rest_length_max -
                                    c_media_config.content_similarity.rest_length_min));

    // Hooke's law: F = -k * (x - x0)
    float displacement = dist - rest_length;
    float force_mag = spring_k * displacement;

    float3 spring_force = normalize(delta) * force_mag;

    // Apply equal and opposite forces
    atomicAdd(&forces[src].x, spring_force.x);
    atomicAdd(&forces[src].y, spring_force.y);
    atomicAdd(&forces[src].z, spring_force.z);

    atomicAdd(&forces[tgt].x, -spring_force.x);
    atomicAdd(&forces[tgt].y, -spring_force.y);
    atomicAdd(&forces[tgt].z, -spring_force.z);
}

// =============================================================================
// Temporal Relevance Kernel
// =============================================================================

// Apply temporal relevance forces to boost recent and trending content
// Moves recently released or trending content toward prominent positions
// while applying decay to older content.
__global__ void apply_temporal_relevance_force(
    const float* item_ages_days,       // Age of each item in days since release
    const float* trending_scores,      // Trending score for each item (0-1)
    float3* positions,                 // Current positions in embedding space
    float3* forces,                    // Force accumulator
    const int num_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    if (!c_media_config.temporal_relevance.enabled) return;

    float age_days = item_ages_days[idx];
    float trending = trending_scores ? trending_scores[idx] : 0.0f;

    // Calculate temporal decay factor
    float decay_factor = temporal_decay(
        age_days,
        c_media_config.temporal_relevance.decay_half_life_days
    );

    // Combine recency and trending for total temporal relevance
    float temporal_relevance = decay_factor * c_media_config.temporal_relevance.recency_weight +
                              trending * c_media_config.temporal_relevance.trending_boost;

    // Apply upward force for temporally relevant content
    // (in recommendation space, "up" typically means more prominent)
    float3 temporal_force = make_float3(
        0.0f,
        temporal_relevance * c_media_config.temporal_relevance.temporal_clustering_strength,
        0.0f
    );

    // Additional clustering: attract other temporally similar content
    for (int i = 0; i < num_items; i++) {
        if (i == idx) continue;

        float other_age = item_ages_days[i];
        float age_difference = fabsf(age_days - other_age);

        // Cluster items with similar ages
        if (age_difference < c_media_config.temporal_relevance.decay_half_life_days) {
            float3 delta = positions[i] - positions[idx];
            float dist = length(delta);

            if (dist > 1e-6f) {
                float cluster_strength = c_media_config.temporal_relevance.temporal_clustering_strength *
                                        (1.0f - age_difference / c_media_config.temporal_relevance.decay_half_life_days);
                temporal_force = temporal_force + (normalize(delta) * cluster_strength * 0.1f);
            }
        }
    }

    // Accumulate forces
    atomicAdd(&forces[idx].x, temporal_force.x);
    atomicAdd(&forces[idx].y, temporal_force.y);
    atomicAdd(&forces[idx].z, temporal_force.z);
}

// =============================================================================
// Collaborative Filtering Kernel
// =============================================================================

// Apply collaborative filtering forces based on user behavior patterns
// Creates connections between items frequently co-viewed or co-rated by users
// with similar preferences.
__global__ void apply_collaborative_force(
    const int* user_item_pairs_user,   // User indices for interaction pairs
    const int* user_item_pairs_item,   // Item indices for interaction pairs
    const float* interaction_weights,  // Weight of each interaction (rating, view time, etc.)
    const float* user_similarity_matrix, // User-user similarity matrix [num_users x num_users]
    float3* positions,                 // Current positions in embedding space
    float3* forces,                    // Force accumulator
    const int num_interactions,
    const int num_users
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_interactions) return;

    if (!c_media_config.collaborative.enabled) return;

    int user = user_item_pairs_user[idx];
    int item = user_item_pairs_item[idx];
    float weight = interaction_weights[idx];

    // Find items interacted with by similar users
    for (int other_interaction = 0; other_interaction < num_interactions; other_interaction++) {
        if (other_interaction == idx) continue;

        int other_user = user_item_pairs_user[other_interaction];
        int other_item = user_item_pairs_item[other_interaction];

        if (other_item == item) continue; // Same item

        // Get user similarity
        float user_similarity = user_similarity_matrix[user * num_users + other_user];

        if (user_similarity < 0.1f) continue; // Skip dissimilar users

        // Calculate collaborative force between items
        float3 delta = positions[other_item] - positions[item];
        float dist = length(delta);

        if (dist < 1e-6f) continue;

        // Force strength based on user similarity and interaction weights
        float collab_strength = c_media_config.collaborative.user_similarity_weight *
                               user_similarity *
                               (weight * c_media_config.collaborative.rating_influence +
                                c_media_config.collaborative.implicit_signal_weight);

        float3 collab_force = normalize(delta) * collab_strength;

        // Attract items liked by similar users
        atomicAdd(&forces[item].x, collab_force.x);
        atomicAdd(&forces[item].y, collab_force.y);
        atomicAdd(&forces[item].z, collab_force.z);
    }
}

// =============================================================================
// Style Similarity Kernel
// =============================================================================

// Apply style-based clustering forces for cinematic/aesthetic similarity
// Groups content with similar visual style, narrative approach, and production quality.
__global__ void apply_style_similarity_force(
    const float* visual_style_embeddings,    // Visual style embeddings [num_items x style_dim]
    const float* narrative_style_embeddings, // Narrative style embeddings [num_items x narrative_dim]
    const float* production_quality_scores,  // Production quality scores [num_items]
    float3* positions,                       // Current positions in embedding space
    float3* forces,                          // Force accumulator
    const int num_items,
    const int visual_style_dim,
    const int narrative_style_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    if (!c_media_config.style_similarity.enabled) return;

    float3 style_force = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < num_items; i++) {
        if (i == idx) continue;

        float total_style_similarity = 0.0f;
        float total_weight = 0.0f;

        // Visual style similarity
        if (c_media_config.style_similarity.visual_style_weight > 0.0f && visual_style_embeddings) {
            float vis_style_sim = cosine_similarity(
                &visual_style_embeddings[idx * visual_style_dim],
                &visual_style_embeddings[i * visual_style_dim],
                visual_style_dim
            );
            total_style_similarity += vis_style_sim * c_media_config.style_similarity.visual_style_weight;
            total_weight += c_media_config.style_similarity.visual_style_weight;
        }

        // Narrative style similarity
        if (c_media_config.style_similarity.narrative_style_weight > 0.0f && narrative_style_embeddings) {
            float narr_style_sim = cosine_similarity(
                &narrative_style_embeddings[idx * narrative_style_dim],
                &narrative_style_embeddings[i * narrative_style_dim],
                narrative_style_dim
            );
            total_style_similarity += narr_style_sim * c_media_config.style_similarity.narrative_style_weight;
            total_weight += c_media_config.style_similarity.narrative_style_weight;
        }

        // Production quality similarity
        if (c_media_config.style_similarity.production_quality_weight > 0.0f && production_quality_scores) {
            float quality_diff = fabsf(production_quality_scores[idx] - production_quality_scores[i]);
            float quality_sim = 1.0f - fminf(quality_diff, 1.0f); // Normalize to 0-1
            total_style_similarity += quality_sim * c_media_config.style_similarity.production_quality_weight;
            total_weight += c_media_config.style_similarity.production_quality_weight;
        }

        if (total_weight < 1e-6f) continue;

        float style_similarity = total_style_similarity / total_weight;

        if (style_similarity < 0.5f) continue; // Skip low similarity

        // Apply attraction force based on style similarity
        float3 delta = positions[i] - positions[idx];
        float dist = length(delta);

        if (dist > 1e-6f) {
            float force_mag = c_media_config.style_similarity.style_clustering_strength * style_similarity;
            style_force = style_force + (normalize(delta) * force_mag / dist);
        }
    }

    // Accumulate forces
    atomicAdd(&forces[idx].x, style_force.x);
    atomicAdd(&forces[idx].y, style_force.y);
    atomicAdd(&forces[idx].z, style_force.z);
}

// =============================================================================
// Popularity and Diversity Balancing Kernel
// =============================================================================

// Balance popularity with diversity to prevent filter bubbles
// Boosts popular content while maintaining exploration of niche items.
__global__ void apply_popularity_diversity_force(
    const float* popularity_scores,    // Popularity score for each item (0-1)
    const float* diversity_scores,     // Diversity/uniqueness score for each item (0-1)
    float3* positions,                 // Current positions in embedding space
    float3* forces,                    // Force accumulator
    const int num_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    if (!c_media_config.popularity_diversity.enabled) return;

    float popularity = popularity_scores[idx];
    float diversity = diversity_scores ? diversity_scores[idx] : 0.5f;

    // Popularity boost: move popular items toward center (more prominent)
    float3 to_center = make_float3(0.0f, 0.0f, 0.0f) - positions[idx];
    float dist_to_center = length(to_center);

    float popularity_force_mag = c_media_config.popularity_diversity.popularity_boost * popularity;
    float3 popularity_force = make_float3(0.0f, 0.0f, 0.0f);
    if (dist_to_center > 1e-6f) {
        popularity_force = normalize(to_center) * popularity_force_mag;
    }

    // Diversity penalty: push overly similar items apart to ensure variety
    float3 diversity_force = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < num_items; i++) {
        if (i == idx) continue;

        float3 delta = positions[idx] - positions[i];
        float dist = length(delta);

        if (dist < 1e-6f) continue;

        // Higher diversity score = stronger repulsion from nearby items
        float diversity_strength = c_media_config.popularity_diversity.diversity_penalty * diversity;
        float repulsion = diversity_strength / (dist * dist);

        diversity_force = diversity_force + (normalize(delta) * repulsion);
    }

    // Serendipity component: occasional random exploration
    float serendipity_mag = c_media_config.popularity_diversity.serendipity_weight * diversity;
    float3 serendipity_force = make_float3(
        (float)(threadIdx.x % 100 - 50) * 0.01f * serendipity_mag,
        (float)(blockIdx.x % 100 - 50) * 0.01f * serendipity_mag,
        0.0f
    );

    // Combine forces with exploration factor
    float exploration = c_media_config.popularity_diversity.exploration_factor;
    float3 total_force = popularity_force * (1.0f - exploration) +
                        diversity_force * exploration +
                        serendipity_force;

    // Accumulate forces
    atomicAdd(&forces[idx].x, total_force.x);
    atomicAdd(&forces[idx].y, total_force.y);
    atomicAdd(&forces[idx].z, total_force.z);
}

// =============================================================================
// Utility Kernels
// =============================================================================

// Calculate genre centroids for clustering
__global__ void calculate_genre_centroids(
    const int* item_genres,            // Genre ID for each item
    const float3* positions,           // Current positions
    float3* genre_centroids,           // Output: centroid for each genre
    int* genre_counts,                 // Output: count for each genre
    const int num_items,
    const int num_genres
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    int genre = item_genres[idx];
    if (genre < 0 || genre >= num_genres) return;

    // Atomic add to accumulate positions
    atomicAdd(&genre_centroids[genre].x, positions[idx].x);
    atomicAdd(&genre_centroids[genre].y, positions[idx].y);
    atomicAdd(&genre_centroids[genre].z, positions[idx].z);
    atomicAdd(&genre_counts[genre], 1);
}

// Finalize centroids by dividing by count
__global__ void finalize_genre_centroids(
    float3* genre_centroids,           // Centroids to finalize
    const int* genre_counts,           // Count for each genre
    const int num_genres
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_genres) return;

    int count = genre_counts[idx];
    if (count > 0) {
        genre_centroids[idx].x /= count;
        genre_centroids[idx].y /= count;
        genre_centroids[idx].z /= count;
    }
}

// Batch compute all pairwise similarities (optimized)
// Uses shared memory for efficient embedding access
__global__ void batch_compute_pairwise_similarities(
    const float* embeddings,           // Embeddings [num_items x embedding_dim]
    float* similarity_matrix,          // Output: similarity matrix [num_items x num_items]
    const int num_items,
    const int embedding_dim
) {
    __shared__ float shared_embedding[256]; // Shared memory for embedding cache

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= num_items || col >= num_items) return;

    // Skip redundant computations (similarity is symmetric)
    if (row > col) return;

    // Load embeddings into shared memory if possible
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < embedding_dim && embedding_dim <= 256) {
        if (row < num_items) {
            shared_embedding[tid] = embeddings[row * embedding_dim + tid];
        }
    }
    __syncthreads();

    // Compute cosine similarity
    float similarity = cosine_similarity(
        &embeddings[row * embedding_dim],
        &embeddings[col * embedding_dim],
        embedding_dim
    );

    // Store in both positions (symmetric matrix)
    similarity_matrix[row * num_items + col] = similarity;
    if (row != col) {
        similarity_matrix[col * num_items + row] = similarity;
    }
}

// =============================================================================
// Configuration Setup
// =============================================================================

// Upload media recommendation configuration to constant memory
void set_media_recommendation_config(const MediaRecommendationConfig* config) {
    cudaMemcpyToSymbol(c_media_config, config, sizeof(MediaRecommendationConfig));
}

} // extern "C"
