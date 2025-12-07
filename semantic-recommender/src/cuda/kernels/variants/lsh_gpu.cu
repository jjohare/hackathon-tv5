#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// LSH structure for GPU
struct LSH_GPU {
    float* random_projections;   // [num_tables, num_projections, embedding_dim]
    int* hash_tables;            // [num_tables, num_buckets, bucket_size]
    int* bucket_counts;          // [num_tables, num_buckets]
    int num_tables;
    int num_projections;
    int num_buckets;
    int bucket_size;
    int embedding_dim;

    __device__ uint32_t compute_hash(
        const __half* embedding,
        int table_id
    ) const {
        // SimHash: random projection + sign bit extraction
        uint32_t hash = 0;

        const float* projections = random_projections +
                                  table_id * num_projections * embedding_dim;

        for (int p = 0; p < num_projections; p++) {
            float projection = 0.0f;

            // Dot product with random vector
            for (int d = 0; d < embedding_dim; d++) {
                projection += __half2float(embedding[d]) *
                            projections[p * embedding_dim + d];
            }

            // Set bit if projection is positive
            if (projection > 0) {
                hash |= (1u << p);
            }
        }

        // Map to bucket
        return hash % num_buckets;
    }

    __device__ void get_bucket_items(
        int table_id,
        uint32_t bucket_id,
        int* output_items,
        int* output_count
    ) const {
        int bucket_offset = table_id * num_buckets * bucket_size +
                           bucket_id * bucket_size;
        int count = bucket_counts[table_id * num_buckets + bucket_id];

        *output_count = min(count, bucket_size);
        for (int i = 0; i < *output_count; i++) {
            output_items[i] = hash_tables[bucket_offset + i];
        }
    }
};

// Initialize random projections for LSH
__global__ void lsh_init_projections(
    float* projections,
    int num_tables,
    int num_projections,
    int embedding_dim,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = num_tables * num_projections * embedding_dim;

    if (idx >= total_size) return;

    // Initialize random number generator
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Sample from standard normal distribution
    projections[idx] = curand_normal(&state);
}

// Insert embeddings into LSH hash tables
__global__ void lsh_insert_batch(
    LSH_GPU lsh,
    const __half* embeddings,    // [num_items, embedding_dim]
    int* item_ids,               // [num_items]
    int num_items
) {
    int item_idx = blockIdx.x;
    if (item_idx >= num_items) return;

    const __half* embedding = embeddings + item_idx * lsh.embedding_dim;
    int item_id = item_ids[item_idx];

    // Hash into all tables
    for (int table = 0; table < lsh.num_tables; table++) {
        uint32_t bucket = lsh.compute_hash(embedding, table);

        // Atomic insert into bucket
        int bucket_offset = table * lsh.num_buckets * lsh.bucket_size +
                           bucket * lsh.bucket_size;
        int count_offset = table * lsh.num_buckets + bucket;

        int pos = atomicAdd(&lsh.bucket_counts[count_offset], 1);
        if (pos < lsh.bucket_size) {
            lsh.hash_tables[bucket_offset + pos] = item_id;
        }
    }
}

// Query LSH for candidate items
__global__ void lsh_search_batch(
    LSH_GPU lsh,
    const __half* queries,       // [batch_size, embedding_dim]
    int* candidates,             // [batch_size, max_candidates]
    int* candidate_counts,       // [batch_size]
    int batch_size,
    int max_candidates
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    const __half* query = queries + query_id * lsh.embedding_dim;

    // Shared memory for deduplication
    __shared__ uint32_t seen_bitmap[4096]; // For up to 128K items

    // Initialize bitmap
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        seen_bitmap[i] = 0;
    }
    __syncthreads();

    __shared__ int local_candidates[1024];
    __shared__ int local_count;

    if (threadIdx.x == 0) {
        local_count = 0;
    }
    __syncthreads();

    // Query all tables in parallel
    for (int table = threadIdx.x; table < lsh.num_tables; table += blockDim.x) {
        uint32_t bucket = lsh.compute_hash(query, table);

        // Get bucket items
        int bucket_offset = table * lsh.num_buckets * lsh.bucket_size +
                           bucket * lsh.bucket_size;
        int count = lsh.bucket_counts[table * lsh.num_buckets + bucket];

        // Add unique items to candidate set
        for (int i = 0; i < min(count, lsh.bucket_size); i++) {
            int item_id = lsh.hash_tables[bucket_offset + i];

            // Check and mark in bitmap (atomic)
            int word_idx = item_id / 32;
            int bit_idx = item_id % 32;
            uint32_t old = atomicOr(&seen_bitmap[word_idx], 1u << bit_idx);

            if (!(old & (1u << bit_idx))) {
                // New candidate
                int pos = atomicAdd(&local_count, 1);
                if (pos < 1024) {
                    local_candidates[pos] = item_id;
                }
            }
        }
    }
    __syncthreads();

    // Write results
    int num_to_write = min(local_count, max_candidates);
    if (threadIdx.x == 0) {
        candidate_counts[query_id] = num_to_write;
    }

    for (int i = threadIdx.x; i < num_to_write; i += blockDim.x) {
        candidates[query_id * max_candidates + i] = local_candidates[i];
    }
}

// Multi-probe LSH: query neighboring buckets
__global__ void lsh_multiprobe_search(
    LSH_GPU lsh,
    const __half* queries,
    int* candidates,
    int* candidate_counts,
    int batch_size,
    int max_candidates,
    int num_probes              // Number of neighboring buckets to probe
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    const __half* query = queries + query_id * lsh.embedding_dim;

    __shared__ uint32_t seen_bitmap[4096];
    __shared__ int local_candidates[2048];
    __shared__ int local_count;

    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        seen_bitmap[i] = 0;
    }
    if (threadIdx.x == 0) local_count = 0;
    __syncthreads();

    // For each table
    for (int table = 0; table < lsh.num_tables; table++) {
        uint32_t base_bucket = lsh.compute_hash(query, table);

        // Probe main bucket and neighbors
        for (int probe = threadIdx.x; probe < num_probes; probe += blockDim.x) {
            // Generate neighboring bucket (flip bits)
            uint32_t bucket = base_bucket ^ probe;
            bucket %= lsh.num_buckets;

            int bucket_offset = table * lsh.num_buckets * lsh.bucket_size +
                               bucket * lsh.bucket_size;
            int count = lsh.bucket_counts[table * lsh.num_buckets + bucket];

            for (int i = 0; i < min(count, lsh.bucket_size); i++) {
                int item_id = lsh.hash_tables[bucket_offset + i];

                int word_idx = item_id / 32;
                int bit_idx = item_id % 32;
                uint32_t old = atomicOr(&seen_bitmap[word_idx], 1u << bit_idx);

                if (!(old & (1u << bit_idx))) {
                    int pos = atomicAdd(&local_count, 1);
                    if (pos < 2048) {
                        local_candidates[pos] = item_id;
                    }
                }
            }
        }
    }
    __syncthreads();

    int num_to_write = min(local_count, max_candidates);
    if (threadIdx.x == 0) {
        candidate_counts[query_id] = num_to_write;
    }

    for (int i = threadIdx.x; i < num_to_write; i += blockDim.x) {
        candidates[query_id * max_candidates + i] = local_candidates[i];
    }
}

// Adaptive LSH: adjust number of tables based on query difficulty
__global__ void lsh_adaptive_search(
    LSH_GPU lsh,
    const __half* queries,
    int* candidates,
    int* candidate_counts,
    int batch_size,
    int target_recall,
    int max_candidates
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    const __half* query = queries + query_id * lsh.embedding_dim;

    __shared__ uint32_t seen_bitmap[4096];
    __shared__ int local_candidates[2048];
    __shared__ int local_count;

    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        seen_bitmap[i] = 0;
    }
    if (threadIdx.x == 0) local_count = 0;
    __syncthreads();

    // Start with few tables, increase until target recall
    for (int num_active_tables = 2;
         num_active_tables <= lsh.num_tables;
         num_active_tables *= 2) {

        for (int table = threadIdx.x;
             table < num_active_tables;
             table += blockDim.x) {

            uint32_t bucket = lsh.compute_hash(query, table);

            int bucket_offset = table * lsh.num_buckets * lsh.bucket_size +
                               bucket * lsh.bucket_size;
            int count = lsh.bucket_counts[table * lsh.num_buckets + bucket];

            for (int i = 0; i < min(count, lsh.bucket_size); i++) {
                int item_id = lsh.hash_tables[bucket_offset + i];

                int word_idx = item_id / 32;
                int bit_idx = item_id % 32;
                uint32_t old = atomicOr(&seen_bitmap[word_idx], 1u << bit_idx);

                if (!(old & (1u << bit_idx))) {
                    int pos = atomicAdd(&local_count, 1);
                    if (pos < 2048) {
                        local_candidates[pos] = item_id;
                    }
                }
            }
        }
        __syncthreads();

        // Check if we have enough candidates
        if (local_count >= target_recall) break;
    }

    int num_to_write = min(local_count, max_candidates);
    if (threadIdx.x == 0) {
        candidate_counts[query_id] = num_to_write;
    }

    for (int i = threadIdx.x; i < num_to_write; i += blockDim.x) {
        candidates[query_id * max_candidates + i] = local_candidates[i];
    }
}
