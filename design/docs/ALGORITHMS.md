# Vector Search Algorithms Guide

## Table of Contents
- [Overview](#overview)
- [HNSW Algorithm](#hnsw-algorithm)
- [LSH (Locality Sensitive Hashing)](#lsh-locality-sensitive-hashing)
- [Product Quantization](#product-quantization)
- [Hybrid Index Strategy](#hybrid-index-strategy)
- [Algorithm Selection Guide](#algorithm-selection-guide)

---

## Overview

This guide provides in-depth explanations of vector similarity search algorithms used in the Media Gateway project.

### Algorithm Comparison Matrix

| Algorithm | Accuracy | Speed | Memory | Updates | Best For |
|-----------|----------|-------|--------|---------|----------|
| **Brute Force** | 100% | Slow (O(ND)) | 1x | Fast | <1M vectors, GPU available |
| **HNSW** | 95-99% | Fast (O(log N)) | 2-3x | Slow | 1M-100M, high recall |
| **LSH** | 80-95% | Fast (O(L×K)) | 1x | Fast | 10M-1B, frequent updates |
| **PQ** | 85-95% | Fast (O(N/8)) | 0.125x | Slow | 100M+, memory constrained |
| **Hybrid** | 90-98% | Very Fast | 0.5-1x | Medium | Production systems |

---

## HNSW Algorithm

### Hierarchical Navigable Small World

HNSW builds a multi-layer graph structure where:
- **Layer 0**: Contains all vectors
- **Higher layers**: Contain progressively fewer vectors (long-range connections)
- **Navigation**: Start at top layer, descend while getting closer to target

### Graph Structure

```
Layer 2:  A ←→ B                     (Sparse, long jumps)
          ↓    ↓
Layer 1:  A ←→ C ←→ D ←→ B           (Medium density)
          ↓    ↓    ↓    ↓
Layer 0:  A-C-E-F-G-D-H-I-B-J-K     (All vectors, short edges)
```

### Core Concepts

#### 1. Layer Assignment

Vectors are assigned to layers probabilistically:

```python
def assign_layer(ml=1.0/math.log(2)):
    """
    ml: Normalization factor (default 1/ln(2))
    Returns: Layer number (0 to max_layer)

    Distribution:
    - 50% of vectors on layer 0 only
    - 25% on layers 0-1
    - 12.5% on layers 0-2
    - etc.
    """
    return math.floor(-math.log(random.uniform(0, 1)) * ml)
```

**Why this works:**
- Most vectors only on layer 0 (local connections)
- Few vectors on high layers (express lanes)
- Balanced tree-like structure

#### 2. Greedy Search

Search process moves toward target at each layer:

```python
def greedy_search(query, entry_point, num_neighbors, layer):
    """
    Greedy best-first search on a single layer

    Args:
        query: Query vector
        entry_point: Starting node
        num_neighbors: Number of closest to track
        layer: Which graph layer

    Returns:
        candidates: num_neighbors closest nodes found
    """
    visited = set()
    candidates = [(distance(query, entry_point), entry_point)]
    results = candidates.copy()

    while candidates:
        # Get closest unvisited candidate
        current_dist, current = heapq.heappop(candidates)

        # If further than worst result, stop
        if current_dist > results[0][0]:
            break

        # Explore neighbors
        for neighbor in get_neighbors(current, layer):
            if neighbor not in visited:
                visited.add(neighbor)
                dist = distance(query, neighbor)

                # Update results if closer
                if dist < results[0][0] or len(results) < num_neighbors:
                    heapq.heappush(candidates, (dist, neighbor))
                    heapq.heappush(results, (dist, neighbor))

                    # Keep only top num_neighbors
                    if len(results) > num_neighbors:
                        heapq.heappop(results)

    return results
```

#### 3. Insert Algorithm

Adding a new vector:

```python
def insert_vector(vector, M=16, ef_construction=200):
    """
    M: Max connections per node (typically 16-48)
    ef_construction: Search quality during construction

    Steps:
    1. Assign layer to new vector
    2. Find entry point (top layer node)
    3. Descend layers, searching for nearest neighbors
    4. Connect to M nearest at each layer
    5. Prune connections to maintain M limit
    """
    # Step 1: Assign layer
    new_layer = assign_layer()

    # Step 2: Find entry point
    entry_point = get_random_node_at_layer(max_layer)
    current_nearest = entry_point

    # Step 3: Navigate from top to target layer
    for layer in range(max_layer, new_layer, -1):
        current_nearest = greedy_search(
            vector,
            current_nearest,
            num_neighbors=1,
            layer=layer
        )[0]

    # Step 4: Connect at each layer (target layer to 0)
    for layer in range(new_layer, -1, -1):
        # Find M nearest neighbors
        candidates = greedy_search(
            vector,
            current_nearest,
            num_neighbors=ef_construction,
            layer=layer
        )

        # Select M best connections
        neighbors = select_neighbors_heuristic(candidates, M, layer)

        # Bidirectional connection
        for neighbor in neighbors:
            add_connection(vector, neighbor, layer)
            add_connection(neighbor, vector, layer)

            # Prune neighbor's connections if exceeds M
            if len(get_neighbors(neighbor, layer)) > M:
                prune_connections(neighbor, M, layer)
```

#### 4. Neighbor Selection Heuristic

Smart neighbor selection maintains graph quality:

```python
def select_neighbors_heuristic(candidates, M, layer, extend_candidates=True):
    """
    Heuristic to select diverse, well-connected neighbors

    Key insight: Prefer neighbors that are:
    1. Close to query
    2. Not redundant (cover different regions)
    3. Well-connected in graph
    """
    # Sort candidates by distance
    candidates = sorted(candidates, key=lambda x: x[0])

    selected = []
    for dist, candidate in candidates:
        if len(selected) >= M:
            break

        # Check if candidate is too close to already selected
        redundant = False
        if extend_candidates:
            for sel_dist, selected_node in selected:
                if distance(candidate, selected_node) < dist:
                    redundant = True
                    break

        if not redundant:
            selected.append((dist, candidate))

    return selected
```

### Search Algorithm

```python
def search_hnsw(query, k=10, ef_search=100):
    """
    Search for k nearest neighbors

    Args:
        query: Query vector
        k: Number of results
        ef_search: Search quality (higher = better recall, slower)

    Returns:
        k nearest neighbors

    Time complexity: O(log N × D × ef_search)
    """
    # Start at highest layer
    entry_point = get_entry_point()
    current_nearest = entry_point

    # Navigate to layer 0
    for layer in range(max_layer, 0, -1):
        current_nearest = greedy_search(
            query,
            current_nearest,
            num_neighbors=1,
            layer=layer
        )[0]

    # Search layer 0 with high quality
    results = greedy_search(
        query,
        current_nearest,
        num_neighbors=ef_search,
        layer=0
    )

    # Return top-k
    return sorted(results, key=lambda x: x[0])[:k]
```

### Parameter Tuning

#### M (Max Connections)

```python
M = 16  # Default, good for most cases
# M = 32  # Higher recall, 2x memory
# M = 8   # Lower memory, reduced recall
```

**Trade-offs:**
- **M=8**: Faster build (30%), lower recall (-5%), less memory
- **M=16**: Balanced (recommended)
- **M=32**: Higher recall (+2%), slower build (+40%), more memory
- **M=48**: Diminishing returns, mostly for very high dimensions

#### ef_construction

```python
ef_construction = 200  # Default for high quality
# ef_construction = 100  # Faster build, slightly lower recall
# ef_construction = 400  # Slower build, marginal recall gain
```

**Guidelines:**
- Start with 200
- Increase if recall < 95%
- Decrease if build time unacceptable

#### ef_search

```python
ef_search = 50   # Fast search, ~90% recall
ef_search = 100  # Balanced, ~95% recall
ef_search = 200  # High recall, ~98%, slower
ef_search = 500  # Very high recall, ~99.5%, much slower
```

**Dynamic adjustment:**
```python
# Adjust based on query complexity
if high_precision_required:
    ef_search = 200
else:
    ef_search = 50
```

### Implementation Example

```python
import hnswlib
import numpy as np

class HNSWIndex:
    def __init__(self, dim, max_elements=1000000):
        """
        Args:
            dim: Embedding dimension
            max_elements: Maximum vectors (can grow)
        """
        self.dim = dim
        self.index = hnswlib.Index(space='cosine', dim=dim)

        # Initialize index
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )

        self.index.set_ef(50)  # Default ef_search

    def add_vectors(self, vectors, ids=None):
        """
        Add vectors to index

        Args:
            vectors: [N, dim] numpy array
            ids: Optional vector IDs (default: 0, 1, 2, ...)
        """
        if ids is None:
            ids = np.arange(len(vectors))

        self.index.add_items(vectors, ids, num_threads=-1)

    def search(self, queries, k=10, ef_search=None):
        """
        Search for nearest neighbors

        Args:
            queries: [batch_size, dim] numpy array
            k: Number of results per query
            ef_search: Override default ef_search

        Returns:
            labels: [batch_size, k] indices
            distances: [batch_size, k] distances
        """
        if ef_search is not None:
            self.index.set_ef(ef_search)

        labels, distances = self.index.knn_query(queries, k=k, num_threads=-1)

        return labels, distances

    def save(self, path):
        """Save index to disk"""
        self.index.save_index(path)

    def load(self, path):
        """Load index from disk"""
        self.index.load_index(path)

# Usage example
index = HNSWIndex(dim=768, max_elements=1000000)
index.add_vectors(embeddings, ids=doc_ids)
labels, distances = index.search(query_embeddings, k=10)
```

### Memory Considerations

```python
def estimate_hnsw_memory(num_vectors, dimension, M=16):
    """
    Estimate HNSW memory usage

    Returns: Memory in GB
    """
    # Base vector storage
    base_memory = num_vectors * dimension * 4  # 4 bytes per float

    # Graph structure
    # Average layer: 1/(1 - 1/2) = 2 layers per vector
    # Each layer: M connections × 4 bytes (int)
    graph_memory = num_vectors * 2 * M * 4

    # Metadata
    metadata = num_vectors * 16  # Layer info, etc.

    total_bytes = base_memory + graph_memory + metadata
    total_gb = total_bytes / 1024**3

    return {
        'base_gb': base_memory / 1024**3,
        'graph_gb': graph_memory / 1024**3,
        'total_gb': total_gb,
        'overhead_ratio': total_gb / (base_memory / 1024**3)
    }

# Example: 1M vectors, 768-dim, M=16
memory = estimate_hnsw_memory(1_000_000, 768, M=16)
# base_gb: 2.86 GB
# graph_gb: 0.12 GB
# total_gb: 2.98 GB
# overhead_ratio: 1.04x (4% overhead)
```

---

## LSH (Locality Sensitive Hashing)

### Core Principle

Map similar vectors to the same hash buckets with high probability.

**Key Insight:**
- Traditional hash: `hash(x) ≠ hash(x + ε)` (different if slightly different)
- LSH: `hash(x) ≈ hash(x + ε)` (same if similar)

### Random Projection LSH

Most common method for cosine similarity:

```python
class RandomProjectionLSH:
    def __init__(self, dim, num_tables=8, hash_length=12):
        """
        Args:
            dim: Embedding dimension
            num_tables (L): Number of hash tables (more = higher recall)
            hash_length (K): Bits per hash (longer = fewer collisions)
        """
        self.dim = dim
        self.L = num_tables
        self.K = hash_length

        # Random projection matrices [L, K, D]
        self.projections = np.random.randn(num_tables, hash_length, dim)

        # Hash tables: {hash_code: [vector_ids]}
        self.hash_tables = [defaultdict(list) for _ in range(num_tables)]

    def hash_vector(self, vector):
        """
        Compute L hash codes for vector

        Args:
            vector: [D] embedding

        Returns:
            hashes: List of L hash codes (bit strings)
        """
        hashes = []
        for l in range(self.L):
            # Project: [K, D] @ [D] = [K]
            projection = self.projections[l] @ vector

            # Threshold at 0: positive -> 1, negative -> 0
            hash_bits = (projection > 0).astype(int)

            # Convert to integer hash code
            hash_code = int(''.join(map(str, hash_bits)), 2)
            hashes.append(hash_code)

        return hashes

    def insert(self, vector, vector_id):
        """
        Insert vector into all L hash tables
        """
        hashes = self.hash_vector(vector)
        for l, hash_code in enumerate(hashes):
            self.hash_tables[l][hash_code].append(vector_id)

    def query(self, vector, top_k=10):
        """
        Find candidate neighbors

        Returns:
            candidates: Set of vector IDs that share buckets
        """
        hashes = self.hash_vector(vector)
        candidates = set()

        for l, hash_code in enumerate(hashes):
            # Get all vectors in same bucket
            bucket = self.hash_tables[l].get(hash_code, [])
            candidates.update(bucket)

        return candidates
```

### Theory: Collision Probability

For random projection LSH with cosine similarity:

```
P(hash collision) = 1 - θ/π

Where θ = angle between vectors

Examples:
- θ = 0° (identical):       P = 1.0  (always collide)
- θ = 45° (0.707 cosine):   P = 0.75
- θ = 90° (orthogonal):     P = 0.5
- θ = 180° (opposite):      P = 0.0
```

For K-bit hash code:
```
P(exact match) = (1 - θ/π)^K

K=8:  (0.75)^8  = 0.10  (10% collision rate)
K=12: (0.75)^12 = 0.032 (3.2% collision rate)
K=16: (0.75)^16 = 0.010 (1% collision rate)
```

For L hash tables (union):
```
P(collision in at least 1 table) = 1 - (1 - P(exact match))^L

L=8, K=12:  1 - (1 - 0.032)^8 = 0.23  (23% chance)
L=16, K=12: 1 - (1 - 0.032)^16 = 0.40 (40% chance)
```

### Parameter Selection

#### Number of Tables (L)

```python
# More tables = higher recall, slower search
L = 4   # Fast (50-80% recall)
L = 8   # Balanced (70-90% recall)
L = 16  # Slow (85-95% recall)
L = 32  # Very slow (90-98% recall)
```

**Rule of thumb:**
- Start with L=8
- Increase if recall < target
- Each doubling adds ~5-10% recall but doubles search time

#### Hash Length (K)

```python
# Longer hash = fewer false positives, more false negatives
K = 8   # Many collisions (broad search)
K = 12  # Balanced
K = 16  # Few collisions (precise search)
K = 20  # Very few collisions (may miss neighbors)
```

**Guidelines:**
- K=12 works for most cases
- Increase K if too many candidates (>10% of database)
- Decrease K if too few candidates (<100)

### Multi-Probe LSH

Improve recall by probing nearby buckets:

```python
def multi_probe_query(self, vector, num_probes=3):
    """
    Query nearby hash buckets by flipping bits

    Args:
        vector: Query vector
        num_probes: Number of buckets to probe per table

    Returns:
        candidates: Expanded candidate set
    """
    base_hashes = self.hash_vector(vector)
    candidates = set()

    for l, base_hash in enumerate(base_hashes):
        # Probe base bucket
        candidates.update(self.hash_tables[l].get(base_hash, []))

        # Probe nearby buckets (flip 1-2 bits)
        for probe_idx in range(1, num_probes):
            # Flip bit at position probe_idx
            perturbed_hash = base_hash ^ (1 << (probe_idx % self.K))
            candidates.update(self.hash_tables[l].get(perturbed_hash, []))

    return candidates

# Multi-probe can increase recall by 10-20% with minimal overhead
```

### LSH Forest

Hierarchical LSH for dynamic updates:

```python
class LSHForest:
    """
    Multiple LSH indices with different random seeds
    Balances recall and update speed
    """
    def __init__(self, dim, num_trees=10, max_leaf_size=100):
        self.trees = [
            RandomProjectionLSH(dim, num_tables=1, hash_length=32)
            for _ in range(num_trees)
        ]

    def insert(self, vector, vector_id):
        for tree in self.trees:
            tree.insert(vector, vector_id)

    def query(self, vector, top_k=10):
        # Aggregate candidates from all trees
        all_candidates = set()
        for tree in self.trees:
            candidates = tree.query(vector)
            all_candidates.update(candidates)

        # Rank candidates
        return rank_candidates(vector, all_candidates, top_k)
```

### Implementation with FAISS

```python
import faiss

class FAISSLSHIndex:
    def __init__(self, dim, num_bits=1024):
        """
        Use FAISS's optimized LSH implementation

        Args:
            dim: Embedding dimension
            num_bits: Hash code length (recommend 4-16 × dim)
        """
        self.dim = dim
        self.index = faiss.IndexLSH(dim, num_bits)

    def add(self, vectors):
        """Add vectors to index"""
        self.index.add(vectors.astype('float32'))

    def search(self, queries, k=10):
        """Search with LSH"""
        distances, indices = self.index.search(queries.astype('float32'), k)
        return indices, distances
```

---

## Product Quantization

### Concept

Compress vectors by:
1. Splitting into subspaces
2. Quantizing each subspace to nearest centroid
3. Storing centroid IDs instead of full vectors

**Example:**
```
Original: 768-dim × 4 bytes = 3,072 bytes
PQ (M=8): 8 subspaces × 1 byte = 8 bytes
Compression: 384x
```

### Algorithm

```python
class ProductQuantizer:
    def __init__(self, dim, num_subspaces=8, num_centroids=256):
        """
        Args:
            dim: Embedding dimension (must be divisible by num_subspaces)
            num_subspaces (M): Number of subspaces
            num_centroids (K): Codebook size per subspace (typically 256)
        """
        self.M = num_subspaces
        self.K = num_centroids
        self.D = dim
        self.subspace_dim = dim // num_subspaces

        # Codebooks: [M, K, subspace_dim]
        self.codebooks = None

    def train(self, training_vectors):
        """
        Learn codebooks via K-means clustering

        Args:
            training_vectors: [N, D] training data
        """
        self.codebooks = []

        for m in range(self.M):
            # Extract subspace
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            subspace_data = training_vectors[:, start:end]

            # Run K-means to find K centroids
            kmeans = KMeans(n_clusters=self.K, n_init=10)
            kmeans.fit(subspace_data)

            # Store centroids [K, subspace_dim]
            self.codebooks.append(kmeans.cluster_centers_)

        self.codebooks = np.array(self.codebooks)  # [M, K, subspace_dim]

    def encode(self, vectors):
        """
        Encode vectors to PQ codes

        Args:
            vectors: [N, D] vectors to encode

        Returns:
            codes: [N, M] PQ codes (each element 0-255)
        """
        N = len(vectors)
        codes = np.zeros((N, self.M), dtype=np.uint8)

        for m in range(self.M):
            # Extract subspace
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            subspace_vectors = vectors[:, start:end]

            # Find nearest centroid in codebook
            distances = cdist(subspace_vectors, self.codebooks[m])
            codes[:, m] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes):
        """
        Reconstruct approximate vectors from codes

        Args:
            codes: [N, M] PQ codes

        Returns:
            vectors: [N, D] reconstructed vectors
        """
        N = len(codes)
        vectors = np.zeros((N, self.D), dtype=np.float32)

        for m in range(self.M):
            start = m * self.subspace_dim
            end = start + self.subspace_dim

            # Lookup centroids
            vectors[:, start:end] = self.codebooks[m][codes[:, m]]

        return vectors

    def search(self, query, database_codes, top_k=10):
        """
        Asymmetric Distance Computation (ADC)

        Args:
            query: [D] query vector (NOT quantized)
            database_codes: [N, M] PQ codes of database
            top_k: Number of results

        Returns:
            indices: [top_k] indices of nearest neighbors
            distances: [top_k] approximate distances
        """
        # Build distance table: [M, K]
        distance_table = np.zeros((self.M, self.K))

        for m in range(self.M):
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            query_subspace = query[start:end]

            # Compute distances to all K centroids
            distance_table[m] = np.linalg.norm(
                self.codebooks[m] - query_subspace[np.newaxis, :],
                axis=1
            )

        # Compute approximate distances
        N = len(database_codes)
        approx_distances = np.zeros(N)

        for n in range(N):
            # Sum distances using lookup table
            approx_distances[n] = sum(
                distance_table[m, database_codes[n, m]]
                for m in range(self.M)
            )

        # Top-k selection
        top_k_indices = np.argpartition(approx_distances, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(approx_distances[top_k_indices])]

        return top_k_indices, approx_distances[top_k_indices]
```

### Training Considerations

#### Sample Size

```python
# Need enough samples per centroid
min_samples = num_centroids * 100  # 100 samples per centroid
recommended_samples = num_centroids * 1000  # 1000 for better quality

# For M=8, K=256:
min_samples = 256 * 100 = 25,600 vectors
recommended_samples = 256 * 1000 = 256,000 vectors
```

#### Subspace Dimension

```python
# Subspace dimension affects quantization quality
dim = 768
M = 8   # subspace_dim = 96 (good)
M = 16  # subspace_dim = 48 (better compression, lower quality)
M = 4   # subspace_dim = 192 (lower compression, higher quality)

# Rule of thumb: subspace_dim should be 32-128
```

### Optimized PQ (OPQ)

Learns rotation matrix to improve quantization:

```python
class OptimizedPQ(ProductQuantizer):
    def train(self, training_vectors):
        """
        Optimized Product Quantization
        1. Learn rotation matrix
        2. Rotate vectors
        3. Train PQ on rotated vectors
        """
        # Learn rotation via PCA or learned optimization
        U, S, Vt = np.linalg.svd(training_vectors, full_matrices=False)
        self.rotation_matrix = Vt.T  # [D, D]

        # Rotate training data
        rotated_vectors = training_vectors @ self.rotation_matrix

        # Train PQ on rotated vectors
        super().train(rotated_vectors)

    def encode(self, vectors):
        # Rotate before encoding
        rotated = vectors @ self.rotation_matrix
        return super().encode(rotated)

    def search(self, query, database_codes, top_k=10):
        # Rotate query
        rotated_query = query @ self.rotation_matrix
        return super().search(rotated_query, database_codes, top_k)

# OPQ typically improves recall by 5-15%
```

### Implementation with FAISS

```python
import faiss

class FAISSPQIndex:
    def __init__(self, dim, num_subspaces=8, nbits=8):
        """
        FAISS Product Quantization index

        Args:
            dim: Embedding dimension
            num_subspaces (M): Must divide dim
            nbits: Bits per subspace (8 = 256 centroids)
        """
        self.dim = dim
        self.index = faiss.IndexPQ(dim, num_subspaces, nbits)

    def train(self, training_vectors):
        """Train codebooks"""
        self.index.train(training_vectors.astype('float32'))

    def add(self, vectors):
        """Add vectors (will be compressed)"""
        self.index.add(vectors.astype('float32'))

    def search(self, queries, k=10):
        """Search with PQ"""
        distances, indices = self.index.search(queries.astype('float32'), k)
        return indices, distances
```

---

## Hybrid Index Strategy

### Multi-Stage Pipeline

Combine algorithms for optimal performance:

```
Stage 1: Coarse Filter (LSH)
  Input:  100M vectors
  Output: 10K candidates (1000x reduction)
  Time:   1-2ms

Stage 2: Medium Filter (PQ)
  Input:  10K candidates
  Output: 1K candidates (10x reduction)
  Time:   2-3ms

Stage 3: Exact Reranking (GPU Brute Force)
  Input:  1K candidates
  Output: Top-10 results
  Time:   1-2ms

Total: 4-7ms for 100M vectors vs 2000ms brute force
```

### Implementation

```python
class HybridIndex:
    def __init__(self, dim, database_vectors):
        """
        Multi-stage hybrid index

        Stage 1: LSH (reduce 1000x)
        Stage 2: PQ (reduce 10x)
        Stage 3: GPU exact (final ranking)
        """
        self.dim = dim
        N = len(database_vectors)

        # Stage 1: LSH with 8 tables
        self.lsh = RandomProjectionLSH(
            dim=dim,
            num_tables=8,
            hash_length=12
        )
        for i, vec in enumerate(database_vectors):
            self.lsh.insert(vec, i)

        # Stage 2: PQ with 8 subspaces
        self.pq = ProductQuantizer(
            dim=dim,
            num_subspaces=8,
            num_centroids=256
        )
        # Train on sample
        train_sample = database_vectors[np.random.choice(N, min(N, 100000), replace=False)]
        self.pq.train(train_sample)
        self.pq_codes = self.pq.encode(database_vectors)

        # Stage 3: GPU tensors for exact ranking
        self.exact_vectors = torch.tensor(
            database_vectors,
            dtype=torch.float32,
            device='cuda'
        )
        self.exact_vectors = torch.nn.functional.normalize(
            self.exact_vectors, p=2, dim=1
        )

    def search(self, query, top_k=10):
        """
        Multi-stage search pipeline

        Args:
            query: [D] query vector
            top_k: Final number of results

        Returns:
            indices: [top_k] indices
            scores: [top_k] similarity scores
        """
        # Stage 1: LSH coarse filter
        stage1_candidates = self.lsh.query(query, top_k=None)

        # Limit candidates
        if len(stage1_candidates) > 10000:
            stage1_candidates = random.sample(list(stage1_candidates), 10000)
        stage1_candidates = list(stage1_candidates)

        if len(stage1_candidates) == 0:
            return np.array([]), np.array([])

        # Stage 2: PQ medium filter
        stage2_indices, _ = self.pq.search(
            query,
            self.pq_codes[stage1_candidates],
            top_k=min(1000, len(stage1_candidates))
        )

        # Map back to original indices
        stage2_candidates = [stage1_candidates[i] for i in stage2_indices]

        # Stage 3: GPU exact ranking
        query_tensor = torch.tensor(query, dtype=torch.float32, device='cuda')
        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=0)

        candidate_vectors = self.exact_vectors[stage2_candidates]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            similarities = torch.mv(candidate_vectors, query_tensor)

        # Top-k selection
        top_k_values, top_k_indices = torch.topk(
            similarities,
            k=min(top_k, len(stage2_candidates))
        )

        # Map back to original indices
        final_indices = [stage2_candidates[i] for i in top_k_indices.cpu().numpy()]
        final_scores = top_k_values.cpu().numpy()

        return np.array(final_indices), np.array(final_scores)

    def benchmark(self, queries, ground_truth_indices):
        """
        Benchmark hybrid index performance

        Args:
            queries: [N, D] test queries
            ground_truth_indices: [N, K] true nearest neighbors

        Returns:
            metrics: Dict with latency, recall, throughput
        """
        latencies = []
        recalls = []

        for i, query in enumerate(queries):
            start = time.time()
            indices, scores = self.search(query, top_k=10)
            latency = (time.time() - start) * 1000  # ms

            # Compute recall
            true_neighbors = set(ground_truth_indices[i])
            found_neighbors = set(indices)
            recall = len(true_neighbors & found_neighbors) / len(true_neighbors)

            latencies.append(latency)
            recalls.append(recall)

        return {
            'mean_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'mean_recall': np.mean(recalls),
            'throughput_qps': 1000 / np.mean(latencies)
        }
```

### Adaptive Stage Gating

Dynamically skip stages based on candidate counts:

```python
def adaptive_search(self, query, top_k=10):
    """
    Adaptively choose pipeline stages
    """
    # Stage 1: LSH
    candidates = self.lsh.query(query)

    # If few candidates, skip PQ and go straight to GPU
    if len(candidates) < 1000:
        return self.gpu_exact_search(query, list(candidates), top_k)

    # If many candidates, use PQ filter
    elif len(candidates) < 100000:
        pq_candidates = self.pq_filter(query, candidates, top_k=1000)
        return self.gpu_exact_search(query, pq_candidates, top_k)

    # If very many candidates, use all three stages
    else:
        pq_candidates = self.pq_filter(query, candidates, top_k=5000)
        refined_candidates = self.hnsw_refine(query, pq_candidates, top_k=1000)
        return self.gpu_exact_search(query, refined_candidates, top_k)
```

---

## Algorithm Selection Guide

### Decision Matrix

```python
def select_algorithm(num_vectors, recall_target, memory_budget_gb, update_frequency):
    """
    Automatic algorithm selection

    Args:
        num_vectors: Database size
        recall_target: Required recall (0.0-1.0)
        memory_budget_gb: Available memory
        update_frequency: 'low', 'medium', 'high'

    Returns:
        Recommended algorithm and parameters
    """
    # Calculate memory requirements
    vector_size_gb = num_vectors * dim * 4 / 1024**3

    # Small datasets: GPU brute force
    if num_vectors < 1_000_000 and memory_budget_gb > vector_size_gb:
        return {
            'algorithm': 'GPU Brute Force',
            'expected_latency_ms': 20 + num_vectors / 50000,
            'expected_recall': 1.0,
            'memory_gb': vector_size_gb
        }

    # Medium datasets, high recall: HNSW
    if num_vectors < 10_000_000 and recall_target > 0.95 and \
       memory_budget_gb > vector_size_gb * 2.5:
        return {
            'algorithm': 'HNSW',
            'parameters': {'M': 16, 'ef_construction': 200, 'ef_search': 100},
            'expected_latency_ms': math.log2(num_vectors) * 2,
            'expected_recall': 0.97,
            'memory_gb': vector_size_gb * 2.5
        }

    # Large datasets, memory constrained: PQ
    if memory_budget_gb < vector_size_gb * 0.5:
        return {
            'algorithm': 'Product Quantization',
            'parameters': {'M': 8, 'K': 256},
            'expected_latency_ms': num_vectors / 100000,
            'expected_recall': 0.88,
            'memory_gb': num_vectors * 8 / 1024**3
        }

    # Frequent updates: LSH
    if update_frequency == 'high':
        return {
            'algorithm': 'LSH',
            'parameters': {'L': 8, 'K': 12},
            'expected_latency_ms': 15,
            'expected_recall': 0.85,
            'memory_gb': vector_size_gb * 1.2
        }

    # Default: Hybrid
    return {
        'algorithm': 'Hybrid (LSH + PQ + GPU)',
        'parameters': {
            'lsh': {'L': 8, 'K': 12},
            'pq': {'M': 8, 'K': 256}
        },
        'expected_latency_ms': 10,
        'expected_recall': 0.92,
        'memory_gb': vector_size_gb * 0.3
    }
```

### Scaling Guidelines

| Database Size | Recommended Algorithm | Expected Latency | Memory Overhead |
|---------------|----------------------|------------------|-----------------|
| <100K | GPU Brute Force | <10ms | 1x |
| 100K-1M | GPU Brute Force or HNSW | 10-50ms | 1-2x |
| 1M-10M | HNSW | 20-100ms | 2-3x |
| 10M-100M | HNSW or Hybrid | 50-200ms | 1-2x |
| 100M-1B | Hybrid (LSH+PQ+GPU) | 10-50ms | 0.3-0.5x |
| >1B | Sharded Hybrid | 20-100ms | 0.2-0.3x |

---

## Summary

### Quick Reference

**When to use each algorithm:**

- **Brute Force + GPU**: Small datasets (<1M), need 100% recall, have GPU
- **HNSW**: Medium datasets (1M-100M), need high recall (95%+), have memory
- **LSH**: Large datasets, frequent updates, can tolerate 85-90% recall
- **PQ**: Very large datasets (100M+), memory constrained, 85-95% recall
- **Hybrid**: Production systems, need balance of speed/accuracy/memory

**Performance targets:**
- Latency: <50ms for most queries
- Recall: >90% for production
- Memory: <2x vector storage
- Throughput: >100 QPS per GPU
