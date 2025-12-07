#!/usr/bin/env python3
"""
MCP Query Interface with Decision Logic Visualization

Displays all recommendation decision logic and fields in real-time.
Runs on DISPLAY=:1 for visual monitoring.
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from collections import deque
from threading import Lock, Thread
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.trt_inference import TensorRTEncoder
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    from utils.gpu_ontology_reasoning import GPUOntologyReasoner
    ONTOLOGY_AVAILABLE = True
except ImportError:
    ONTOLOGY_AVAILABLE = False
    print("[Warning] Ontology reasoning not available")

app = Flask(__name__)


class BatchProcessor:
    """
    Batch request processor for high-throughput query processing

    Accumulates incoming requests and processes them in batches using TensorRT
    to maximize GPU utilization and achieve 1000+ QPS.
    """

    def __init__(self, encoder, max_batch_size=32, max_wait_ms=50):
        """
        Initialize batch processor

        Args:
            encoder: TensorRT or SentenceTransformer encoder
            max_batch_size: Maximum batch size (matches TensorRT engine)
            max_wait_ms: Maximum wait time before processing partial batch (ms)
        """
        self.encoder = encoder
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds

        # Request queue
        self.queue = deque()
        self.lock = Lock()

        # Processing thread
        self.running = False
        self.processor_thread = None

    def start(self):
        """Start batch processing thread"""
        if not self.running:
            self.running = True
            self.processor_thread = Thread(target=self._process_loop, daemon=True)
            self.processor_thread.start()

    def stop(self):
        """Stop batch processing thread"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)

    def _process_loop(self):
        """Main processing loop - runs in background thread"""
        while self.running:
            batch_queries = []
            batch_futures = []

            # Collect batch
            start_time = time.time()
            while len(batch_queries) < self.max_batch_size:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.max_wait_ms and batch_queries:
                    break

                # Try to get request from queue
                with self.lock:
                    if self.queue:
                        query, future = self.queue.popleft()
                        batch_queries.append(query)
                        batch_futures.append(future)
                    elif batch_queries:
                        # Have some requests and queue is empty - process now
                        break
                    else:
                        # Queue empty, no requests - wait a bit
                        time.sleep(0.001)  # 1ms sleep to avoid busy waiting
                        continue

            # Process batch if we have requests
            if batch_queries:
                try:
                    # Encode all queries in single TensorRT call
                    embeddings = self.encoder.encode(
                        batch_queries,
                        batch_size=len(batch_queries),
                        convert_to_tensor=True
                    )

                    # Return results to individual requests
                    for i, future in enumerate(batch_futures):
                        future['result'] = embeddings[i]
                        future['ready'] = True

                except Exception as e:
                    # Return error to all requests in batch
                    for future in batch_futures:
                        future['error'] = str(e)
                        future['ready'] = True

    def encode(self, query, timeout=5.0):
        """
        Encode query (batched)

        Args:
            query: Query text
            timeout: Maximum wait time for result (seconds)

        Returns:
            Embedding tensor
        """
        # Create future for result
        future = {'result': None, 'error': None, 'ready': False}

        # Add to queue
        with self.lock:
            self.queue.append((query, future))

        # Wait for result
        start = time.time()
        while not future['ready']:
            if time.time() - start > timeout:
                raise TimeoutError(f"Batch encoding timeout after {timeout}s")
            time.sleep(0.0001)  # 0.1ms sleep

        if future['error']:
            raise RuntimeError(f"Batch encoding error: {future['error']}")

        return future['result']


class QueryInterfaceBackend:
    """Backend for query interface with full decision logging"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Interface] Initializing on {self.device}...")

        # Load model (with TensorRT if available)
        engine_path = Path(__file__).parent.parent.parent / "models/sentence_transformer_fp16_sm86.trt"

        if TRT_AVAILABLE and engine_path.exists():
            print(f"[Interface] Loading TensorRT engine from {engine_path}")
            self.encoder = TensorRTEncoder(
                engine_path=str(engine_path),
                device=str(self.device)
            )
            self.backend_type = "TensorRT FP16"
        else:
            print(f"[Interface] Loading SentenceTransformer model")
            self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.encoder.to(self.device)
            self.backend_type = f"PyTorch ({self.device})"

        # Load embeddings
        self.load_data()

        # Initialize ontology reasoner if available
        self.ontology_reasoner = None
        if ONTOLOGY_AVAILABLE:
            try:
                print(f"[Interface] Loading ontology reasoner...")
                self.ontology_reasoner = GPUOntologyReasoner(
                    base_path=str(Path(__file__).parent.parent.parent)
                )
                print(f"[Interface] Ontology reasoning enabled")
            except Exception as e:
                print(f"[Warning] Failed to load ontology reasoner: {e}")

        # Initialize batch processor for high-throughput queries
        self.batch_processor = BatchProcessor(
            encoder=self.encoder,
            max_batch_size=32,
            max_wait_ms=50
        )
        self.batch_processor.start()

        print(f"[Interface] Ready! Backend: {self.backend_type}")
        print(f"[Interface] Batch processor enabled (batch_size=32, max_wait=50ms)")

    def load_data(self):
        """Load media embeddings and metadata"""
        base_path = Path(__file__).parent.parent.parent / "data/embeddings/media"

        if not base_path.exists():
            print(f"[Warning] Data path not found: {base_path}")
            print("[Info] Creating sample data for demo...")
            self.create_sample_data()
            return

        vectors = np.load(base_path / "content_vectors.npy")
        self.media_embeddings = torch.from_numpy(vectors).to(self.device)

        self.media_metadata = {}
        self.media_ids = []
        with open(base_path / "metadata.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                media_id = item['media_id']
                self.media_ids.append(media_id)
                self.media_metadata[media_id] = item

        print(f"[Data] Loaded {len(self.media_ids)} items")

    def create_sample_data(self):
        """Create sample data for demo"""
        self.media_ids = [f"movie_{i}" for i in range(100)]
        self.media_metadata = {}

        genres_pool = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller', 'Romance']
        sample_titles = [
            "The Matrix", "Inception", "Interstellar", "The Godfather", "Pulp Fiction",
            "The Dark Knight", "Forrest Gump", "Fight Club", "The Shawshank Redemption",
            "Goodfellas", "The Silence of the Lambs", "Saving Private Ryan"
        ]

        for i, media_id in enumerate(self.media_ids):
            self.media_metadata[media_id] = {
                'media_id': media_id,
                'title': sample_titles[i % len(sample_titles)] + f" #{i}",
                'genres': [genres_pool[i % len(genres_pool)], genres_pool[(i+1) % len(genres_pool)]],
                'year': 1990 + (i % 30),
                'language': 'en',
                'avg_rating': 3.0 + (i % 20) / 10.0
            }

        # Create random embeddings
        self.media_embeddings = torch.randn(100, 384, device=self.device)
        self.media_embeddings = F.normalize(self.media_embeddings, p=2, dim=1)

    def process_query(self, query: str, limit: int = 10, filters: dict = None, use_batch: bool = False) -> dict:
        """
        Process query with full decision logging

        Args:
            query: Query text
            limit: Maximum number of results
            filters: Optional filters (genres, year_range, min_rating)
            use_batch: Use batch processor for high-throughput encoding

        Returns:
            Results with decision log and performance metrics
        """
        decision_log = {
            'query': query,
            'timestamp': time.time(),
            'backend': self.backend_type,
            'device': str(self.device),
            'batch_mode': use_batch,
            'steps': []
        }

        # Step 1: Query Encoding
        start = time.time()

        if use_batch:
            # Use batch processor (queues request for batch processing)
            query_embedding = self.batch_processor.encode(query)
        else:
            # Direct encoding (bypass batch processor)
            query_embedding = self.encoder.encode(
                query,
                convert_to_tensor=True,
                device=self.device if hasattr(self.encoder, 'device') else None
            )

        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding, device=self.device)

        encoding_time = (time.time() - start) * 1000

        decision_log['steps'].append({
            'step': 1,
            'name': 'Query Encoding',
            'duration_ms': round(encoding_time, 3),
            'output': {
                'shape': list(query_embedding.shape),
                'norm': round(float(torch.norm(query_embedding)), 4),
                'sample_values': query_embedding[:5].cpu().tolist()
            }
        })

        # Step 2: Normalization
        start = time.time()
        # Handle both 1D and 2D tensors (TensorRT returns 2D with batch dim)
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.squeeze(0)  # Remove batch dimension
        query_norm = F.normalize(query_embedding, p=2, dim=0)
        norm_time = (time.time() - start) * 1000

        decision_log['steps'].append({
            'step': 2,
            'name': 'L2 Normalization',
            'duration_ms': round(norm_time, 3),
            'output': {
                'normalized_norm': round(float(torch.norm(query_norm)), 4)
            }
        })

        # Step 3: Similarity Computation (GPU)
        start = time.time()
        similarities = torch.matmul(self.media_embeddings, query_norm)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        similarity_time = (time.time() - start) * 1000

        decision_log['steps'].append({
            'step': 3,
            'name': 'GPU Similarity Computation',
            'duration_ms': round(similarity_time, 3),
            'output': {
                'num_items': len(self.media_ids),
                'max_similarity': round(float(torch.max(similarities)), 4),
                'mean_similarity': round(float(torch.mean(similarities)), 4),
                'min_similarity': round(float(torch.min(similarities)), 4)
            }
        })

        # Step 4: Top-K Selection
        start = time.time()
        k = min(limit * 3, len(self.media_ids))
        top_k_vals, top_k_indices = torch.topk(similarities, k=k)
        topk_time = (time.time() - start) * 1000

        decision_log['steps'].append({
            'step': 4,
            'name': 'Top-K Selection',
            'duration_ms': round(topk_time, 3),
            'parameters': {'k': k, 'limit': limit},
            'output': {
                'candidates': k,
                'top_score': round(float(top_k_vals[0]), 4)
            }
        })

        # Step 5: Ontology Reasoning (if available)
        ontology_time = 0
        ontology_scores = {}

        if self.ontology_reasoner:
            start = time.time()
            # Compute ontology scores for top candidates
            for idx in top_k_indices.cpu().numpy():
                media_id = self.media_ids[idx]
                # Compute ontology similarity (using first result as query for demo)
                query_id = self.media_ids[top_k_indices[0].item()]
                onto_score = self.ontology_reasoner.ontology_similarity(query_id, media_id)
                genre_score = self.ontology_reasoner.genre_similarity(query_id, media_id)

                # Get shared ontology classes
                query_classes = set(self.ontology_reasoner.movie_ontology_classes.get(query_id, []))
                candidate_classes = set(self.ontology_reasoner.movie_ontology_classes.get(media_id, []))
                shared_classes = list(query_classes & candidate_classes)

                ontology_scores[media_id] = {
                    'ontology_score': onto_score,
                    'genre_score': genre_score,
                    'shared_classes': shared_classes[:5],
                    'total_classes': len(candidate_classes)
                }

            ontology_time = (time.time() - start) * 1000

            decision_log['steps'].append({
                'step': 5,
                'name': 'Ontology Reasoning',
                'duration_ms': round(ontology_time, 3),
                'output': {
                    'candidates_evaluated': len(ontology_scores),
                    'avg_ontology_score': round(sum(s['ontology_score'] for s in ontology_scores.values()) / len(ontology_scores) if ontology_scores else 0, 4),
                    'avg_genre_score': round(sum(s['genre_score'] for s in ontology_scores.values()) / len(ontology_scores) if ontology_scores else 0, 4),
                    'ontology_classes_found': sum(1 for s in ontology_scores.values() if s['shared_classes'])
                }
            })

        # Step 6: Filtering & Hybrid Ranking
        start = time.time()
        results = []
        filtered_count = 0

        for idx, sim in zip(top_k_indices.cpu().numpy(), top_k_vals.cpu().numpy()):
            media_id = self.media_ids[idx]
            metadata = self.media_metadata[media_id]

            # Apply filters
            filter_reason = None
            if filters:
                if 'genres' in filters and filters['genres']:
                    if not any(g in metadata.get('genres', []) for g in filters['genres']):
                        filter_reason = f"Genre mismatch (has {metadata.get('genres', [])})"
                        filtered_count += 1
                        continue

                if 'min_rating' in filters:
                    if metadata.get('avg_rating', 0) < filters['min_rating']:
                        filter_reason = f"Rating too low ({metadata.get('avg_rating', 0)} < {filters['min_rating']})"
                        filtered_count += 1
                        continue

                if 'year_range' in filters:
                    year = metadata.get('year', 0)
                    if not (filters['year_range'][0] <= year <= filters['year_range'][1]):
                        filter_reason = f"Year out of range ({year} not in {filters['year_range']})"
                        filtered_count += 1
                        continue

            # Build result with ontology info
            result = {
                'rank': len(results) + 1,
                'id': media_id,
                'title': metadata['title'],
                'similarity_score': round(float(sim), 4),
                'metadata': {
                    'genres': metadata.get('genres', []),
                    'year': metadata.get('year'),
                    'language': metadata.get('language', 'en'),
                    'rating': metadata.get('avg_rating', 0.0)
                }
            }

            # Add ontology information if available and compute hybrid score
            if media_id in ontology_scores:
                result['ontology'] = {
                    'ontology_score': round(ontology_scores[media_id]['ontology_score'], 4),
                    'genre_score': round(ontology_scores[media_id]['genre_score'], 4),
                    'shared_classes': ontology_scores[media_id]['shared_classes'],
                    'total_classes': ontology_scores[media_id]['total_classes']
                }

                # Compute hybrid score
                weights = {'semantic': 0.7, 'ontology': 0.2, 'genre': 0.1}
                hybrid_score = round(
                    weights['semantic'] * float(sim) +
                    weights['ontology'] * ontology_scores[media_id]['ontology_score'] +
                    weights['genre'] * ontology_scores[media_id]['genre_score'],
                    4
                )
                result['hybrid_score'] = hybrid_score
                result['score'] = hybrid_score  # Unified score field (hybrid when available)
            else:
                # No ontology info - use semantic score only
                result['score'] = round(float(sim), 4)

            results.append(result)

            if len(results) >= limit:
                break

        filter_time = (time.time() - start) * 1000

        decision_log['steps'].append({
            'step': 6 if self.ontology_reasoner else 5,
            'name': 'Filtering & Hybrid Ranking',
            'duration_ms': round(filter_time, 3),
            'filters_applied': filters or {},
            'weights': {'semantic': 0.7, 'ontology': 0.2, 'genre': 0.1} if self.ontology_reasoner else {'semantic': 1.0},
            'output': {
                'candidates_evaluated': k,
                'items_filtered': filtered_count,
                'items_returned': len(results),
                'hybrid_scoring_enabled': self.ontology_reasoner is not None
            }
        })

        # Final timing
        total_time = sum(step['duration_ms'] for step in decision_log['steps'])

        return {
            'results': results,
            'decision_log': decision_log,
            'performance': {
                'total_time_ms': round(total_time, 3),
                'encoding_time_ms': round(encoding_time, 3),
                'similarity_time_ms': round(similarity_time, 3),
                'items_searched': len(self.media_ids),
                'results_returned': len(results)
            }
        }


# Global backend instance
backend = None


@app.route('/')
def index():
    """Serve main interface"""
    return render_template('query_interface.html', backend_type=backend.backend_type)


@app.route('/api/query', methods=['POST'])
def api_query():
    """Process single query and return decision log"""
    data = request.json
    query = data.get('query', '')
    limit = data.get('limit', 10)
    filters = data.get('filters', {})
    use_batch = data.get('use_batch', False)  # Optional batch mode

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    result = backend.process_query(query, limit, filters, use_batch=use_batch)
    return jsonify(result)


@app.route('/api/query/batch', methods=['POST'])
def api_query_batch():
    """
    Process multiple queries in batch for high-throughput

    Request body:
        {
            "queries": ["query1", "query2", ...],  # List of queries
            "limit": 10,  # Results per query (default: 10)
            "filters": {...}  # Optional filters
        }

    Response:
        {
            "results": [
                {
                    "query": "query1",
                    "results": [...],
                    "performance": {...}
                },
                ...
            ],
            "batch_performance": {
                "total_queries": N,
                "total_time_ms": X,
                "avg_time_per_query_ms": Y,
                "qps": Z
            }
        }
    """
    data = request.json

    # Support both single query and array of queries
    queries = data.get('queries', [])
    if isinstance(queries, str):
        queries = [queries]

    if not queries:
        # Try single query field for backward compatibility
        single_query = data.get('query', '')
        if single_query:
            queries = [single_query]

    if not queries:
        return jsonify({'error': 'queries (array) or query (string) is required'}), 400

    limit = data.get('limit', 10)
    filters = data.get('filters', {})

    # Process all queries using batch processor
    batch_start = time.time()
    batch_results = []

    for query in queries:
        try:
            result = backend.process_query(query, limit, filters, use_batch=True)
            batch_results.append({
                'query': query,
                'results': result['results'],
                'performance': result['performance']
            })
        except Exception as e:
            batch_results.append({
                'query': query,
                'error': str(e)
            })

    batch_time = (time.time() - batch_start) * 1000

    # Batch performance metrics
    response = {
        'results': batch_results,
        'batch_performance': {
            'total_queries': len(queries),
            'total_time_ms': round(batch_time, 3),
            'avg_time_per_query_ms': round(batch_time / len(queries), 3) if queries else 0,
            'qps': round(len(queries) / (batch_time / 1000), 2) if batch_time > 0 else 0
        }
    }

    return jsonify(response)


@app.route('/api/status')
def api_status():
    """System status"""
    return jsonify({
        'backend': backend.backend_type,
        'device': str(backend.device),
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'items_loaded': len(backend.media_ids),
        'batch_processor': {
            'enabled': True,
            'max_batch_size': backend.batch_processor.max_batch_size,
            'max_wait_ms': backend.batch_processor.max_wait_ms * 1000,
            'queue_size': len(backend.batch_processor.queue),
            'running': backend.batch_processor.running
        }
    })


def main():
    global backend

    print("=" * 80)
    print("MCP Query Interface - Decision Logic Visualization")
    print("=" * 80)

    backend = QueryInterfaceBackend()

    print(f"\n🌐 Starting server on http://0.0.0.0:5000")
    print(f"📊 Backend: {backend.backend_type}")
    print(f"💾 Items loaded: {len(backend.media_ids)}")
    print(f"\n🖥️  Open in browser on DISPLAY=:1")
    print("=" * 80)

    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
