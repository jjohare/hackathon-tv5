#!/usr/bin/env python3
"""
Comprehensive data validation script
Validates all generated data and database populations
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from pymilvus import connections, Collection, utility
from neo4j import GraphDatabase

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
AGENTDB_DIR = DATA_DIR / "agentdb"

# Database config
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j"

# Expected dimensions
EMBEDDING_DIM = 384


class ValidationReport:
    """Collects validation results"""

    def __init__(self):
        self.checks = []
        self.errors = []
        self.warnings = []

    def add_check(self, category: str, check: str, passed: bool, details: str = ""):
        self.checks.append({
            'category': category,
            'check': check,
            'passed': passed,
            'details': details
        })

        if not passed:
            self.errors.append(f"{category}: {check} - {details}")

    def add_warning(self, category: str, message: str):
        self.warnings.append(f"{category}: {message}")

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("📊 Validation Summary")
        print("=" * 60)

        passed = sum(1 for c in self.checks if c['passed'])
        failed = sum(1 for c in self.checks if not c['passed'])

        print(f"\nTotal Checks: {len(self.checks)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {len(self.warnings)}")

        if self.errors:
            print("\n❌ Errors:")
            for error in self.errors:
                print(f"   - {error}")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")

        # Save report
        report_path = DATA_DIR / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_checks': len(self.checks),
                    'passed': passed,
                    'failed': failed,
                    'warnings': len(self.warnings)
                },
                'checks': self.checks,
                'errors': self.errors,
                'warnings': self.warnings
            }, f, indent=2)

        print(f"\n💾 Report saved to {report_path}")

        return failed == 0


def validate_processed_data(report: ValidationReport):
    """Validate processed MovieLens data"""
    print("\n🎬 Validating Processed Data")
    print("-" * 60)

    # Check movies
    movies_path = PROCESSED_DIR / "media" / "movies.jsonl"
    if movies_path.exists():
        movie_count = 0
        with open(movies_path) as f:
            for line in f:
                movie_count += 1
                if movie_count == 1:
                    # Validate first entry structure
                    movie = json.loads(line)
                    required_fields = ['media_id', 'metadata', 'classification', 'identifiers']
                    for field in required_fields:
                        if field not in movie:
                            report.add_check('Processed Data', f'Movie has {field}', False)
                            break
                    else:
                        report.add_check('Processed Data', 'Movie structure', True)

        report.add_check('Processed Data', 'Movies file exists', True, f'{movie_count:,} movies')
        print(f"   ✓ Movies: {movie_count:,}")

        # Check expected range
        if movie_count < 60000:
            report.add_warning('Processed Data', f'Only {movie_count:,} movies (expected ~62K)')
    else:
        report.add_check('Processed Data', 'Movies file exists', False, 'File not found')

    # Check genome
    genome_path = PROCESSED_DIR / "media" / "genome_scores.json"
    if genome_path.exists():
        with open(genome_path) as f:
            genome_data = json.load(f)
        report.add_check('Processed Data', 'Genome file exists', True, f'{len(genome_data):,} movies')
        print(f"   ✓ Genome: {len(genome_data):,} movies")
    else:
        report.add_check('Processed Data', 'Genome file exists', False, 'File not found')

    # Check ratings
    ratings_path = PROCESSED_DIR / "interactions" / "ratings.jsonl"
    if ratings_path.exists():
        rating_count = 0
        with open(ratings_path) as f:
            rating_count = sum(1 for _ in f)
        report.add_check('Processed Data', 'Ratings file exists', True, f'{rating_count:,} ratings')
        print(f"   ✓ Ratings: {rating_count:,}")

        if rating_count < 20000000:
            report.add_warning('Processed Data', f'Only {rating_count:,} ratings (expected ~25M)')
    else:
        report.add_check('Processed Data', 'Ratings file exists', False, 'File not found')


def validate_synthetic_data(report: ValidationReport):
    """Validate synthetic user and platform data"""
    print("\n👥 Validating Synthetic Data")
    print("-" * 60)

    # Check demographics
    demo_path = SYNTHETIC_DIR / "users" / "demographics.jsonl"
    if demo_path.exists():
        user_count = 0
        with open(demo_path) as f:
            user_count = sum(1 for _ in f)
        report.add_check('Synthetic Data', 'Demographics file exists', True, f'{user_count:,} users')
        print(f"   ✓ Demographics: {user_count:,} users")
    else:
        report.add_check('Synthetic Data', 'Demographics file exists', False, 'File not found')

    # Check platform data
    platform_path = SYNTHETIC_DIR / "platforms" / "availability.jsonl"
    if platform_path.exists():
        avail_count = 0
        with open(platform_path) as f:
            avail_count = sum(1 for _ in f)
        report.add_check('Synthetic Data', 'Platform file exists', True, f'{avail_count:,} records')
        print(f"   ✓ Platform availability: {avail_count:,} records")
    else:
        report.add_check('Synthetic Data', 'Platform file exists', False, 'File not found')


def validate_embeddings(report: ValidationReport):
    """Validate embedding files"""
    print("\n🔢 Validating Embeddings")
    print("-" * 60)

    # Check media embeddings
    media_emb_path = EMBEDDINGS_DIR / "media" / "content_vectors.npy"
    if media_emb_path.exists():
        embeddings = np.load(media_emb_path)
        report.add_check('Embeddings', 'Media embeddings exist', True, f'Shape: {embeddings.shape}')
        print(f"   ✓ Media vectors: {embeddings.shape}")

        # Validate dimension
        if embeddings.shape[1] == EMBEDDING_DIM:
            report.add_check('Embeddings', f'Media dimension is {EMBEDDING_DIM}', True)
        else:
            report.add_check('Embeddings', f'Media dimension is {EMBEDDING_DIM}', False,
                           f'Got {embeddings.shape[1]}')

        # Validate normalization
        norms = np.linalg.norm(embeddings, axis=1)
        is_normalized = np.allclose(norms, 1.0, rtol=1e-4)
        report.add_check('Embeddings', 'Media vectors normalized', is_normalized,
                        f'Mean norm: {np.mean(norms):.4f}')
    else:
        report.add_check('Embeddings', 'Media embeddings exist', False, 'File not found')

    # Check user embeddings
    user_emb_path = EMBEDDINGS_DIR / "users" / "preference_vectors.npy"
    if user_emb_path.exists():
        embeddings = np.load(user_emb_path)
        report.add_check('Embeddings', 'User embeddings exist', True, f'Shape: {embeddings.shape}')
        print(f"   ✓ User vectors: {embeddings.shape}")

        # Validate dimension
        if embeddings.shape[1] == EMBEDDING_DIM:
            report.add_check('Embeddings', f'User dimension is {EMBEDDING_DIM}', True)
        else:
            report.add_check('Embeddings', f'User dimension is {EMBEDDING_DIM}', False,
                           f'Got {embeddings.shape[1]}')

        # Validate normalization
        norms = np.linalg.norm(embeddings, axis=1)
        is_normalized = np.allclose(norms, 1.0, rtol=1e-4)
        report.add_check('Embeddings', 'User vectors normalized', is_normalized,
                        f'Mean norm: {np.mean(norms):.4f}')
    else:
        report.add_check('Embeddings', 'User embeddings exist', False, 'File not found')

    # Check metadata alignment
    metadata_path = EMBEDDINGS_DIR / "media" / "metadata.jsonl"
    user_ids_path = EMBEDDINGS_DIR / "users" / "user_ids.json"

    if media_emb_path.exists() and metadata_path.exists():
        media_emb_count = len(np.load(media_emb_path))
        metadata_count = 0
        with open(metadata_path) as f:
            metadata_count = sum(1 for _ in f)

        aligned = media_emb_count == metadata_count
        report.add_check('Embeddings', 'Media metadata aligned', aligned,
                        f'Embeddings: {media_emb_count}, Metadata: {metadata_count}')

    if user_emb_path.exists() and user_ids_path.exists():
        user_emb_count = len(np.load(user_emb_path))
        with open(user_ids_path) as f:
            user_ids = json.load(f)
        user_ids_count = len(user_ids)

        aligned = user_emb_count == user_ids_count
        report.add_check('Embeddings', 'User IDs aligned', aligned,
                        f'Embeddings: {user_emb_count}, IDs: {user_ids_count}')


def validate_milvus(report: ValidationReport):
    """Validate Milvus vector database"""
    print("\n🔍 Validating Milvus")
    print("-" * 60)

    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        report.add_check('Milvus', 'Connection', True)
        print(f"   ✓ Connected to Milvus")

        # Check collections
        collections = utility.list_collections()
        expected = ['media_vectors', 'user_vectors']

        for coll_name in expected:
            if coll_name in collections:
                collection = Collection(coll_name)
                count = collection.num_entities
                report.add_check('Milvus', f'{coll_name} exists', True, f'{count:,} entities')
                print(f"   ✓ {coll_name}: {count:,} entities")

                # Check if loaded
                try:
                    collection.load()
                    report.add_check('Milvus', f'{coll_name} loadable', True)
                except Exception as e:
                    report.add_check('Milvus', f'{coll_name} loadable', False, str(e))
            else:
                report.add_check('Milvus', f'{coll_name} exists', False, 'Collection not found')

        connections.disconnect("default")

    except Exception as e:
        report.add_check('Milvus', 'Connection', False, str(e))
        report.add_warning('Milvus', 'Could not connect - is Milvus running?')


def validate_neo4j(report: ValidationReport):
    """Validate Neo4j graph database"""
    print("\n🕸️  Validating Neo4j")
    print("-" * 60)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session() as session:
            # Check connection
            result = session.run("RETURN 1")
            result.single()
            report.add_check('Neo4j', 'Connection', True)
            print(f"   ✓ Connected to Neo4j")

            # Check node counts
            node_types = ['Media', 'User', 'Genre', 'Tag', 'Platform']
            for node_type in node_types:
                result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                count = result.single()["count"]
                report.add_check('Neo4j', f'{node_type} nodes exist', count > 0, f'{count:,} nodes')
                print(f"   ✓ {node_type}: {count:,} nodes")

            # Check relationship counts
            rel_types = ['RATED', 'HAS_GENRE', 'HAS_TAG', 'AVAILABLE_ON']
            for rel_type in rel_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                count = result.single()["count"]
                report.add_check('Neo4j', f'{rel_type} relationships exist', count > 0, f'{count:,} relationships')
                print(f"   ✓ {rel_type}: {count:,} relationships")

        driver.close()

    except Exception as e:
        report.add_check('Neo4j', 'Connection', False, str(e))
        report.add_warning('Neo4j', 'Could not connect - is Neo4j running?')


def validate_agentdb(report: ValidationReport):
    """Validate AgentDB RL data"""
    print("\n🧠 Validating AgentDB")
    print("-" * 60)

    # Check policies
    policies_path = AGENTDB_DIR / "policies.jsonl"
    if policies_path.exists():
        policy_count = 0
        with open(policies_path) as f:
            policy_count = sum(1 for _ in f)
        report.add_check('AgentDB', 'Policies file exists', True, f'{policy_count:,} policies')
        print(f"   ✓ Policies: {policy_count:,}")
    else:
        report.add_check('AgentDB', 'Policies file exists', False, 'File not found')

    # Check episodes
    episodes_path = AGENTDB_DIR / "episodes.jsonl"
    if episodes_path.exists():
        episode_count = 0
        with open(episodes_path) as f:
            episode_count = sum(1 for _ in f)
        report.add_check('AgentDB', 'Episodes file exists', True, f'{episode_count:,} episodes')
        print(f"   ✓ Episodes: {episode_count:,}")
    else:
        report.add_check('AgentDB', 'Episodes file exists', False, 'File not found')

    # Check replay buffers
    algorithms = ['q_learning', 'sarsa', 'dqn', 'actor_critic', 'ppo', 'decision_transformer']
    found_buffers = 0
    for algo in algorithms:
        buffer_path = AGENTDB_DIR / f"replay_buffer_{algo}.jsonl"
        if buffer_path.exists():
            found_buffers += 1

    report.add_check('AgentDB', 'Replay buffers created', found_buffers > 0,
                    f'{found_buffers}/{len(algorithms)} buffers')
    print(f"   ✓ Replay buffers: {found_buffers}/{len(algorithms)}")


def run_benchmark_queries(report: ValidationReport):
    """Run benchmark queries to test performance"""
    print("\n⚡ Running Benchmark Queries")
    print("-" * 60)

    # Milvus search benchmark
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection("media_vectors")
        collection.load()

        # Load a query vector
        embeddings = np.load(EMBEDDINGS_DIR / "media" / "content_vectors.npy")
        query_vector = embeddings[0].tolist()

        # Benchmark search
        import time
        start = time.time()
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"ef": 100},
            limit=10
        )
        duration = (time.time() - start) * 1000

        passed = duration < 100  # Should be <100ms
        report.add_check('Performance', 'Milvus search <100ms', passed, f'{duration:.2f}ms')
        print(f"   {'✓' if passed else '✗'} Milvus search: {duration:.2f}ms")

        connections.disconnect("default")

    except Exception as e:
        report.add_warning('Performance', f'Could not benchmark Milvus: {e}')

    # Neo4j query benchmark
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session() as session:
            # Benchmark user preferences query
            import time
            start = time.time()
            session.run("""
                MATCH (u:User)-[r:RATED]->(m:Media)
                WHERE u.user_id = 'ml-1'
                RETURN m.title, r.rating
                ORDER BY r.rating DESC
                LIMIT 10
            """)
            duration = (time.time() - start) * 1000

            passed = duration < 1000  # Should be <1s
            report.add_check('Performance', 'Neo4j query <1s', passed, f'{duration:.2f}ms')
            print(f"   {'✓' if passed else '✗'} Neo4j query: {duration:.2f}ms")

        driver.close()

    except Exception as e:
        report.add_warning('Performance', f'Could not benchmark Neo4j: {e}')


def main():
    """Main validation execution"""
    print("=" * 60)
    print("🔍 Data Validation Script")
    print("=" * 60)

    report = ValidationReport()

    # Run all validations
    validate_processed_data(report)
    validate_synthetic_data(report)
    validate_embeddings(report)
    validate_milvus(report)
    validate_neo4j(report)
    validate_agentdb(report)
    run_benchmark_queries(report)

    # Print summary
    success = report.print_summary()

    if success:
        print("\n✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
