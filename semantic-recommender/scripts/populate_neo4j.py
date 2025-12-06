#!/usr/bin/env python3
"""
Populate Neo4j graph database with relationships
Creates nodes for users, media, genres, tags, platforms
Builds interaction graph with ratings, views, preferences
"""

import json
from pathlib import Path
from typing import List, Dict
from neo4j import GraphDatabase
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j"  # Change in production

# Batch sizes
BATCH_SIZE = 1000


class Neo4jPopulator:
    """Handles Neo4j population"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create uniqueness constraints"""
        print("\n🔒 Creating constraints...")

        constraints = [
            "CREATE CONSTRAINT media_id IF NOT EXISTS FOR (m:Media) REQUIRE m.media_id IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
            "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT platform_id IF NOT EXISTS FOR (p:Platform) REQUIRE p.platform_id IS UNIQUE",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"   ✓ {constraint.split()[2]}")
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"   ⚠️  {constraint.split()[2]}: {e}")

        print("✅ Constraints created")

    def create_indexes(self):
        """Create indexes for performance"""
        print("\n📇 Creating indexes...")

        indexes = [
            "CREATE INDEX media_title IF NOT EXISTS FOR (m:Media) ON (m.title)",
            "CREATE INDEX media_year IF NOT EXISTS FOR (m:Media) ON (m.year)",
            "CREATE INDEX user_archetype IF NOT EXISTS FOR (u:User) ON (u.archetype)",
        ]

        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                    print(f"   ✓ {index.split()[2]}")
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"   ⚠️  {index.split()[2]}: {e}")

        print("✅ Indexes created")

    def clear_database(self):
        """Clear all nodes and relationships"""
        print("\n🗑️  Clearing existing data...")

        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]

            if count > 0:
                print(f"   Found {count:,} existing nodes")
                session.run("MATCH (n) DETACH DELETE n")
                print("   ✓ Cleared")
            else:
                print("   Database already empty")

    def create_media_nodes(self):
        """Create media nodes from movies.jsonl"""
        print("\n🎬 Creating Media nodes...")

        movies_path = PROCESSED_DIR / "media" / "movies.jsonl"

        # Load movies
        movies = []
        with open(movies_path) as f:
            for line in f:
                movies.append(json.loads(line))

        print(f"   Loaded {len(movies):,} movies")

        # Insert in batches
        query = """
        UNWIND $batch AS movie
        CREATE (m:Media {
            media_id: movie.media_id,
            title: movie.metadata.title,
            year: movie.metadata.year,
            genres: movie.classification.genres,
            imdb_id: movie.identifiers.imdb_id,
            tmdb_id: movie.identifiers.tmdb_id
        })
        """

        with self.driver.session() as session:
            for i in tqdm(range(0, len(movies), BATCH_SIZE), desc="Creating Media nodes"):
                batch = movies[i:i+BATCH_SIZE]
                session.run(query, batch=batch)

        print(f"✅ Created {len(movies):,} Media nodes")
        return len(movies)

    def create_genre_nodes_and_relationships(self):
        """Create genre nodes and link to media"""
        print("\n🎭 Creating Genre nodes and relationships...")

        query = """
        MATCH (m:Media)
        UNWIND m.genres AS genre_name
        MERGE (g:Genre {name: genre_name})
        MERGE (m)-[:HAS_GENRE]->(g)
        """

        with self.driver.session() as session:
            session.run(query)

        # Count genres
        with self.driver.session() as session:
            result = session.run("MATCH (g:Genre) RETURN count(g) as count")
            count = result.single()["count"]

        print(f"✅ Created {count} Genre nodes")
        return count

    def create_tag_nodes_and_relationships(self):
        """Create tag nodes from genome data"""
        print("\n🏷️  Creating Tag nodes and relationships...")

        genome_path = PROCESSED_DIR / "media" / "genome_scores.json"

        with open(genome_path) as f:
            genome_data = json.load(f)

        print(f"   Loaded genome data for {len(genome_data):,} movies")

        # Create tags and relationships
        query = """
        UNWIND $batch AS item
        MATCH (m:Media {media_id: item.media_id})
        UNWIND item.tags AS tag_data
        MERGE (t:Tag {name: tag_data.name})
        MERGE (m)-[r:HAS_TAG]->(t)
        SET r.relevance = tag_data.relevance
        """

        # Prepare batch data (only top 10 tags per movie for relevance > 0.5)
        batch_data = []
        for ml_id, tags in genome_data.items():
            # Filter and sort tags
            relevant_tags = [(tag, score) for tag, score in tags.items() if score > 0.5]
            relevant_tags.sort(key=lambda x: x[1], reverse=True)
            top_tags = relevant_tags[:10]

            if top_tags:
                batch_data.append({
                    'media_id': f'ml-{ml_id}',
                    'tags': [{'name': tag, 'relevance': score} for tag, score in top_tags]
                })

        print(f"   Processing {len(batch_data):,} movies with relevant tags...")

        with self.driver.session() as session:
            for i in tqdm(range(0, len(batch_data), BATCH_SIZE), desc="Creating Tag relationships"):
                batch = batch_data[i:i+BATCH_SIZE]
                session.run(query, batch=batch)

        # Count tags
        with self.driver.session() as session:
            result = session.run("MATCH (t:Tag) RETURN count(t) as count")
            count = result.single()["count"]

        print(f"✅ Created {count} Tag nodes")
        return count

    def create_user_nodes(self):
        """Create user nodes from demographics"""
        print("\n👥 Creating User nodes...")

        users_path = SYNTHETIC_DIR / "users" / "demographics.jsonl"

        if not users_path.exists():
            print("   ⚠️  Demographics file not found, skipping")
            return 0

        # Load users
        users = []
        with open(users_path) as f:
            for line in f:
                users.append(json.loads(line))

        print(f"   Loaded {len(users):,} users")

        # Insert in batches
        query = """
        UNWIND $batch AS user
        CREATE (u:User {
            user_id: user.user_id,
            archetype: user.archetype,
            age_group: user.demographics.age_group,
            gender: user.demographics.gender,
            location: user.demographics.location,
            language: user.demographics.language
        })
        """

        with self.driver.session() as session:
            for i in tqdm(range(0, len(users), BATCH_SIZE), desc="Creating User nodes"):
                batch = users[i:i+BATCH_SIZE]
                session.run(query, batch=batch)

        print(f"✅ Created {len(users):,} User nodes")
        return len(users)

    def create_rating_relationships(self):
        """Create RATED relationships from ratings"""
        print("\n⭐ Creating RATED relationships...")

        ratings_path = PROCESSED_DIR / "interactions" / "ratings.jsonl"

        # Count total ratings
        with open(ratings_path) as f:
            total = sum(1 for _ in f)

        print(f"   Processing {total:,} ratings...")

        query = """
        UNWIND $batch AS rating
        MATCH (u:User {user_id: rating.user_id})
        MATCH (m:Media {media_id: rating.media_id})
        CREATE (u)-[r:RATED]->(m)
        SET r.rating = rating.rating,
            r.timestamp = rating.timestamp,
            r.time_of_day = rating.context.time_of_day
        """

        batch = []
        processed = 0

        with self.driver.session() as session:
            with open(ratings_path) as f:
                pbar = tqdm(total=total, desc="Creating RATED relationships")

                for line in f:
                    rating = json.loads(line)
                    batch.append(rating)
                    processed += 1

                    if len(batch) >= BATCH_SIZE:
                        session.run(query, batch=batch)
                        pbar.update(len(batch))
                        batch = []

                # Insert remaining
                if batch:
                    session.run(query, batch=batch)
                    pbar.update(len(batch))

                pbar.close()

        print(f"✅ Created {processed:,} RATED relationships")
        return processed

    def create_platform_nodes_and_relationships(self):
        """Create platform nodes and AVAILABLE_ON relationships"""
        print("\n🌐 Creating Platform nodes and AVAILABLE_ON relationships...")

        availability_path = SYNTHETIC_DIR / "platforms" / "availability.jsonl"

        if not availability_path.exists():
            print("   ⚠️  Platform data not found, skipping")
            return 0

        # Create platform nodes
        platforms = [
            {'platform_id': 'tv5monde', 'name': 'TV5MONDE', 'type': 'free'},
            {'platform_id': 'netflix', 'name': 'Netflix', 'type': 'subscription'},
            {'platform_id': 'amazon', 'name': 'Amazon Prime Video', 'type': 'subscription'},
            {'platform_id': 'mubi', 'name': 'MUBI', 'type': 'subscription'},
            {'platform_id': 'criterion', 'name': 'Criterion Channel', 'type': 'subscription'},
            {'platform_id': 'disney', 'name': 'Disney+', 'type': 'subscription'},
            {'platform_id': 'hbo', 'name': 'HBO Max', 'type': 'subscription'},
            {'platform_id': 'apple', 'name': 'Apple TV+', 'type': 'subscription'},
        ]

        query = """
        UNWIND $platforms AS platform
        CREATE (p:Platform {
            platform_id: platform.platform_id,
            name: platform.name,
            type: platform.type
        })
        """

        with self.driver.session() as session:
            session.run(query, platforms=platforms)

        print(f"   ✓ Created {len(platforms)} Platform nodes")

        # Create AVAILABLE_ON relationships
        query = """
        UNWIND $batch AS avail
        MATCH (m:Media {media_id: avail.media_id})
        UNWIND avail.platforms AS platform_data
        MATCH (p:Platform {platform_id: platform_data.platform_id})
        CREATE (m)-[r:AVAILABLE_ON]->(p)
        SET r.regions = platform_data.regions,
            r.quality = platform_data.quality_options,
            r.subtitles = platform_data.subtitle_languages
        """

        availability = []
        with open(availability_path) as f:
            for line in f:
                data = json.loads(line)
                if data['platforms']:  # Only include if available somewhere
                    availability.append(data)

        print(f"   Processing {len(availability):,} availability records...")

        rel_count = 0
        with self.driver.session() as session:
            for i in tqdm(range(0, len(availability), BATCH_SIZE), desc="Creating AVAILABLE_ON relationships"):
                batch = availability[i:i+BATCH_SIZE]
                session.run(query, batch=batch)
                rel_count += sum(len(a['platforms']) for a in batch)

        print(f"✅ Created {rel_count:,} AVAILABLE_ON relationships")
        return rel_count

    def generate_stats(self):
        """Generate graph statistics"""
        print("\n" + "=" * 60)
        print("📊 Graph Statistics")
        print("=" * 60)

        queries = {
            'Media nodes': "MATCH (m:Media) RETURN count(m) as count",
            'User nodes': "MATCH (u:User) RETURN count(u) as count",
            'Genre nodes': "MATCH (g:Genre) RETURN count(g) as count",
            'Tag nodes': "MATCH (t:Tag) RETURN count(t) as count",
            'Platform nodes': "MATCH (p:Platform) RETURN count(p) as count",
            'RATED relationships': "MATCH ()-[r:RATED]->() RETURN count(r) as count",
            'HAS_GENRE relationships': "MATCH ()-[r:HAS_GENRE]->() RETURN count(r) as count",
            'HAS_TAG relationships': "MATCH ()-[r:HAS_TAG]->() RETURN count(r) as count",
            'AVAILABLE_ON relationships': "MATCH ()-[r:AVAILABLE_ON]->() RETURN count(r) as count",
        }

        stats = {}
        with self.driver.session() as session:
            for label, query in queries.items():
                result = session.run(query)
                count = result.single()["count"]
                stats[label] = count
                print(f"  {label:30s}: {count:,}")

        # Total counts
        total_nodes = sum(v for k, v in stats.items() if 'nodes' in k)
        total_rels = sum(v for k, v in stats.items() if 'relationships' in k)

        print(f"\n  {'Total Nodes':30s}: {total_nodes:,}")
        print(f"  {'Total Relationships':30s}: {total_rels:,}")

        # Save stats
        stats_path = DATA_DIR / "neo4j_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Stats saved to {stats_path}")

        return stats


def main():
    """Main execution"""
    print("=" * 60)
    print("🚀 Neo4j Graph Population Script")
    print("=" * 60)
    print()

    populator = None

    try:
        # Connect to Neo4j
        print(f"📡 Connecting to Neo4j at {NEO4J_URI}")
        populator = Neo4jPopulator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        print("✅ Connected to Neo4j")

        # Clear and setup
        populator.clear_database()
        populator.create_constraints()
        populator.create_indexes()

        # Create nodes
        media_count = populator.create_media_nodes()
        genre_count = populator.create_genre_nodes_and_relationships()
        tag_count = populator.create_tag_nodes_and_relationships()
        user_count = populator.create_user_nodes()
        platform_rels = populator.create_platform_nodes_and_relationships()

        # Create relationships
        rating_count = populator.create_rating_relationships()

        # Generate stats
        stats = populator.generate_stats()

        print("\n" + "=" * 60)
        print("✅ NEO4J POPULATION COMPLETE")
        print("=" * 60)
        print(f"\nGraph Created:")
        print(f"  Nodes: {stats.get('Media nodes', 0) + stats.get('User nodes', 0) + stats.get('Genre nodes', 0) + stats.get('Tag nodes', 0) + stats.get('Platform nodes', 0):,}")
        print(f"  Relationships: {stats.get('RATED relationships', 0) + stats.get('HAS_GENRE relationships', 0) + stats.get('HAS_TAG relationships', 0) + stats.get('AVAILABLE_ON relationships', 0):,}")
        print(f"\nGraph ready for traversal queries")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if populator:
            populator.close()
            print("\n👋 Disconnected from Neo4j")


if __name__ == '__main__':
    main()
