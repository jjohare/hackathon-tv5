#!/usr/bin/env python3
"""
Load synthetic media metadata and user data into PostgreSQL.
Complements Milvus vector storage with relational data.
"""

import psycopg2
from psycopg2.extras import execute_batch
import pyarrow.parquet as pq
import json
from tqdm import tqdm
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database_schema(conn):
    """Create PostgreSQL schema for media data."""

    with conn.cursor() as cur:
        # Media content table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS media_content (
                content_id BIGINT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                genre VARCHAR(100),
                mood VARCHAR(50),
                cultural_context JSONB,
                popularity FLOAT,
                release_year INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id UUID PRIMARY KEY,
                demographics JSONB,
                psychographic_state FLOAT[],
                taste_clusters VARCHAR[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Interactions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id BIGSERIAL PRIMARY KEY,
                user_id UUID REFERENCES users(user_id),
                content_id BIGINT REFERENCES media_content(content_id),
                interaction_type VARCHAR(20),
                watch_time FLOAT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create indexes
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_media_genre ON media_content(genre);
            CREATE INDEX IF NOT EXISTS idx_media_mood ON media_content(mood);
            CREATE INDEX IF NOT EXISTS idx_media_year ON media_content(release_year);
            CREATE INDEX IF NOT EXISTS idx_media_popularity ON media_content(popularity);
            CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_content ON interactions(content_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp);
        """)

        conn.commit()
        logger.info("Database schema created successfully")


def load_media_content(conn, parquet_path, batch_size=5000):
    """Load media content from parquet to PostgreSQL."""

    logger.info(f"Loading media content from {parquet_path}")

    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        with conn.cursor() as cur:
            total_inserted = 0

            for i in tqdm(range(0, len(df), batch_size), desc="Loading media content"):
                batch = df.iloc[i:i+batch_size]

                # Prepare data
                data = []
                for _, row in batch.iterrows():
                    # Parse cultural context if string
                    cultural_context = row.get('cultural_context', {})
                    if isinstance(cultural_context, str):
                        cultural_context = json.loads(cultural_context)

                    data.append((
                        int(row['content_id']),
                        row['title'],
                        row.get('description', ''),
                        row.get('genre', ''),
                        row.get('mood', ''),
                        json.dumps(cultural_context),
                        float(row.get('popularity', 0.0)),
                        int(row.get('release_year', 2000))
                    ))

                # Insert batch
                execute_batch(cur, """
                    INSERT INTO media_content
                    (content_id, title, description, genre, mood, cultural_context, popularity, release_year)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (content_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        updated_at = CURRENT_TIMESTAMP
                """, data)

                total_inserted += len(batch)

            conn.commit()
            logger.info(f"Inserted {total_inserted} media records")

            return total_inserted

    except Exception as e:
        logger.error(f"Error loading media content: {e}")
        conn.rollback()
        return 0


def load_users(conn, parquet_path, batch_size=5000):
    """Load users from parquet to PostgreSQL."""

    logger.info(f"Loading users from {parquet_path}")

    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        with conn.cursor() as cur:
            total_inserted = 0

            for i in tqdm(range(0, len(df), batch_size), desc="Loading users"):
                batch = df.iloc[i:i+batch_size]

                data = []
                for _, row in batch.iterrows():
                    # Parse demographics if string
                    demographics = row.get('demographics', {})
                    if isinstance(demographics, str):
                        demographics = json.loads(demographics)

                    # Parse psychographic state
                    psycho_state = row.get('psychographic_state', [0, 0, 0])
                    if isinstance(psycho_state, str):
                        psycho_state = json.loads(psycho_state)

                    # Parse taste clusters
                    taste_clusters = row.get('taste_clusters', [])
                    if isinstance(taste_clusters, str):
                        taste_clusters = json.loads(taste_clusters)

                    data.append((
                        str(row['user_id']),
                        json.dumps(demographics),
                        psycho_state,
                        taste_clusters
                    ))

                # Insert batch
                execute_batch(cur, """
                    INSERT INTO users
                    (user_id, demographics, psychographic_state, taste_clusters)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        demographics = EXCLUDED.demographics,
                        updated_at = CURRENT_TIMESTAMP
                """, data)

                total_inserted += len(batch)

            conn.commit()
            logger.info(f"Inserted {total_inserted} user records")

            return total_inserted

    except Exception as e:
        logger.error(f"Error loading users: {e}")
        conn.rollback()
        return 0


def load_interactions(conn, parquet_path, batch_size=10000):
    """Load interactions from parquet to PostgreSQL."""

    logger.info(f"Loading interactions from {parquet_path}")

    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        with conn.cursor() as cur:
            total_inserted = 0

            for i in tqdm(range(0, len(df), batch_size), desc="Loading interactions"):
                batch = df.iloc[i:i+batch_size]

                data = []
                for _, row in batch.iterrows():
                    data.append((
                        str(row['user_id']),
                        int(row['content_id']),
                        row.get('interaction_type', 'view'),
                        float(row.get('watch_time', 0.0)),
                        row.get('timestamp')
                    ))

                # Insert batch
                execute_batch(cur, """
                    INSERT INTO interactions
                    (user_id, content_id, interaction_type, watch_time, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, data)

                total_inserted += len(batch)

            conn.commit()
            logger.info(f"Inserted {total_inserted} interaction records")

            return total_inserted

    except Exception as e:
        logger.error(f"Error loading interactions: {e}")
        conn.rollback()
        return 0


def verify_data(conn):
    """Verify loaded data with summary statistics."""

    with conn.cursor() as cur:
        logger.info("\n=== Database Statistics ===")

        # Media content stats
        cur.execute("SELECT COUNT(*) FROM media_content")
        logger.info(f"Total media items: {cur.fetchone()[0]:,}")

        cur.execute("""
            SELECT genre, COUNT(*) as count
            FROM media_content
            GROUP BY genre
            ORDER BY count DESC
            LIMIT 5
        """)
        logger.info("Top 5 genres:")
        for genre, count in cur.fetchall():
            logger.info(f"  {genre}: {count:,}")

        # User stats
        cur.execute("SELECT COUNT(*) FROM users")
        logger.info(f"\nTotal users: {cur.fetchone()[0]:,}")

        # Interaction stats
        cur.execute("SELECT COUNT(*) FROM interactions")
        logger.info(f"Total interactions: {cur.fetchone()[0]:,}")

        cur.execute("""
            SELECT interaction_type, COUNT(*) as count
            FROM interactions
            GROUP BY interaction_type
            ORDER BY count DESC
        """)
        logger.info("Interactions by type:")
        for interaction_type, count in cur.fetchall():
            logger.info(f"  {interaction_type}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Load synthetic media data to PostgreSQL")
    parser.add_argument("--data-dir", type=str,
                        default="/data/synthetic/tv5_media",
                        help="Directory with parquet files")
    parser.add_argument("--db-host", type=str, default="localhost",
                        help="PostgreSQL host")
    parser.add_argument("--db-port", type=int, default=5432,
                        help="PostgreSQL port")
    parser.add_argument("--db-name", type=str, default="tv5_media",
                        help="Database name")
    parser.add_argument("--db-user", type=str, default="postgres",
                        help="Database user")
    parser.add_argument("--db-password", type=str, default="postgres",
                        help="Database password")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch size for insertion")

    args = parser.parse_args()

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password
    )

    logger.info(f"Connected to PostgreSQL at {args.db_host}:{args.db_port}")

    # Create schema
    create_database_schema(conn)

    data_dir = Path(args.data_dir)

    # Load media content
    media_files = sorted(data_dir.glob("media/**/*.parquet"))
    if media_files:
        for media_file in media_files[:5]:  # Load first 5 batches for testing
            load_media_content(conn, media_file, args.batch_size)

    # Load users
    user_files = sorted(data_dir.glob("users/**/*.parquet"))
    if user_files:
        for user_file in user_files[:2]:  # Load first 2 batches
            load_users(conn, user_file, args.batch_size)

    # Load interactions
    interaction_files = sorted(data_dir.glob("interactions/**/*.parquet"))
    if interaction_files:
        for interaction_file in interaction_files[:5]:  # Load first 5 batches
            load_interactions(conn, interaction_file, args.batch_size)

    # Verify data
    verify_data(conn)

    conn.close()
    logger.info("Data loading complete!")


if __name__ == "__main__":
    main()
