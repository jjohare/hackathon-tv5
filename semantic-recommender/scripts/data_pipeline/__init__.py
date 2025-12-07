"""Data pipeline module for semantic recommender system.

This module contains scripts for the complete data generation pipeline:
- Parsing input datasets (MovieLens)
- Generating synthetic data (user profiles, platform data)
- Creating embeddings (GPU-accelerated with SBERT)
- Populating vector databases (Milvus)
- Populating graph databases (Neo4j)
- Populating RL policy databases (AgentDB)
- Validating data integrity

Each script is standalone and can be run independently, but the recommended
execution order is: parse -> generate -> embeddings -> populate -> validate
"""

__version__ = "1.0.0"
__all__ = [
    "parse_movielens",
    "generate_user_profiles",
    "generate_platform_data",
    "generate_embeddings",
    "populate_milvus",
    "populate_neo4j",
    "populate_agentdb",
    "validate_data",
]
