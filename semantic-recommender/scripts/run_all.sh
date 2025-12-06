#!/bin/bash
# Master execution script for data generation pipeline
# Run on GCP A100 VM

set -e  # Exit on error

echo "======================================================================"
echo "🚀 TV5 Media Recommendation System - Data Generation Pipeline"
echo "======================================================================"
echo ""
echo "Environment Check:"
python3 --version
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo ""
echo "======================================================================"
echo ""

# Phase 1: Parse MovieLens
echo "📊 Phase 1: Parsing MovieLens Dataset"
echo "----------------------------------------------------------------------"
python3 scripts/parse_movielens.py
echo ""

# Phase 2: Generate Synthetic Data
echo "🎭 Phase 2: Generating Synthetic Data"
echo "----------------------------------------------------------------------"
echo "Generating user profiles..."
python3 scripts/generate_user_profiles.py
echo ""

echo "Generating platform data..."
python3 scripts/generate_platform_data.py
echo ""

# Phase 3: Generate Embeddings (A100)
echo "🚀 Phase 3: Generating Embeddings on GPU"
echo "----------------------------------------------------------------------"
python3 scripts/generate_embeddings.py
echo ""

# Phase 4: Populate Databases
echo "💾 Phase 4: Populating Databases"
echo "----------------------------------------------------------------------"
echo "Populating Milvus vector database..."
python3 scripts/populate_milvus.py
echo ""

echo "Populating Neo4j graph database..."
python3 scripts/populate_neo4j.py
echo ""

echo "Populating AgentDB RL policies..."
python3 scripts/populate_agentdb.py
echo ""

# Phase 5: Validation
echo "✅ Phase 5: Validation"
echo "----------------------------------------------------------------------"
python3 scripts/validate_data.py
echo ""

echo "======================================================================"
echo "✅ DATA GENERATION PIPELINE COMPLETE"
echo "======================================================================"
echo ""
echo "Final Statistics:"
echo "----------------------------------------------------------------------"
[ -f data/processed/processing_stats.json ] && cat data/processed/processing_stats.json
[ -f data/embeddings/embedding_stats.json ] && cat data/embeddings/embedding_stats.json
[ -f data/milvus_stats.json ] && cat data/milvus_stats.json
[ -f data/neo4j_stats.json ] && cat data/neo4j_stats.json
[ -f data/agentdb/agentdb_stats.json ] && cat data/agentdb/agentdb_stats.json
echo ""
echo "Validation Report:"
echo "----------------------------------------------------------------------"
[ -f data/validation_report.json ] && cat data/validation_report.json
echo ""
echo "All data and databases ready for semantic recommendation system!"
echo ""
