#!/bin/bash
# Complete TMDB Semantic Enrichment Pipeline
# Upgrades from title-only to full semantic embeddings

set -e  # Exit on error

echo "========================================================================"
echo "TMDB Semantic Enrichment Pipeline"
echo "========================================================================"
echo ""

# Check for TMDB API key
if [ -z "$TMDB_API_KEY" ]; then
    echo "❌ ERROR: TMDB_API_KEY environment variable not set"
    echo ""
    echo "Get your API key:"
    echo "  1. Visit: https://www.themoviedb.org/settings/api"
    echo "  2. Register for free account"
    echo "  3. Request API key (instant approval)"
    echo "  4. Export key: export TMDB_API_KEY='your_key_here'"
    echo ""
    exit 1
fi

echo "✅ TMDB API key found"
echo ""

# Get base path (script directory -> parent -> parent)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_PATH="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "Base path: $BASE_PATH"
echo ""

# Stage 1b: TMDB API Enrichment (7-8 hours)
echo "========================================================================"
echo "Stage 1b: TMDB API Enrichment"
echo "========================================================================"
echo "Fetching full metadata for 1.3M movies from TMDB API..."
echo "Estimated time: 7-8 hours with 50 req/sec rate limiting"
echo ""

cd "$SCRIPT_DIR"

python3 stage1b_enrich_tmdb.py \
    --base-path "$BASE_PATH" \
    --checkpoint-interval 10000

if [ $? -ne 0 ]; then
    echo "❌ Stage 1b failed"
    exit 1
fi

echo ""
echo "✅ Stage 1b complete"
echo ""

# Stage 2b: Rich Text Generation (2 minutes)
echo "========================================================================"
echo "Stage 2b: Rich Text Generation"
echo "========================================================================"
echo "Generating comprehensive semantic text from enriched metadata..."
echo "Estimated time: 2 minutes"
echo ""

python3 stage2b_generate_rich_text.py \
    --base-path "$BASE_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Stage 2b failed"
    exit 1
fi

echo ""
echo "✅ Stage 2b complete"
echo ""

# Stage 3: GPU Embedding Generation (15 minutes)
echo "========================================================================"
echo "Stage 3: TensorRT Embedding Generation"
echo "========================================================================"
echo "Generating embeddings with TensorRT acceleration..."
echo "Estimated time: 15 minutes on A100 GPU"
echo ""

# Create output directory for new embeddings
mkdir -p "$BASE_PATH/data/embeddings/tmdb_full_semantic"

# Run modified stage3 for rich text
python3 stage3_gpu_embeddings.py \
    --base-path "$BASE_PATH" \
    --batch-size 32 \
    --checkpoint-interval 10000 \
    --input-file "data/processed/tmdb/movies_rich_text.jsonl" \
    --output-dir "data/embeddings/tmdb_full_semantic" \
    --text-field "rich_text"

if [ $? -ne 0 ]; then
    echo "❌ Stage 3 failed"
    exit 1
fi

echo ""
echo "✅ Stage 3 complete"
echo ""

# Validation Test
echo "========================================================================"
echo "Validation: Semantic Upgrade Test"
echo "========================================================================"
echo "Comparing OLD (title-only) vs NEW (full semantic) embeddings..."
echo ""

cd "$BASE_PATH/scripts"

python3 test_semantic_upgrade.py \
    --base-path "$BASE_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Validation test failed"
    exit 1
fi

echo ""
echo "✅ Validation complete"
echo ""

# Final summary
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Output Files:"
echo "  - data/processed/tmdb/movies_enriched.jsonl (TMDB enriched metadata)"
echo "  - data/processed/tmdb/movies_rich_text.jsonl (semantic text)"
echo "  - data/embeddings/tmdb_full_semantic/content_vectors.npy (NEW embeddings)"
echo "  - docs/SEMANTIC_UPGRADE_REPORT.md (comparison report)"
echo ""
echo "Next Steps:"
echo "  1. Review report: cat docs/SEMANTIC_UPGRADE_REPORT.md"
echo "  2. Update production to use: data/embeddings/tmdb_full_semantic/"
echo "  3. Deploy to recommendation engine"
echo ""
echo "✅ All stages complete!"
