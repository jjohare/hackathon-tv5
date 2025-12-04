#!/bin/bash
set -e

# Configuration
CONFIG_PATH="/home/devuser/workspace/hackathon-tv5/config/datadesigner/media_dataset.yaml"
OUTPUT_DIR="/data/synthetic/tv5_media/"
LOG_DIR="/var/log/datadesigner/"

# Create directories
mkdir -p $OUTPUT_DIR $LOG_DIR

echo "==========================================="
echo "TV5MONDE Synthetic Dataset Generation"
echo "==========================================="
echo "Configuration: $CONFIG_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Log Directory: $LOG_DIR"
echo ""

# Stage 1: Generate taxonomy (small, high quality)
echo "[Stage 1/5] Generating taxonomy..."
python -m datadesigner.generate \
    --config $CONFIG_PATH \
    --stage taxonomy_generation \
    --output $OUTPUT_DIR/taxonomy \
    --log $LOG_DIR/taxonomy.log \
    --max-workers 4

echo "✓ Taxonomy generation complete"
echo ""

# Stage 2: Generate media content (parallel batches)
echo "[Stage 2/5] Generating media content (100M records)..."
echo "This will generate 100 batches of 1M records each"

# Split into 100 jobs of 1M each
for i in {0..99}; do
    echo "Starting batch $((i+1))/100"
    python -m datadesigner.generate \
        --config $CONFIG_PATH \
        --stage media_content_generation \
        --output $OUTPUT_DIR/media/batch_$i \
        --batch-start $((i * 1000000)) \
        --batch-size 1000000 \
        --log $LOG_DIR/media_$i.log &

    # Run 8 parallel jobs
    if [ $((i % 8)) -eq 7 ]; then
        echo "Waiting for batch $((i-7))-$i to complete..."
        wait
    fi
done

# Wait for remaining jobs
wait
echo "✓ Media content generation complete"
echo ""

# Stage 3: Generate users
echo "[Stage 3/5] Generating users (10M records)..."
python -m datadesigner.generate \
    --config $CONFIG_PATH \
    --stage user_generation \
    --output $OUTPUT_DIR/users \
    --log $LOG_DIR/users.log \
    --max-workers 8

echo "✓ User generation complete"
echo ""

# Stage 4: Generate interactions
echo "[Stage 4/5] Generating interactions (1B records)..."
python -m datadesigner.generate \
    --config $CONFIG_PATH \
    --stage interaction_generation \
    --output $OUTPUT_DIR/interactions \
    --log $LOG_DIR/interactions.log \
    --streaming true \
    --chunk-size 10000000

echo "✓ Interaction generation complete"
echo ""

# Stage 5: Generate embeddings
echo "[Stage 5/5] Generating embeddings..."
python /home/devuser/workspace/hackathon-tv5/scripts/generate_embeddings.py \
    --input-dir $OUTPUT_DIR/media \
    --output-dir /home/devuser/workspace/hackathon-tv5/data/embedded \
    --num-workers 8

echo "✓ Embedding generation complete"
echo ""

# Stage 6: Quality validation
echo "[Validation] Running quality checks..."
python -m datadesigner.validate \
    --config $CONFIG_PATH \
    --data $OUTPUT_DIR \
    --report $OUTPUT_DIR/quality_report.html

echo "✓ Quality validation complete"
echo ""

# Summary
echo "==========================================="
echo "Dataset Generation Complete!"
echo "==========================================="
echo "Total size: $(du -sh $OUTPUT_DIR)"
echo "Quality report: $OUTPUT_DIR/quality_report.html"
echo ""
echo "Next steps:"
echo "1. Review quality report"
echo "2. Load data to Milvus: python scripts/load_to_milvus.py"
echo "3. Load metadata to PostgreSQL: python scripts/load_to_postgres.py"
