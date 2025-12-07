#!/bin/bash
# TMDB Dataset Download - Quick Start Example
# This script demonstrates how to download the TMDB dataset

set -e  # Exit on error

echo "==================================================================="
echo "TMDB Movies Dataset Download - Quick Start"
echo "==================================================================="
echo ""

# Step 1: Check if Kaggle credentials exist
echo "Step 1: Checking Kaggle credentials..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo ""
    echo "To set up Kaggle API:"
    echo "1. Visit: https://www.kaggle.com/settings/account"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New Token'"
    echo "4. Save kaggle.json to ~/.kaggle/kaggle.json"
    echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
else
    echo "✅ Found: ~/.kaggle/kaggle.json"

    # Check permissions
    PERMS=$(stat -c "%a" ~/.kaggle/kaggle.json)
    if [ "$PERMS" != "600" ]; then
        echo "⚠️  Warning: Permissions are $PERMS (should be 600)"
        echo "   Run: chmod 600 ~/.kaggle/kaggle.json"
    else
        echo "✅ Permissions: 600"
    fi
fi

echo ""

# Step 2: Install dependencies
echo "Step 2: Installing dependencies..."
pip install -q kaggle
echo "✅ Kaggle CLI installed"

echo ""

# Step 3: Run download script
echo "Step 3: Downloading TMDB dataset..."
echo "   Dataset: asaniczka/tmdb-movies-dataset-2023-930k-movies"
echo "   Destination: data/raw/tmdb/"
echo ""

python scripts/download_tmdb_dataset.py

echo ""
echo "==================================================================="
echo "Download Complete!"
echo "==================================================================="
echo ""
echo "Next steps:"
echo "1. Verify data: python scripts/validate_data.py --source tmdb"
echo "2. Process data: python scripts/generate_platform_data.py --source tmdb"
echo "3. Generate embeddings: python scripts/generate_embeddings.py --source tmdb"
echo ""
