#!/bin/bash
# A100 Deployment and Testing Script
# Automates deployment to GCP A100 VM and runs comprehensive tests

set -e  # Exit on error

echo "========================================================================"
echo "A100 GPU Deployment and Testing"
echo "========================================================================"

# Configuration
INSTANCE_NAME="semantics-testbed-a100"
ZONE="us-central1-a"
PACKAGE_FILE="/tmp/semantic-recommender-deploy.tar.gz"
REMOTE_DIR="/home/devuser/semantic-recommender"

# Check if package exists
if [ ! -f "$PACKAGE_FILE" ]; then
    echo "❌ Package not found: $PACKAGE_FILE"
    echo "Run: tar -czf /tmp/semantic-recommender-deploy.tar.gz scripts/ data/embeddings/"
    exit 1
fi

echo "✅ Package found: $PACKAGE_FILE ($(ls -lh $PACKAGE_FILE | awk '{print $5}'))"

# Step 1: Transfer package
echo ""
echo "📤 Step 1: Transferring package to A100 VM..."
gcloud compute scp "$PACKAGE_FILE" "$INSTANCE_NAME:/home/devuser/" --zone "$ZONE"

if [ $? -eq 0 ]; then
    echo "✅ Transfer complete"
else
    echo "❌ Transfer failed"
    exit 1
fi

# Step 2: Extract and setup
echo ""
echo "📦 Step 2: Extracting package on A100 VM..."
gcloud compute ssh "$INSTANCE_NAME" --zone "$ZONE" --command "
    cd /home/devuser
    rm -rf semantic-recommender
    mkdir -p semantic-recommender
    tar -xzf semantic-recommender-deploy.tar.gz -C semantic-recommender
    cd semantic-recommender
    mkdir -p results
    echo '✅ Package extracted'
    ls -lh scripts/ | head -5
    ls -lh data/embeddings/media/
    ls -lh data/embeddings/users/
"

# Step 3: Install dependencies
echo ""
echo "🔧 Step 3: Installing PyTorch and dependencies..."
gcloud compute ssh "$INSTANCE_NAME" --zone "$ZONE" --command "
    # Check if PyTorch already installed
    if python3 -c 'import torch' 2>/dev/null; then
        echo '✅ PyTorch already installed'
        python3 -c 'import torch; print(f\"  Version: {torch.__version__}\"); print(f\"  CUDA: {torch.cuda.is_available()}\")'
    else
        echo '📥 Installing PyTorch...'
        pip install --user torch --index-url https://download.pytorch.org/whl/cu121
        echo '✅ PyTorch installed'
    fi

    # Install numpy if needed
    python3 -c 'import numpy' 2>/dev/null || pip install --user numpy

    # Verify installation
    python3 -c '
import torch
print(\"\\n\" + \"=\"*70)
print(\"Environment Verification\")
print(\"=\"*70)
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA Available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"GPU: {torch.cuda.get_device_name(0)}\")
    print(f\"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")
    print(f\"CUDA Version: {torch.version.cuda}\")
print(\"=\"*70)
'
"

# Step 4: Run comprehensive tests
echo ""
echo "🧪 Step 4: Running comprehensive test suite..."
gcloud compute ssh "$INSTANCE_NAME" --zone "$ZONE" --command "
    cd /home/devuser/semantic-recommender
    python3 scripts/test_a100_comprehensive.py 2>&1 | tee results/test_output.log
"

# Step 5: Download results
echo ""
echo "📥 Step 5: Downloading test results..."
gcloud compute scp "$INSTANCE_NAME:$REMOTE_DIR/results/a100_test_results.json" ./results/ --zone "$ZONE"
gcloud compute scp "$INSTANCE_NAME:$REMOTE_DIR/results/test_output.log" ./results/ --zone "$ZONE"

echo ""
echo "========================================================================"
echo "✅ Deployment and Testing Complete!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  • Local: ./results/a100_test_results.json"
echo "  • Log:   ./results/test_output.log"
echo ""
echo "Next steps:"
echo "  1. Review results/a100_test_results.json"
echo "  2. Document findings in docs/A100_TEST_RESULTS.md"
echo "  3. Commit and push results to GitHub"
echo ""
