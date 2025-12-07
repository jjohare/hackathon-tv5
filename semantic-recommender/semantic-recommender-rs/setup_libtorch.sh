#!/bin/bash
# PyTorch 2.5.1 libtorch configuration for Rust compilation
# Source this file before running cargo build: source setup_libtorch.sh

# Get the absolute path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
source "${SCRIPT_DIR}/venv/bin/activate"

# Set LIBTORCH to the torch installation
export LIBTORCH="${SCRIPT_DIR}/venv/lib/python3.13/site-packages/torch"

# Add libtorch lib directory to library path
export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH}"

# Tell torch-sys to use the PyTorch installation
export LIBTORCH_USE_PYTORCH=1

# Bypass version check (torch-sys expects 2.3.0, we have 2.5.1)
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Optional: Set CUDA paths if needed
export CUDA_LIBRARY_PATH="${LIBTORCH}/lib"

echo "PyTorch 2.5.1 libtorch environment configured:"
echo "  Python: $(which python)"
echo "  LIBTORCH: ${LIBTORCH}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "  LIBTORCH_USE_PYTORCH: ${LIBTORCH_USE_PYTORCH}"
echo ""
echo "Verifying libtorch libraries:"
ls -lh "${LIBTORCH}/lib/libtorch"*.so 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Verifying PyTorch installation:"
python -c "import torch; print('  PyTorch version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
echo ""
echo "Ready for cargo build!"
