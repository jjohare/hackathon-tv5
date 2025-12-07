#!/bin/bash
set -e

# Activate venv with PyTorch 2.5.1
source /home/devuser/workspace/hackathon-tv5/semantic-recommender/venv-pytorch-2.5/bin/activate

# Setup libtorch from PyTorch 2.5.1
export LIBTORCH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export PYTHONPATH=$LIBTORCH:$PYTHONPATH

echo "Building with PyTorch 2.5.1 libtorch..."
echo "LIBTORCH=$LIBTORCH"
echo "Python: $(which python3)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"

cd /home/devuser/workspace/hackathon-tv5/semantic-recommender/semantic-recommender-rs
cargo build --release --bins 2>&1 | tee build.log
