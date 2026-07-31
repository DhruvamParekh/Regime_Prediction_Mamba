#!/bin/bash
# =====================================================================
# setup_env.sh
# =====================================================================
# One-time GPU environment setup for the regime-prediction pipeline.
#
# ⚠️  GPU REQUIRED — Mamba SSM only runs on an NVIDIA GPU with CUDA
#     12.6. This script (and the whole pipeline) will fail on CPU.
#
# Run this once per fresh runtime/machine, BEFORE running main.py.
# On Google Colab: Runtime → change runtime type → GPU, then run this
# script, then RESTART THE RUNTIME, then run main.py.
# =====================================================================

set -e

echo "Step 1 — Installing CUDA 12.6 toolkit..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -q
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -qq
apt-get install -y cuda-toolkit-12-6 -qq

echo "Step 2 — Pointing environment to CUDA 12.6..."
export CUDA_HOME="/usr/local/cuda-12.6"
export PATH="/usr/local/cuda-12.6/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}"

echo "Step 3 — Verifying nvcc..."
nvcc --version

echo "Step 4 — Installing PyTorch built for CUDA 12.6..."
pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 --quiet

echo "Step 5 — Installing causal-conv1d, mamba-ssm==2.3.1, einops..."
pip install "causal-conv1d>=1.4.0" --no-build-isolation --quiet
pip install mamba-ssm==2.3.1 --no-build-isolation --quiet
pip install einops --quiet

echo ""
echo "Setup complete. Now RESTART THE RUNTIME, then run:"
echo "    python main.py"
