#!/bin/bash
set -e
echo "[ACT-CI-MAIN] Setting up lightweight ACT main environment for CI..."

# Step 1: Conda check
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found on this system."
    echo "[INFO] Please install Miniconda or Anaconda first from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo "[ABORT] Exiting setup..."
    exit 1
fi

# Step 2: Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Step 3: Create and activate main environment (act-main)
if ! conda env list | grep -q "^act-main "; then
    echo "[ACT-CI-MAIN] Creating conda env: act-main..."
    conda create -y -n act-main python=3.9
else
    echo "[ACT-CI-MAIN] Conda env 'act-main' already exists."
fi

echo "[ACT-CI-MAIN] Activating ACT-main environment..."
conda activate act-main

echo "[ACT-CI-MAIN] Installing ACT requirements..."
pip install -r ../setup/main_requirements.txt

# Skip Gurobi for CI to save disk space
echo "[ACT-CI-MAIN] Skipping Gurobi installation for CI..."

echo "[ACT-CI-MAIN] Main environment setup complete."
echo "[ACT-CI-MAIN] Ready for HybridZ verification with act-main environment."