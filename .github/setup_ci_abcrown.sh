#!/bin/bash
set -e
echo "[ACT-CI-abCrown] Setting up lightweight ACT abCrown environment for CI..."

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

# Step 3: Create and activate abcrown environment (act-abcrown)
if ! conda env list | grep -q "^act-abcrown "; then
    echo "[ACT-CI-abCrown] Creating conda env: act-abcrown..."
    conda create -y -n act-abcrown python=3.9
else
    echo "[ACT-CI-abCrown] Conda env 'act-abcrown' already exists."
fi

echo "[ACT-CI-abCrown] Activating ACT-abCrown environment..."
conda activate act-abcrown

echo "[ACT-CI-abCrown] Installing abCrown requirements..."
pip install -r ../setup/abcrown_requirements.txt

# Step 4: Create empty config file for abcrown CLI parameter mode
echo "[ACT-CI-abCrown] Creating empty_config.yaml for CLI-only abcrown runs..."
echo "{}" > ../verifier/abstract_constraint_solver/abcrown/empty_config.yaml

# Step 5: Patch the abcrown module __init__.py
ABCROWN_SUBMODULE_DIR="../modules/abcrown"
INIT_RELATIVE_PATH="complete_verifier/__init__.py"
INIT_FULL_PATH="$ABCROWN_SUBMODULE_DIR/$INIT_RELATIVE_PATH"

echo "[ACT-CI-abCrown] Patching abCrown __init__.py to prevent circular import..."
if grep -q "^from abcrown import abCrown" "$INIT_FULL_PATH"; then
    echo "[ACT-CI-abCrown] Found problematic line. Commenting it out..."
    sed -i 's/^from abcrown import abCrown/# from abcrown import abCrown/' "$INIT_FULL_PATH"

    echo "[ACT-CI-abCrown] Marking file as assume-unchanged within submodule..."
    pushd "$ABCROWN_SUBMODULE_DIR" > /dev/null
    git update-index --assume-unchanged "$INIT_RELATIVE_PATH" \
        && echo "[ACT-CI-abCrown] Successfully marked as assume-unchanged." \
        || echo "[WARN] Git mark failed. You may need to check submodule status manually."
    popd > /dev/null
else
    echo "[ACT-CI-abCrown] No patch needed. __init__.py already safe."
fi

echo "[ACT-CI-abCrown] abCrown environment setup complete."
echo "[ACT-CI-abCrown] Ready for abCrown verification with act-abcrown environment."