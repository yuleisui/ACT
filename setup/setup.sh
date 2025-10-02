#!/bin/bash
set -e
echo "[ACT] Setting up main ACT environment..."

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
    echo "[ACT] Creating conda env: act-main..."
    conda create -y -n act-main python=3.9
else
    echo "[ACT] Conda env 'act-main' already exists."
fi

echo "[ACT] Activating ACT-main environment..."
conda activate act-main

echo "[ACT] Installing ACT requirements..."
pip install -r main_requirements.txt

# Step 4: Install gurobi via conda for ACT main environment solving
echo "[ACT] Installing Gurobi for act-main environment..."
conda config --add channels http://conda.anaconda.org/gurobi
conda install -y gurobi 

# Step 5: Create and activate abcrown environment (act-abcrown)
if ! conda env list | grep -q "^act-abcrown "; then
    echo "[ACT] Creating conda env: act-abcrown..."
    conda create -y -n act-abcrown python=3.9
else
    echo "[ACT] Conda env 'act-abcrown' already exists."
fi

echo "[ACT] Activating ACT-ABCROWN environment..."
conda activate act-abcrown

# Step 6: Install ABCROWN dependencies
echo "[ACT] Installing ABCROWN requirements..."
pip install -r abcrown_requirements.txt

# Step 7: Call ERAN environment setup script (note: cannot activate sub-environment here)
echo "[ACT] Setting up ERAN sub-environment..."
source eran_env_setup.sh

# Step 8: Create empty config file for abcrown CLI parameter mode
echo "[ACT] Creating empty_config.yaml for CLI-only abcrown runs..."
echo "{}" > ../verifier/abstract_constraint_solver/abcrown/empty_config.yaml

# Step 9: Patch the abcrown module __init__.py
ABCROWN_SUBMODULE_DIR="../modules/abcrown"
INIT_RELATIVE_PATH="complete_verifier/__init__.py"
INIT_FULL_PATH="$ABCROWN_SUBMODULE_DIR/$INIT_RELATIVE_PATH"

echo "[ACT] Patching ABCROWN __init__.py to prevent circular import..."
if grep -q "^from abcrown import ABCROWN" "$INIT_FULL_PATH"; then
    echo "[ACT] Found problematic line. Commenting it out..."
    sed -i 's/^from abcrown import ABCROWN/# from abcrown import ABCROWN/' "$INIT_FULL_PATH"

    echo "[ACT] Marking file as assume-unchanged within submodule..."
    pushd "$ABCROWN_SUBMODULE_DIR" > /dev/null
    git update-index --assume-unchanged "$INIT_RELATIVE_PATH" \
        && echo "[ACT] Successfully marked as assume-unchanged." \
        || echo "[WARN] Git mark failed. You may need to check submodule status manually."
    popd > /dev/null
else
    echo "[ACT] No patch needed. __init__.py already safe."
fi

export ACTHOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

#Step 10: Set up the Gurobi license path
export GRB_LICENSE_FILE=$ACTHOME/gurobi/gurobi.lic
echo "[ACT] Gurobi license path configured for this shell: $GRB_LICENSE_FILE"

echo "[ACT] Setup complete. Now you can run with 'conda activate act-main' and subprocess call abCrown, ERAN and Hybrid Zonotope."