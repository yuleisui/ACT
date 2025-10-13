#!/bin/bash
set -e

# Configuration flags and parameters
ACT_CI_MODE=${ACT_CI_MODE:-false}
COMPONENT=${1:-"all"}  # Default to all components if no argument provided

# Function to show usage
show_usage() {
    echo "Usage: source [COMPONENT]"
    echo "COMPONENT can be: main, abcrown, eran, all (default)"
    echo "Set ACT_CI_MODE=true for CI installation"
    echo ""
    echo "Examples:"
    echo "  source setup.sh              # Install all components (local mode)"
    echo "  source setup.sh main         # Install only main environment"
    echo "  ACT_CI_MODE=true source setup.sh main  # Install main for CI"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

if [ "$ACT_CI_MODE" = "true" ]; then
    echo "[ACT-CI] Setting up ACT environment for CI (component: $COMPONENT)..."
else
    echo "[ACT] Setting up ACT environment (component: $COMPONENT)..."
fi

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

# Function to setup main environment
setup_main() {
    echo "[ACT] Setting up main environment..."
    
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
    if [ "$ACT_CI_MODE" = "true" ]; then
        echo "[ACT-CI] Skipping Gurobi installation for CI..."
    else
        echo "[ACT] Installing Gurobi for act-main environment..."
        conda config --add channels http://conda.anaconda.org/gurobi
        conda install -y gurobi 
    fi
}

# Function to setup abcrown environment
setup_abcrown() {
    echo "[ACT] Setting up abCrown environment..."
    
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

    # Step 8: Create empty config file for abcrown CLI parameter mode
    echo "[ACT] Creating empty_config.yaml for CLI-only abcrown runs..."
    echo "{}" > ../act/wrapper_exts/abcrown/empty_config.yaml

    # Step 9: Patch the abcrown module __init__.py
    ABCROWN_SUBMODULE_DIR="../modules/abcrown"
    INIT_RELATIVE_PATH="complete_verifier/__init__.py"
    INIT_FULL_PATH="$ABCROWN_SUBMODULE_DIR/$INIT_RELATIVE_PATH"

    echo "[ACT] Patching ABCROWN __init__.py to prevent circular import..."
    if grep -q "^from abcrown import abCrown" "$INIT_FULL_PATH"; then
        echo "[ACT] Found problematic line. Commenting it out..."
        sed -i 's/^from abcrown import abCrown/# from abcrown import abCrown/' "$INIT_FULL_PATH"

        if [ "$ACT_CI_MODE" = "false" ]; then
            echo "[ACT] Marking file as assume-unchanged within submodule..."
            pushd "$ABCROWN_SUBMODULE_DIR" > /dev/null
            git update-index --assume-unchanged "$INIT_RELATIVE_PATH" \
                && echo "[ACT] Successfully marked as assume-unchanged." \
                || echo "[WARN] Git mark failed. You may need to check submodule status manually."
            popd > /dev/null
        fi
    else
        echo "[ACT] No patch needed. __init__.py already safe."
    fi
}

# Function to setup ERAN environment (calls eran_env_setup.sh)
setup_eran() {
    echo "[ACT] Setting up ERAN environment..."
    # Step 7: Call ERAN environment setup script
    if [ "$ACT_CI_MODE" = "true" ]; then
        echo "[ACT] Using CI mode for ERAN setup..."
        ACT_CI_MODE=true source eran_env_setup.sh
    else
        echo "[ACT] Using normal mode for ERAN setup..."
        source eran_env_setup.sh
    fi
}

# Main setup logic based on component selection
case "$COMPONENT" in
    "main")
        setup_main
        ;;
    "abcrown")
        setup_abcrown
        ;;
    "eran")
        setup_eran
        ;;
    "all")
        setup_main
        setup_abcrown
        setup_eran
        ;;
    *)
        echo "[ERROR] Unknown component: $COMPONENT"
        show_usage
        exit 1
        ;;
esac

# Final setup steps (for all or when not in CI mode)
if [ "$ACT_CI_MODE" = "false" ] && [ "$COMPONENT" = "all" ]; then
    export ACTHOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    export GRB_LICENSE_FILE=$ACTHOME/gurobi/gurobi.lic
    echo "[ACT] Gurobi license path configured for this shell: $GRB_LICENSE_FILE"
fi

echo "[ACT] Setup complete for component: $COMPONENT"
if [ "$ACT_CI_MODE" = "false" ]; then
    echo "[ACT] Now you can run with 'conda activate act-main' and subprocess call abCrown, ERAN and Hybrid Zonotope."
fi