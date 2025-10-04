#!/bin/bash
set -e
echo "[ACT-CI-ERAN] Setting up ACT ERAN environment for CI..."

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

# Step 3: Create and activate ERAN environment (act-eran)
if ! conda env list | grep -q "^act-eran "; then
    echo "[ACT-CI-ERAN] Creating conda env: act-eran..."
    conda create -y -n act-eran python=3.8
else
    echo "[ACT-CI-ERAN] Conda env 'act-eran' already exists."
fi

echo "[ACT-CI-ERAN] Activating ACT-ERAN environment..."
conda activate act-eran

# Step 4: Install dependencies
echo "[ACT-CI-ERAN] Installing conda dependencies..."
conda install -y -c conda-forge \
    gmp=6.2.1 \
    mpfr=4.1.0 \
    cmake make autoconf libtool m4 pkg-config \
    gcc_linux-64 gxx_linux-64

echo "[ACT-CI-ERAN] Installing Python packages..."
pip install -r ../setup/eran_requirements.txt

# Step 5: Setup ERAN components
ERAN_DIR="$(cd ../modules/eran && pwd)"
echo "[ACT-CI-ERAN] Switching to $ERAN_DIR"
pushd "$ERAN_DIR" > /dev/null

echo "[ACT-CI-ERAN] Installing Gurobi from source (C++ support needed)..."
if [ ! -f "gurobi9.1.2_linux64.tar.gz" ]; then
    wget -q https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
fi
if [ ! -d "gurobi912" ]; then
    tar -xf gurobi9.1.2_linux64.tar.gz
    cd gurobi912/linux64/src/build
    sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
    make -j$(nproc)
    cp libgurobi_c++.a ../../lib/
    cd ../../
    cp lib/libgurobi91.so $CONDA_PREFIX/lib/
    echo "[ACT-CI-ERAN] Installing Gurobi Python interface..."
    python setup.py install
    cd ../../
    echo "[ACT-CI-ERAN] Gurobi installed successfully"
    rm -f gurobi9.1.2_linux64.tar.gz
else
    echo "[ACT-CI-ERAN] Gurobi directory already exists, skipping"
fi

export GUROBI_HOME="$ERAN_DIR/gurobi912/linux64"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$GUROBI_HOME/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export CPPFLAGS="-I$CONDA_PREFIX/include -I$GUROBI_HOME/include $CPPFLAGS"
export LDFLAGS="-L$CONDA_PREFIX/lib -L$GUROBI_HOME/lib $LDFLAGS"

# Install cddlib
if [ ! -f "cddlib-0.94m.tar.gz" ]; then
    echo "[ACT-CI-ERAN] Downloading cddlib..."
    wget -q https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
fi
if [ ! -d "cddlib-0.94m" ]; then
    echo "[ACT-CI-ERAN] Installing cddlib from source..."
    tar -xf cddlib-0.94m.tar.gz
    cd cddlib-0.94m
    autoreconf -i
    ./configure --prefix=$CONDA_PREFIX --disable-static --enable-shared
    make -j$(nproc)
    make install
    cd ..
    rm -rf cddlib-0.94m cddlib-0.94m.tar.gz
    # fix: ELINA expects headers directly under include/
    ln -sf $CONDA_PREFIX/include/cddlib/* $CONDA_PREFIX/include/
    echo "[ACT-CI-ERAN] cddlib installation completed"
else
    echo "[ACT-CI-ERAN] cddlib already installed, skipping"
fi

# Install ELINA
if [ ! -d "ELINA" ]; then
    echo "[ACT-CI-ERAN] Cloning ELINA..."
    git clone https://github.com/eth-sri/ELINA.git
fi

cd ELINA
if [ ! -f "elina_installed" ]; then
    echo "[ACT-CI-ERAN] Installing ELINA..."
    export CDD_PREFIX=$CONDA_PREFIX
    export CPPFLAGS="-I$CDD_PREFIX/include -I$CDD_PREFIX/include/cddlib $CPPFLAGS"
    export LDFLAGS="-L$CDD_PREFIX/lib $LDFLAGS"

    echo "[ACT-CI-ERAN] Setting relaxed compiler flags for ELINA compilation..."
    export CFLAGS="-Wno-incompatible-pointer-types -Wno-error=incompatible-pointer-types $CFLAGS"
    export CXXFLAGS="-Wno-incompatible-pointer-types -Wno-error=incompatible-pointer-types $CXXFLAGS"

    ./configure -use-deeppoly -use-gurobi -use-fconv \
        -prefix $CONDA_PREFIX \
        -gmp-prefix $CONDA_PREFIX \
        -mpfr-prefix $CONDA_PREFIX \
        -cdd-prefix $CDD_PREFIX
        
    make -j$(nproc)
    make install
    touch elina_installed
    echo "[ACT-CI-ERAN] ELINA installation completed"
else
    echo "[ACT-CI-ERAN] ELINA already installed, skipping"
fi
cd ..

# Install DeepG
if [ ! -d "deepg" ]; then
    echo "[ACT-CI-ERAN] Cloning DeepG..."
    git clone https://github.com/eth-sri/deepg.git
fi

cd deepg/code
if [ ! -f "deepg_installed" ]; then
    echo "[ACT-CI-ERAN] Installing DeepG..."
    mkdir -p build
    make shared_object
    cp ./build/libgeometric.so $CONDA_PREFIX/lib/
    touch deepg_installed
    echo "[ACT-CI-ERAN] DeepG installation completed"
else
    echo "[ACT-CI-ERAN] DeepG already installed, skipping"
fi
cd ../..

popd > /dev/null

echo "[ACT-CI-ERAN] Installing compatible libstdc++..."
conda install -y -c conda-forge "libstdcxx-ng>=12.0.0"

echo "[ACT-CI-ERAN] Configuring ELINA Python interface..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$ERAN_DIR/ELINA/python_interface" > "$SITE_PACKAGES/elina.pth"

echo "[ACT-CI-ERAN] ERAN environment setup completed!"

