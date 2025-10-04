#!/bin/bash
set -e

# Configuration flags
ACT_CI_MODE=${ACT_CI_MODE:-false}

if [ "$ACT_CI_MODE" = "true" ]; then
    echo "[ERAN-CI] Setting up ERAN environment with conda dependencies for CI..."
else
    echo "[ERAN] Setting up ERAN environment with conda dependencies..."
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^act-eran "; then
    echo "[ERAN] Creating conda env 'act-eran'..."
    conda create -y -n act-eran python=3.8
else
    echo "[ERAN] Conda env 'act-eran' already exists."
fi

echo "[ERAN] Activating ERAN environment..."
conda activate act-eran

echo "[ERAN] Installing conda dependencies..."
conda install -y -c conda-forge \
    gmp=6.2.1 \
    mpfr=4.1.0 \
    cmake \
    make \
    gcc_linux-64 \
    gxx_linux-64 \
    autoconf \
    libtool \
    m4 \
    pkg-config

echo "[ERAN] Installing Python packages..."
pip install -r eran_requirements.txt
ERAN_DIR="$(cd ../modules/eran && pwd)"
echo "[ERAN] Switching to $ERAN_DIR"
pushd "$ERAN_DIR" > /dev/null

echo "[ERAN] Installing Gurobi from source (C++ support needed)..."
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
    echo "[ERAN] Installing Gurobi Python interface..."
    python setup.py install
    cd ../../
    echo "[ERAN] Gurobi installed successfully"
    rm -f gurobi9.1.2_linux64.tar.gz
    echo "[ERAN] Cleaned up Gurobi installation tarball"
else
    echo "[ERAN] Gurobi directory already exists, skipping"
fi

export CONDA_PREFIX=${CONDA_PREFIX}
export GUROBI_HOME="$ERAN_DIR/gurobi912/linux64"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export CPPFLAGS="-I${CONDA_PREFIX}/include -I${GUROBI_HOME}/include ${CPPFLAGS}"
export LDFLAGS="-L${CONDA_PREFIX}/lib -L${GUROBI_HOME}/lib ${LDFLAGS}"

if [ ! -f "cddlib-0.94m.tar.gz" ]; then
    echo "[ERAN] Downloading cddlib..."
    wget -q https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
fi
if [ ! -d "cddlib-0.94m" ]; then
    echo "[ERAN] Installing cddlib from source..."
    tar -xf cddlib-0.94m.tar.gz
    cd cddlib-0.94m
    autoreconf -i
    ./configure --prefix=$CONDA_PREFIX --disable-static --enable-shared
    make -j$(nproc)
    make install
    cd ..
    rm -f cddlib-0.94m.tar.gz
    rm -rf cddlib-0.94m

    if [ "$ACT_CI_MODE" = "true" ]; then
        ln -sf $CONDA_PREFIX/include/cddlib/* $CONDA_PREFIX/include/
    fi
    echo "[ERAN] cddlib installation completed"
else
    echo "[ERAN] cddlib already installed, skipping"
fi

if [ ! -d "ELINA" ]; then
    echo "[ERAN] Cloning ELINA..."
    git clone https://github.com/eth-sri/ELINA.git
fi

cd ELINA
if [ ! -f "elina_installed" ]; then
    echo "[ERAN] Installing ELINA..."
    export CDD_PREFIX=$CONDA_PREFIX
    export CPPFLAGS="-I$CDD_PREFIX/include -I$CDD_PREFIX/include/cddlib $CPPFLAGS"
    export LDFLAGS="-L$CDD_PREFIX/lib $LDFLAGS"

    if [ "$ACT_CI_MODE" = "true" ]; then
        echo "[ERAN] Setting relaxed compiler flags for ELINA compilation..."
        export CFLAGS="-Wno-incompatible-pointer-types -Wno-error=incompatible-pointer-types $CFLAGS"
        export CXXFLAGS="-Wno-incompatible-pointer-types -Wno-error=incompatible-pointer-types $CXXFLAGS"
    fi

    ./configure -use-deeppoly -use-gurobi -use-fconv -prefix $CONDA_PREFIX -gmp-prefix $CONDA_PREFIX -mpfr-prefix $CONDA_PREFIX -cdd-prefix $CONDA_PREFIX
    make -j$(nproc)
    make install
    touch elina_installed
    echo "[ERAN] ELINA installation completed"
fi
cd ..

if [ ! -d "deepg" ]; then
    echo "[ERAN] Cloning DeepG..."
    git clone https://github.com/eth-sri/deepg.git
fi

cd deepg/code
if [ ! -f "deepg_installed" ]; then
    echo "[ERAN] Installing DeepG..."
    mkdir -p build
    export GUROBI_HOME="$ERAN_DIR/gurobi912/linux64"
    make shared_object
    cp ./build/libgeometric.so $CONDA_PREFIX/lib/
    touch deepg_installed
    echo "[ERAN] DeepG installation completed"
fi
cd ../..

popd > /dev/null

echo "[ERAN] Installing compatible libstdc++..."
conda install -y -c conda-forge "libstdcxx-ng>=12.0.0"

echo "[ERAN] Configuring ELINA Python interface..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
ELINA_PYTHON_PATH="$ERAN_DIR/ELINA/python_interface"
echo "$ELINA_PYTHON_PATH" > "$SITE_PACKAGES/elina.pth"

echo "[ERAN] ERAN environment setup completed!"