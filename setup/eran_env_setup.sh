#!/bin/bash
set -e
echo "[ERAN] Setting up ERAN environment..."

# 初始化 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"

# 创建 ERAN 环境
if ! conda env list | grep -q "^act-eran "; then
    echo "[ERAN] Creating conda env: act-eran..."
    conda create -y -n act-eran python=3.8 # use Python 3.8 to incorporate the ERAN requirements for onnx (1.8.0)
else
    echo "[ERAN] Conda env 'act-eran' already exists."
fi

echo "[ERAN] Activating ERAN environment..."
conda activate act-eran

# 安装系统级依赖（sudo）
echo "[ERAN] Installing system-level dependencies..."
# 更新包列表
sudo apt-get update -y

# 注意：不包含 libgmp-dev 和 libmpfr-dev，因为 ERAN install.sh 会从源码编译特定版本
REQUIRED_APT_PACKAGES=(m4 build-essential autoconf libtool texlive-latex-base)
for pkg in "${REQUIRED_APT_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" &> /dev/null; then
        echo "[ERAN] Installing $pkg..."
        sudo apt-get install -y "$pkg"
    else
        echo "[ERAN] $pkg already installed."
    fi
done

ERAN_DIR="$(cd ../modules/eran && pwd)"
echo "[ERAN] Switching to $ERAN_DIR"
pushd "$ERAN_DIR" > /dev/null

# 安装 CMake（如缺）
if ! command -v cmake &> /dev/null; then
    echo "[ERAN] cmake not found. Installing..."
    INSTALL_DIR="$(pwd)/cmake-3.19.7-Linux-x86_64"
    wget -q -N https://github.com/Kitware/CMake/releases/download/v3.19.7/cmake-3.19.7-Linux-x86_64.sh
    mkdir -p "$INSTALL_DIR" 
    sudo bash ./cmake-3.19.7-Linux-x86_64.sh --skip-license --prefix="$INSTALL_DIR"
    sudo rm -f /usr/bin/cmake
    sudo ln -s "$INSTALL_DIR/bin/cmake" /usr/bin/cmake
    echo "[ERAN] cmake installed..."
    echo "[DEBUG] Where is cmake? $(which cmake)"
    ls -l /usr/bin/cmake
else
    echo "[ERAN] cmake already available."
fi
 
if [ -f "install.sh" ]; then
    echo "[ERAN] Found install.sh, but using manual installation steps for better control..."
    
    # 手动执行 ERAN 依赖安装步骤（基于官方文档）
    echo "[ERAN] Installing GMP..."
    if [ ! -f "gmp-6.1.2.tar.xz" ]; then
        wget -q https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
    fi
    if [ ! -d "gmp-6.1.2" ]; then
        tar -xf gmp-6.1.2.tar.xz
        cd gmp-6.1.2
        ./configure --enable-cxx
        make -j$(nproc)
        sudo make install
        cd ..
        echo "[ERAN] GMP installed successfully"
    else
        echo "[ERAN] GMP directory already exists, skipping"
    fi
    
    echo "[ERAN] Installing MPFR..."
    if [ ! -f "mpfr-4.1.0.tar.xz" ]; then
        wget -q https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
    fi
    if [ ! -d "mpfr-4.1.0" ]; then
        tar -xf mpfr-4.1.0.tar.xz
        cd mpfr-4.1.0
        ./configure --with-gmp=/usr/local
        make -j$(nproc)
        sudo make install
        cd ..
        echo "[ERAN] MPFR installed successfully"
    else
        echo "[ERAN] MPFR directory already exists, skipping"
    fi
    
    echo "[ERAN] Installing cddlib..."
    if [ ! -f "cddlib-0.94m.tar.gz" ]; then
        wget -q https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
    fi
    if [ ! -d "cddlib-0.94m" ]; then
        tar -xf cddlib-0.94m.tar.gz
        cd cddlib-0.94m
        ./configure
        make -j$(nproc)
        sudo make install
        cd ..
        echo "[ERAN] cddlib installed successfully"
    else
        echo "[ERAN] cddlib directory already exists, skipping"
    fi
    
    echo "[ERAN] Installing Gurobi..."
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
        sudo cp lib/libgurobi91.so /usr/local/lib
        # 使用正确的 Python 环境安装 Gurobi Python 接口（不使用 sudo）
        echo "[ERAN] Installing Gurobi Python interface..."
        python setup.py install --user
        cd ../../
        echo "[ERAN] Gurobi installed successfully"
    else
        echo "[ERAN] Gurobi directory already exists, skipping"
    fi
    
    # 设置环境变量
    export GUROBI_HOME="$(pwd)/gurobi912/linux64"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export CPATH="${CPATH}:${GUROBI_HOME}/include"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib:${GUROBI_HOME}/lib"
    
    echo "[ERAN] Installing ELINA..."
    if [ ! -d "ELINA" ]; then
        git clone https://github.com/eth-sri/ELINA.git
        cd ELINA
        ./configure -use-deeppoly -use-gurobi -use-fconv
        make -j$(nproc)
        sudo make install
        cd ..
        echo "[ERAN] ELINA installed successfully"
    else
        echo "[ERAN] ELINA directory already exists, skipping"
    fi
    
    echo "[ERAN] Installing DeepG..."
    if [ ! -d "deepg" ]; then
        git clone https://github.com/eth-sri/deepg.git
        cd deepg/code
        mkdir -p build
        make shared_object
        sudo cp ./build/libgeometric.so /usr/lib
        cd ../..
        echo "[ERAN] DeepG installed successfully"
    else
        echo "[ERAN] DeepG directory already exists, skipping"
    fi
    
    # 更新库路径
    sudo ldconfig
    
    echo "[ERAN] Manual installation completed successfully"
    
    # 设置ELINA Python接口路径（在popd之前）
    echo "[ERAN] Configuring ELINA Python interface path..."
    ELINA_PYTHON_PATH="$(pwd)/ELINA/python_interface"
    echo "[ERAN] ELINA Python interface path: $ELINA_PYTHON_PATH"
    
else
    echo "[ERAN] ERAN install.sh not found at $ERAN_DIR"
    exit 1
fi

popd > /dev/null

# 安装 Python 包
echo "[ERAN] Installing Python packages for ERAN..."
pip install -r eran_requirements.txt

# 安装conda版本的Gurobi（提供gurobipy模块）
echo "[ERAN] Installing Gurobi via conda for Python interface..."
conda config --add channels http://conda.anaconda.org/gurobi
conda install -y gurobi

# 添加ELINA Python接口到Python路径
echo "[ERAN] Configuring ELINA Python interface path..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "[ERAN] Using ELINA Python interface path: $ELINA_PYTHON_PATH"
echo "$ELINA_PYTHON_PATH" > "$SITE_PACKAGES/elina.pth"
echo "[ERAN] Created elina.pth in: $SITE_PACKAGES"

echo "[ERAN] ERAN environment setup complete."
