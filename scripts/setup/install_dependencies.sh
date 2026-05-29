#!/bin/bash
set -e # Exit on error

# Bare-metal host install for SuperSLAM (Ubuntu 22.04 or 24.04). Mirrors the Docker stack
# (CUDA 12.9 + TensorRT 10.11 + cuDNN 9). The self-contained Dockerfiles
# (Dockerfile.tensorrt10/tensorrt11) are the canonical, build-verified path; this reproduces
# it on a host. Idempotent: re-running re-applies the same apt installs (no-ops if present).
#
# TensorRT 10+ is required: LightGlue's opset-18 ONNX needs the TensorRT 10 ONNX parser. For
# TensorRT 11, swap libnvinfer10/libnvonnxparsers10 for the *11 packages (find the exact
# version with `apt-cache madison libnvinfer11`).

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export DEBIAN_FRONTEND=noninteractive

# --- Map Ubuntu version to NVIDIA CUDA repo path ---
. /etc/os-release
case "${VERSION_ID:-}" in
  24.04) CUDA_REPO="ubuntu2404" ;;
  22.04) CUDA_REPO="ubuntu2204" ;;
  *)
    echo "!! Unsupported Ubuntu '${VERSION_ID:-unknown}' (need 22.04 or 24.04)." >&2
    echo "   You can force one: CUDA_REPO=ubuntu2404 bash $0" >&2
    CUDA_REPO="${CUDA_REPO:-ubuntu2404}" ;;
esac
echo "-- Ubuntu ${VERSION_ID:-?} -> CUDA repo: $CUDA_REPO"

SHELL_NAME=$(basename "${SHELL:-bash}")
case "$SHELL_NAME" in
  zsh) SHELL_CONFIG="$HOME/.zshrc" ;;
  fish) SHELL_CONFIG="$HOME/.config/fish/config.fish" ;;
  *) SHELL_CONFIG="$HOME/.bashrc" ;;
esac
echo "-- Using shell config: $SHELL_CONFIG"

# --- NVIDIA CUDA repository ---
echo "-- Setting up NVIDIA CUDA repository ($CUDA_REPO)..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends gnupg2 ca-certificates curl wget
wget "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
rm -f cuda-keyring_1.1-1_all.deb

# --- CUDA toolkit (slim: nvcc + cudart-dev + libraries-dev, not the full metapackage) ---
echo "-- Installing CUDA (slim)..."
sudo apt-get install -y --no-install-recommends \
    cuda-nvcc-12-9 \
    cuda-cudart-dev-12-9 \
    cuda-libraries-dev-12-9 \
    cuda-command-line-tools-12-9

# --- Build + math + OpenCV/Eigen/Boost/yaml-cpp/spdlog ---
echo "-- Installing general dependencies..."
sudo apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build g++ git pkg-config gfortran ccache make \
    clang-tidy clang-format clangd cppcheck gdb lldb valgrind \
    python3-dev python3-pip unzip \
    libboost-all-dev libssl-dev \
    libeigen3-dev libopencv-dev python3-opencv \
    libyaml-cpp-dev libspdlog-dev \
    libgl1-mesa-dev libgtk-3-dev libwayland-dev libxkbcommon-dev \
    wayland-protocols libegl1-mesa-dev libepoxy-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev libatlas-base-dev

# --- TensorRT 10.11 + cuDNN 9 for CUDA 12.9 (pinned versions) ---
echo "-- Installing TensorRT 10 + cuDNN 9..."
sudo apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12=9.23.2.1-1 \
    libcudnn9-dev-cuda-12=9.23.2.1-1 \
    libnvinfer10=10.11.0.33-1+cuda12.9 \
    libnvinfer-plugin10=10.11.0.33-1+cuda12.9 \
    libnvinfer-plugin-dev=10.11.0.33-1+cuda12.9 \
    libnvinfer-bin=10.11.0.33-1+cuda12.9 \
    libnvinfer-dev=10.11.0.33-1+cuda12.9 \
    libnvinfer-headers-dev=10.11.0.33-1+cuda12.9 \
    libnvinfer-headers-plugin-dev=10.11.0.33-1+cuda12.9 \
    libnvonnxparsers10=10.11.0.33-1+cuda12.9 \
    libnvonnxparsers-dev=10.11.0.33-1+cuda12.9

# --- Environment (CUDA + trtexec) ---
if [[ "$SHELL_NAME" == "fish" ]]; then
    echo "set -x PATH /usr/local/cuda-12.9/bin /usr/src/tensorrt/bin \$PATH" >> "$SHELL_CONFIG"
    echo "set -x LD_LIBRARY_PATH /usr/local/cuda-12.9/lib64 \$LD_LIBRARY_PATH" >> "$SHELL_CONFIG"
else
    echo "export PATH=/usr/local/cuda-12.9/bin:/usr/src/tensorrt/bin:\$PATH" >> "$SHELL_CONFIG"
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:\$LD_LIBRARY_PATH" >> "$SHELL_CONFIG"
fi

echo ""
echo "-- Setup complete! 'source $SHELL_CONFIG' or restart your terminal, then build:"
echo "     cmake -S . -B build-tensorrt -G Ninja -DCMAKE_BUILD_TYPE=Release \\"
echo "       -DCMAKE_CUDA_ARCHITECTURES=\"86;89;120\""
echo "     cmake --build build-tensorrt -j\"\$(nproc)\""
echo ""
echo "-- GPU-free core only (no TensorRT): add -DSUPERSLAM_BUILD_TRT=OFF, or run 'make test-superslam'."
echo "-- Models: uv run python scripts/models/download_onnx_engine_superpoint.py  (then bash scripts/rebuild_engines.sh)"
echo "-- Datasets: uv run python scripts/datasets/download_kitti.py --out ~/datasets/kitti"
