#!/bin/bash
set -e  # Exit on error

# Function to prompt user for yes/no input
prompt_yes_no() {
    while true; do
        printf "%s (yes/no): " "$1"
        read yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Detect shell and set appropriate configuration file
#!/bin/bash

detect_shell() {
    # Detect shell and set appropriate configuration file
    local shell_name

    # Try to detect the shell from $SHELL first
    if [[ -n "$SHELL" ]]; then
        shell_name=$(basename "$SHELL")
    else
        # Fallback to parsing the parent process name if $SHELL is not set
        shell_name=$(basename "$(ps -o comm= -p $PPID)")
    fi

    local shell_config

    case $shell_name in
        zsh)
            shell_config="$HOME/.zshrc"
            ;;
        bash)
            shell_config="$HOME/.bashrc"
            ;;
        fish)
            shell_config="$HOME/.config/fish/config.fish"
            ;;
        tcsh)
            shell_config="$HOME/.tcshrc"
            ;;
        *)
            shell_config="$HOME/.profile"
            ;;
    esac

    echo "$shell_config"
}

SHELL_CONFIG=$(detect_shell)
echo "-- Detected shell: $SHELL, using config file: $SHELL_CONFIG"

# Set non-interactive frontend for apt
export DEBIAN_FRONTEND=noninteractive

# Add NVIDIA package repository and install CUDA 11.8
echo "-- Setting up NVIDIA CUDA 11.8 repository..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    gnupg2 \
    software-properties-common \
    curl

# Add NVIDIA CUDA 11.8 repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8
rm cuda-keyring_1.1-1_all.deb

# Install CUDA Toolkit 11.8
echo "-- Installing CUDA Toolkit 11.8..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    cuda-toolkit-11-8 \
    cuda-libraries-dev-11-8 \
    cuda-command-line-tools-11-8 \
    cuda-compiler-11-8 \
    cuda-cudart-dev-11-8 \
    cuda-cupti-dev-11-8 \
    cuda-nvml-dev-11-8 \
    cuda-nvtx-11-8

# Update environment variables for CUDA
echo "-- Updating environment variables for CUDA..."
if [[ "$SHELL_NAME" == "fish" ]]; then
    echo "set -x PATH /usr/local/cuda-11.8/bin \$PATH" >> $SHELL_CONFIG
    echo "set -x LD_LIBRARY_PATH /usr/local/cuda-11.8/lib64 \$LD_LIBRARY_PATH" >> $SHELL_CONFIG
else
    echo "export PATH=/usr/local/cuda-11.8/bin:\$PATH" >> $SHELL_CONFIG
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH" >> $SHELL_CONFIG
fi
source $SHELL_CONFIG

# Install general dependencies
echo "-- Installing general dependencies..."
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    ninja-build \
    git \
    pkg-config \
    gfortran \
    python3-dev \
    python3-pip \
    libglew-dev \
    libboost-all-dev \
    libssl-dev \
    wget \
    unzip \
    curl \
    libgl1-mesa-dev \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols \
    libegl1-mesa-dev \
    libc++-dev \
    libepoxy-dev \
    libglew-dev \
    libeigen3-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    openexr \
    libatlas-base-dev \
    libopencv-dev \
    python3-opencv \
    libyaml-cpp-dev

# Install CUDA-related packages
echo "-- Installing TensorRT and cuDNN for CUDA 11.8..."
sudo apt-get install -y --no-install-recommends \
    libcudnn8=8.9.7.29-1+cuda11.8 \
    libcudnn8-dev=8.9.7.29-1+cuda11.8 \
    libnvinfer8=8.5.3-1+cuda11.8 \
    libnvinfer-plugin8=8.5.3-1+cuda11.8 \
    libnvinfer-plugin-dev=8.5.3-1+cuda11.8 \
    libnvinfer-bin=8.5.3-1+cuda11.8 \
    libnvinfer-dev=8.5.3-1+cuda11.8 \
    libnvinfer-samples=8.5.3-1+cuda11.8 \
    libnvonnxparsers8=8.5.3-1+cuda11.8 \
    libnvonnxparsers-dev=8.5.3-1+cuda11.8 \
    libnvparsers8=8.5.3-1+cuda11.8 \
    libnvparsers-dev=8.5.3-1+cuda11.8 \
    tensorrt=8.5.3.1-1+cuda11.8 \
    tensorrt-dev=8.5.3.1-1+cuda11.8 \
    tensorrt-libs=8.5.3.1-1+cuda11.8

# Optional ROS 2 Humble installation
if prompt_yes_no "-- Do you want to install ROS 2 Humble and run ROS 2 examples?"; then
    echo "-- Installing ROS 2 Humble..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository universe -y
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        python3-colcon-common-extensions

    # Source ROS in shell config
    echo "-- Configuring ROS 2 environment for $SHELL_NAME..."
    if [[ "$SHELL_NAME" == "fish" ]]; then
        echo "source /opt/ros/humble/setup.fish" >> $SHELL_CONFIG
    else
        echo "source /opt/ros/humble/setup.bash" >> $SHELL_CONFIG
    fi
    source $SHELL_CONFIG
else
    echo "-- Skipping ROS 2 Humble installation."
fi

# Set repository directory
SUPER_SLAM_DIR=$(pwd)
echo "-- Running setup in repository directory: $SUPER_SLAM_DIR"

# Check if libtorch is already downloaded and extracted
if [ ! -d "thirdparty/libtorch" ]; then
    echo "-- Downloading libtorch..."
    mkdir -p thirdparty
    wget -q https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip -d thirdparty
    rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
else
    echo "-- libtorch already exists in thirdparty/libtorch"
fi

# Check if Pangolin is already downloaded and extracted
if [ ! -d "thirdparty/Pangolin" ]; then
    echo "-- Setting up Pangolin..."
    wget https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.9.2.zip
    unzip v0.9.2.zip -d thirdparty
    mv thirdparty/Pangolin-0.9.2 thirdparty/Pangolin
    rm v0.9.2.zip
else
    echo "-- Pangolin already exists in thirdparty/Pangolin"
fi

# Build third-party dependencies in parallel
echo "-- Building third-party dependencies..."
(
    cd thirdparty/Pangolin
    mkdir -p build && cd build
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
    sudo ninja install
) &

(
    cd thirdparty/g2o
    mkdir -p build && cd build
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j$(nproc)
) &

(
    cd thirdparty/DBoW3
    mkdir -p build && cd build
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j$(nproc)
) &

wait  # Wait for all background processes

# Set environment variables
echo "-- Setting environment variables for $SHELL_NAME..."
if [[ "$SHELL_NAME" == "fish" ]]; then
    echo "set -x LD_LIBRARY_PATH /usr/local/cuda/lib64 ${SUPER_SLAM_DIR}/thirdparty/libtorch/lib \$LD_LIBRARY_PATH" >> $SHELL_CONFIG
else
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${SUPER_SLAM_DIR}/thirdparty/libtorch/lib:\$LD_LIBRARY_PATH" >> $SHELL_CONFIG
fi


echo ""
echo "-- Setup complete! Please run 'source $SHELL_CONFIG' or restart your terminal."
echo "-- Then build the project with: bash build.sh"
