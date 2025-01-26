# Stage 1: Build stage
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install all dependencies, OpenCV, Eigen3, TensorRT and cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    libyaml-cpp-dev \
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
    tensorrt-libs=8.5.3.1-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Set working directory
WORKDIR /home/SuperSLAM

# Copy project files
COPY . .

# Download libtorch with CUDA support and Pangolin
RUN mkdir -p thirdparty \
    && wget -q https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip -d thirdparty \
    && rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip \
    && wget https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.9.2.zip \
    && unzip v0.9.2.zip -d thirdparty \
    && mv thirdparty/Pangolin-0.9.2 thirdparty/Pangolin \
    && rm v0.9.2.zip

# Build third-party dependencies in parallel using Ninja
RUN (cd thirdparty/Pangolin && \
     mkdir -p build && \
     cd build && \
     cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release && \
     ninja -j$(nproc)) & \
    (cd thirdparty/g2o && \
     mkdir -p build && \
     cd build && \
     cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release && \
     ninja -j$(nproc)) & \
    (cd thirdparty/DBoW3 && \
     mkdir -p build && \
     cd build && \
     cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release && \
     ninja -j$(nproc)) & \
    wait

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/SuperSLAM/thirdparty/libtorch/lib:$LD_LIBRARY_PATH

# Default command
CMD ["/bin/bash"]