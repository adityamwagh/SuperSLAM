FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
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
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-pip \
    libeigen3-dev \
    libglew-dev \
    libboost-all-dev \
    libssl-dev \
    libopencv-dev \
    python3-opencv \
    libyaml-cpp-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT 8.5.3 and cuDNN for CUDA 11.8
RUN apt-get update && apt-get install -y \
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

# Set working directory
WORKDIR /home/SuperSLAM

# Copy project files
COPY . .

# Download and install libtorch with CUDA support
RUN mkdir -p thirdparty && \
    wget -q https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip -d thirdparty && \
    rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

# Build third-party dependencies
RUN chmod +x install_dependencies.sh && \
    ./install_dependencies.sh

# # Build the project
# RUN mkdir build && cd build && \
#     cmake .. -GNinja \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DTorch_DIR=/SuperSLAM/thirdparty/libtorch/share/cmake/Torch \
#         -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 \
#         -DEigen3_DIR=/usr/lib/cmake/eigen3 && \
#     ninja

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/SuperSLAM/thirdparty/libtorch/lib:$LD_LIBRARY_PATH

# Default command
CMD ["/bin/bash"]