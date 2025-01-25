# SuperSLAM Project - Local Development Setup

This guide will walk you through setting up the SuperSLAM project locally on your machine without using Docker. This setup mirrors the environment defined in the Dockerfile and is suitable for development and testing.

## Prerequisites

- **Ubuntu 22.04**: The setup is tested on Ubuntu 22.04. Other distributions may require adjustments.
- **NVIDIA GPU**: Ensure you have an NVIDIA GPU with CUDA 11.8 support.
- **NVIDIA Drivers**: Install the latest NVIDIA drivers compatible with CUDA 11.8.
- **CUDA Toolkit 11.8**: Install CUDA 11.8 from the [NVIDIA website](https://developer.nvidia.com/cuda-11-8-0-download-archive).
- **cuDNN 8.9.7**: Install cuDNN compatible with CUDA 11.8 from the [NVIDIA website](https://developer.nvidia.com/cudnn).

## Step 1: Install General Dependencies

Update your system and install the required general dependencies:

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    build-essential \
    cmake \
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
    curl
```

## Step 2: Install OpenCV Dependencies

Install dependencies for OpenCV:

```bash
sudo apt-get install -y \
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
    python3-opencv
```

## Step 3: Install libyaml-cpp

Install `libyaml-cpp` for YAML file parsing:

```bash
sudo apt-get install -y libyaml-cpp-dev
```

## Step 4: Install Eigen3

Install Eigen3, a C++ template library for linear algebra:

```bash
sudo apt-get install -y libeigen3-dev
```

## Step 5: Install TensorRT 8.5.3 and cuDNN for CUDA 11.8

Install TensorRT and cuDNN for CUDA 11.8:

```bash
sudo apt-get update && sudo apt-get install -y \
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
```

## Step 6: Install ROS 2 Humble

Follow these steps to install ROS 2 Humble:

1. Add the ROS 2 repository:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

2. Install ROS 2 Humble Desktop:

```bash
sudo apt-get update && sudo apt-get install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions
```

3. Source the ROS 2 environment in your `.bashrc`:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Step 7: Set Up the Project

1. Clone the SuperSLAM repository:

```bash
git clone https://github.com/your-repo/SuperSLAM.git --recursive
cd SuperSLAM
```

2. Download and install `libtorch` with CUDA support:

```bash
mkdir -p thirdparty
wget -q https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip -d thirdparty
rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
```

3. Build third-party dependencies:

```bash
chmod +x ./utils/build_deps.sh
./utils/build_deps.sh
```

## Step 8: Set Environment Variables

Add the following to your `.bashrc` to set the `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/SuperSLAM/thirdparty/libtorch/lib:$LD_LIBRARY_PATH
```

Reload your `.bashrc`:

```bash
source ~/.bashrc
```

## Step 9: Verify the Setup

1. Verify CUDA installation:

```bash
nvcc --version
```

2. Verify TensorRT installation:

```bash
dpkg -l | grep tensorrt
```

3. Verify ROS 2 installation:

```bash
ros2 --version
```

4. Verify `libtorch` installation:

Ensure the `libtorch` directory exists in `thirdparty/`.

## Step 10: Build and Run the Project

1. Build the project:

```bash
sh build.sh
```

## Convert model (Optional)
The converted model is already provided in the [weights](./weights) folder, if you are using the pretrained model officially provided by [SuperPoint and SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), you do not need to go through this step.

The default image size param is 320x240, if you need to modify the image size in the `utils/config.yaml` file, you should delete the old `.engine` file in the weights dir.