# SuperSLAM: Framework for Deep Learning based Visual SLAM

(Work in Progress) SuperSLAM is a deep learning based visual SLAM system that combines recent advances in learned feature detection and matching with the mapping capabilities of ORB_SLAM2.

It utilizes SuperPoint for keypoint detection and description and SuperGlue for robust feature matching between frames. These matches are then used by ORB_SLAM2 to estimate camera poses and build a map of the environment.

## Environment required

* CUDA==11.6
* TensorRT==8.4.1.5
* OpenCV>=4.0
* Eigen
* yaml-cpp
* DBoW3
* DBoW2
* Ubuntu 20.04

## Installation

Clone the repository and the submodules.

```bash
git clone https://github.com/adityamwagh/SuperSLAM.git --recursive
cd SuperSLAM
```

### Automatically Install Dependencies

```bash
sh ./install_dependencies.sh
```

You can use the included script to build the dependencies or install using the APT package manager.

### Manually Install Dependencies

#### OpenCV

```bash
sudo apt-get install -y libopencv-dev
```

#### Eigen

```bash
sudo apt install libeigen3-dev
```

#### Pangolin

```bash
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
```

Pangolin is split into a few components so you can include just what you need. Most dependencies are optional so you can pick and mix for your needs. Rather than enforcing a particular package manager, you can use a simple script to generate a list of (required, recommended or all) packages for installation for that manager (e.g. apt, port, brew, dnf, pacman, vcpkg):

```bash
# See what package manager and packages are recommended
./scripts/install_prerequisites.sh --dry-run recommended

# install recommended prerequisites for pangolin
./scripts/install_prerequisites.sh recommended

# Configure and build
cmake -B build
cmake --build build

# with Ninja for faster builds (sudo apt install ninja-build)
cmake -B build -GNinja
cmake --build build
```

## Acknowledgements

* [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
* [TensorRT](https://github.com/NVIDIA/TensorRT)
* [AirVO](https://github.com/xukuanHIT/AirVO)
* [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)
