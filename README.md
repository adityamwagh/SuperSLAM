# SuperSLAM: Open Source Framework for Deep Learning based Monocular Visual SLAM

> (Work in Progress) Very early stage development that I do in my free time now. This might break once in a while.
 
SuperSLAM is a deep learning based visual SLAM system that combines recent advances in learned feature detection and matching with the mapping capabilities of ORB_SLAM2. 

It utilizes SuperPoint for keypoint detection and description and SuperGlue for robust feature matching between frames. These matches are then used by ORB_SLAM2 to estimate camera poses and build a map of the environment.

One of the future goals is to eventually replace the ORB_SLAM2 backend with a more modern and performant implementation and also release the code under a MIT Licence.

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
sh ./build_dependencies.sh
```

You can use the included script to build the dependencies or install using the APT package manager.

### Manually Install Dependencies

```bash
# OpenCV
sudo apt-get install -y libopencv-dev

# Eigen

sudo apt install libeigen3-dev

# Pangolin

git clone --recursive https://github.com/stevenlovegrove/Pangolin.git

# Pangolin is split into a few components so you can include just what you need. Most dependencies are optional so you can pick and mix for your needs.
# Rather than enforcing a particular package manager, you can use a simple script to generate a list of (required, recommended or all) packages for installation for that manager (e.g. apt, port, brew, dnf, pacman, vcpkg):

# See what package manager and packages are recommended
./scripts/install_prerequisites.sh --dry-run recommended

# Override the package manager choice and install all packages
./scripts/install_prerequisites.sh -m brew all
```

## Convert model(Optional)
The converted model is already provided in the [weights](./weights) folder, if you are using the pretrained model officially provided by [SuperPoint and SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), you do not need to go through this step.
```bash
python convert2onnx/convert_superpoint_to_onnx.py --weight_file superpoint_pth_file_path --output_dir superpoint_onnx_file_dir
python convert2onnx/convert_superglue_to_onnx.py --weight_file superglue_pth_file_path --output_dir superglue_onnx_file_dir
```

## Build and run
```bash
git clone https://github.com/adityamwagh/SuperSLAM.git
cd SuperSLAM
sh ./build.sh
```

The default image size param is 320x240, if you need to modify the image size in the config file, you should delete the old .engine file in the weights dir.

## Acknowledgements

Lot of code is borrowed from these repositories. Thanks to the authors for opensourcing them!
* [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
* [TensorRT](https://github.com/NVIDIA/TensorRT)
* [AirVO](https://github.com/xukuanHIT/AirVO)
* [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)
* [superslam](https://github.com/klammecr/superslam)
* [SuperPoint-SuperGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT)
