# SuperSLAM: Open Source System for Deep Learning based Visual SLAM

> Alpha Software
 
SuperSLAM is a deep learning based visual SLAM system that combines learned feature detection and matching with a classical SLAM pipeline.

## Installation

### Local Installation

Easiest way to get started would be the `setup.sh` script.

See [INSTALL.md](INSTALL.md) for more information.

### Docker/Podman Installation


### Using Docker

1. Build the Docker image:

```bash
docker build -t superslam .
```

2. Run the container with GPU support:

```bash
docker run --gpus all -it --rm superslam
```

3. For development with mounted source code:

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  superslam
```

### Using Podman

1. Build the Podman image:

```bash
podman build -t superslam .
```

2. Run the container with GPU support:

```bash
podman run --security-opt=label=disable --device /dev/dri:/dev/dri --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it --rm superslam
```

3. For development with mounted source code:

```bash
podman run --security-opt=label=disable --device /dev/dri:/dev/dri --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  superslam
```

## Build and run
```bash
git clone https://github.com/adityamwagh/SuperSLAM.git
cd SuperSLAM
sh ./build.sh
```

This will create `libSuperSLAM.so` in the `lib` folder and the executables `mono_kitti_rerun` and `stereo_kitti_rerun` in appropriate subfolders in `examples` folder.

## Troubleshooting
- **CUDA Errors**: Ensure your NVIDIA drivers and CUDA toolkit are correctly installed.
- **ROS 2 Issues**: Verify that the ROS 2 environment is sourced correctly.
- **Missing Dependencies**: Double-check that all dependencies listed above are installed.

## Contributing

Contributions to the SuperSLAM project are welcome! Please ensure that any changes are well-documented and tested.

## License

This project is licensed under the [LGPL](LICENSE).

For any questions or issues, please open an issue on the GitHub repository.

## Star History

<a href="https://www.star-history.com/#adityamwagh/SuperSLAM&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=adityamwagh/SuperSLAM&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=adityamwagh/SuperSLAM&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=adityamwagh/SuperSLAM&type=Date" />
 </picture>
</a>

## Acknowledgements

Lot of code is borrowed from these repositories. Thanks to the authors for opensourcing them!
* [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
* [TensorRT](https://github.com/NVIDIA/TensorRT)
* [AirVO](https://github.com/xukuanHIT/AirVO)
* [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)
* [superslam](https://github.com/klammecr/superslam)
* [SuperPoint-SuperGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT)
