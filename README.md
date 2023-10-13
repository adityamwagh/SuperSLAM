# SuperSLAM: Framework for deep learning based SLAM

SuperSLAM is a deep learning based visual SLAM system that combines recent advances in learned feature detection and matching with the mapping capabilities of ORB_SLAM2. 

It utilizes SuperPoint for keypoint detection and description and SuperGlue for robust feature matching between frames. These matches are then used by ORB_SLAM2 to estimate camera poses and build a map of the environment.

## Environment required
* CUDA==11.6
* TensorRT==8.4.1.5
* OpenCV>=4.0
* Eigen
* yaml-cpp
* DBoW3
* DBoW2

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
* [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) 
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) 
* [TensorRT](https://github.com/NVIDIA/TensorRT) 
* [AirVO](https://github.com/xukuanHIT/AirVO)
* [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)
