#ifndef SUPERGLUETRT_H_
#define SUPERGLUETRT_H_

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "SuperPointTRT.h"  // For TensorRT deleters


struct MatchResult {
  std::vector<cv::DMatch> matches;
  cv::Mat scores;
};

class SuperGlueTRT {
 public:
  explicit SuperGlueTRT(const std::string& engine_file, int image_width,
                        int image_height);
  ~SuperGlueTRT();

  bool initialize();
  bool match(const std::vector<cv::KeyPoint>& keypoints0,
             const cv::Mat& descriptors0,
             const std::vector<cv::KeyPoint>& keypoints1,
             const cv::Mat& descriptors1, MatchResult& result);

 private:
  struct TensorInfo {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    size_t size = 0;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    std::string name;
  };

  std::string engine_file_;
  int image_width_;
  int image_height_;

  std::unique_ptr<nvinfer1::IRuntime, SuperSLAM::TRTRuntimeDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, SuperSLAM::TRTEngineDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, SuperSLAM::TRTContextDeleter>
      context_;
  cudaStream_t stream_;

  std::vector<TensorInfo> input_tensors_;
  std::vector<TensorInfo> output_tensors_;

  bool loadEngine();
  bool allocateBuffers();
  void freeBuffers();
  bool allocateDynamicTensors(const std::vector<cv::KeyPoint>& keypoints0,
                              const std::vector<cv::KeyPoint>& keypoints1);
  bool prepareInputs(const std::vector<cv::KeyPoint>& keypoints0,
                     const cv::Mat& descriptors0,
                     const std::vector<cv::KeyPoint>& keypoints1,
                     const cv::Mat& descriptors1);
  bool postprocessOutputs(const std::vector<cv::KeyPoint>& keypoints0,
                          const std::vector<cv::KeyPoint>& keypoints1,
                          MatchResult& result);

  size_t getElementSize(nvinfer1::DataType dtype);
  size_t getTensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
  void printTensorInfo(const std::string& name, const nvinfer1::Dims& dims,
                       nvinfer1::DataType dtype);

  // Helper functions for input preparation
  void normalizeKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                          int imageWidth, int imageHeight, float* output);
  void copyDescriptors(const cv::Mat& descriptors, float* output);
};

typedef std::shared_ptr<SuperGlueTRT> SuperGlueTRTPtr;

#endif  // SUPERGLUETRT_H_