#ifndef SUPERPOINTTRT_H_
#define SUPERPOINTTRT_H_

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

// Custom deleters for TensorRT objects
// Note: In TensorRT 10.x, the destroy() method is replaced with delete
namespace SuperSLAM {

struct TRTRuntimeDeleter {
  void operator()(nvinfer1::IRuntime* rt) { delete rt; }
};

struct TRTEngineDeleter {
  void operator()(nvinfer1::ICudaEngine* engine) { delete engine; }
};

struct TRTContextDeleter {
  void operator()(nvinfer1::IExecutionContext* context) { delete context; }
};

}  // namespace SuperSLAM

class SuperPointTRT {
 public:
  explicit SuperPointTRT(const std::string& engine_file, int max_keypoints,
                         double keypoint_threshold, int remove_borders);
  ~SuperPointTRT();

  bool initialize();
  bool infer(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
             cv::Mat& descriptors);

 private:
  struct TensorInfo {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    size_t size = 0;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    std::string name;
  };

  std::unique_ptr<nvinfer1::IRuntime, SuperSLAM::TRTRuntimeDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, SuperSLAM::TRTEngineDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, SuperSLAM::TRTContextDeleter>
      context_;
  cudaStream_t stream_;

  std::vector<TensorInfo> input_tensors_;
  std::vector<TensorInfo> output_tensors_;

  int input_height_ = 0;
  int input_width_ = 0;

  // Configuration parameters
  std::string engine_file_;
  int max_keypoints_;
  double keypoint_threshold_;
  int remove_borders_;

  bool loadEngine();
  bool allocateBuffers();
  bool allocateDynamicBuffers(int height, int width);
  void freeBuffers();
  bool preprocessImage(const cv::Mat& image);
  bool postprocessOutputs(std::vector<cv::KeyPoint>& keypoints,
                          cv::Mat& descriptors);

  size_t getElementSize(nvinfer1::DataType dtype);
  size_t getTensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
  void printTensorInfo(const std::string& name, const nvinfer1::Dims& dims,
                       nvinfer1::DataType dtype);
};

typedef std::shared_ptr<SuperPointTRT> SuperPointTRTPtr;

#endif  // SUPERPOINTTRT_H_