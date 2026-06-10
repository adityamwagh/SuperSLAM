#ifndef EIGENPLACESTRT_H_
#define EIGENPLACESTRT_H_

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv4/opencv2/core.hpp>
#include <string>

#include "PlaceRecognizer.h" // superslam::IPlaceRecognizer, CosineDescriptorIndex
#include "SuperPoint.h"      // reuse superslam::TRT*Deleter

// EigenPlaces (ResNet18) global-descriptor place recognizer on TensorRT.
// Produce one L2-normalized descriptor per image; delegate retrieval to
// CosineDescriptorIndex (shared with the unit-test stub). Run on the loop
// worker thread. Build the engine from convert_eigenplaces_to_onnx.py via
// rebuild_engines.sh.
class EigenPlaces : public superslam::IPlaceRecognizer {
public:
  // input_width and input_height: the resolution to resize images to before
  // inference.
  EigenPlaces(const std::string& engine_file, int input_width, int input_height);
  ~EigenPlaces();

  bool initialize();

  // superslam::IPlaceRecognizer
  cv::Mat compute_global_descriptor(const cv::Mat& image) override;
  void add(size_t keyframe_id, const cv::Mat& global_descriptor) override {
    index_.add(keyframe_id, global_descriptor);
  }
  std::vector<superslam::LoopCandidate>
  query(const cv::Mat& global_descriptor, size_t excludeRecent, int topK) override {
    return index_.query(global_descriptor, excludeRecent, topK, min_score_);
  }

private:
  bool load_engine();
  bool allocate_buffers();
  void preprocess(const cv::Mat& image,
                  float* dst) const; // ImageNet-normalized NCHW

  std::unique_ptr<nvinfer1::IRuntime, superslam::TRTRuntimeDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, superslam::TRTEngineDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, superslam::TRTContextDeleter> context_;
  cudaStream_t stream_ = nullptr;

  std::string engine_file_;
  int input_width_;
  int input_height_;
  int desc_dim_ = 0;
  float min_score_ = 0.75f;

  std::string input_name_, output_name_;
  nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;
  void* d_input_ = nullptr;  // device input  [1,3,H,W]
  void* d_output_ = nullptr; // device output [1,D]
  std::vector<float> h_input_;
  std::vector<char> h_output_;

  superslam::CosineDescriptorIndex index_;
};

#endif // EIGENPLACESTRT_H_
