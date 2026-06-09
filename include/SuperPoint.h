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

#include "DescriptorGather.h"    // launch_gather_descriptors
#include "DescriptorPool.h"      // superslam::DescriptorPool, DeviceDescriptors
#include "InferenceInterfaces.h" // superslam::IFeatureExtractor, Features

// Custom deleters for TensorRT objects.
namespace superslam {

struct TRTRuntimeDeleter {
  void operator()(nvinfer1::IRuntime* rt) { delete rt; }
};

struct TRTEngineDeleter {
  void operator()(nvinfer1::ICudaEngine* engine) { delete engine; }
};

struct TRTContextDeleter {
  void operator()(nvinfer1::IExecutionContext* context) { delete context; }
};

} // namespace superslam

class SuperPoint : public superslam::IFeatureExtractor {
public:
  explicit SuperPoint(const std::string& engine_file,
                      int max_keypoints,
                      double keypoint_threshold,
                      int remove_borders);
  ~SuperPoint();

  bool initialize();
  // Run the host path: CPU grid-sample and D2H of the dense grid.
  bool infer(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

  // superslam::IFeatureExtractor device path: gather descriptors on the GPU
  // into a DescriptorPool slot.
  superslam::Features extract(const cv::Mat& image) override;
  // Run the batched stereo path: one {2,1,H,W} infer for L and R, then gather both slices.
  std::pair<superslam::Features, superslam::Features> extract_stereo(const cv::Mat& left,
                                                                     const cv::Mat& right) override;

private:
  struct TensorInfo {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    size_t size = 0;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype{};
    std::string name;
  };

  std::unique_ptr<nvinfer1::IRuntime, superslam::TRTRuntimeDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, superslam::TRTEngineDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, superslam::TRTContextDeleter> context_;
  cudaStream_t stream_ = nullptr;

  std::vector<TensorInfo> input_tensors_;
  std::vector<TensorInfo> output_tensors_;

  int input_height_ = 0;
  int input_width_ = 0;
  int input_batch_ = 1; // Current bound batch size (1 = mono, 2 = stereo).

  // Device descriptor gather (extract() path).
  static constexpr int descriptor_dim = 256;
  static constexpr int descriptor_pool_slots =
      8; // Live frames in flight (left, right, keyframe, plus headroom).
  std::unique_ptr<superslam::DescriptorPool> pool_;
  int* cell_h_dev_ = nullptr; // [max_keypoints_] per-keypoint grid row, device
  int* cell_w_dev_ = nullptr; // [max_keypoints_] per-keypoint grid col, device

  // Configuration parameters
  std::string engine_file_;
  int max_keypoints_;
  double keypoint_threshold_;
  int remove_borders_;

  bool load_engine();
  bool allocate_buffers();
  bool allocate_dynamic_buffers(int height, int width);
  void free_buffers();
  bool preprocess_image(const cv::Mat& image);
  bool postprocess_outputs(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

  // Run preprocess, H2D, and enqueue, leaving all outputs resident on the
  // device. Do not copy outputs to host or synchronize; the caller
  // chooses which outputs to read back.
  bool run_inference(const cv::Mat& image);
  // Run the device path used by extract(): D2H the scores only, select keypoints
  // on the host, then gather their descriptors on the GPU into a DescriptorPool
  // slot.
  bool infer_device(const cv::Mat& image,
                    std::vector<cv::KeyPoint>& keypoints,
                    superslam::DeviceDescriptors& descriptors);
  // Run the batched stereo device path: preprocess L and R into a {2,1,H,W} input, one enqueue,
  // then select keypoints and gather descriptors for each batch slice.
  bool infer_device_stereo(const cv::Mat& left,
                           const cv::Mat& right,
                           std::vector<cv::KeyPoint>& kp_left,
                           superslam::DeviceDescriptors& desc_left,
                           std::vector<cv::KeyPoint>& kp_right,
                           superslam::DeviceDescriptors& desc_right);
  // Select keypoints (host scores scan) and gather descriptors on device for one image or
  // batch slice. `scores_host` points at this slice's score map; `grid_device` at its FP16
  // descriptor grid. The mono and stereo device paths share this.
  bool select_and_gather(const void* scores_host,
                         bool scores_half,
                         int score_height,
                         int score_width,
                         const void* grid_device,
                         int desc_channels,
                         int desc_height,
                         int desc_width,
                         std::vector<cv::KeyPoint>& keypoints,
                         superslam::DeviceDescriptors& descriptors);

  size_t get_element_size(nvinfer1::DataType dtype);
  size_t get_tensor_size(const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
  void
  print_tensor_info(const std::string& name, const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
};

typedef std::shared_ptr<SuperPoint> SuperPointPtr;

#endif // SUPERPOINTTRT_H_
