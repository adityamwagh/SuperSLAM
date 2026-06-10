#ifndef LIGHTGLUETRT_H_
#define LIGHTGLUETRT_H_

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "InferenceInterfaces.h" // MatchResult, superslam::IFeatureMatcher
#include "SuperPoint.h"          // TensorRT deleters

// LightGlue matcher (TensorRT). Implement superslam::IFeatureMatcher.
//
// Expect this ONNX I/O (matches utils/convert_lightglue_to_onnx.py):
//   inputs : kpts0 [1,N,2] f32, kpts1 [1,M,2] f32,
//            desc0 [1,N,256] f32, desc1 [1,M,256] f32   ([N,256], not
//            [256,N])
//   outputs: matches0 [1,N0] int32 (index into set1 per query kpt, -1 if
//   none),
//            mscores0 [1,N0] f32
// Normalize keypoints in the wrapper (LightGlue convention:
//   (kpt - size/2) / (max(w,h)/2)); the export must NOT re-normalize.
// Hold the deserialized LightGlue engine, shareable across multiple execution contexts (the
// tracking matcher and the loop-closure matcher).
struct LightGlueEngine {
  std::unique_ptr<nvinfer1::IRuntime, superslam::TRTRuntimeDeleter> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, superslam::TRTEngineDeleter> engine;
};

class LightGlue : public superslam::IFeatureMatcher {
public:
  // Primary path: deserialize the engine from file (owns a new shared LightGlueEngine).
  explicit LightGlue(const std::string& engine_file, int image_width, int image_height);
  // Shared path: reuse an already-loaded engine; this instance builds only its own context,
  // stream, and buffers. Use shared_engine() of a primary LightGlue.
  LightGlue(std::shared_ptr<LightGlueEngine> shared_engine, int image_width, int image_height);
  ~LightGlue();

  bool initialize();
  // Return the deserialized engine to share with a second LightGlue (no re-deserialize).
  std::shared_ptr<LightGlueEngine> shared_engine() const { return engine_; }
  bool match(const std::vector<cv::KeyPoint>& keypoints0,
             const cv::Mat& descriptors0,
             const std::vector<cv::KeyPoint>& keypoints1,
             const cv::Mat& descriptors1,
             MatchResult& result);

  // superslam::IFeatureMatcher
  MatchResult match(const std::vector<cv::KeyPoint>& kp0,
                    const cv::Mat& d0,
                    const std::vector<cv::KeyPoint>& kp1,
                    const cv::Mat& d1) override;
  // Run the device path: descriptors FP16-resident in DescriptorPool slots, D2D-copied
  // into the engine's FP16 desc inputs.
  MatchResult match(const std::vector<cv::KeyPoint>& kp0,
                    const superslam::DeviceDescriptors& d0,
                    const std::vector<cv::KeyPoint>& kp1,
                    const superslam::DeviceDescriptors& d1) override;
  // Copy FP16 device slot to host CV_32F (rows L2-normalized by the gather).
  cv::Mat descriptors_to_host(const superslam::DeviceDescriptors& d) override;

private:
  struct TensorInfo {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    size_t size = 0;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype{};
    std::string name;
  };

  std::string engine_file_;
  int image_width_;
  int image_height_;

  std::shared_ptr<LightGlueEngine>
      engine_; // Shared, deserialized once; backs one or more contexts.
  std::unique_ptr<nvinfer1::IExecutionContext, superslam::TRTContextDeleter> context_;
  cudaStream_t stream_ = nullptr;

  std::vector<TensorInfo> input_tensors_;
  std::vector<TensorInfo> output_tensors_;

  bool load_engine();
  bool allocate_buffers();
  void free_buffers();
  bool allocate_dynamic_tensors(int n0, int n1);
  bool prepare_inputs(const std::vector<cv::KeyPoint>& keypoints0,
                      const cv::Mat& descriptors0,
                      const std::vector<cv::KeyPoint>& keypoints1,
                      const cv::Mat& descriptors1);
  bool postprocess_outputs(MatchResult& result);

  size_t get_element_size(nvinfer1::DataType dtype);
  size_t get_tensor_size(const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
  void normalize_keypoints(const std::vector<cv::KeyPoint>& keypoints, float* output);
  // Normalize keypoints (LightGlue convention) and store into a kpts input
  // tensor's host buffer, respecting its dtype. The host and device match paths
  // share this.
  void store_keypoints(TensorInfo* dst, const std::vector<cv::KeyPoint>& kps);
  TensorInfo* find_tensor(std::vector<TensorInfo>& tensors, const std::string& name);
};

typedef std::shared_ptr<LightGlue> LightGluePtr;

#endif // LIGHTGLUETRT_H_
