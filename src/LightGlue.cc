#include "LightGlue.h"

#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

#include "Logging.h"

namespace {
class LGLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      SLOG_WARN("[TensorRT] {}", msg);
  }
};
LGLogger gLgLogger;
} // namespace

LightGlue::LightGlue(const std::string& engine_file, int image_width, int image_height)
    : engine_file_(engine_file), image_width_(image_width), image_height_(image_height) {
  cudaStreamCreate(&stream_);
}

LightGlue::LightGlue(std::shared_ptr<LightGlueEngine> shared_engine,
                     int image_width,
                     int image_height)
    : image_width_(image_width), image_height_(image_height), engine_(std::move(shared_engine)) {
  cudaStreamCreate(&stream_);
}

LightGlue::~LightGlue() {
  free_buffers();
  cudaStreamDestroy(stream_);
}

bool LightGlue::initialize() {
  SLOG_INFO("LightGlue: Initializing...");
  if (!load_engine())
    return false;
  if (!allocate_buffers())
    return false;
  SLOG_INFO("LightGlue: Initialization successful");
  return true;
}

bool LightGlue::load_engine() {
  // Shared path: the sharing ctor set engine_; skip deserialize and reuse the weights.
  if (!engine_) {
    SLOG_INFO("LightGlue: Loading engine from {}", engine_file_);
    std::ifstream file(engine_file_, std::ios::binary);
    if (!file.good()) {
      SLOG_ERROR("LightGlue: Cannot read engine file {}", engine_file_);
      return false;
    }
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    auto holder = std::make_shared<LightGlueEngine>();
    holder->runtime.reset(nvinfer1::createInferRuntime(gLgLogger));
    if (!holder->runtime)
      return false;
    holder->engine.reset(holder->runtime->deserializeCudaEngine(data.data(), size));
    if (!holder->engine) {
      SLOG_ERROR("LightGlue: Failed to deserialize engine (TensorRT version mismatch?)");
      return false;
    }
    engine_ = std::move(holder);
    SLOG_INFO("LightGlue: Engine loaded successfully");
  } else {
    SLOG_INFO("LightGlue: Sharing an already-loaded engine (no re-deserialize)");
  }

  // Each instance gets its own execution context (own stream and I/O buffers).
  context_.reset(engine_->engine->createExecutionContext());
  if (!context_)
    return false;
  return true;
}

size_t LightGlue::get_element_size(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kINT8:
  case nvinfer1::DataType::kBOOL:
    return 1;
  default:
    return 4;
  }
}

size_t LightGlue::get_tensor_size(const nvinfer1::Dims& dims, nvinfer1::DataType dtype) {
  size_t size = get_element_size(dtype);
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0)
      return 0; // dynamic
    size *= dims.d[i];
  }
  return size;
}

bool LightGlue::allocate_buffers() {
  int n = engine_->engine->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    TensorInfo t;
    t.name = engine_->engine->getIOTensorName(i);
    t.dims = engine_->engine->getTensorShape(t.name.c_str());
    t.dtype = engine_->engine->getTensorDataType(t.name.c_str());
    t.size = get_tensor_size(t.dims,
                             t.dtype); // 0 if dynamic, then allocated per-frame
    if (t.size > 0) {
      if (cudaMalloc(&t.devicePtr, t.size) != cudaSuccess)
        return false;
      if (cudaMallocHost(&t.hostPtr, t.size) != cudaSuccess)
        return false;
    }
    if (engine_->engine->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
      input_tensors_.push_back(t);
    else
      output_tensors_.push_back(t);
  }
  return true;
}

void LightGlue::free_buffers() {
  for (auto& t : input_tensors_) {
    if (t.devicePtr)
      cudaFree(t.devicePtr);
    if (t.hostPtr)
      cudaFreeHost(t.hostPtr);
  }
  for (auto& t : output_tensors_) {
    if (t.devicePtr)
      cudaFree(t.devicePtr);
    if (t.hostPtr)
      cudaFreeHost(t.hostPtr);
  }
  input_tensors_.clear();
  output_tensors_.clear();
}

LightGlue::TensorInfo* LightGlue::find_tensor(std::vector<TensorInfo>& tensors,
                                              const std::string& name) {
  for (auto& t : tensors)
    if (t.name == name)
      return &t;
  return nullptr;
}

// LightGlue keypoint normalization: (kpt - size/2) / (max(w,h)/2).
void LightGlue::normalize_keypoints(const std::vector<cv::KeyPoint>& keypoints, float* output) {
  const float scale = std::max(image_width_, image_height_) / 2.0f;
  const float cx = image_width_ / 2.0f;
  const float cy = image_height_ / 2.0f;
  for (size_t i = 0; i < keypoints.size(); ++i) {
    output[i * 2 + 0] = (keypoints[i].pt.x - cx) / scale;
    output[i * 2 + 1] = (keypoints[i].pt.y - cy) / scale;
  }
}

bool LightGlue::allocate_dynamic_tensors(int n0, int n1) {
  auto ensure_capacity = [](TensorInfo& t, size_t needed) -> bool {
    if (t.devicePtr != nullptr && needed == t.size)
      return true;
    if (t.devicePtr)
      cudaFree(t.devicePtr);
    if (t.hostPtr)
      cudaFreeHost(t.hostPtr);
    t.devicePtr = nullptr;
    t.hostPtr = nullptr;
    if (cudaMalloc(&t.devicePtr, needed) != cudaSuccess)
      return false;
    if (cudaMallocHost(&t.hostPtr, needed) != cudaSuccess)
      return false;
    t.size = needed;
    return true;
  };

  for (auto& t : input_tensors_) {
    nvinfer1::Dims d = t.dims;
    if (t.name == "kpts0") {
      d.d[1] = n0;
      d.d[2] = 2;
    } else if (t.name == "desc0") {
      d.d[1] = n0;
      d.d[2] = 256;
    } else if (t.name == "kpts1") {
      d.d[1] = n1;
      d.d[2] = 2;
    } else if (t.name == "desc1") {
      d.d[1] = n1;
      d.d[2] = 256;
    } else {
      continue;
    }
    if (!context_->setInputShape(t.name.c_str(), d)) {
      SLOG_ERROR("LightGlue: Failed to set input shape for {}", t.name);
      return false;
    }
    if (!ensure_capacity(t, get_tensor_size(d, t.dtype))) {
      SLOG_ERROR("LightGlue: Failed to (re)allocate input {}", t.name);
      return false;
    }
  }

  for (auto& t : output_tensors_) {
    nvinfer1::Dims d = context_->getTensorShape(t.name.c_str());
    if (!ensure_capacity(t, get_tensor_size(d, t.dtype))) {
      SLOG_ERROR("LightGlue: Failed to (re)allocate output {}", t.name);
      return false;
    }
  }
  return true;
}

// Store float values into a tensor's host buffer respecting its dtype (fp16
// engines take fp16 inputs).
static void store_floats(void* host_ptr, nvinfer1::DataType dtype, const std::vector<float>& v) {
  if (dtype == nvinfer1::DataType::kHALF) {
    __half* h = static_cast<__half*>(host_ptr);
    for (size_t i = 0; i < v.size(); ++i)
      h[i] = __float2half(v[i]);
  } else {
    std::memcpy(host_ptr, v.data(), v.size() * sizeof(float));
  }
}

void LightGlue::store_keypoints(TensorInfo* dst, const std::vector<cv::KeyPoint>& kps) {
  // LightGlue keypoint normalization: (kpt - size/2)/(max(w,h)/2).
  const float scale = std::max(image_width_, image_height_) / 2.0f;
  const float cx = image_width_ / 2.0f, cy = image_height_ / 2.0f;
  std::vector<float> v(kps.size() * 2);
  for (size_t i = 0; i < kps.size(); ++i) {
    v[i * 2 + 0] = (kps[i].pt.x - cx) / scale;
    v[i * 2 + 1] = (kps[i].pt.y - cy) / scale;
  }
  store_floats(dst->hostPtr, dst->dtype, v);
}

bool LightGlue::prepare_inputs(const std::vector<cv::KeyPoint>& keypoints0,
                               const cv::Mat& descriptors0,
                               const std::vector<cv::KeyPoint>& keypoints1,
                               const cv::Mat& descriptors1) {
  TensorInfo* kp0 = find_tensor(input_tensors_, "kpts0");
  TensorInfo* kp1 = find_tensor(input_tensors_, "kpts1");
  TensorInfo* d0 = find_tensor(input_tensors_, "desc0");
  TensorInfo* d1 = find_tensor(input_tensors_, "desc1");
  if (!kp0 || !kp1 || !d0 || !d1) {
    SLOG_ERROR("LightGlue: Expected inputs kpts0/kpts1/desc0/desc1 not found "
               "in engine");
    return false;
  }

  // Descriptors are [N, 256] CV_32F.
  auto desc_vec = [](const cv::Mat& desc) {
    cv::Mat d = (desc.type() == CV_32F) ? desc : cv::Mat();
    if (d.empty())
      desc.convertTo(d, CV_32F);
    if (!d.isContinuous())
      d = d.clone();
    return std::vector<float>(reinterpret_cast<const float*>(d.data),
                              reinterpret_cast<const float*>(d.data) + d.total());
  };

  store_keypoints(kp0, keypoints0);
  store_keypoints(kp1, keypoints1);
  store_floats(d0->hostPtr, d0->dtype, desc_vec(descriptors0));
  store_floats(d1->hostPtr, d1->dtype, desc_vec(descriptors1));
  return true;
}

bool LightGlue::match(const std::vector<cv::KeyPoint>& keypoints0,
                      const cv::Mat& descriptors0,
                      const std::vector<cv::KeyPoint>& keypoints1,
                      const cv::Mat& descriptors1,
                      MatchResult& result) {
  if (!context_)
    return false;
  const int n0 = static_cast<int>(keypoints0.size());
  const int n1 = static_cast<int>(keypoints1.size());
  if (n0 == 0 || n1 == 0)
    return false;

  if (!allocate_dynamic_tensors(n0, n1))
    return false;
  if (!prepare_inputs(keypoints0, descriptors0, keypoints1, descriptors1))
    return false;

  for (auto& t : input_tensors_) {
    if (!context_->setTensorAddress(t.name.c_str(), t.devicePtr))
      return false;
    if (cudaMemcpyAsync(t.devicePtr, t.hostPtr, t.size, cudaMemcpyHostToDevice, stream_) !=
        cudaSuccess)
      return false;
  }
  for (auto& t : output_tensors_) {
    if (!context_->setTensorAddress(t.name.c_str(), t.devicePtr))
      return false;
  }
  if (!context_->enqueueV3(stream_)) {
    SLOG_ERROR("LightGlue: enqueueV3 failed");
    return false;
  }
  for (auto& t : output_tensors_) {
    if (cudaMemcpyAsync(t.hostPtr, t.devicePtr, t.size, cudaMemcpyDeviceToHost, stream_) !=
        cudaSuccess)
      return false;
  }
  cudaStreamSynchronize(stream_);
  return postprocess_outputs(result);
}

bool LightGlue::postprocess_outputs(MatchResult& result) {
  result.matches.clear();
  TensorInfo* m = find_tensor(output_tensors_, "matches0");
  TensorInfo* s = find_tensor(output_tensors_, "mscores0");
  if (!m) {
    SLOG_ERROR("LightGlue: output 'matches0' not found");
    return false;
  }

  nvinfer1::Dims md = context_->getTensorShape("matches0");
  // matches0 is [1, N0]: for each query keypoint, the matched index in set1 or
  // -1. Fixed shape (N0 known from input). Filter the -1 entries here on CPU.
  const int N0 = md.nbDims >= 1 ? md.d[md.nbDims - 1] : 0;
  if (N0 <= 0)
    return true; // no keypoints this frame

  // matches0 stays int32 (indices) even in fp16 engines; mscores0 may be fp16.
  const int32_t* idx = static_cast<const int32_t*>(m->hostPtr);
  const bool sc_half = s && s->dtype == nvinfer1::DataType::kHALF;
  auto get_score = [&](int i) -> float {
    if (!s)
      return 1.0f;
    return sc_half ? __half2float(static_cast<const __half*>(s->hostPtr)[i])
                   : static_cast<const float*>(s->hostPtr)[i];
  };

  for (int i = 0; i < N0; ++i) {
    const int j = idx[i];
    if (j < 0)
      continue; // unmatched query keypoint
    cv::DMatch dm;
    dm.queryIdx = i;
    dm.trainIdx = j;
    dm.distance = 1.0f - get_score(i); // similarity to distance
    result.matches.push_back(dm);
  }
  return true;
}

// superslam::IFeatureMatcher. Thin wrapper over the 5-arg match.
MatchResult LightGlue::match(const std::vector<cv::KeyPoint>& kp0,
                             const cv::Mat& d0,
                             const std::vector<cv::KeyPoint>& kp1,
                             const cv::Mat& d1) {
  MatchResult r;
  match(kp0, d0, kp1, d1, r);
  return r;
}

// Device path: upload keypoints H2D; copy descriptors D2D from their FP16
// pool slots into the engine's desc inputs.
MatchResult LightGlue::match(const std::vector<cv::KeyPoint>& kp0,
                             const superslam::DeviceDescriptors& d0,
                             const std::vector<cv::KeyPoint>& kp1,
                             const superslam::DeviceDescriptors& d1) {
  MatchResult r;
  if (!context_ || d0.empty() || d1.empty())
    return r;
  const int n0 = static_cast<int>(kp0.size());
  const int n1 = static_cast<int>(kp1.size());
  if (n0 == 0 || n1 == 0)
    return r;

  if (!allocate_dynamic_tensors(n0, n1))
    return r;

  TensorInfo* kp0_t = find_tensor(input_tensors_, "kpts0");
  TensorInfo* kp1_t = find_tensor(input_tensors_, "kpts1");
  TensorInfo* d0_t = find_tensor(input_tensors_, "desc0");
  TensorInfo* d1_t = find_tensor(input_tensors_, "desc1");
  if (!kp0_t || !kp1_t || !d0_t || !d1_t) {
    SLOG_ERROR("LightGlue: Expected inputs kpts0/kpts1/desc0/desc1 not found "
               "in engine");
    return r;
  }
  if (d0_t->dtype != nvinfer1::DataType::kHALF || d1_t->dtype != nvinfer1::DataType::kHALF) {
    SLOG_ERROR("LightGlue: device match requires FP16 desc input bindings "
               "(rebuild the engine with fp16 I/O via scripts/rebuild_engines.sh)");
    return r;
  }

  store_keypoints(kp0_t, kp0);
  store_keypoints(kp1_t, kp1);

  // Set addresses; H2D the keypoints, D2D the descriptors from the pool slots.
  for (auto& t : input_tensors_)
    if (!context_->setTensorAddress(t.name.c_str(), t.devicePtr))
      return r;
  if (cudaMemcpyAsync(kp0_t->devicePtr,
                      kp0_t->hostPtr,
                      kp0_t->size,
                      cudaMemcpyHostToDevice,
                      stream_) != cudaSuccess ||
      cudaMemcpyAsync(kp1_t->devicePtr,
                      kp1_t->hostPtr,
                      kp1_t->size,
                      cudaMemcpyHostToDevice,
                      stream_) != cudaSuccess)
    return r;
  // Slot holds [N,256] FP16 contiguous, which is the desc input layout; copy exactly
  // its bytes.
  const size_t bytes0 = static_cast<size_t>(n0) * d0.dim * sizeof(__half);
  const size_t bytes1 = static_cast<size_t>(n1) * d1.dim * sizeof(__half);
  if (bytes0 > d0_t->size || bytes1 > d1_t->size) {
    SLOG_ERROR("LightGlue: desc slot ({}/{} B) exceeds engine input ({}/{} B)",
               bytes0,
               bytes1,
               d0_t->size,
               d1_t->size);
    return r;
  }
  if (cudaMemcpyAsync(d0_t->devicePtr, d0.data, bytes0, cudaMemcpyDeviceToDevice, stream_) !=
          cudaSuccess ||
      cudaMemcpyAsync(d1_t->devicePtr, d1.data, bytes1, cudaMemcpyDeviceToDevice, stream_) !=
          cudaSuccess)
    return r;

  for (auto& t : output_tensors_)
    if (!context_->setTensorAddress(t.name.c_str(), t.devicePtr))
      return r;
  if (!context_->enqueueV3(stream_)) {
    SLOG_ERROR("LightGlue: enqueueV3 failed (device match)");
    return r;
  }
  for (auto& t : output_tensors_)
    if (cudaMemcpyAsync(t.hostPtr, t.devicePtr, t.size, cudaMemcpyDeviceToHost, stream_) !=
        cudaSuccess)
      return r;
  cudaStreamSynchronize(stream_);
  postprocess_outputs(r);
  return r;
}

// FP16 device slot to host CV_32F (rows L2-normalized by the gather kernel).
cv::Mat LightGlue::descriptors_to_host(const superslam::DeviceDescriptors& d) {
  if (d.empty())
    return cv::Mat();
  const size_t count = static_cast<size_t>(d.count) * d.dim;
  std::vector<__half> tmp(count);
  if (cudaMemcpy(tmp.data(), d.data, count * sizeof(__half), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    SLOG_ERROR("LightGlue: descriptors_to_host D2H failed");
    return cv::Mat();
  }
  cv::Mat out(d.count, d.dim, CV_32F);
  float* o = out.ptr<float>();
  for (size_t i = 0; i < count; ++i)
    o[i] = __half2float(tmp[i]);
  return out;
}
