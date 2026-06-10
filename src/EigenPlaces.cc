#include "EigenPlaces.h"

#include <cuda_fp16.h>

#include <cstring>
#include <fstream>
#include <opencv4/opencv2/imgproc.hpp>

#include "Logging.h"

namespace {
class EPLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      SLOG_INFO("[TensorRT/EigenPlaces] {}", msg);
  }
};
EPLogger gLogger;

// ImageNet normalization constants.
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

size_t element_size(nvinfer1::DataType d) {
  return d == nvinfer1::DataType::kHALF ? 2 : 4;
}
} // namespace

EigenPlaces::EigenPlaces(const std::string& engine_file, int input_width, int input_height)
    : engine_file_(engine_file), input_width_(input_width), input_height_(input_height) {
  if (const char* s = std::getenv("SUPERSLAM_LOOP_MIN_SCORE"))
    min_score_ = static_cast<float>(std::atof(s));
}

EigenPlaces::~EigenPlaces() {
  if (d_input_)
    cudaFree(d_input_);
  if (d_output_)
    cudaFree(d_output_);
  if (stream_)
    cudaStreamDestroy(stream_);
}

bool EigenPlaces::initialize() {
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    SLOG_ERROR("EigenPlaces: failed to create CUDA stream");
    return false;
  }
  return load_engine() && allocate_buffers();
}

bool EigenPlaces::load_engine() {
  std::ifstream file(engine_file_, std::ios::binary);
  if (!file.good()) {
    SLOG_ERROR("EigenPlaces: cannot read engine {} (rebuild with rebuild_engines.sh)",
               engine_file_);
    return false;
  }
  file.seekg(0, std::ios::end);
  const size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  file.read(data.data(), size);

  runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  if (!runtime_)
    return false;
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
  if (!engine_) {
    SLOG_ERROR("EigenPlaces: deserialize failed (TensorRT version / GPU arch "
               "mismatch)");
    return false;
  }
  context_.reset(engine_->createExecutionContext());
  return context_ != nullptr;
}

bool EigenPlaces::allocate_buffers() {
  // Identify the single input and single output tensor.
  const int n = engine_->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = name;
    } else {
      output_name_ = name;
      output_dtype_ = engine_->getTensorDataType(name);
    }
  }
  if (input_name_.empty() || output_name_.empty()) {
    SLOG_ERROR("EigenPlaces: expected one input and one output tensor");
    return false;
  }

  // Fix the input shape [1,3,H,W]; the output descriptor dim follows.
  const nvinfer1::Dims4 in{1, 3, input_height_, input_width_};
  if (!context_->setInputShape(input_name_.c_str(), in)) {
    SLOG_ERROR("EigenPlaces: setInputShape failed");
    return false;
  }
  const nvinfer1::Dims out_dims = context_->getTensorShape(output_name_.c_str());
  desc_dim_ = 1;
  for (int i = 1; i < out_dims.nbDims; ++i) // skip batch
    desc_dim_ *= out_dims.d[i];

  const size_t in_bytes = static_cast<size_t>(3) * input_height_ * input_width_ * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(desc_dim_) * element_size(output_dtype_);
  h_input_.resize(static_cast<size_t>(3) * input_height_ * input_width_);
  h_output_.resize(out_bytes);
  if (cudaMalloc(&d_input_, in_bytes) != cudaSuccess ||
      cudaMalloc(&d_output_, out_bytes) != cudaSuccess) {
    SLOG_ERROR("EigenPlaces: cudaMalloc failed");
    return false;
  }
  SLOG_INFO("EigenPlaces: ready ({}x{} -> {}-d descriptor)",
            input_width_,
            input_height_,
            desc_dim_);
  return true;
}

void EigenPlaces::preprocess(const cv::Mat& image, float* dst) const {
  cv::Mat rgb;
  if (image.channels() == 1)
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  else
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, rgb, cv::Size(input_width_, input_height_));
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  // HWC to CHW with per-channel ImageNet normalization.
  const int hw = input_height_ * input_width_;
  for (int y = 0; y < input_height_; ++y) {
    const float* row = rgb.ptr<float>(y);
    for (int x = 0; x < input_width_; ++x) {
      for (int c = 0; c < 3; ++c) {
        const float v = row[x * 3 + c];
        dst[c * hw + y * input_width_ + x] = (v - kMean[c]) / kStd[c];
      }
    }
  }
}

cv::Mat EigenPlaces::compute_global_descriptor(const cv::Mat& image) {
  if (!context_)
    return cv::Mat();
  preprocess(image, h_input_.data());

  const size_t in_bytes = h_input_.size() * sizeof(float);
  context_->setInputShape(input_name_.c_str(), nvinfer1::Dims4{1, 3, input_height_, input_width_});
  context_->setTensorAddress(input_name_.c_str(), d_input_);
  context_->setTensorAddress(output_name_.c_str(), d_output_);
  cudaMemcpyAsync(d_input_, h_input_.data(), in_bytes, cudaMemcpyHostToDevice, stream_);
  if (!context_->enqueueV3(stream_)) {
    SLOG_ERROR("EigenPlaces: inference failed");
    return cv::Mat();
  }
  cudaMemcpyAsync(h_output_.data(), d_output_, h_output_.size(), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // Read descriptor (fp32 or fp16) into a row vector and L2-normalize.
  cv::Mat desc(1, desc_dim_, CV_32F);
  float* out = desc.ptr<float>(0);
  if (output_dtype_ == nvinfer1::DataType::kHALF) {
    const __half* h = reinterpret_cast<const __half*>(h_output_.data());
    for (int i = 0; i < desc_dim_; ++i)
      out[i] = __half2float(h[i]);
  } else {
    std::memcpy(out, h_output_.data(), desc_dim_ * sizeof(float));
  }
  cv::normalize(desc, desc, 1.0, 0.0, cv::NORM_L2);
  return desc;
}
