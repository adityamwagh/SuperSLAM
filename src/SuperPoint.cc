#include "SuperPoint.h"

#include <cuda_fp16.h>
#include <fstream>
#include <iostream>

#include "Logging.h"
#include "Profiling.h"

// TensorRT logger in an anonymous namespace for internal linkage.
namespace {
class SimpleLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      SLOG_INFO("[TensorRT] {}", msg);
    }
  }
};

SimpleLogger gLogger;
} // namespace

SuperPoint::SuperPoint(const std::string& engine_file,
                       int max_keypoints,
                       double keypoint_threshold,
                       int remove_borders)
    : engine_file_(engine_file), max_keypoints_(max_keypoints),
      keypoint_threshold_(keypoint_threshold), remove_borders_(remove_borders), stream_(nullptr) {}

SuperPoint::~SuperPoint() {
  free_buffers();
  // Smart pointers release the TensorRT objects.
  if (stream_)
    cudaStreamDestroy(stream_);
}

bool SuperPoint::initialize() {
  SLOG_INFO("SuperPoint: Initializing...");

  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    SLOG_ERROR("SuperPoint: Failed to create CUDA stream");
    return false;
  }

  if (!load_engine()) {
    return false;
  }

  if (!allocate_buffers()) {
    return false;
  }

  // Device descriptor gather: FP16 slot pool and per-keypoint cell-index
  // scratch buffers.
  pool_ = std::make_unique<superslam::DescriptorPool>(descriptor_pool_slots,
                                                      max_keypoints_,
                                                      descriptor_dim);
  if (cudaMalloc(&cell_h_dev_, max_keypoints_ * sizeof(int)) != cudaSuccess ||
      cudaMalloc(&cell_w_dev_, max_keypoints_ * sizeof(int)) != cudaSuccess) {
    SLOG_ERROR("SuperPoint: Failed to allocate descriptor cell-index buffers");
    return false;
  }

  SLOG_INFO("SuperPoint: Initialization successful");
  return true;
}

bool SuperPoint::load_engine() {
  SLOG_INFO("SuperPoint: Loading engine from {}", engine_file_);

  std::ifstream file(engine_file_, std::ios::binary);
  if (!file.good()) {
    SLOG_ERROR("SuperPoint: Cannot read engine file {}", engine_file_);
    SLOG_ERROR("SuperPoint: Please rebuild the engine using rebuild_engines.sh");
    return false;
  }

  file.seekg(0, std::ifstream::end);
  size_t size = file.tellg();
  file.seekg(0, std::ifstream::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  if (!runtime_) {
    SLOG_ERROR("SuperPoint: Failed to create runtime");
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  if (!engine_) {
    SLOG_ERROR("SuperPoint: Failed to deserialize engine");
    SLOG_ERROR("SuperPoint: This is likely due to TensorRT version mismatch.");
    SLOG_ERROR("SuperPoint: The engine was built with a different TensorRT "
               "version.");
    SLOG_ERROR("SuperPoint: Please rebuild the engine using rebuild_engines.sh");
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    SLOG_ERROR("SuperPoint: Failed to create execution context");
    return false;
  }

  SLOG_INFO("SuperPoint: Engine loaded successfully");
  return true;
}

bool SuperPoint::allocate_buffers() {
  SLOG_INFO("SuperPoint: Allocating buffers...");

  int num_io_tensors = engine_->getNbIOTensors();

  for (int i = 0; i < num_io_tensors; ++i) {
    TensorInfo tensor;
    tensor.name = engine_->getIOTensorName(i);
    tensor.dims = engine_->getTensorShape(tensor.name.c_str());
    tensor.dtype = engine_->getTensorDataType(tensor.name.c_str());

    print_tensor_info(tensor.name, tensor.dims, tensor.dtype);

    if (engine_->getTensorIOMode(tensor.name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
      if (tensor.dims.nbDims >= 2) {
        input_height_ = tensor.dims.d[tensor.dims.nbDims - 2];
        input_width_ = tensor.dims.d[tensor.dims.nbDims - 1];
      }

      // For dynamic shapes, defer memory allocation.
      if (input_height_ <= 0 || input_width_ <= 0) {
        SLOG_INFO("SuperPoint: Dynamic input shape detected, deferring buffer "
                  "allocation");
        tensor.size = 0;    // Mark as not allocated
        input_height_ = -1; // Mark as dynamic
        input_width_ = -1;
      } else {
        tensor.size = get_tensor_size(tensor.dims, tensor.dtype);

        // Allocate device memory
        if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
          SLOG_ERROR("SuperPoint: Failed to allocate device memory for {}", tensor.name);
          return false;
        }

        // Allocate host memory
        if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
          SLOG_ERROR("SuperPoint: Failed to allocate host memory for {}", tensor.name);
          return false;
        }
      }

      input_tensors_.push_back(tensor);
    } else {
      // For output tensors with dynamic shapes, also defer allocation
      tensor.size = get_tensor_size(tensor.dims, tensor.dtype);
      if (tensor.size == 0 || tensor.dims.d[tensor.dims.nbDims - 2] <= 0 ||
          tensor.dims.d[tensor.dims.nbDims - 1] <= 0) {
        SLOG_INFO("SuperPoint: Dynamic output shape detected, deferring buffer "
                  "allocation");
        tensor.size = 0; // Mark as not allocated
      } else {
        // Allocate device memory
        if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
          SLOG_ERROR("SuperPoint: Failed to allocate device memory for {}", tensor.name);
          return false;
        }

        // Allocate host memory
        if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
          SLOG_ERROR("SuperPoint: Failed to allocate host memory for {}", tensor.name);
          return false;
        }
      }

      output_tensors_.push_back(tensor);
    }
  }

  SLOG_INFO("SuperPoint: Input dimensions: {}x{}", input_height_, input_width_);
  SLOG_INFO("SuperPoint: Buffers allocated successfully");
  return true;
}

bool SuperPoint::allocate_dynamic_buffers(int height, int width) {
  SLOG_INFO("SuperPoint: Allocating dynamic buffers for {}x{}", height, width);

  // Allocate input tensors
  for (auto& tensor : input_tensors_) {
    if (tensor.size == 0) { // Not allocated yet
      // Calculate size for current input dimensions
      tensor.size = height * width * get_element_size(tensor.dtype);

      // Allocate device memory
      if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
        SLOG_ERROR("SuperPoint: Failed to allocate device memory for {}", tensor.name);
        return false;
      }

      // Allocate host memory
      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        SLOG_ERROR("SuperPoint: Failed to allocate host memory for {}", tensor.name);
        return false;
      }
    }
  }

  // Allocate output tensors based on current context shapes
  for (auto& tensor : output_tensors_) {
    if (tensor.size == 0) { // Not allocated yet
      nvinfer1::Dims output_dims = context_->getTensorShape(tensor.name.c_str());
      tensor.dims = output_dims;
      tensor.size = get_tensor_size(output_dims, tensor.dtype);

      if (tensor.size > 0) {
        // Allocate device memory
        if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
          SLOG_ERROR("SuperPoint: Failed to allocate device memory for {}", tensor.name);
          return false;
        }

        // Allocate host memory
        if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
          SLOG_ERROR("SuperPoint: Failed to allocate host memory for {}", tensor.name);
          return false;
        }
      }
    }
  }

  return true;
}

void SuperPoint::free_buffers() {
  for (auto& tensor : input_tensors_) {
    if (tensor.devicePtr)
      cudaFree(tensor.devicePtr);
    if (tensor.hostPtr)
      cudaFreeHost(tensor.hostPtr);
  }
  for (auto& tensor : output_tensors_) {
    if (tensor.devicePtr)
      cudaFree(tensor.devicePtr);
    if (tensor.hostPtr)
      cudaFreeHost(tensor.hostPtr);
  }
  input_tensors_.clear();
  output_tensors_.clear();

  if (cell_h_dev_)
    cudaFree(cell_h_dev_);
  if (cell_w_dev_)
    cudaFree(cell_w_dev_);
  cell_h_dev_ = nullptr;
  cell_w_dev_ = nullptr;
  pool_.reset(); // Release the FP16 slot allocations.
}

bool SuperPoint::run_inference(const cv::Mat& image) {
  if (!context_) {
    SLOG_ERROR("SuperPoint: Context not initialized");
    return false;
  }

  // Preprocess the image and set dynamic shapes if needed.
  if (!preprocess_image(image)) {
    SLOG_ERROR("SuperPoint: Failed to preprocess image");
    return false;
  }

  // For dynamic shapes, get the actual output sizes after setting the input
  // shape.
  if (input_height_ > 0 && input_width_ > 0) {
    // Update output tensor sizes based on current input
    for (auto& tensor : output_tensors_) {
      nvinfer1::Dims output_dims = context_->getTensorShape(tensor.name.c_str());
      tensor.dims = output_dims;
    }
  }

  // Set tensor addresses for inputs
  for (auto& tensor : input_tensors_) {
    if (!context_->setTensorAddress(tensor.name.c_str(), tensor.devicePtr)) {
      SLOG_ERROR("SuperPoint: Failed to set input tensor address for {}", tensor.name);
      return false;
    }

    // Calculate actual size to copy (for dynamic shapes)
    size_t copy_size = input_height_ * input_width_ * get_element_size(tensor.dtype);

    // Copy input to device
    if (cudaMemcpyAsync(tensor.devicePtr,
                        tensor.hostPtr,
                        copy_size,
                        cudaMemcpyHostToDevice,
                        stream_) != cudaSuccess) {
      SLOG_ERROR("SuperPoint: Failed to copy input to device");
      return false;
    }
  }

  // Set tensor addresses for outputs
  for (auto& tensor : output_tensors_) {
    if (!context_->setTensorAddress(tensor.name.c_str(), tensor.devicePtr)) {
      SLOG_ERROR("SuperPoint: Failed to set output tensor address for {}", tensor.name);
      return false;
    }
  }

  // Execute inference
  if (!context_->enqueueV3(stream_)) {
    SLOG_ERROR("SuperPoint: Failed to execute inference");
    return false;
  }
  return true;
}

// Host path: run inference, copy all outputs to the host, CPU grid-sample in
// postprocess.
bool SuperPoint::infer(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors) {
  if (!run_inference(image))
    return false;

  // Copy every output tensor device to host.
  for (auto& tensor : output_tensors_) {
    size_t output_size = get_tensor_size(tensor.dims, tensor.dtype);
    if (output_size > tensor.size) {
      SLOG_ERROR("SuperPoint: Output size ({}) exceeds allocated buffer ({})",
                 output_size,
                 tensor.size);
      output_size = tensor.size; // Clamp to the allocated size.
    }
    if (cudaMemcpyAsync(tensor.hostPtr,
                        tensor.devicePtr,
                        output_size,
                        cudaMemcpyDeviceToHost,
                        stream_) != cudaSuccess) {
      SLOG_ERROR("SuperPoint: Failed to copy output from device");
      return false;
    }
  }
  cudaStreamSynchronize(stream_);
  return postprocess_outputs(keypoints, descriptors);
}

bool SuperPoint::preprocess_image(const cv::Mat& image) {
  // For dynamic shapes, allocate buffers after setting the input shape.
  if (input_height_ == -1 && input_width_ == -1) {
    // First time with dynamic shapes: set the input shape and allocate
    // buffers.
    int target_height = image.rows;
    int target_width = image.cols;

    // Set input shape
    nvinfer1::Dims input_dims = input_tensors_[0].dims;
    input_dims.d[0] = 1; // mono path: batch 1 (dynamic-batch engine leaves d[0] dynamic)
    input_dims.d[input_dims.nbDims - 2] = target_height;
    input_dims.d[input_dims.nbDims - 1] = target_width;

    if (!context_->setInputShape(input_tensors_[0].name.c_str(), input_dims)) {
      SLOG_ERROR("SuperPoint: Failed to set input shape");
      return false;
    }

    // Allocate buffers for this input size.
    if (!allocate_dynamic_buffers(target_height, target_width)) {
      SLOG_ERROR("SuperPoint: Failed to allocate dynamic buffers");
      return false;
    }

    input_height_ = target_height;
    input_width_ = target_width;
  }

  if (input_tensors_.empty() || input_tensors_[0].hostPtr == nullptr) {
    SLOG_ERROR("SuperPoint: No input tensors allocated");
    return false;
  }

  cv::Mat processed_image;

  // Convert to grayscale if needed
  if (image.channels() == 3) {
    cv::cvtColor(image, processed_image, cv::COLOR_BGR2GRAY);
  } else {
    processed_image = image.clone();
  }

  // For subsequent calls, check whether to resize the input or reallocate.
  int target_height = processed_image.rows;
  int target_width = processed_image.cols;

  // If input size changed, update shape
  if (target_height != input_height_ || target_width != input_width_) {
    nvinfer1::Dims input_dims = input_tensors_[0].dims;
    input_dims.d[input_dims.nbDims - 2] = target_height;
    input_dims.d[input_dims.nbDims - 1] = target_width;

    if (!context_->setInputShape(input_tensors_[0].name.c_str(), input_dims)) {
      SLOG_ERROR("SuperPoint: Failed to set input shape");
      return false;
    }

    input_height_ = target_height;
    input_width_ = target_width;
  }

  // Resize to the expected input size.
  if (processed_image.rows != target_height || processed_image.cols != target_width) {
    cv::resize(processed_image, processed_image, cv::Size(target_width, target_height));
  }

  // Convert to float and normalize to [0, 1]
  processed_image.convertTo(processed_image, CV_32F, 1.0 / 255.0);

  // Copy to input tensor (assuming NCHW format: 1x1xHxW)
  float* input_ptr = static_cast<float*>(input_tensors_[0].hostPtr);
  std::memcpy(input_ptr, processed_image.data, target_height * target_width * sizeof(float));

  return true;
}

bool SuperPoint::postprocess_outputs(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
  if (output_tensors_.size() < 2) {
    SLOG_ERROR("SuperPoint: Expected at least 2 output tensors (scores, "
               "descriptors)");
    return false;
  }

  keypoints.clear();

  // Outputs may be fp32 or fp16. Read element-wise respecting the tensor dtype.
  auto& scores_dims = output_tensors_[0].dims;
  auto& desc_dims = output_tensors_[1].dims;
  const bool scores_half = output_tensors_[0].dtype == nvinfer1::DataType::kHALF;
  const bool desc_half = output_tensors_[1].dtype == nvinfer1::DataType::kHALF;
  const void* scores_raw = output_tensors_[0].hostPtr;
  const void* desc_raw = output_tensors_[1].hostPtr;
  auto get_score = [&](int idx) -> float {
    return scores_half ? __half2float(static_cast<const __half*>(scores_raw)[idx])
                       : static_cast<const float*>(scores_raw)[idx];
  };
  auto get_desc = [&](int idx) -> float {
    return desc_half ? __half2float(static_cast<const __half*>(desc_raw)[idx])
                     : static_cast<const float*>(desc_raw)[idx];
  };

  // Calculate dimensions
  int score_height = scores_dims.d[scores_dims.nbDims - 2];
  int score_width = scores_dims.d[scores_dims.nbDims - 1];
  int desc_channels = desc_dims.d[desc_dims.nbDims - 3];
  int desc_height = desc_dims.d[desc_dims.nbDims - 2];
  int desc_width = desc_dims.d[desc_dims.nbDims - 1];

  SLOG_INFO("SuperPoint: Scores shape: {}x{}", score_height, score_width);
  SLOG_INFO("SuperPoint: Descriptors shape: {}x{}x{}", desc_channels, desc_height, desc_width);

  // Extract keypoints above threshold (full-resolution heatmap scan, CPU)
  std::vector<std::pair<float, std::pair<int, int>>> candidates;
  for (int h = remove_borders_; h < score_height - remove_borders_; ++h) {
    for (int w = remove_borders_; w < score_width - remove_borders_; ++w) {
      float score = get_score(h * score_width + w);
      if (score > keypoint_threshold_) {
        candidates.emplace_back(score, std::make_pair(h, w));
      }
    }
  }

  // Sort by score and take top K
  std::sort(candidates.begin(), candidates.end(), std::greater<>());

  int num_keypoints = std::min(static_cast<int>(candidates.size()), max_keypoints_);
  keypoints.reserve(num_keypoints);

  // Scale factor from score map to original image
  float scale_x = static_cast<float>(input_width_) / score_width;
  float scale_y = static_cast<float>(input_height_) / score_height;

  std::vector<cv::Point2f> keypoint_positions;
  keypoint_positions.reserve(num_keypoints);

  for (int i = 0; i < num_keypoints; ++i) {
    float score = candidates[i].first;
    int h = candidates[i].second.first;
    int w = candidates[i].second.second;

    // Convert to original image coordinates
    float x = w * scale_x;
    float y = h * scale_y;

    keypoints.emplace_back(x, y, 1.0f, -1, score);
    keypoint_positions.emplace_back(x, y);
  }

  // Extract descriptors for keypoints (CPU grid sampling and L2 normalize).
  if (num_keypoints > 0) {
    descriptors = cv::Mat::zeros(num_keypoints, desc_channels, CV_32F);

    for (int i = 0; i < num_keypoints; ++i) {
      int h = candidates[i].second.first;
      int w = candidates[i].second.second;

      // Scale coordinates to descriptor map
      int desc_h = std::min(h / 8, desc_height - 1); // Assume 8x downsampling.
      int desc_w = std::min(w / 8, desc_width - 1);

      // Extract descriptor
      float* desc_row = descriptors.ptr<float>(i);
      for (int c = 0; c < desc_channels; ++c) {
        int idx = c * desc_height * desc_width + desc_h * desc_width + desc_w;
        desc_row[c] = get_desc(idx);
      }
    }

    // Normalize descriptors
    for (int i = 0; i < num_keypoints; ++i) {
      cv::Mat desc_row = descriptors.row(i);
      cv::normalize(desc_row, desc_row, 1.0, 0.0, cv::NORM_L2);
    }
  }

  SLOG_INFO("SuperPoint: Extracted {} keypoints", num_keypoints);
  return true;
}

size_t SuperPoint::get_element_size(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
    return 1;
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kBOOL:
    return 1;
  default:
    return 4;
  }
}

size_t SuperPoint::get_tensor_size(const nvinfer1::Dims& dims, nvinfer1::DataType dtype) {
  size_t size = get_element_size(dtype);
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      // Dynamic dimension, return 0 to indicate unknown size
      return 0;
    }
    size *= dims.d[i];
  }
  return size;
}

void SuperPoint::print_tensor_info(const std::string& name,
                                   const nvinfer1::Dims& dims,
                                   nvinfer1::DataType dtype) {
  std::string shape_str = "";
  for (int i = 0; i < dims.nbDims; ++i) {
    shape_str += std::to_string(dims.d[i]);
    if (i < dims.nbDims - 1) {
      shape_str += "x";
    }
  }

  std::string type_str;
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    type_str = "FLOAT";
    break;
  case nvinfer1::DataType::kHALF:
    type_str = "HALF";
    break;
  case nvinfer1::DataType::kINT8:
    type_str = "INT8";
    break;
  case nvinfer1::DataType::kINT32:
    type_str = "INT32";
    break;
  case nvinfer1::DataType::kBOOL:
    type_str = "BOOL";
    break;
  default:
    type_str = "UNKNOWN";
    break;
  }

  SLOG_INFO("SuperPoint: Tensor {} - Shape: {}, Type: {}", name, shape_str, type_str);
}

// Device path: run inference, D2H the scores only, select keypoints on the host,
// then gather their descriptors on the GPU.
bool SuperPoint::infer_device(const cv::Mat& image,
                              std::vector<cv::KeyPoint>& keypoints,
                              superslam::DeviceDescriptors& descriptors) {
  // Per-stage timing under SUPERSLAM_PROFILE: infer, keypoint scan, gather.
  auto prof_t = std::chrono::steady_clock::now();
  auto prof_mark = [&prof_t](const char* label) {
    if (superslam::Profiler::enabled())
      superslam::Profiler::instance().add(
          label,
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prof_t)
              .count());
    prof_t = std::chrono::steady_clock::now();
  };

  if (!run_inference(image))
    return false;
  if (output_tensors_.size() < 2) {
    SLOG_ERROR("SuperPoint: Expected >=2 output tensors (scores, descriptors)");
    return false;
  }

  TensorInfo& scores_t = output_tensors_[0];
  TensorInfo& desc_t = output_tensors_[1];

  // Pick up the actual (dynamic) output shapes.
  scores_t.dims = context_->getTensorShape(scores_t.name.c_str());
  desc_t.dims = context_->getTensorShape(desc_t.name.c_str());

  // Copy ONLY the scores back to the host; the descriptor grid stays on the
  // device.
  size_t scores_size = get_tensor_size(scores_t.dims, scores_t.dtype);
  if (scores_size > scores_t.size)
    scores_size = scores_t.size;
  if (cudaMemcpyAsync(scores_t.hostPtr,
                      scores_t.devicePtr,
                      scores_size,
                      cudaMemcpyDeviceToHost,
                      stream_) != cudaSuccess) {
    SLOG_ERROR("SuperPoint: Failed to copy scores from device");
    return false;
  }
  cudaStreamSynchronize(stream_);
  prof_mark("sp_gpu_infer"); // enqueue, scores D2H, and sync

  const bool scores_half = scores_t.dtype == nvinfer1::DataType::kHALF;
  const void* scores_raw = scores_t.hostPtr;
  auto get_score = [&](int idx) -> float {
    return scores_half ? __half2float(static_cast<const __half*>(scores_raw)[idx])
                       : static_cast<const float*>(scores_raw)[idx];
  };

  const int score_height = scores_t.dims.d[scores_t.dims.nbDims - 2];
  const int score_width = scores_t.dims.d[scores_t.dims.nbDims - 1];
  const int desc_channels = desc_t.dims.d[desc_t.dims.nbDims - 3];
  const int desc_height = desc_t.dims.d[desc_t.dims.nbDims - 2];
  const int desc_width = desc_t.dims.d[desc_t.dims.nbDims - 1];

  // The device gather kernel reads the grid as FP16; the engine's `descriptors`
  // output is built with --outputIOFormats=fp16, a half binding.
  if (desc_t.dtype != nvinfer1::DataType::kHALF) {
    SLOG_ERROR("SuperPoint: device gather requires an FP16 descriptor grid binding "
               "(rebuild the engine with fp16 I/O via scripts/rebuild_engines.sh)");
    return false;
  }
  if (desc_channels != descriptor_dim) {
    SLOG_ERROR("SuperPoint: descriptor channels {} != pool dim {}", desc_channels, descriptor_dim);
    return false;
  }

  return select_and_gather(scores_raw,
                           scores_half,
                           score_height,
                           score_width,
                           desc_t.devicePtr,
                           desc_channels,
                           desc_height,
                           desc_width,
                           keypoints,
                           descriptors);
}

// Shared keypoint-select (host) and on-device gather for one image or batch slice. `scores_host`
// and `grid_device` point at this slice's score map or FP16 descriptor grid (caller applies the
// per-batch offset).
bool SuperPoint::select_and_gather(const void* scores_host,
                                   bool scores_half,
                                   int score_height,
                                   int score_width,
                                   const void* grid_device,
                                   int desc_channels,
                                   int desc_height,
                                   int desc_width,
                                   std::vector<cv::KeyPoint>& keypoints,
                                   superslam::DeviceDescriptors& descriptors) {
  auto get_score = [&](int idx) -> float {
    return scores_half ? __half2float(static_cast<const __half*>(scores_host)[idx])
                       : static_cast<const float*>(scores_host)[idx];
  };

  std::vector<std::pair<float, std::pair<int, int>>> candidates;
  for (int h = remove_borders_; h < score_height - remove_borders_; ++h)
    for (int w = remove_borders_; w < score_width - remove_borders_; ++w) {
      const float score = get_score(h * score_width + w);
      if (score > keypoint_threshold_)
        candidates.emplace_back(score, std::make_pair(h, w));
    }
  std::sort(candidates.begin(), candidates.end(), std::greater<>());
  const int num_keypoints = std::min(static_cast<int>(candidates.size()), max_keypoints_);

  keypoints.clear();
  keypoints.reserve(num_keypoints);
  const float scale_x = static_cast<float>(input_width_) / score_width;
  const float scale_y = static_cast<float>(input_height_) / score_height;

  std::vector<int> cell_h(num_keypoints), cell_w(num_keypoints);
  for (int i = 0; i < num_keypoints; ++i) {
    const float score = candidates[i].first;
    const int h = candidates[i].second.first;
    const int w = candidates[i].second.second;
    keypoints.emplace_back(w * scale_x, h * scale_y, 1.0f, -1, score);
    cell_h[i] = std::min(h / 8, desc_height - 1); // 8x downsample, nearest cell
    cell_w[i] = std::min(w / 8, desc_width - 1);
  }

  descriptors = pool_->make(num_keypoints);
  if (num_keypoints == 0)
    return true;
  if (descriptors.empty()) {
    SLOG_ERROR("SuperPoint: descriptor pool exhausted (no free slot)");
    return false;
  }

  cudaMemcpyAsync(cell_h_dev_,
                  cell_h.data(),
                  num_keypoints * sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream_);
  cudaMemcpyAsync(cell_w_dev_,
                  cell_w.data(),
                  num_keypoints * sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream_);
  superslam::launch_gather_descriptors(grid_device,
                                       desc_channels,
                                       desc_height,
                                       desc_width,
                                       cell_h_dev_,
                                       cell_w_dev_,
                                       num_keypoints,
                                       descriptors.data,
                                       stream_);
  cudaStreamSynchronize(stream_);
  return true;
}

// Batched stereo device path: preprocess L and R into one {2,1,H,W} input, a single enqueue, then
// select and gather each batch slice.
bool SuperPoint::infer_device_stereo(const cv::Mat& left,
                                     const cv::Mat& right,
                                     std::vector<cv::KeyPoint>& kp_left,
                                     superslam::DeviceDescriptors& desc_left,
                                     std::vector<cv::KeyPoint>& kp_right,
                                     superslam::DeviceDescriptors& desc_right) {
  if (!context_)
    return false;
  if (left.rows != right.rows || left.cols != right.cols) {
    SLOG_ERROR("SuperPoint: stereo pair must share resolution (rectified)");
    return false;
  }

  // Grayscale and normalize each image to CV_32F [0,1] at the engine input size.
  auto prep = [](const cv::Mat& img, int h, int w) {
    cv::Mat g;
    if (img.channels() == 3)
      cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    else
      g = img.clone();
    if (g.rows != h || g.cols != w)
      cv::resize(g, g, cv::Size(w, h));
    g.convertTo(g, CV_32F, 1.0 / 255.0);
    return g;
  };
  const int h = left.rows, w = left.cols;
  const cv::Mat gl = prep(left, h, w), gr = prep(right, h, w);

  // Bind a batch-2 input shape and (re)size all I/O buffers to match.
  nvinfer1::Dims in;
  in.nbDims = 4;
  in.d[0] = 2;
  in.d[1] = 1;
  in.d[2] = h;
  in.d[3] = w;
  if (!context_->setInputShape(input_tensors_[0].name.c_str(), in)) {
    SLOG_ERROR("SuperPoint: setInputShape {{2,1,{}x{}}} failed", h, w);
    return false;
  }
  input_height_ = h;
  input_width_ = w;
  input_batch_ = 2;

  auto ensure = [](TensorInfo& t, size_t need) -> bool {
    if (t.devicePtr && need == t.size)
      return true;
    if (t.devicePtr)
      cudaFree(t.devicePtr);
    if (t.hostPtr)
      cudaFreeHost(t.hostPtr);
    t.devicePtr = t.hostPtr = nullptr;
    if (cudaMalloc(&t.devicePtr, need) != cudaSuccess ||
        cudaMallocHost(&t.hostPtr, need) != cudaSuccess)
      return false;
    t.size = need;
    return true;
  };
  TensorInfo& input_t = input_tensors_[0];
  if (!ensure(input_t, static_cast<size_t>(2) * h * w * get_element_size(input_t.dtype))) {
    SLOG_ERROR("SuperPoint: failed to size batched input buffer");
    return false;
  }
  for (auto& t : output_tensors_) {
    t.dims = context_->getTensorShape(t.name.c_str());
    if (!ensure(t, get_tensor_size(t.dims, t.dtype))) {
      SLOG_ERROR("SuperPoint: failed to size batched output {}", t.name);
      return false;
    }
  }

  // Pack both images into the input buffer, H2D, set output addresses, one enqueue.
  float* host_in = static_cast<float*>(input_t.hostPtr);
  std::memcpy(host_in, gl.data, static_cast<size_t>(h) * w * sizeof(float));
  std::memcpy(host_in + static_cast<size_t>(h) * w,
              gr.data,
              static_cast<size_t>(h) * w * sizeof(float));
  context_->setTensorAddress(input_t.name.c_str(), input_t.devicePtr);
  if (cudaMemcpyAsync(input_t.devicePtr,
                      input_t.hostPtr,
                      input_t.size,
                      cudaMemcpyHostToDevice,
                      stream_) != cudaSuccess)
    return false;
  for (auto& t : output_tensors_)
    context_->setTensorAddress(t.name.c_str(), t.devicePtr);
  if (!context_->enqueueV3(stream_)) {
    SLOG_ERROR("SuperPoint: stereo enqueueV3 failed");
    return false;
  }

  TensorInfo& scores_t = output_tensors_[0];
  TensorInfo& desc_t = output_tensors_[1];
  // D2H the whole batched scores grid; the descriptor grid stays on the device.
  if (cudaMemcpyAsync(scores_t.hostPtr,
                      scores_t.devicePtr,
                      scores_t.size,
                      cudaMemcpyDeviceToHost,
                      stream_) != cudaSuccess)
    return false;
  cudaStreamSynchronize(stream_);

  const bool scores_half = scores_t.dtype == nvinfer1::DataType::kHALF;
  const int sh = scores_t.dims.d[scores_t.dims.nbDims - 2];
  const int sw = scores_t.dims.d[scores_t.dims.nbDims - 1];
  const int dc = desc_t.dims.d[desc_t.dims.nbDims - 3];
  const int dh = desc_t.dims.d[desc_t.dims.nbDims - 2];
  const int dw = desc_t.dims.d[desc_t.dims.nbDims - 1];
  if (desc_t.dtype != nvinfer1::DataType::kHALF || dc != descriptor_dim) {
    SLOG_ERROR("SuperPoint: stereo gather needs FP16 grid with {} channels", descriptor_dim);
    return false;
  }

  // Per-batch-slice select and gather (slice b at score offset b*sh*sw, grid offset b*dc*dh*dw).
  const size_t score_plane = static_cast<size_t>(sh) * sw * get_element_size(scores_t.dtype);
  const size_t grid_plane = static_cast<size_t>(dc) * dh * dw * get_element_size(desc_t.dtype);
  const char* scores_base = static_cast<const char*>(scores_t.hostPtr);
  const char* grid_base = static_cast<const char*>(desc_t.devicePtr);
  const bool ok_l = select_and_gather(scores_base,
                                      scores_half,
                                      sh,
                                      sw,
                                      grid_base,
                                      dc,
                                      dh,
                                      dw,
                                      kp_left,
                                      desc_left);
  const bool ok_r = select_and_gather(scores_base + score_plane,
                                      scores_half,
                                      sh,
                                      sw,
                                      grid_base + grid_plane,
                                      dc,
                                      dh,
                                      dw,
                                      kp_right,
                                      desc_right);
  return ok_l && ok_r;
}

// superslam::IFeatureExtractor. Scores live in keypoint.response.
superslam::Features SuperPoint::extract(const cv::Mat& image) {
  superslam::Features f;
  infer_device(image, f.keypoints, f.descriptors);
  return f;
}

// Batched stereo: one {2,1,H,W} infer for L and R.
std::pair<superslam::Features, superslam::Features>
SuperPoint::extract_stereo(const cv::Mat& left, const cv::Mat& right) {
  SUPERSLAM_PROFILE_SCOPE("sp_extract_stereo");
  superslam::Features l, r;
  infer_device_stereo(left, right, l.keypoints, l.descriptors, r.keypoints, r.descriptors);
  return {std::move(l), std::move(r)};
}
