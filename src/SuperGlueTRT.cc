#include "SuperGlueTRT.h"

#include <cfloat>

#include "Logging.h"

// Forward declarations for SuperGLUE decode functions
void decode(float* scores, int h, int w, std::vector<int>& indices0,
            std::vector<int>& indices1, std::vector<double>& mscores0,
            std::vector<double>& mscores1);

// Simple logger for TensorRT 10.11.0
class SimpleLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      SLOG_WARN("[TensorRT] {}", msg);
    }
  }
};

static SimpleLogger gLogger;

SuperGlueTRT::SuperGlueTRT(const std::string& engine_file, int image_width,
                           int image_height)
    : engine_file_(engine_file),
      image_width_(image_width),
      image_height_(image_height) {
  cudaStreamCreate(&stream_);
}

SuperGlueTRT::~SuperGlueTRT() {
  freeBuffers();
  // Smart pointers automatically clean up TensorRT objects
  cudaStreamDestroy(stream_);
}

bool SuperGlueTRT::initialize() {
  SLOG_INFO("SuperGlueTRT: Initializing...");

  if (!loadEngine()) {
    SLOG_ERROR("SuperGlueTRT: Failed to load engine");
    return false;
  }

  if (!allocateBuffers()) {
    SLOG_ERROR("SuperGlueTRT: Failed to allocate buffers");
    return false;
  }

  SLOG_INFO("SuperGlueTRT: Initialization successful");
  return true;
}

bool SuperGlueTRT::loadEngine() {
  SLOG_INFO("SuperGlueTRT: Loading engine from {}", engine_file_);

  std::ifstream file(engine_file_, std::ios::binary);
  if (!file.good()) {
    SLOG_ERROR("SuperGlueTRT: Cannot read engine file {}", engine_file_);
    return false;
  }

  file.seekg(0, std::ifstream::end);
  size_t size = file.tellg();
  file.seekg(0, std::ifstream::beg);

  std::vector<char> engineData(size);
  file.read(engineData.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  if (!runtime_) {
    SLOG_ERROR("SuperGlueTRT: Failed to create runtime");
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
  if (!engine_) {
    SLOG_ERROR("SuperGlueTRT: Failed to deserialize engine");
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    SLOG_ERROR("SuperGlueTRT: Failed to create execution context");
    return false;
  }

  SLOG_INFO("SuperGlueTRT: Engine loaded successfully");
  return true;
}

bool SuperGlueTRT::allocateBuffers() {
  SLOG_INFO("SuperGlueTRT: Allocating buffers...");

  int numIOTensors = engine_->getNbIOTensors();

  for (int i = 0; i < numIOTensors; ++i) {
    TensorInfo tensor;
    tensor.name = engine_->getIOTensorName(i);
    tensor.dims = engine_->getTensorShape(tensor.name.c_str());
    tensor.dtype = engine_->getTensorDataType(tensor.name.c_str());
    tensor.size = getTensorSize(tensor.dims, tensor.dtype);

    printTensorInfo(tensor.name, tensor.dims, tensor.dtype);

    if (tensor.size == 0) {
      // Dynamic tensor - defer allocation until actual shape is known
      SLOG_DEBUG("SuperGlueTRT: Deferring allocation for dynamic tensor {}",
                 tensor.name);
      tensor.devicePtr = nullptr;
      tensor.hostPtr = nullptr;
    } else {
      // Allocate device memory
      if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
        SLOG_ERROR("SuperGlueTRT: Failed to allocate device memory for {}",
                   tensor.name);
        return false;
      }

      // Allocate host memory
      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        SLOG_ERROR("SuperGlueTRT: Failed to allocate host memory for {}",
                   tensor.name);
        return false;
      }
    }

    if (engine_->getTensorIOMode(tensor.name.c_str()) ==
        nvinfer1::TensorIOMode::kINPUT) {
      input_tensors_.push_back(tensor);
    } else {
      output_tensors_.push_back(tensor);
    }
  }

  SLOG_INFO("SuperGlueTRT: Buffers allocated successfully");
  return true;
}

void SuperGlueTRT::freeBuffers() {
  for (auto& tensor : input_tensors_) {
    if (tensor.devicePtr) cudaFree(tensor.devicePtr);
    if (tensor.hostPtr) cudaFreeHost(tensor.hostPtr);
  }
  for (auto& tensor : output_tensors_) {
    if (tensor.devicePtr) cudaFree(tensor.devicePtr);
    if (tensor.hostPtr) cudaFreeHost(tensor.hostPtr);
  }
  input_tensors_.clear();
  output_tensors_.clear();
}

bool SuperGlueTRT::allocateDynamicTensors(
    const std::vector<cv::KeyPoint>& keypoints0,
    const std::vector<cv::KeyPoint>& keypoints1) {
  // Define actual shapes for dynamic tensors
  int n0 = static_cast<int>(keypoints0.size());
  int n1 = static_cast<int>(keypoints1.size());

  for (auto& tensor : input_tensors_) {
    if (tensor.size == 0) {  // This is a dynamic tensor
      nvinfer1::Dims actualDims = tensor.dims;

      if (tensor.name == "keypoints_0") {
        actualDims.d[1] = n0;  // Set actual number of keypoints
        actualDims.d[2] = 2;   // xy coordinates
      } else if (tensor.name == "scores_0") {
        actualDims.d[1] = n0;  // Set actual number of keypoints
      } else if (tensor.name == "descriptors_0") {
        actualDims.d[1] = 256;  // SuperPoint descriptor dimension
        actualDims.d[2] = n0;   // Set actual number of keypoints
      } else if (tensor.name == "keypoints_1") {
        actualDims.d[1] = n1;  // Set actual number of keypoints
        actualDims.d[2] = 2;   // xy coordinates
      } else if (tensor.name == "scores_1") {
        actualDims.d[1] = n1;  // Set actual number of keypoints
      } else if (tensor.name == "descriptors_1") {
        actualDims.d[1] = 256;  // SuperPoint descriptor dimension
        actualDims.d[2] = n1;   // Set actual number of keypoints
      }

      // Set the input shape in the execution context
      if (!context_->setInputShape(tensor.name.c_str(), actualDims)) {
        SLOG_ERROR("SuperGlueTRT: Failed to set input shape for {}",
                   tensor.name);
        return false;
      }

      // Calculate size with actual dimensions
      tensor.size = getTensorSize(actualDims, tensor.dtype);

      // Allocate memory
      if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
        SLOG_ERROR(
            "SuperGlueTRT: Failed to allocate device memory for dynamic tensor "
            "{}",
            tensor.name);
        return false;
      }

      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        SLOG_ERROR(
            "SuperGlueTRT: Failed to allocate host memory for dynamic tensor "
            "{}",
            tensor.name);
        return false;
      }

      SLOG_DEBUG("SuperGlueTRT: Allocated dynamic tensor {} with size {} bytes",
                 tensor.name, tensor.size);
    }
  }

  // Handle output tensors - their shapes are determined after setting input
  // shapes
  for (auto& tensor : output_tensors_) {
    if (tensor.size == 0) {  // This is a dynamic tensor
      // Get the actual output shape from the context
      nvinfer1::Dims outputDims = context_->getTensorShape(tensor.name.c_str());

      // Calculate size with actual dimensions
      tensor.size = getTensorSize(outputDims, tensor.dtype);

      // Allocate memory
      if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
        SLOG_ERROR(
            "SuperGlueTRT: Failed to allocate device memory for dynamic output "
            "tensor {}",
            tensor.name);
        return false;
      }

      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        SLOG_ERROR(
            "SuperGlueTRT: Failed to allocate host memory for dynamic output "
            "tensor {}",
            tensor.name);
        return false;
      }

      SLOG_DEBUG(
          "SuperGlueTRT: Allocated dynamic output tensor {} with size {} bytes",
          tensor.name, tensor.size);
    }
  }

  return true;
}

bool SuperGlueTRT::match(const std::vector<cv::KeyPoint>& keypoints0,
                         const cv::Mat& descriptors0,
                         const std::vector<cv::KeyPoint>& keypoints1,
                         const cv::Mat& descriptors1, MatchResult& result) {
  if (!context_) {
    SLOG_ERROR("SuperGlueTRT: Context not initialized");
    return false;
  }

  SLOG_DEBUG(
      "SuperGlueTRT: Matching {} keypoints from image0 with {} keypoints from "
      "image1",
      keypoints0.size(), keypoints1.size());

  // Allocate dynamic tensors with actual shapes
  if (!allocateDynamicTensors(keypoints0, keypoints1)) {
    SLOG_ERROR("SuperGlueTRT: Failed to allocate dynamic tensors");
    return false;
  }

  // Prepare inputs
  if (!prepareInputs(keypoints0, descriptors0, keypoints1, descriptors1)) {
    SLOG_ERROR("SuperGlueTRT: Failed to prepare inputs");
    return false;
  }

  // Set tensor addresses for inputs
  for (auto& tensor : input_tensors_) {
    if (!context_->setTensorAddress(tensor.name.c_str(), tensor.devicePtr)) {
      SLOG_ERROR("SuperGlueTRT: Failed to set input tensor address for {}",
                 tensor.name);
      return false;
    }

    // Copy input to device
    if (cudaMemcpyAsync(tensor.devicePtr, tensor.hostPtr, tensor.size,
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
      SLOG_ERROR("SuperGlueTRT: Failed to copy input to device");
      return false;
    }
  }

  // Set tensor addresses for outputs
  for (auto& tensor : output_tensors_) {
    if (!context_->setTensorAddress(tensor.name.c_str(), tensor.devicePtr)) {
      SLOG_ERROR("SuperGlueTRT: Failed to set output tensor address for {}",
                 tensor.name);
      return false;
    }
  }

  // Execute inference
  if (!context_->enqueueV3(stream_)) {
    SLOG_ERROR("SuperGlueTRT: Failed to execute inference");
    return false;
  }

  // Copy outputs from device
  for (auto& tensor : output_tensors_) {
    if (cudaMemcpyAsync(tensor.hostPtr, tensor.devicePtr, tensor.size,
                        cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
      SLOG_ERROR("SuperGlueTRT: Failed to copy output from device");
      return false;
    }
  }

  // Synchronize stream
  cudaStreamSynchronize(stream_);

  // Postprocess outputs
  return postprocessOutputs(keypoints0, keypoints1, result);
}

bool SuperGlueTRT::prepareInputs(const std::vector<cv::KeyPoint>& keypoints0,
                                 const cv::Mat& descriptors0,
                                 const std::vector<cv::KeyPoint>& keypoints1,
                                 const cv::Mat& descriptors1) {
  if (input_tensors_.size() < 6) {
    SLOG_ERROR("SuperGlueTRT: Expected 6 input tensors");
    return false;
  }

  // Expected input order: keypoints_0, scores_0, descriptors_0, keypoints_1,
  // scores_1, descriptors_1

  // Prepare keypoints_0
  float* kp0_ptr = static_cast<float*>(input_tensors_[0].hostPtr);
  normalizeKeypoints(keypoints0, image_width_, image_height_, kp0_ptr);

  // Prepare scores_0
  float* scores0_ptr = static_cast<float*>(input_tensors_[1].hostPtr);
  for (size_t i = 0; i < keypoints0.size(); ++i) {
    scores0_ptr[i] = keypoints0[i].response;
  }

  // Prepare descriptors_0
  float* desc0_ptr = static_cast<float*>(input_tensors_[2].hostPtr);
  copyDescriptors(descriptors0, desc0_ptr);

  // Prepare keypoints_1
  float* kp1_ptr = static_cast<float*>(input_tensors_[3].hostPtr);
  normalizeKeypoints(keypoints1, image_width_, image_height_, kp1_ptr);

  // Prepare scores_1
  float* scores1_ptr = static_cast<float*>(input_tensors_[4].hostPtr);
  for (size_t i = 0; i < keypoints1.size(); ++i) {
    scores1_ptr[i] = keypoints1[i].response;
  }

  // Prepare descriptors_1
  float* desc1_ptr = static_cast<float*>(input_tensors_[5].hostPtr);
  copyDescriptors(descriptors1, desc1_ptr);

  return true;
}

bool SuperGlueTRT::postprocessOutputs(
    const std::vector<cv::KeyPoint>& keypoints0,
    const std::vector<cv::KeyPoint>& keypoints1, MatchResult& result) {
  if (output_tensors_.empty()) {
    SLOG_ERROR("SuperGlueTRT: No output tensors");
    return false;
  }

  result.matches.clear();

  // Get matching scores output
  float* scoresPtr = static_cast<float*>(output_tensors_[0].hostPtr);

  // Get actual runtime dimensions from the execution context
  nvinfer1::Dims actualDims =
      context_->getTensorShape(output_tensors_[0].name.c_str());

  // SuperGlue output is typically [N0+1, N1+1] where N0, N1 are number of
  // keypoints
  int rows = actualDims.d[actualDims.nbDims - 2];
  int cols = actualDims.d[actualDims.nbDims - 1];

  SLOG_DEBUG("SuperGlueTRT: Matching scores shape: {}x{}", rows, cols);

  // Validate dimensions
  if (rows <= 0 || cols <= 0) {
    SLOG_ERROR("SuperGlueTRT: Invalid output dimensions: {}x{}", rows, cols);
    return false;
  }

  // Use reference SuperGLUE decode logic instead of softmax
  std::vector<int> indices0, indices1;
  std::vector<double> mscores0, mscores1;

  decode(scoresPtr, rows, cols, indices0, indices1, mscores0, mscores1);

  // Convert to cv::DMatch format
  for (size_t i = 0; i < indices0.size(); ++i) {
    if (indices0[i] >= 0 && indices0[i] < static_cast<int>(keypoints1.size())) {
      cv::DMatch match;
      match.queryIdx = static_cast<int>(i);
      match.trainIdx = indices0[i];
      match.distance = 1.0 - (mscores0[i] + mscores1[indices0[i]]) /
                                 2.0;  // Reference formula
      result.matches.push_back(match);
    }
  }

  // Create scores matrix for visualization
  result.scores = cv::Mat(rows, cols, CV_32F, scoresPtr).clone();

  SLOG_INFO("SuperGlueTRT: Found {} matches", result.matches.size());
  return true;
}

void SuperGlueTRT::normalizeKeypoints(
    const std::vector<cv::KeyPoint>& keypoints, int imageWidth, int imageHeight,
    float* output) {
  for (size_t i = 0; i < keypoints.size(); ++i) {
    // Use SuperGLUE reference normalization: (x - w/2) / (max(w,h) * 0.7)
    float maxDim = std::max(imageWidth, imageHeight) * 0.7f;
    output[i * 2] = (keypoints[i].pt.x - imageWidth / 2.0f) / maxDim;
    output[i * 2 + 1] = (keypoints[i].pt.y - imageHeight / 2.0f) / maxDim;
  }
}

void SuperGlueTRT::copyDescriptors(const cv::Mat& descriptors, float* output) {
  if (descriptors.type() == CV_32F) {
    std::memcpy(output, descriptors.data, descriptors.total() * sizeof(float));
  } else {
    // Convert to float if needed
    cv::Mat floatDesc;
    descriptors.convertTo(floatDesc, CV_32F);
    std::memcpy(output, floatDesc.data, floatDesc.total() * sizeof(float));
  }
}

size_t SuperGlueTRT::getElementSize(nvinfer1::DataType dtype) {
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

size_t SuperGlueTRT::getTensorSize(const nvinfer1::Dims& dims,
                                   nvinfer1::DataType dtype) {
  size_t size = getElementSize(dtype);
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] == -1) {
      // Dynamic dimension detected - cannot determine size yet
      return 0;
    }
    size *= dims.d[i];
  }
  return size;
}

void SuperGlueTRT::printTensorInfo(const std::string& name,
                                   const nvinfer1::Dims& dims,
                                   nvinfer1::DataType dtype) {
  std::string shape_str;
  for (int i = 0; i < dims.nbDims; ++i) {
    shape_str += std::to_string(dims.d[i]);
    if (i < dims.nbDims - 1) shape_str += "x";
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

  SLOG_DEBUG("SuperGlueTRT: Tensor {} - Shape: {}, Type: {}", name, shape_str,
             type_str);
}

// Helper functions from reference SuperGLUE implementation
void where_negative_one(const int* flag_data, const int* data, int size,
                        std::vector<int>& indices) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      indices.push_back(data[i]);
    } else {
      indices.push_back(-1);
    }
  }
}

void max_matrix(const float* data, int* indices, float* values, int h, int w,
                int dim) {
  if (dim == 2) {
    for (int i = 0; i < h - 1; ++i) {
      float max_value = -FLT_MAX;
      int max_indices = 0;
      for (int j = 0; j < w - 1; ++j) {
        if (max_value < data[i * w + j]) {
          max_value = data[i * w + j];
          max_indices = j;
        }
      }
      values[i] = max_value;
      indices[i] = max_indices;
    }
  } else if (dim == 1) {
    for (int i = 0; i < w - 1; ++i) {
      float max_value = -FLT_MAX;
      int max_indices = 0;
      for (int j = 0; j < h - 1; ++j) {
        if (max_value < data[j * w + i]) {
          max_value = data[j * w + i];
          max_indices = j;
        }
      }
      values[i] = max_value;
      indices[i] = max_indices;
    }
  }
}

void equal_gather(const int* indices0, const int* indices1, int* mutual,
                  int size) {
  for (int i = 0; i < size; ++i) {
    if (indices0[indices1[i]] == i) {
      mutual[i] = 1;
    } else {
      mutual[i] = 0;
    }
  }
}

void where_exp(const int* flag_data, float* data, std::vector<double>& mscores0,
               int size) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      mscores0.push_back(std::exp(data[i]));
    } else {
      mscores0.push_back(0);
    }
  }
}

void where_gather(const int* flag_data, int* indices,
                  std::vector<double>& mscores0, std::vector<double>& mscores1,
                  int size) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      mscores1.push_back(mscores0[indices[i]]);
    } else {
      mscores1.push_back(0);
    }
  }
}

void and_threshold(const int* mutual0, int* valid0,
                   const std::vector<double>& mscores0, double threshold) {
  for (int i = 0; i < mscores0.size(); ++i) {
    if (mutual0[i] == 1 && mscores0[i] > threshold) {
      valid0[i] = 1;
    } else {
      valid0[i] = 0;
    }
  }
}

void and_gather(const int* mutual1, const int* valid0, const int* indices1,
                int* valid1, int size) {
  for (int i = 0; i < size; ++i) {
    if (mutual1[i] == 1 && valid0[indices1[i]] == 1) {
      valid1[i] = 1;
    } else {
      valid1[i] = 0;
    }
  }
}

void decode(float* scores, int h, int w, std::vector<int>& indices0,
            std::vector<int>& indices1, std::vector<double>& mscores0,
            std::vector<double>& mscores1) {
  std::vector<int> max_indices0(h - 1);
  std::vector<int> max_indices1(w - 1);
  std::vector<float> max_values0(h - 1);
  std::vector<float> max_values1(w - 1);
  max_matrix(scores, max_indices0.data(), max_values0.data(), h, w, 2);
  max_matrix(scores, max_indices1.data(), max_values1.data(), h, w, 1);

  // Debug: Log max values to understand score range
  SLOG_DEBUG("SuperGLUE decode: Max score values (first 5): {}, {}, {}, {}, {}",
             max_values0[0], max_values0[1], max_values0[2], max_values0[3],
             max_values0[4]);

  std::vector<int> mutual0(h - 1);
  std::vector<int> mutual1(w - 1);
  equal_gather(max_indices1.data(), max_indices0.data(), mutual0.data(), h - 1);
  equal_gather(max_indices0.data(), max_indices1.data(), mutual1.data(), w - 1);
  where_exp(mutual0.data(), max_values0.data(), mscores0, h - 1);
  where_gather(mutual1.data(), max_indices1.data(), mscores0, mscores1, w - 1);
  std::vector<int> valid0(h - 1);
  std::vector<int> valid1(w - 1);
  and_threshold(mutual0.data(), valid0.data(), mscores0,
                0.0);  // Set to 0 to allow all matches for debugging

  // Debug: Count valid matches before final step
  int validCount = 0;
  for (int i = 0; i < h - 1; ++i) {
    if (valid0[i] == 1) validCount++;
  }
  SLOG_DEBUG("SuperGLUE decode: Valid matches after threshold: {}", validCount);

  and_gather(mutual1.data(), valid0.data(), max_indices1.data(), valid1.data(),
             w - 1);
  where_negative_one(valid0.data(), max_indices0.data(), h - 1, indices0);
  where_negative_one(valid1.data(), max_indices1.data(), w - 1, indices1);
  // Vectors automatically clean up their memory
}