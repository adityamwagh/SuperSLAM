#include "SuperGlueTRT.h"

// Simple logger for TensorRT 10.11.0
class SimpleLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TensorRT] " << msg << "\n";
    }
  }
};

static SimpleLogger gLogger;

SuperGlueTRT::SuperGlueTRT(const SuperGlueConfig& config) : config_(config) {
  cudaStreamCreate(&stream_);
}

SuperGlueTRT::~SuperGlueTRT() {
  freeBuffers();
  if (context_) delete context_;
  if (engine_) delete engine_;
  if (runtime_) delete runtime_;
  cudaStreamDestroy(stream_);
}

bool SuperGlueTRT::initialize() {
  std::cout << "SuperGlueTRT: Initializing..." << "\n";

  if (!loadEngine()) {
    std::cerr << "SuperGlueTRT: Failed to load engine" << "\n";
    return false;
  }

  if (!allocateBuffers()) {
    std::cerr << "SuperGlueTRT: Failed to allocate buffers" << "\n";
    return false;
  }

  std::cout << "SuperGlueTRT: Initialization successful" << "\n";
  return true;
}

bool SuperGlueTRT::loadEngine() {
  std::cout << "SuperGlueTRT: Loading engine from " << config_.engine_file
            << "\n";

  std::ifstream file(config_.engine_file, std::ios::binary);
  if (!file.good()) {
    std::cerr << "SuperGlueTRT: Cannot read engine file " << config_.engine_file
              << "\n";
    return false;
  }

  file.seekg(0, std::ifstream::end);
  size_t size = file.tellg();
  file.seekg(0, std::ifstream::beg);

  std::vector<char> engineData(size);
  file.read(engineData.data(), size);
  file.close();

  runtime_ = nvinfer1::createInferRuntime(gLogger);
  if (!runtime_) {
    std::cerr << "SuperGlueTRT: Failed to create runtime" << "\n";
    return false;
  }

  engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
  if (!engine_) {
    std::cerr << "SuperGlueTRT: Failed to deserialize engine" << "\n";
    return false;
  }

  context_ = engine_->createExecutionContext();
  if (!context_) {
    std::cerr << "SuperGlueTRT: Failed to create execution context"
              << "\n";
    return false;
  }

  std::cout << "SuperGlueTRT: Engine loaded successfully" << "\n";
  return true;
}

bool SuperGlueTRT::allocateBuffers() {
  std::cout << "SuperGlueTRT: Allocating buffers..." << "\n";

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
      std::cout << "SuperGlueTRT: Deferring allocation for dynamic tensor "
                << tensor.name << "\n";
      tensor.devicePtr = nullptr;
      tensor.hostPtr = nullptr;
    } else {
      // Allocate device memory
      if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperGlueTRT: Failed to allocate device memory for "
                  << tensor.name << "\n";
        return false;
      }

      // Allocate host memory
      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperGlueTRT: Failed to allocate host memory for "
                  << tensor.name << "\n";
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

  std::cout << "SuperGlueTRT: Buffers allocated successfully" << "\n";
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
        std::cerr << "SuperGlueTRT: Failed to set input shape for "
                  << tensor.name << "\n";
        return false;
      }

      // Calculate size with actual dimensions
      tensor.size = getTensorSize(actualDims, tensor.dtype);

      // Allocate memory
      if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperGlueTRT: Failed to allocate device memory for "
                     "dynamic tensor "
                  << tensor.name << "\n";
        return false;
      }

      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperGlueTRT: Failed to allocate host memory for dynamic "
                     "tensor "
                  << tensor.name << "\n";
        return false;
      }

      std::cout << "SuperGlueTRT: Allocated dynamic tensor " << tensor.name
                << " with size " << tensor.size << " bytes" << "\n";
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
        std::cerr << "SuperGlueTRT: Failed to allocate device memory for "
                     "dynamic output tensor "
                  << tensor.name << "\n";
        return false;
      }

      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperGlueTRT: Failed to allocate host memory for dynamic "
                     "output tensor "
                  << tensor.name << "\n";
        return false;
      }

      std::cout << "SuperGlueTRT: Allocated dynamic output tensor "
                << tensor.name << " with size " << tensor.size << " bytes"
                << "\n";
    }
  }

  return true;
}

bool SuperGlueTRT::match(const std::vector<cv::KeyPoint>& keypoints0,
                         const cv::Mat& descriptors0,
                         const std::vector<cv::KeyPoint>& keypoints1,
                         const cv::Mat& descriptors1, MatchResult& result) {
  if (!context_) {
    std::cerr << "SuperGlueTRT: Context not initialized" << "\n";
    return false;
  }

  // Allocate dynamic tensors with actual shapes
  if (!allocateDynamicTensors(keypoints0, keypoints1)) {
    std::cerr << "SuperGlueTRT: Failed to allocate dynamic tensors"
              << "\n";
    return false;
  }

  // Prepare inputs
  if (!prepareInputs(keypoints0, descriptors0, keypoints1, descriptors1)) {
    std::cerr << "SuperGlueTRT: Failed to prepare inputs" << "\n";
    return false;
  }

  // Set tensor addresses for inputs
  for (auto& tensor : input_tensors_) {
    if (!context_->setTensorAddress(tensor.name.c_str(), tensor.devicePtr)) {
      std::cerr << "SuperGlueTRT: Failed to set input tensor address for "
                << tensor.name << "\n";
      return false;
    }

    // Copy input to device
    if (cudaMemcpyAsync(tensor.devicePtr, tensor.hostPtr, tensor.size,
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
      std::cerr << "SuperGlueTRT: Failed to copy input to device" << "\n";
      return false;
    }
  }

  // Set tensor addresses for outputs
  for (auto& tensor : output_tensors_) {
    if (!context_->setTensorAddress(tensor.name.c_str(), tensor.devicePtr)) {
      std::cerr << "SuperGlueTRT: Failed to set output tensor address for "
                << tensor.name << "\n";
      return false;
    }
  }

  // Execute inference
  if (!context_->enqueueV3(stream_)) {
    std::cerr << "SuperGlueTRT: Failed to execute inference" << "\n";
    return false;
  }

  // Copy outputs from device
  for (auto& tensor : output_tensors_) {
    if (cudaMemcpyAsync(tensor.hostPtr, tensor.devicePtr, tensor.size,
                        cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
      std::cerr << "SuperGlueTRT: Failed to copy output from device"
                << "\n";
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
    std::cerr << "SuperGlueTRT: Expected 6 input tensors" << "\n";
    return false;
  }

  // Expected input order: keypoints_0, scores_0, descriptors_0, keypoints_1,
  // scores_1, descriptors_1

  // Prepare keypoints_0
  float* kp0_ptr = static_cast<float*>(input_tensors_[0].hostPtr);
  normalizeKeypoints(keypoints0, config_.image_width, config_.image_height,
                     kp0_ptr);

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
  normalizeKeypoints(keypoints1, config_.image_width, config_.image_height,
                     kp1_ptr);

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
    std::cerr << "SuperGlueTRT: No output tensors" << "\n";
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

  std::cout << "SuperGlueTRT: Matching scores shape: " << rows << "x" << cols
            << "\n";

  // Validate dimensions
  if (rows <= 0 || cols <= 0) {
    std::cerr << "SuperGlueTRT: Invalid output dimensions: " << rows << "x"
              << cols << "\n";
    return false;
  }

  // Create scores matrix for visualization
  result.scores = cv::Mat(rows, cols, CV_32F, scoresPtr).clone();

  // Extract matches using Hungarian algorithm or simple max approach
  std::vector<bool> matched0(keypoints0.size(), false);
  std::vector<bool> matched1(keypoints1.size(), false);

  // Simple greedy matching - find best match for each keypoint
  for (int i = 0; i < static_cast<int>(keypoints0.size()) && i < rows - 1;
       ++i) {
    float maxScore = -1.0f;
    int bestMatch = -1;

    for (int j = 0; j < static_cast<int>(keypoints1.size()) && j < cols - 1;
         ++j) {
      if (!matched1[j]) {
        float score = scoresPtr[i * cols + j];
        if (score > maxScore && score > 0.2f) {  // Threshold for good matches
          maxScore = score;
          bestMatch = j;
        }
      }
    }

    if (bestMatch >= 0) {
      // Verify mutual best match
      float bestReverseScore = -1.0f;
      int bestReverseMatch = -1;

      for (int k = 0; k < static_cast<int>(keypoints0.size()) && k < rows - 1;
           ++k) {
        if (!matched0[k]) {
          float score = scoresPtr[k * cols + bestMatch];
          if (score > bestReverseScore) {
            bestReverseScore = score;
            bestReverseMatch = k;
          }
        }
      }

      if (bestReverseMatch == i) {  // Mutual best match
        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = bestMatch;
        match.distance = 1.0f - maxScore;  // Convert score to distance
        result.matches.push_back(match);

        matched0[i] = true;
        matched1[bestMatch] = true;
      }
    }
  }

  std::cout << "SuperGlueTRT: Found " << result.matches.size() << " matches"
            << "\n";
  return true;
}

void SuperGlueTRT::normalizeKeypoints(
    const std::vector<cv::KeyPoint>& keypoints, int imageWidth, int imageHeight,
    float* output) {
  for (size_t i = 0; i < keypoints.size(); ++i) {
    // Normalize to [-1, 1]
    output[i * 2] = (2.0f * keypoints[i].pt.x / imageWidth) - 1.0f;
    output[i * 2 + 1] = (2.0f * keypoints[i].pt.y / imageHeight) - 1.0f;
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
  std::cout << "SuperGlueTRT: Tensor " << name << " - Shape: ";
  for (int i = 0; i < dims.nbDims; ++i) {
    std::cout << dims.d[i];
    if (i < dims.nbDims - 1) std::cout << "x";
  }
  std::cout << ", Type: ";
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      std::cout << "FLOAT";
      break;
    case nvinfer1::DataType::kHALF:
      std::cout << "HALF";
      break;
    case nvinfer1::DataType::kINT8:
      std::cout << "INT8";
      break;
    case nvinfer1::DataType::kINT32:
      std::cout << "INT32";
      break;
    case nvinfer1::DataType::kBOOL:
      std::cout << "BOOL";
      break;
    default:
      std::cout << "UNKNOWN";
      break;
  }
  std::cout << "\n";
}