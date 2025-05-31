#include "SuperPointTRT.h"

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

SuperPointTRT::SuperPointTRT(const SuperPointConfig& config) : config_(config) {
  cudaStreamCreate(&stream_);
}

SuperPointTRT::~SuperPointTRT() {
  freeBuffers();
  if (context_) delete context_;
  if (engine_) delete engine_;
  if (runtime_) delete runtime_;
  cudaStreamDestroy(stream_);
}

bool SuperPointTRT::initialize() {
  std::cout << "SuperPointTRT: Initializing..." << "\n";

  if (!loadEngine()) {
    std::cerr << "SuperPointTRT: Failed to load engine" << "\n";
    return false;
  }

  if (!allocateBuffers()) {
    std::cerr << "SuperPointTRT: Failed to allocate buffers" << "\n";
    return false;
  }

  std::cout << "SuperPointTRT: Initialization successful" << "\n";
  return true;
}

bool SuperPointTRT::loadEngine() {
  std::cout << "SuperPointTRT: Loading engine from " << config_.engine_file
            << "\n";

  std::ifstream file(config_.engine_file, std::ios::binary);
  if (!file.good()) {
    std::cerr << "SuperPointTRT: Cannot read engine file "
              << config_.engine_file << "\n";
    std::cerr << "SuperPointTRT: Please rebuild the engine for TensorRT "
                 "10.11.0 using:"
              << "\n";
    std::cerr << "trtexec --onnx=" << config_.onnx_file
              << " --saveEngine=" << config_.engine_file << " --fp16"
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
    std::cerr << "SuperPointTRT: Failed to create runtime" << "\n";
    return false;
  }

  engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
  if (!engine_) {
    std::cerr << "SuperPointTRT: Failed to deserialize engine" << "\n";
    std::cerr
        << "SuperPointTRT: This is likely due to TensorRT version mismatch."
        << "\n";
    std::cerr << "SuperPointTRT: The engine was built with a different "
                 "TensorRT version."
              << "\n";
    std::cerr << "SuperPointTRT: Please rebuild the engine for TensorRT "
                 "10.11.0 using:"
              << "\n";
    std::cerr << "trtexec --onnx=" << config_.onnx_file
              << " --saveEngine=" << config_.engine_file << " --fp16"
              << "\n";
    return false;
  }

  context_ = engine_->createExecutionContext();
  if (!context_) {
    std::cerr << "SuperPointTRT: Failed to create execution context"
              << "\n";
    return false;
  }

  std::cout << "SuperPointTRT: Engine loaded successfully" << "\n";
  return true;
}

bool SuperPointTRT::allocateBuffers() {
  std::cout << "SuperPointTRT: Allocating buffers..." << "\n";

  int numIOTensors = engine_->getNbIOTensors();

  for (int i = 0; i < numIOTensors; ++i) {
    TensorInfo tensor;
    tensor.name = engine_->getIOTensorName(i);
    tensor.dims = engine_->getTensorShape(tensor.name.c_str());
    tensor.dtype = engine_->getTensorDataType(tensor.name.c_str());

    printTensorInfo(tensor.name, tensor.dims, tensor.dtype);

    if (engine_->getTensorIOMode(tensor.name.c_str()) ==
        nvinfer1::TensorIOMode::kINPUT) {
      if (tensor.dims.nbDims >= 2) {
        input_height_ = tensor.dims.d[tensor.dims.nbDims - 2];
        input_width_ = tensor.dims.d[tensor.dims.nbDims - 1];
      }

      // For dynamic shapes, allocate maximum possible size or defer allocation
      if (input_height_ <= 0 || input_width_ <= 0) {
        // Allocate a reasonable maximum size for dynamic shapes
        const size_t max_size = 1 * 1 * 1024 * 1024 *
                                getElementSize(tensor.dtype);  // 1M pixels max
        tensor.size = max_size;
        input_height_ = -1;  // Mark as dynamic
        input_width_ = -1;
      } else {
        tensor.size = getTensorSize(tensor.dims, tensor.dtype);
      }

      // Allocate device memory
      if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperPointTRT: Failed to allocate device memory for "
                  << tensor.name << "\n";
        return false;
      }

      // Allocate host memory
      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperPointTRT: Failed to allocate host memory for "
                  << tensor.name << "\n";
        return false;
      }

      input_tensors_.push_back(tensor);
    } else {
      // For output tensors with dynamic shapes, also allocate max size
      tensor.size = getTensorSize(tensor.dims, tensor.dtype);
      if (tensor.size == 0) {
        const size_t max_size =
            10 * 1024 * 1024 *
            getElementSize(tensor.dtype);  // 10MB max for outputs
        tensor.size = max_size;
      }

      // Allocate device memory
      if (cudaMalloc(&tensor.devicePtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperPointTRT: Failed to allocate device memory for "
                  << tensor.name << "\n";
        return false;
      }

      // Allocate host memory
      if (cudaMallocHost(&tensor.hostPtr, tensor.size) != cudaSuccess) {
        std::cerr << "SuperPointTRT: Failed to allocate host memory for "
                  << tensor.name << "\n";
        return false;
      }

      output_tensors_.push_back(tensor);
    }
  }

  std::cout << "SuperPointTRT: Input dimensions: " << input_height_ << "x"
            << input_width_ << "\n";
  std::cout << "SuperPointTRT: Buffers allocated successfully" << "\n";
  return true;
}

void SuperPointTRT::freeBuffers() {
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

bool SuperPointTRT::infer(const cv::Mat& image,
                          std::vector<cv::KeyPoint>& keypoints,
                          cv::Mat& descriptors) {
  if (!context_) {
    std::cerr << "SuperPointTRT: Context not initialized" << "\n";
    return false;
  }

  // Preprocess image (this sets dynamic shapes if needed)
  if (!preprocessImage(image)) {
    std::cerr << "SuperPointTRT: Failed to preprocess image" << "\n";
    return false;
  }

  // For dynamic shapes, we need to get the actual output sizes after setting
  // input shape
  if (input_height_ > 0 && input_width_ > 0) {
    // Update output tensor sizes based on current input
    for (auto& tensor : output_tensors_) {
      nvinfer1::Dims outputDims = context_->getTensorShape(tensor.name.c_str());
      tensor.dims = outputDims;
    }
  }

  // Set tensor addresses for inputs
  for (auto& tensor : input_tensors_) {
    if (!context_->setTensorAddress(tensor.name.c_str(), tensor.devicePtr)) {
      std::cerr << "SuperPointTRT: Failed to set input tensor address for "
                << tensor.name << "\n";
      return false;
    }

    // Calculate actual size to copy (for dynamic shapes)
    size_t copySize =
        input_height_ * input_width_ * getElementSize(tensor.dtype);

    // Copy input to device
    if (cudaMemcpyAsync(tensor.devicePtr, tensor.hostPtr, copySize,
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
      std::cerr << "SuperPointTRT: Failed to copy input to device" << "\n";
      return false;
    }
  }

  // Set tensor addresses for outputs
  for (auto& tensor : output_tensors_) {
    if (!context_->setTensorAddress(tensor.name.c_str(), tensor.devicePtr)) {
      std::cerr << "SuperPointTRT: Failed to set output tensor address for "
                << tensor.name << "\n";
      return false;
    }
  }

  // Execute inference
  if (!context_->enqueueV3(stream_)) {
    std::cerr << "SuperPointTRT: Failed to execute inference" << "\n";
    return false;
  }

  // Copy output from device
  for (auto& tensor : output_tensors_) {
    // For dynamic shapes, calculate actual output size
    size_t outputSize = getTensorSize(tensor.dims, tensor.dtype);
    if (outputSize > tensor.size) {
      std::cerr << "SuperPointTRT: Output size (" << outputSize
                << ") exceeds allocated buffer (" << tensor.size << ")"
                << "\n";
      outputSize = tensor.size;  // Use allocated size to prevent overflow
    }

    if (cudaMemcpyAsync(tensor.hostPtr, tensor.devicePtr, outputSize,
                        cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
      std::cerr << "SuperPointTRT: Failed to copy output from device"
                << "\n";
      return false;
    }
  }

  // Synchronize stream
  cudaStreamSynchronize(stream_);

  // Postprocess outputs
  return postprocessOutputs(keypoints, descriptors);
}

bool SuperPointTRT::preprocessImage(const cv::Mat& image) {
  if (input_tensors_.empty()) {
    std::cerr << "SuperPointTRT: No input tensors allocated" << "\n";
    return false;
  }

  cv::Mat processedImage;

  // Convert to grayscale if needed
  if (image.channels() == 3) {
    cv::cvtColor(image, processedImage, cv::COLOR_BGR2GRAY);
  } else {
    processedImage = image.clone();
  }

  // For dynamic shapes, always use current image dimensions
  int target_height = processedImage.rows;
  int target_width = processedImage.cols;

  // Always update input tensor shape for dynamic inputs (pyramid levels have
  // different sizes)
  nvinfer1::Dims inputDims = input_tensors_[0].dims;
  if (inputDims.d[inputDims.nbDims - 2] <= 0 ||
      inputDims.d[inputDims.nbDims - 1] <= 0 ||
      inputDims.d[inputDims.nbDims - 2] != target_height ||
      inputDims.d[inputDims.nbDims - 1] != target_width) {
    inputDims.d[inputDims.nbDims - 2] = target_height;
    inputDims.d[inputDims.nbDims - 1] = target_width;

    if (!context_->setInputShape(input_tensors_[0].name.c_str(), inputDims)) {
      std::cerr << "SuperPointTRT: Failed to set input shape" << "\n";
      return false;
    }

    input_height_ = target_height;
    input_width_ = target_width;
  }

  // Resize to expected input size
  if (processedImage.rows != target_height ||
      processedImage.cols != target_width) {
    cv::resize(processedImage, processedImage,
               cv::Size(target_width, target_height));
  }

  // Convert to float and normalize to [0, 1]
  processedImage.convertTo(processedImage, CV_32F, 1.0 / 255.0);

  // Copy to input tensor (assuming NCHW format: 1x1xHxW)
  float* inputPtr = static_cast<float*>(input_tensors_[0].hostPtr);
  std::memcpy(inputPtr, processedImage.data,
              target_height * target_width * sizeof(float));

  return true;
}

bool SuperPointTRT::postprocessOutputs(std::vector<cv::KeyPoint>& keypoints,
                                       cv::Mat& descriptors) {
  if (output_tensors_.size() < 2) {
    std::cerr << "SuperPointTRT: Expected at least 2 output tensors (scores, "
                 "descriptors)"
              << "\n";
    return false;
  }

  keypoints.clear();

  // Get scores output (first output tensor)
  float* scoresPtr = static_cast<float*>(output_tensors_[0].hostPtr);
  auto& scoresDims = output_tensors_[0].dims;

  // Get descriptors output (second output tensor)
  float* descriptorsPtr = static_cast<float*>(output_tensors_[1].hostPtr);
  auto& descDims = output_tensors_[1].dims;

  // Calculate dimensions
  int scoreHeight = scoresDims.d[scoresDims.nbDims - 2];
  int scoreWidth = scoresDims.d[scoresDims.nbDims - 1];
  int descChannels = descDims.d[descDims.nbDims - 3];
  int descHeight = descDims.d[descDims.nbDims - 2];
  int descWidth = descDims.d[descDims.nbDims - 1];

  std::cout << "SuperPointTRT: Scores shape: " << scoreHeight << "x"
            << scoreWidth << "\n";
  std::cout << "SuperPointTRT: Descriptors shape: " << descChannels << "x"
            << descHeight << "x" << descWidth << "\n";

  // Extract keypoints above threshold
  std::vector<std::pair<float, std::pair<int, int>>> candidates;

  for (int h = config_.remove_borders; h < scoreHeight - config_.remove_borders;
       ++h) {
    for (int w = config_.remove_borders;
         w < scoreWidth - config_.remove_borders; ++w) {
      float score = scoresPtr[h * scoreWidth + w];
      if (score > config_.keypoint_threshold) {
        candidates.emplace_back(score, std::make_pair(h, w));
      }
    }
  }

  // Sort by score and take top K
  std::sort(candidates.begin(), candidates.end(), std::greater<>());

  int numKeypoints =
      std::min(static_cast<int>(candidates.size()), config_.max_keypoints);
  keypoints.reserve(numKeypoints);

  // Scale factor from score map to original image
  float scaleX = static_cast<float>(input_width_) / scoreWidth;
  float scaleY = static_cast<float>(input_height_) / scoreHeight;

  std::vector<cv::Point2f> keypointPositions;
  keypointPositions.reserve(numKeypoints);

  for (int i = 0; i < numKeypoints; ++i) {
    float score = candidates[i].first;
    int h = candidates[i].second.first;
    int w = candidates[i].second.second;

    // Convert to original image coordinates
    float x = w * scaleX;
    float y = h * scaleY;

    keypoints.emplace_back(x, y, 1.0f, -1, score);
    keypointPositions.emplace_back(x, y);
  }

  // Extract descriptors for keypoints
  if (numKeypoints > 0) {
    descriptors = cv::Mat::zeros(numKeypoints, descChannels, CV_32F);

    for (int i = 0; i < numKeypoints; ++i) {
      int h = candidates[i].second.first;
      int w = candidates[i].second.second;

      // Scale coordinates to descriptor map
      int descH = std::min(h / 8, descHeight - 1);  // Assuming 8x downsampling
      int descW = std::min(w / 8, descWidth - 1);

      // Extract descriptor
      float* descRow = descriptors.ptr<float>(i);
      for (int c = 0; c < descChannels; ++c) {
        int idx = c * descHeight * descWidth + descH * descWidth + descW;
        descRow[c] = descriptorsPtr[idx];
      }
    }

    // Normalize descriptors
    for (int i = 0; i < numKeypoints; ++i) {
      cv::Mat descRow = descriptors.row(i);
      cv::normalize(descRow, descRow, 1.0, 0.0, cv::NORM_L2);
    }
  }

  std::cout << "SuperPointTRT: Extracted " << numKeypoints << " keypoints"
            << "\n";
  return true;
}

size_t SuperPointTRT::getElementSize(nvinfer1::DataType dtype) {
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

size_t SuperPointTRT::getTensorSize(const nvinfer1::Dims& dims,
                                    nvinfer1::DataType dtype) {
  size_t size = getElementSize(dtype);
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      // Dynamic dimension, return 0 to indicate unknown size
      return 0;
    }
    size *= dims.d[i];
  }
  return size;
}

void SuperPointTRT::printTensorInfo(const std::string& name,
                                    const nvinfer1::Dims& dims,
                                    nvinfer1::DataType dtype) {
  std::cout << "SuperPointTRT: Tensor " << name << " - Shape: ";
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