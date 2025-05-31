#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

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

template <typename T>
struct TensorRTDestroy {
  void operator()(T* obj) const {
    if (obj) delete obj;
  }
};

template <typename T>
using TensorRTUniquePtr = std::unique_ptr<T, TensorRTDestroy<T>>;

bool buildEngineFromOnnx(const std::string& onnxFile,
                         const std::string& engineFile) {
  std::cout << "Building engine from " << onnxFile << " to " << engineFile
            << "\n";

  // Create builder
  TensorRTUniquePtr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(gLogger));
  if (!builder) {
    std::cerr << "Failed to create TensorRT builder" << "\n";
    return false;
  }

  // Create network
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TensorRTUniquePtr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    std::cerr << "Failed to create network" << "\n";
    return false;
  }

  // Create ONNX parser
  TensorRTUniquePtr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, gLogger));
  if (!parser) {
    std::cerr << "Failed to create ONNX parser" << "\n";
    return false;
  }

  // Parse ONNX file
  if (!parser->parseFromFile(
          onnxFile.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "Failed to parse ONNX file" << "\n";
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << "Parser error " << i << ": " << parser->getError(i)->desc()
                << "\n";
    }
    return false;
  }

  // Create builder config
  TensorRTUniquePtr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  if (!config) {
    std::cerr << "Failed to create builder config" << "\n";
    return false;
  }

  // Set configuration
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             1U << 30);           // 1GB
  config->setFlag(nvinfer1::BuilderFlag::kFP16);  // Enable FP16

  // Print network info
  std::cout << "Network inputs: " << network->getNbInputs() << "\n";
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto input = network->getInput(i);
    auto dims = input->getDimensions();
    std::cout << "Input " << i << ": " << input->getName() << " - ";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j];
      if (j < dims.nbDims - 1) std::cout << "x";
    }
    std::cout << "\n";
  }

  std::cout << "Network outputs: " << network->getNbOutputs() << "\n";
  for (int i = 0; i < network->getNbOutputs(); ++i) {
    auto output = network->getOutput(i);
    auto dims = output->getDimensions();
    std::cout << "Output " << i << ": " << output->getName() << " - ";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j];
      if (j < dims.nbDims - 1) std::cout << "x";
    }
    std::cout << "\n";
  }

  // Build serialized network
  std::cout << "Building TensorRT engine..." << "\n";
  TensorRTUniquePtr<nvinfer1::IHostMemory> serializedEngine(
      builder->buildSerializedNetwork(*network, *config));
  if (!serializedEngine) {
    std::cerr << "Failed to build serialized engine" << "\n";
    return false;
  }

  // Save engine to file
  std::ofstream engineFile_stream(engineFile, std::ios::binary);
  if (!engineFile_stream) {
    std::cerr << "Failed to open engine file for writing: " << engineFile
              << "\n";
    return false;
  }

  engineFile_stream.write(static_cast<const char*>(serializedEngine->data()),
                          serializedEngine->size());
  engineFile_stream.close();

  std::cout << "Engine saved to " << engineFile
            << " (size: " << serializedEngine->size() << " bytes)" << "\n";
  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <onnx_file> <engine_file>"
              << "\n";
    return 1;
  }

  std::string onnxFile = argv[1];
  std::string engineFile = argv[2];

  // Initialize CUDA
  cudaSetDevice(0);

  if (buildEngineFromOnnx(onnxFile, engineFile)) {
    std::cout << "Successfully built engine!" << "\n";
    return 0;
  } else {
    std::cerr << "Failed to build engine!" << "\n";
    return 1;
  }
}