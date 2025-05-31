#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <Logging.h>
#include <memory>
#include <vector>

// Simple logger for TensorRT 10.11.0
class SimpleLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
      if (severity <= Severity::kWARNING) { SLOG_WARN("[TensorRT] {}", msg); }
    }
};

static SimpleLogger gLogger;

template <typename T> struct TensorRTDestroy {
    void operator()(T* obj) const {
      if (obj) { delete obj; }
    }
};

template <typename T> using TensorRTUniquePtr = std::unique_ptr<T, TensorRTDestroy<T>>;

bool buildEngineFromOnnx(const std::string& onnxFile, const std::string& engineFile) {
  SLOG_INFO("Building engine from {} to {}", onnxFile, engineFile);

  // Create builder
  TensorRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
  if (!builder) {
    SLOG_ERROR("Failed to create TensorRT builder");
    return false;
  }

  // Create network
  const auto explicitBatch
    = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TensorRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
  if (!network) {
    SLOG_ERROR("Failed to create network");
    return false;
  }

  // Create ONNX parser
  TensorRTUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
  if (!parser) {
    SLOG_ERROR("Failed to create ONNX parser");
    return false;
  }

  // Parse ONNX file
  if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    SLOG_ERROR("Failed to parse ONNX file");
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      SLOG_ERROR("Parser error {}: {}", i, parser->getError(i)->desc());
    }
    return false;
  }

  // Create builder config
  TensorRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!config) {
    SLOG_ERROR("Failed to create builder config");
    return false;
  }

  // Set configuration
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             8ULL << 30);        // 8GB
  config->setFlag(nvinfer1::BuilderFlag::kFP16); // Enable FP16

  // Print network info
  SLOG_INFO("Network inputs: {}", network->getNbInputs());
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto input = network->getInput(i);
    auto dims  = input->getDimensions();
    SLOG_INFO("Input {}: {} - ", i, input->getName());
    for (int j = 0; j < dims.nbDims; ++j) {
      SLOG_INFO("{}", dims.d[ j ]);
      if (j < dims.nbDims - 1) { SLOG_INFO("x"); }
    }
  }

  SLOG_INFO("Network outputs: {}", network->getNbOutputs());
  for (int i = 0; i < network->getNbOutputs(); ++i) {
    auto output = network->getOutput(i);
    auto dims   = output->getDimensions();
    SLOG_INFO("Output {}: {} - ", i, output->getName());
    for (int j = 0; j < dims.nbDims; ++j) {
      SLOG_INFO("{}", dims.d[ j ]);
      if (j < dims.nbDims - 1) { SLOG_INFO("x"); }
    }
  }

  // Build serialized network
  SLOG_INFO("Building TensorRT engine...");
  TensorRTUniquePtr<nvinfer1::IHostMemory> serializedEngine(
    builder->buildSerializedNetwork(*network, *config));
  if (!serializedEngine) {
    SLOG_ERROR("Failed to build serialized engine");
    return false;
  }

  // Save engine to file
  std::ofstream engineFile_stream(engineFile, std::ios::binary);
  if (!engineFile_stream) {
    SLOG_ERROR("Failed to open engine file for writing");
    return false;
  }

  engineFile_stream.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
  engineFile_stream.close();

  SLOG_INFO("Engine saved to {}", engineFile);
  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    SLOG_ERROR("Usage: {} <onnx_file> <engine_file>", argv[ 0 ]);
    return 1;
  }

  // Initialize logging
  SuperSLAM::Logger::initialize();

  // Initialize CUDA
  cudaSetDevice(0);

  std::string onnxFile   = argv[ 1 ];
  std::string engineFile = argv[ 2 ];

  if (buildEngineFromOnnx(onnxFile, engineFile)) {
    SLOG_INFO("Successfully built engine!");
    return 0;
  } else {
    SLOG_ERROR("Failed to build engine!");
    return 1;
  }
}
