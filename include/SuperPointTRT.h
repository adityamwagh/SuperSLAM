#ifndef SUPERPOINTTRT_H_
#define SUPERPOINTTRT_H_

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv4/opencv2/opencv.hpp>
#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "ReadConfig.h"

class SuperPointTRT {
public:
    explicit SuperPointTRT(const SuperPointConfig& config);
    ~SuperPointTRT();

    bool initialize();
    bool infer(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    
private:
    struct TensorInfo {
        void* devicePtr = nullptr;
        void* hostPtr = nullptr;
        size_t size = 0;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype;
        std::string name;
    };

    SuperPointConfig config_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;
    
    std::vector<TensorInfo> input_tensors_;
    std::vector<TensorInfo> output_tensors_;
    
    int input_height_ = 0;
    int input_width_ = 0;
    
    bool loadEngine();
    bool allocateBuffers();
    void freeBuffers();
    bool preprocessImage(const cv::Mat& image);
    bool postprocessOutputs(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    
    size_t getElementSize(nvinfer1::DataType dtype);
    size_t getTensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
    void printTensorInfo(const std::string& name, const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
};

typedef std::shared_ptr<SuperPointTRT> SuperPointTRTPtr;

#endif // SUPERPOINTTRT_H_