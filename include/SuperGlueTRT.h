#ifndef SUPERGLUETRT_H_
#define SUPERGLUETRT_H_

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

struct MatchResult {
    std::vector<cv::DMatch> matches;
    cv::Mat scores;
};

class SuperGlueTRT {
public:
    explicit SuperGlueTRT(const SuperGlueConfig& config);
    ~SuperGlueTRT();

    bool initialize();
    bool match(const std::vector<cv::KeyPoint>& keypoints0, const cv::Mat& descriptors0,
               const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& descriptors1,
               MatchResult& result);
    
private:
    struct TensorInfo {
        void* devicePtr = nullptr;
        void* hostPtr = nullptr;
        size_t size = 0;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype;
        std::string name;
    };

    SuperGlueConfig config_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;
    
    std::vector<TensorInfo> input_tensors_;
    std::vector<TensorInfo> output_tensors_;
    
    bool loadEngine();
    bool allocateBuffers();
    void freeBuffers();
    bool allocateDynamicTensors(const std::vector<cv::KeyPoint>& keypoints0,
                               const std::vector<cv::KeyPoint>& keypoints1);
    bool prepareInputs(const std::vector<cv::KeyPoint>& keypoints0, const cv::Mat& descriptors0,
                       const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& descriptors1);
    bool postprocessOutputs(const std::vector<cv::KeyPoint>& keypoints0,
                           const std::vector<cv::KeyPoint>& keypoints1,
                           MatchResult& result);
    
    size_t getElementSize(nvinfer1::DataType dtype);
    size_t getTensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
    void printTensorInfo(const std::string& name, const nvinfer1::Dims& dims, nvinfer1::DataType dtype);
    
    // Helper functions for input preparation
    void normalizeKeypoints(const std::vector<cv::KeyPoint>& keypoints, 
                           int imageWidth, int imageHeight, 
                           float* output);
    void copyDescriptors(const cv::Mat& descriptors, float* output);
};

typedef std::shared_ptr<SuperGlueTRT> SuperGlueTRTPtr;

#endif // SUPERGLUETRT_H_