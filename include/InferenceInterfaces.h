#pragma once
#include <opencv4/opencv2/core.hpp>

#include <utility>
#include <vector>

#include "DescriptorPool.h"

// Hold the result of matching two feature sets: query and train index pairs
// (cv::DMatch) and, optionally, a raw score matrix. Keep in the global
// namespace.
struct MatchResult {
  std::vector<cv::DMatch> matches;
  cv::Mat scores;
};

namespace superslam {

// Hold backend-agnostic feature-extractor output. Per-keypoint score lives in
// cv::KeyPoint::response.
struct Features {
  std::vector<cv::KeyPoint> keypoints;
  DeviceDescriptors descriptors; // [N x D], resident in a DescriptorPool slot (FP16)
};

// Pluggable feature extractor.
class IFeatureExtractor {
public:
  virtual ~IFeatureExtractor() = default;
  virtual Features extract(const cv::Mat& image) = 0;
  // Extract a rectified stereo pair. Default: two single-image calls. SuperPoint overrides
  // this with one batched {2,1,H,W} infer.
  virtual std::pair<Features, Features> extract_stereo(const cv::Mat& left, const cv::Mat& right) {
    return {extract(left), extract(right)};
  }
};

// Pluggable feature matcher. Two descriptor sources: host cv::Mat (loop
// closure, descriptors in the host keyframe DB) and DeviceDescriptors (live
// tracking, descriptors resident on the device).
class IFeatureMatcher {
public:
  virtual ~IFeatureMatcher() = default;
  // Run the host path (loop closure): descriptors are host cv::Mat, uploaded
  // internally.
  virtual MatchResult match(const std::vector<cv::KeyPoint>& kp0,
                            const cv::Mat& d0,
                            const std::vector<cv::KeyPoint>& kp1,
                            const cv::Mat& d1) = 0;
  // Run the device path (live tracking): descriptors already on the device (FP16 pool
  // slots).
  virtual MatchResult match(const std::vector<cv::KeyPoint>& kp0,
                            const DeviceDescriptors& d0,
                            const std::vector<cv::KeyPoint>& kp1,
                            const DeviceDescriptors& d1) = 0;
  // Copy device descriptors to a host cv::Mat (CV_32F, L2-normalized rows).
  // Empty handle maps to empty Mat.
  virtual cv::Mat descriptors_to_host(const DeviceDescriptors& d) = 0;
};

} // namespace superslam
