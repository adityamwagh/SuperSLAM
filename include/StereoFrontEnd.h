#pragma once
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <opencv4/opencv2/core.hpp>

#include "InferenceInterfaces.h"
#include "StereoFrame.h"

namespace superslam {

// Convert a stereo image pair to a StereoFrame. Hold backend-agnostic
// inference interfaces (no knowledge of TensorRT). Leave pose at identity for
// the estimator to set.
class StereoFrontEnd {
public:
  StereoFrontEnd(IFeatureExtractor* ext,
                 IFeatureMatcher* matcher,
                 const gtsam::Cal3_S2Stereo& K,
                 float min_disparity = 1.0f)
      : ext_(ext), matcher_(matcher), K_(K), min_disparity_(min_disparity) {}

  StereoFrame process(const cv::Mat& left, const cv::Mat& right, double timestamp);

private:
  IFeatureExtractor* ext_;
  IFeatureMatcher* matcher_;
  gtsam::Cal3_S2Stereo K_;
  float min_disparity_;
};

} // namespace superslam
