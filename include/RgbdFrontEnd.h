#pragma once
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <opencv4/opencv2/core.hpp>

#include "InferenceInterfaces.h"
#include "StereoFrame.h"

namespace superslam {

// Convert an RGB-D image and depth map to a StereoFrame. Depth gives the
// right-image coordinate directly (uR = uL - bf/Z); the front end needs no
// feature matcher. Undistort keypoints with camera_matrix and dist_coeffs.
// Sample depth at the raw pixel (it is registered to the raw image). The
// emitted StereoFrame uses the same backend as stereo.
class RgbdFrontEnd {
public:
  RgbdFrontEnd(IFeatureExtractor* ext,
               const gtsam::Cal3_S2Stereo& K,
               double depth_factor,
               double max_depth,
               const cv::Mat& camera_matrix,
               const cv::Mat& dist_coeffs)
      : ext_(ext), K_(K), depth_factor_(depth_factor), max_depth_(max_depth),
        camera_matrix_(camera_matrix), dist_coeffs_(dist_coeffs) {}

  StereoFrame process(const cv::Mat& gray, const cv::Mat& depth, double timestamp);

private:
  IFeatureExtractor* ext_;
  gtsam::Cal3_S2Stereo K_;
  double depth_factor_;
  double max_depth_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
};

} // namespace superslam
