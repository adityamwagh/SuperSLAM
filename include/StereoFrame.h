#pragma once
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <opencv4/opencv2/core.hpp>

#include <vector>

#include "DescriptorPool.h"

namespace superslam {

// Lean, GTSAM-native per-frame data. Pose is Twc (camera-in-world).
struct StereoFrame {
  double timestamp = 0.0;
  std::vector<cv::KeyPoint> keypoints_left; // left, undistorted
  DeviceDescriptors descriptors_left;       // device-resident, for frame-to-keyframe matching
  std::vector<gtsam::StereoPoint2> stereo;  // (uL, uR, v); uR = NaN if no stereo
  std::vector<char> has_depth;              // 1 if stereo-valid
  gtsam::Pose3 pose;                        // Twc (camera-in-world)

  // World point for stereo feature i: pose (Twc) * camera-frame
  // backprojection.
  gtsam::Point3 backproject(int i, const gtsam::Cal3_S2Stereo& K) const;
};

} // namespace superslam
