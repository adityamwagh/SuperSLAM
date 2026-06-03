#include "StereoFrontEnd.h"

#include <cmath>
#include <limits>

#include "Profiling.h"

namespace superslam {

StereoFrame StereoFrontEnd::process(const cv::Mat& left, const cv::Mat& right, double timestamp) {
  Features L, R;
  {
    SUPERSLAM_PROFILE_SCOPE("fe_extract_stereo"); // One batched {2,1,H,W} SuperPoint infer.
    std::pair<Features, Features> lr = ext_->extract_stereo(left, right);
    L = std::move(lr.first);
    R = std::move(lr.second);
  }

  StereoFrame f;
  f.timestamp = timestamp;
  f.keypoints_left = L.keypoints;
  f.descriptors_left = L.descriptors;
  const size_t n = L.keypoints.size();
  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  f.stereo.assign(n, gtsam::StereoPoint2());
  f.has_depth.assign(n, 0);
  for (size_t i = 0; i < n; ++i) // Default: monocular-only (no valid uR).
    f.stereo[i] = gtsam::StereoPoint2(L.keypoints[i].pt.x, kNaN, L.keypoints[i].pt.y);

  MatchResult m;
  {
    SUPERSLAM_PROFILE_SCOPE("fe_lg_stereo_match");
    m = matcher_->match(L.keypoints, L.descriptors, R.keypoints, R.descriptors);
  }
  for (const cv::DMatch& d : m.matches) {
    const int i = d.queryIdx, j = d.trainIdx;
    if (i < 0 || j < 0 || i >= static_cast<int>(n) || j >= static_cast<int>(R.keypoints.size()))
      continue;
    const float uL = L.keypoints[i].pt.x, v = L.keypoints[i].pt.y;
    const float uR = R.keypoints[j].pt.x;
    if (uL - uR < min_disparity_)
      continue; // Disparity floor.
    if (std::abs(L.keypoints[i].pt.y - R.keypoints[j].pt.y) > 2.0f)
      continue; // Rectified row check.
    f.stereo[i] = gtsam::StereoPoint2(uL, uR, v);
    f.has_depth[i] = 1;
  }
  return f;
}

} // namespace superslam
