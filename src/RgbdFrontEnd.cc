#include "RgbdFrontEnd.h"

#include <opencv4/opencv2/calib3d.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

namespace superslam {

namespace {
double sampleDepth(const cv::Mat& depth, int u, int v, double factor) {
  if (u < 0 || v < 0 || u >= depth.cols || v >= depth.rows)
    return 0.0;
  if (depth.type() == CV_16U)
    return static_cast<double>(depth.at<uint16_t>(v, u)) / factor;
  if (depth.type() == CV_32F)
    return static_cast<double>(depth.at<float>(v, u)) / factor;
  return 0.0;
}
} // namespace

StereoFrame RgbdFrontEnd::process(const cv::Mat& gray, const cv::Mat& depth, double timestamp) {
  const Features L = ext_->extract(gray);
  const size_t n = L.keypoints.size();

  std::vector<cv::Point2f> raw(n), undist(n);
  for (size_t i = 0; i < n; ++i)
    raw[i] = L.keypoints[i].pt;
  const bool hasDist = !dist_coeffs_.empty() && cv::countNonZero(dist_coeffs_) > 0;
  if (hasDist && n > 0)
    cv::undistortPoints(raw, undist, camera_matrix_, dist_coeffs_, cv::noArray(), camera_matrix_);
  else
    undist = raw;

  StereoFrame f;
  f.timestamp = timestamp;
  f.keypoints_left = L.keypoints;
  f.descriptors_left = L.descriptors;
  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  const double bf = K_.fx() * K_.baseline();
  f.stereo.assign(n, gtsam::StereoPoint2());
  f.has_depth.assign(n, 0);

  for (size_t i = 0; i < n; ++i) {
    f.keypoints_left[i].pt = undist[i];
    const double Z =
        sampleDepth(depth, std::lround(raw[i].x), std::lround(raw[i].y), depth_factor_);
    const double uL = undist[i].x, v = undist[i].y;
    if (Z > 0.0 && Z < max_depth_) {
      f.stereo[i] = gtsam::StereoPoint2(uL, uL - bf / Z, v);
      f.has_depth[i] = 1;
    } else {
      f.stereo[i] = gtsam::StereoPoint2(uL, kNaN, v);
    }
  }
  return f;
}

} // namespace superslam
