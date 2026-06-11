#pragma once
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <opencv4/opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "EigenPlaces.h"
#include "LightGlue.h"
#include "LoopCloser.h"
#include "RerunViewer.h"
#include "RgbdFrontEnd.h"
#include "StereoFrontEnd.h"
#include "SuperPoint.h"
#include "VoEstimator.h"

namespace superslam {

// Public facade: the one object users construct. Own the SuperPoint and LightGlue
// front-end and the GTSAM stereo VO; everything internal is Twc.
class SuperSLAM {
public:
  explicit SuperSLAM(const std::string& config_path, bool use_viewer = false);

  // Process a rectified stereo pair. Return Tcw (world to cam) cv::Mat;
  // internally the pose is Twc.
  cv::Mat track_stereo(const cv::Mat& left, const cv::Mat& right, double timestamp);

  // Process an RGB image and its aligned depth map. Return Tcw (world to cam).
  cv::Mat track_rgbd(const cv::Mat& rgb, const cv::Mat& depth, double timestamp);

  enum class TrajectoryFormat { KITTI, TUM };
  void save_trajectory(const std::string& path, TrajectoryFormat format) const;
  void save_trajectory_kitti(const std::string& path) const;
  void save_map(const std::string& path) const;
  void shutdown() {}

  // Return the number of accepted loop closures so far (diagnostics or benchmark).
  size_t loop_closure_count() const;

private:
  gtsam::Cal3_S2Stereo K_;
  std::shared_ptr<SuperPoint> extractor_;
  std::shared_ptr<LightGlue> matcher_;
  std::shared_ptr<LightGlue> loop_matcher_; // dedicated matcher for the loop thread
  std::unique_ptr<StereoFrontEnd> frontend_;
  std::unique_ptr<RgbdFrontEnd> rgbd_frontend_;
  std::unique_ptr<VoEstimator> estimator_;
  std::unique_ptr<RerunViewer> viewer_;  // null unless use_viewer
  std::vector<gtsam::Pose3> trajectory_; // Twc
  std::vector<double> timestamps_;
  bool use_viewer_;
  bool loop_enabled_ = false;
};

} // namespace superslam
