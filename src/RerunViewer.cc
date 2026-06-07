#include "RerunViewer.h"

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Logging.h"

// Compiling stub. The Frame, KeyFrame, MapPoint, and Map draw bodies are
// no-ops; the constructor sets up the Rerun recording stream.

namespace superslam {

RerunViewer::RerunViewer() {
  // Headless or CI: if SUPERSLAM_RRD is set, record to that .rrd file; otherwise
  // spawn the interactive viewer.
  if (const char* rrd_path = std::getenv("SUPERSLAM_RRD")) {
    rec.save(rrd_path).exit_on_failure();
  } else {
    rec.spawn().exit_on_failure();
  }

  // World origin
  rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);

  // Stereo camera view coordinates
  rec.log_static("world/current_camera", rerun::ViewCoordinates::RDF);
  rec.log_static("world/current_left_camera", rerun::ViewCoordinates::RDF);
  rec.log_static("world/current_right_camera", rerun::ViewCoordinates::RDF);

  // Plot series setup.
  rec.log_static("plots/loop_deep_score",
                 rerun::SeriesLine()
                     .with_color({255, 0, 0})
                     .with_name("Loop Closure Deep Score")
                     .with_width(2));
  rec.log_static("plots/frontend_inlier_ratio",
                 rerun::SeriesLine()
                     .with_color({0, 255, 255})
                     .with_name("Frontend landmark inlier ratio")
                     .with_width(2));

  rec.set_time_sequence("max_keyframe_id", 0);
  rec.set_time_sequence("currentframe_id", 0);
  SLOG_INFO("RerunViewer initialized");
}

void RerunViewer::set_cameras(float fx_left,
                              float fy_left,
                              float cx_left,
                              float cy_left,
                              float fx_right,
                              float fy_right,
                              float cx_right,
                              float cy_right,
                              float baseline) {
  std::lock_guard<std::mutex> lock(viewer_data_mutex_);
  fx_left_ = fx_left;
  fy_left_ = fy_left;
  cx_left_ = cx_left;
  cy_left_ = cy_left;
  fx_right_ = fx_right;
  fy_right_ = fy_right;
  cx_right_ = cx_right;
  cy_right_ = cy_right;
  baseline_ = baseline;
  cameras_set_ = true;
}

void RerunViewer::close() {
  std::lock_guard<std::mutex> lock(viewer_data_mutex_);
  viewer_running_ = false;
}

void RerunViewer::log_info(const std::string& msg, const std::string& log_type) {
  auto it = log_color.find(log_type);
  const rerun::Color color = (it != log_color.end()) ? it->second : rerun::Color(255, 255, 255);
  rec.log("logs", rerun::TextLog(msg).with_color(color));
}

void RerunViewer::log_info_mkf(const std::string& msg,
                               unsigned long maxkeyframe_id,
                               const std::string& log_type) {
  rec.set_time_sequence("max_keyframe_id", static_cast<int64_t>(maxkeyframe_id));
  log_info(msg, log_type);
}

void RerunViewer::plot(const std::string& plot_name, double value, unsigned long maxkeyframe_id) {
  rec.set_time_sequence("max_keyframe_id", static_cast<int64_t>(maxkeyframe_id));
  rec.log("plots/" + plot_name, rerun::Scalar(value));
}

// Thread-management hooks: plain flag toggles.
void RerunViewer::request_finish() {
  std::lock_guard<std::mutex> lock(mutex_finish_);
  finish_requested_ = true;
}

bool RerunViewer::is_finished() {
  std::lock_guard<std::mutex> lock(mutex_finish_);
  return finished_;
}

void RerunViewer::request_stop() {
  std::lock_guard<std::mutex> lock(mutex_stop_);
  stop_requested_ = true;
}

bool RerunViewer::is_stopped() {
  std::lock_guard<std::mutex> lock(mutex_stop_);
  return stopped_;
}

void RerunViewer::release() {
  std::lock_guard<std::mutex> lock(mutex_stop_);
  stopped_ = false;
}

bool RerunViewer::stop() {
  std::lock_guard<std::mutex> lock(mutex_stop_);
  if (stop_requested_) {
    stopped_ = true;
    return true;
  }
  return false;
}

void RerunViewer::run() {}

bool RerunViewer::check_finish() {
  std::lock_guard<std::mutex> lock(mutex_finish_);
  return finish_requested_;
}

void RerunViewer::set_finish() {
  std::lock_guard<std::mutex> lock(mutex_finish_);
  finished_ = true;
}

void RerunViewer::draw_frame(const StereoFrame& frame,
                             const std::vector<gtsam::Pose3>& traj,
                             const gtsam::Cal3_S2Stereo& K) {
  std::vector<rerun::Position3D> path;
  path.reserve(traj.size());
  for (const auto& p : traj) {
    const auto& t = p.translation();
    path.emplace_back(static_cast<float>(t.x()),
                      static_cast<float>(t.y()),
                      static_cast<float>(t.z()));
  }
  rec.log("world/trajectory", rerun::Points3D(path).with_radii(0.3f));

  std::vector<rerun::Position3D> cloud;
  for (size_t i = 0; i < frame.has_depth.size(); ++i) {
    if (!frame.has_depth[i])
      continue;
    const auto X = frame.backproject(static_cast<int>(i), K);
    cloud.emplace_back(static_cast<float>(X.x()),
                       static_cast<float>(X.y()),
                       static_cast<float>(X.z()));
  }
  rec.log("world/cloud", rerun::Points3D(cloud));
}

} // namespace superslam
