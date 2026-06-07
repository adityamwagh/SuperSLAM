#ifndef RERUNVIEWER_H
#define RERUNVIEWER_H

#include <memory> // For std::shared_ptr
#include <mutex>  // For std::mutex
#include <opencv2/opencv.hpp>
#include <rerun.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "StereoFrame.h" // StereoFrame, gtsam Pose3 and Cal3_S2Stereo

namespace superslam {

// Stream the live trajectory, stereo cameras, on-demand point cloud, text logs, and
// plots to a Rerun viewer. Spawn the interactive viewer, or record to the .rrd path in
// SUPERSLAM_RRD.
class RerunViewer {
public:
  typedef std::shared_ptr<RerunViewer> Ptr;

  RerunViewer();

  void set_cameras(float fx_left,
                   float fy_left,
                   float cx_left,
                   float cy_left,
                   float fx_right,
                   float fy_right,
                   float cx_right,
                   float cy_right,
                   float baseline);

  void close();

  // Draw the camera trajectory (Twc) and the current frame's on-demand stereo
  // cloud.
  void draw_frame(const StereoFrame& frame,
                  const std::vector<gtsam::Pose3>& trajectory,
                  const gtsam::Cal3_S2Stereo& K);

  // Log in rerun viewer
  void log_info(const std::string& msg, const std::string& log_type);

  // Log in rerun viewer with id of most recent active keyframe as index
  void
  log_info_mkf(const std::string& msg, unsigned long maxkeyframe_id, const std::string& log_type);

  void plot(const std::string& plot_name, double value, unsigned long maxkeyframe_id);

  // Thread management hooks (no-ops).
  void request_finish();
  bool is_finished();
  void request_stop();
  bool is_stopped();
  void release();
  bool stop();
  void run();

  bool check_finish();
  void set_finish();

private:
  const rerun::RecordingStream rec = rerun::RecordingStream("SuperSLAM_Viewer");

  // Camera parameters
  float fx_left_{0}, fy_left_{0}, cx_left_{0}, cy_left_{0};
  float fx_right_{0}, fy_right_{0}, cx_right_{0}, cy_right_{0};
  float baseline_{0};
  bool cameras_set_{false};

  bool viewer_running_{true};
  bool finish_requested_{false};
  bool finished_{false};
  bool stop_requested_{false};
  bool stopped_{false};

  std::mutex viewer_data_mutex_;
  std::mutex mutex_finish_;
  std::mutex mutex_stop_;

  // Different color for logging info of different components
  std::unordered_map<std::string, rerun::Color> log_color{
      {"vo", rerun::Color(255, 255, 255)},
      {"frontend", rerun::Color(0, 255, 255)},
      {"backend", rerun::Color(0, 255, 0)},
      {"loopclosing", rerun::Color(255, 165, 0)}};
};

} // namespace superslam

#endif // RERUNVIEWER_H
