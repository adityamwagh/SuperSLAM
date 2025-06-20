#ifndef RERUNVIEWER_H
#define RERUNVIEWER_H

#include <memory>  // For std::shared_ptr
#include <mutex>   // For std::mutex
#include <opencv2/opencv.hpp>
#include <rerun.hpp>
#include <thread>         // for std::thread
#include <unordered_map>  // For std::unordered_map

#include "Frame.h"
#include "KeyFrame.h"  // Added for KeyFrame type
#include "Map.h"
#include "MapPoint.h"  // Added for MapPoint type

namespace SuperSLAM {
class RerunViewer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef std::shared_ptr<RerunViewer> Ptr;

  RerunViewer();

  void SetMap(Map* map);

  void SetCameras(float fx_left, float fy_left, float cx_left, float cy_left,
                  float fx_right, float fy_right, float cx_right,
                  float cy_right, float baseline);

  void Close();

  void AddCurrentFrame(Frame* current_frame);

  void AddStereoFrames(Frame* current_frame, const cv::Mat& left_image,
                       const cv::Mat& right_image);

  void UpdateMap();

  // Log in rerun viewer
  void LogInfo(std::string msg, std::string log_type);

  // Log in rerun viewer with id of most recent active keyframe as index
  void LogInfoMKF(std::string msg, unsigned long maxkeyframe_id,
                  std::string log_type);

  void Plot(std::string plot_name, double value, unsigned long maxkeyframe_id);

  // Log keyframe in rerun viewer
  void LogKeyFrame(KeyFrame* keyframe);

  // Add methods that might be called from Tracking or System
  void RequestFinish();
  bool isFinished();
  void RequestStop();
  bool isStopped();
  void Release();
  bool Stop();  // Added based on old RerunViewer
  void Run();   // Added based on old RerunViewer

  // Helper methods for thread management (from old RerunViewer, ensure
  // consistency)
  bool CheckFinish();
  void SetFinish();

 private:
  const rerun::RecordingStream rec = rerun::RecordingStream("SuperSLAM_Viewer");

  Frame* current_frame_{nullptr};
  Map* map_{nullptr};

  // Stereo images for display
  cv::Mat current_left_image_;
  cv::Mat current_right_image_;
  bool has_stereo_images_{false};

  // Camera parameters
  float fx_left_, fy_left_, cx_left_, cy_left_;
  float fx_right_, fy_right_, cx_right_, cy_right_;
  float baseline_;
  bool cameras_set_{false};

  bool viewer_running_{true};
  bool mbFinishRequested{false};
  bool mbFinished{false};
  bool mbStopRequested{false};
  bool mbStopped{false};

  std::unordered_map<unsigned long, KeyFrame*> all_keyframes_;
  std::unordered_map<unsigned long, KeyFrame*> active_keyframes_;
  std::unordered_map<unsigned long, MapPoint*> active_landmarks_;

  std::mutex viewer_data_mutex_;
  std::mutex mMutexFinish;
  std::mutex mMutexStop;

  // Different color for logging info of different components
  std::unordered_map<std::string, rerun::Color> log_color{
      {"vo", rerun::Color(255, 255, 255)},
      {"frontend", rerun::Color(0, 255, 255)},
      {"backend", rerun::Color(0, 255, 0)},
      {"loopclosing", rerun::Color(255, 165, 0)}};

  std::thread* mptViewer{nullptr};
  double mT;
};

}  // namespace SuperSLAM

#endif  // RERUNVIEWER_H