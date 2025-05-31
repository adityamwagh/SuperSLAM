/**
 * This file is part of SuperSLAM.
 *
 * Copyright (C) Aditya Wagh <adityamwagh at outlook dot com>
 * For more information see <https://github.com/adityamwagh/SuperSLAM>
 *
 * SuperSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SuperSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SuperSLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RERUNVIEWER_H
#define RERUNVIEWER_H

#include <memory>
#include <mutex>
#include <opencv4/opencv2/opencv.hpp>
#include <rerun.hpp>
#include <string>
#include <thread>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "System.h"
#include "Tracking.h"

namespace SuperSLAM {

class System;
class Tracking;
class Map;
class KeyFrame;
class MapPoint;

/**
 * @brief Rerun.io-based visualization for SuperSLAM
 *
 * This class provides real-time visualization of SLAM data using Rerun.io,
 * including camera trajectory, map points, keyframes, and feature tracks.
 */
class RerunViewer {
 public:
  RerunViewer(System* pSystem, Tracking* pTracking, Map* pMap,
              const std::string& strSettingPath);
  ~RerunViewer();

  // Main viewer thread
  void Run();

  // Control functions
  void RequestFinish();
  void RequestStop();
  bool isFinished();
  bool isStopped();
  void Release();

  // Logging functions for real-time visualization
  void LogFrame(const cv::Mat& image,
                const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& pose,
                double timestamp);
  void LogTrajectory(const cv::Mat& pose, double timestamp);
  void LogMapPoints(const std::vector<MapPoint*>& map_points);
  void LogKeyFrame(KeyFrame* pKF);
  void LogFeatureMatches(const std::vector<cv::KeyPoint>& kpts1,
                         const std::vector<cv::KeyPoint>& kpts2,
                         const std::vector<cv::DMatch>& matches);
  void LogSuperPointFeatures(const cv::Mat& image,
                             const std::vector<cv::KeyPoint>& keypoints,
                             const cv::Mat& descriptors);
  void LogLoopClosure(KeyFrame* pKF1, KeyFrame* pKF2,
                      const std::vector<cv::DMatch>& matches);

 private:
  // Rerun recorder
  std::shared_ptr<rerun::RecordingStream> rec_;

  // SLAM components
  System* mpSystem;
  Tracking* mpTracker;
  Map* mpMap;

  // Threading
  std::thread* mptViewer;

  // Timing
  double mT;  // 1/fps in ms

  // Viewpoint parameters
  float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

  // Control flags
  bool mbFinishRequested;
  bool mbFinished;
  std::mutex mMutexFinish;

  bool mbStopped;
  bool mbStopRequested;
  std::mutex mMutexStop;

  // Data tracking
  std::vector<cv::Point3f> trajectory_points_;
  std::vector<double> trajectory_timestamps_;
  std::mutex mMutexTrajectory;

  std::vector<cv::Point3f> map_points_;
  std::mutex mMutexMapPoints;

  // Current frame data
  cv::Mat current_image_;
  std::vector<cv::KeyPoint> current_keypoints_;
  cv::Mat current_pose_;
  double current_timestamp_;
  std::mutex mMutexCurrentFrame;

  // Active data structures (like the example)
  std::unordered_map<unsigned long, KeyFrame*> all_keyframes_;
  std::unordered_map<unsigned long, KeyFrame*> active_keyframes_;
  std::unordered_map<unsigned long, MapPoint*> active_landmarks_;
  std::mutex viewer_data_mutex_;

  // Visualization settings
  struct VisualizationSettings {
    bool show_trajectory = true;
    bool show_map_points = true;
    bool show_keyframes = true;
    bool show_current_frame = true;
    bool show_feature_tracks = true;
    bool show_loop_closures = true;

    // Colors
    std::array<float, 3> trajectory_color = {0.0f, 1.0f, 0.0f};    // Green
    std::array<float, 3> map_points_color = {1.0f, 0.0f, 0.0f};    // Red
    std::array<float, 3> keyframe_color = {0.0f, 0.0f, 1.0f};      // Blue
    std::array<float, 3> current_pose_color = {1.0f, 1.0f, 0.0f};  // Yellow
    std::array<float, 3> features_color = {0.0f, 1.0f, 1.0f};      // Cyan

    // Sizes
    float trajectory_line_width = 2.0f;
    float map_point_size = 0.02f;
    float keyframe_size = 0.1f;
    float feature_size = 3.0f;
  } settings_;

  // Helper functions
  bool CheckFinish();
  void SetFinish();
  bool Stop();

  void UpdateVisualization();
  void DrawTrajectory();
  void DrawMapPoints();
  void DrawKeyFrames();
  void DrawCurrentFrame();

  // Conversion utilities
  rerun::Position3D cvMatToRerunPosition(const cv::Mat& pose);
  rerun::Transform3D cvMatToRerunTransform(const cv::Mat& pose);
  std::vector<rerun::Position3D> keypointsToRerunPoints(
      const std::vector<cv::KeyPoint>& keypoints, float depth = 1.0f);

  void LoadViewerSettings(const std::string& strSettingPath);
};

}  // namespace SuperSLAM

#endif  // RERUNVIEWER_H