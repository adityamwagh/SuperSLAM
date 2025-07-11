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

#ifndef SYSTEM_H
#define SYSTEM_H

#include <opencv4/opencv2/imgproc/types_c.h>

#include <memory>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "RerunViewer.h"
#include "SPVocabulary.h"
#include "Tracking.h"

namespace SuperSLAM {

class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class System {
 public:
  // Input sensor
  enum eSensor { MONOCULAR = 0, STEREO = 1, RGBD = 2 };

 public:
  // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and
  // Viewer threads.
  System(const std::string& strVocFile, const std::string& strSettingsFile,
         const eSensor sensor, const bool bUseViewer = true);

  // Destructor
  ~System();

  // Proccess the given stereo frame. Images must be synchronized and rectified.
  // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to
  // grayscale. Returns the camera pose (empty if tracking fails).
  cv::Mat TrackStereo(const cv::Mat& imLeft, const cv::Mat& imRight,
                      const double& timestamp);

  // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
  // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to
  // grayscale. Input depthmap: Float (CV_32F). Returns the camera pose (empty
  // if tracking fails).
  cv::Mat TrackRGBD(const cv::Mat& im, const cv::Mat& depthmap,
                    const double& timestamp);

  // Proccess the given monocular frame
  // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to
  // grayscale. Returns the camera pose (empty if tracking fails).
  cv::Mat TrackMonocular(const cv::Mat& im, const double& timestamp);

  // This stops local mapping thread (map building) and performs only camera
  // tracking.
  void ActivateLocalizationMode();
  // This resumes local mapping thread and performs SLAM again.
  void DeactivateLocalizationMode();

  // Returns true if there have been a big map change (loop closure, global BA)
  // since last call to this function
  bool MapChanged();

  // Reset the system (clear map)
  void Reset();

  // All threads will be requested to finish.
  // It waits until all threads have finished.
  // This function must be called before saving the trajectory.
  void Shutdown();

  // Save camera trajectory in the TUM RGB-D dataset format.
  // Only for stereo and RGB-D. This method does not work for monocular.
  // Call first Shutdown()
  // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
  void SaveTrajectoryTUM(const std::string& filename);

  // Save keyframe poses in the TUM RGB-D dataset format.
  // This method works for all sensor input.
  // Call first Shutdown()
  // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
  void SaveKeyFrameTrajectoryTUM(const std::string& filename);

  // Save camera trajectory in the KITTI dataset format.
  // Only for stereo and RGB-D. This method does not work for monocular.
  // Call first Shutdown()
  // See format details at:
  // http://www.cvlibs.net/datasets/kitti/eval_odometry.php
  void SaveTrajectoryKITTI(const std::string& filename);

  // TODO: Save/Load functions
  // SaveMap(const string &filename);
  // LoadMap(const string &filename);

  // Information from most recent processed frame
  // You can call this right after TrackMonocular (or stereo or RGBD)
  int GetTrackingState();
  std::vector<MapPoint*> GetTrackedMapPoints();
  std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

 private:
  // Input sensor
  eSensor mSensor;

  // ORB vocabulary used for place recognition and feature matching.
  std::unique_ptr<ORBVocabulary> mpVocabulary;

  // KeyFrame database for place recognition (relocalization and loop
  // detection).
  std::unique_ptr<KeyFrameDatabase> mpKeyFrameDatabase;

  // Map structure that stores the pointers to all KeyFrames and MapPoints.
  std::unique_ptr<Map> mpMap;

  // Tracker. It receives a frame and computes the associated camera pose.
  // It also decides when to insert a new keyframe, create some new MapPoints
  // and performs relocalization if tracking fails.
  std::unique_ptr<Tracking> mpTracker;

  // Local Mapper. It manages the local map and performs local bundle
  // adjustment.
  std::unique_ptr<LocalMapping> mpLocalMapper;

  // Loop Closer. It searches loops with every new keyframe. If there is a loop
  // it performs a pose graph optimization and full bundle adjustment (in a new
  // thread) afterwards.
  std::unique_ptr<LoopClosing> mpLoopCloser;

  // The viewer draws the map and the current camera pose. It uses Rerun.io.
  std::unique_ptr<RerunViewer> mpViewer;

  // System threads: Local Mapping, Loop Closing, Viewer.
  // The Tracking thread "lives" in the main execution thread that creates the
  // System object.
  std::unique_ptr<std::thread> mptLocalMapping;
  std::unique_ptr<std::thread> mptLoopClosing;
  std::unique_ptr<std::thread> mptViewer;

  // Reset flag
  std::mutex mMutexReset;
  bool mbReset;

  // Change mode flags
  std::mutex mMutexMode;
  bool mbActivateLocalizationMode;
  bool mbDeactivateLocalizationMode;

  // Tracking state
  int mTrackingState;
  std::vector<MapPoint*> mTrackedMapPoints;
  std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
  std::mutex mMutexState;
};

}  // namespace SuperSLAM

#endif  // SYSTEM_H
