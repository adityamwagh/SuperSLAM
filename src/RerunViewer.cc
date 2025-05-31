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

#include "RerunViewer.h"

#include <chrono>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>

namespace SuperSLAM {

RerunViewer::RerunViewer(System* pSystem, Tracking* pTracking, Map* pMap,
                         const std::string& strSettingPath)
    : mpSystem(pSystem),
      mpTracker(pTracking),
      mpMap(pMap),
      mbFinishRequested(false),
      mbFinished(false),
      mbStopped(false),
      mbStopRequested(false),
      mptViewer(nullptr) {
  // Initialize Rerun
  rec_ = std::make_shared<rerun::RecordingStream>("SuperSLAM");
  rec_->spawn().exit_on_failure();

  // Set up the 3D scene with better organization
  rec_->log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
  rec_->log_static("world/camera", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
  rec_->log_static("camera/image", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
  
  // Initialize the scene with some context
  rec_->log_static("world/origin", rerun::Points3D({{0.0, 0.0, 0.0}}).with_colors({{255, 255, 255}}).with_radii(0.1f));
  rec_->log_static("world/axes", rerun::Arrows3D::from_vectors({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}).with_colors({{255, 0, 0}, {0, 255, 0}, {0, 0, 255}}));

  // Load settings
  LoadViewerSettings(strSettingPath);

  std::cout << "RerunViewer: Initialized successfully" << "\n";
  std::cout << "RerunViewer: View at http://localhost:9090" << "\n";
}

RerunViewer::~RerunViewer() {
  if (mptViewer) {
    mptViewer->join();
    delete mptViewer;
  }
}

void RerunViewer::LoadViewerSettings(const std::string& strSettingPath) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  if (!fSettings.isOpened()) {
    std::cerr << "RerunViewer: Failed to open settings file at: "
              << strSettingPath << "\n";
    // Use default settings
    mT = 33.0;  // ~30 FPS
    return;
  }

  float fps = fSettings["Camera.fps"];
  if (fps == 0) fps = 30.0f;
  mT = 1000.0 / fps;  // Convert to ms

  // Load viewpoint settings (optional)
  mViewpointX = fSettings["Viewer.ViewpointX"];
  mViewpointY = fSettings["Viewer.ViewpointY"];
  mViewpointZ = fSettings["Viewer.ViewpointZ"];
  mViewpointF = fSettings["Viewer.ViewpointF"];

  fSettings.release();
}

void RerunViewer::Run() {
  mbFinished = false;
  mbStopped = false;

  std::cout << "RerunViewer: Starting viewer thread" << "\n";

  while (true) {
    // Check if we should stop
    if (CheckFinish()) break;
    if (Stop()) {
      while (isStopped() && !CheckFinish()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
    }

    // Update visualization
    UpdateVisualization();

    // Sleep to maintain frame rate
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(mT)));
  }

  SetFinish();
}

void RerunViewer::UpdateVisualization() {
  if (!mpMap) return;

  // Update active data
  {
    std::unique_lock<std::mutex> lock(viewer_data_mutex_);
    
    // Update active keyframes
    std::vector<KeyFrame*> all_kfs = mpMap->GetAllKeyFrames();
    for (KeyFrame* pKF : all_kfs) {
      if (pKF && !pKF->isBad()) {
        all_keyframes_[pKF->mnId] = pKF;
        active_keyframes_[pKF->mnId] = pKF;
      }
    }
    
    // Update active landmarks
    std::vector<MapPoint*> all_mps = mpMap->GetAllMapPoints();
    for (MapPoint* pMP : all_mps) {
      if (pMP && !pMP->isBad()) {
        active_landmarks_[pMP->mnId] = pMP;
      }
    }
  }

  // Draw map points
  if (settings_.show_map_points) {
    DrawMapPoints();
  }

  // Draw keyframes
  if (settings_.show_keyframes) {
    DrawKeyFrames();
  }

  // Draw trajectory
  if (settings_.show_trajectory) {
    DrawTrajectory();
  }

  // Draw current frame
  if (settings_.show_current_frame) {
    DrawCurrentFrame();
  }
}

void RerunViewer::LogFrame(const cv::Mat& image,
                           const std::vector<cv::KeyPoint>& keypoints,
                           const cv::Mat& pose, double timestamp) {
  {
    std::unique_lock<std::mutex> lock(mMutexCurrentFrame);
    current_image_ = image.clone();
    current_keypoints_ = keypoints;
    current_pose_ = pose.clone();
    current_timestamp_ = timestamp;
  }

  // Set timeline for this frame
  rec_->set_time_seconds("timeline", timestamp);

  // Log the image
  if (!image.empty()) {
    // Convert OpenCV image to Rerun format
    rerun::archetypes::Image rerun_image;
    if (image.channels() == 3) {
      cv::Mat rgb_image;
      cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
      std::vector<uint8_t> rgb_data(
          rgb_image.data,
          rgb_image.data + rgb_image.total() * rgb_image.elemSize());
      rerun_image = rerun::archetypes::Image::from_rgb24(
          rgb_data, {static_cast<size_t>(rgb_image.cols),
                     static_cast<size_t>(rgb_image.rows)});
    } else {
      std::vector<uint8_t> gray_data(
          image.data, image.data + image.total() * image.elemSize());
      rerun_image = rerun::archetypes::Image::from_greyscale8(
          gray_data,
          {static_cast<size_t>(image.cols), static_cast<size_t>(image.rows)});
    }

    rec_->log("camera/image", rerun_image);

    // Log keypoints on the image - ensure they are always visible
    if (!keypoints.empty()) {
      std::vector<rerun::Position2D> points;
      std::vector<float> radii;
      std::vector<rerun::Color> colors;

      for (const auto& kp : keypoints) {
        points.emplace_back(kp.pt.x, kp.pt.y);
        radii.push_back(1.0f);  // Small fixed radius for keypoints
        // Better color mapping for visibility
        float intensity = std::min(1.0f, std::max(0.3f, kp.response * 2.0f));
        colors.emplace_back(static_cast<uint8_t>(255 * intensity), 255, 50);
      }

      rec_->log("camera/keypoints",
                rerun::Points2D(points).with_radii(radii).with_colors(colors));
      
      std::cout << "RerunViewer: Logged " << keypoints.size() << " keypoints at timestamp " << timestamp << "\n";
    } else {
      // Clear previous keypoints if none detected
      rec_->log("camera/keypoints", rerun::Clear());
      std::cout << "RerunViewer: No keypoints detected\n";
    }
  }

  // Log camera pose and trajectory
  if (!pose.empty()) {
    LogTrajectory(pose, timestamp);
  }
}

void RerunViewer::LogTrajectory(const cv::Mat& pose, double timestamp) {
  if (pose.empty()) return;

  // Set timeline for pose logging
  rec_->set_time_seconds("timeline", timestamp);

  {
    std::unique_lock<std::mutex> lock(mMutexTrajectory);
    
    // Extract position from pose matrix - convert from Tcw to world position
    cv::Mat R = pose.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = pose.rowRange(0, 3).col(3);
    cv::Mat pos_world = -R.t() * t;  // Convert camera pose to world position
    
    cv::Point3f position(pos_world.at<float>(0), pos_world.at<float>(1),
                         pos_world.at<float>(2));
    trajectory_points_.push_back(position);
    trajectory_timestamps_.push_back(timestamp);

    // Limit trajectory length to avoid memory issues
    const size_t max_trajectory_points = 10000;
    if (trajectory_points_.size() > max_trajectory_points) {
      trajectory_points_.erase(trajectory_points_.begin());
      trajectory_timestamps_.erase(trajectory_timestamps_.begin());
    }
  }

  // Log current camera pose with better entity path
  rec_->log("world/camera", cvMatToRerunTransform(pose));
  
  // Also log the trajectory path in real-time
  DrawTrajectory();
  
  std::cout << "RerunViewer: Logged pose at timestamp " << timestamp << ", position: (" 
            << trajectory_points_.back().x << ", " << trajectory_points_.back().y 
            << ", " << trajectory_points_.back().z << ")\n";
}

void RerunViewer::LogMapPoints(const std::vector<MapPoint*>& map_points) {
  std::vector<rerun::Position3D> points;
  std::vector<rerun::Color> colors;

  for (MapPoint* pMP : map_points) {
    if (!pMP || pMP->isBad()) continue;

    cv::Mat pos = pMP->GetWorldPos();
    if (pos.empty() || pos.rows < 3) continue;
    
    points.emplace_back(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));

    // Color by observations count
    int obs = pMP->Observations();
    uint8_t intensity = static_cast<uint8_t>(std::min(255, std::max(50, obs * 20)));
    colors.emplace_back(255 - intensity, intensity, 100);  // Red to green gradient
  }

  if (!points.empty()) {
    rec_->log("world/map_points",
              rerun::Points3D(points).with_colors(colors).with_radii(0.03f));
    std::cout << "RerunViewer: Logged " << points.size() << " map points\n";
  }
}

void RerunViewer::LogKeyFrame(KeyFrame* pKF) {
  if (!pKF || pKF->isBad()) return;

  cv::Mat pose = pKF->GetPose();
  if (pose.empty()) return;

  // Log keyframe pose
  std::string entity_path = "world/keyframes/" + std::to_string(pKF->mnId);
  rec_->log(entity_path, cvMatToRerunTransform(pose));

  // Log keyframe connections (covisibility graph)
  std::vector<KeyFrame*> connected_kfs = pKF->GetVectorCovisibleKeyFrames();
  if (!connected_kfs.empty()) {
    std::vector<rerun::Position3D> line_points;
    cv::Mat current_pos = pKF->GetCameraCenter();

    for (KeyFrame* pConnectedKF : connected_kfs) {
      if (!pConnectedKF || pConnectedKF->isBad()) continue;

      cv::Mat connected_pos = pConnectedKF->GetCameraCenter();
      line_points.emplace_back(current_pos.at<float>(0),
                               current_pos.at<float>(1),
                               current_pos.at<float>(2));
      line_points.emplace_back(connected_pos.at<float>(0),
                               connected_pos.at<float>(1),
                               connected_pos.at<float>(2));
    }

    if (!line_points.empty()) {
      rec_->log(
          "world/covisibility_graph",
          rerun::LineStrips3D({line_points})
              .with_colors({{100, 100, 100, 128}}));  // Semi-transparent gray
    }
  }
}

void RerunViewer::LogFeatureMatches(const std::vector<cv::KeyPoint>& kpts1,
                                    const std::vector<cv::KeyPoint>& kpts2,
                                    const std::vector<cv::DMatch>& matches) {
  if (matches.empty()) return;

  // Log matched features
  std::vector<rerun::Position2D> points1, points2;
  std::vector<rerun::Color> colors;

  for (const auto& match : matches) {
    if (match.queryIdx < kpts1.size() && match.trainIdx < kpts2.size()) {
      points1.emplace_back(kpts1[match.queryIdx].pt.x,
                           kpts1[match.queryIdx].pt.y);
      points2.emplace_back(kpts2[match.trainIdx].pt.x,
                           kpts2[match.trainIdx].pt.y);

      // Color by match distance/quality
      uint8_t quality =
          static_cast<uint8_t>(255 - std::min(255.0f, match.distance * 5.0f));
      colors.emplace_back(quality, 255, 0);
    }
  }

  rec_->log("matches/frame1", rerun::Points2D(points1).with_colors(colors));
  rec_->log("matches/frame2", rerun::Points2D(points2).with_colors(colors));
}

void RerunViewer::LogSuperPointFeatures(
    const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& descriptors) {
  // Log image with SuperPoint features
  LogFrame(image, keypoints, cv::Mat(), 0.0);

  // Log additional SuperPoint-specific data
  if (!descriptors.empty()) {
    // Log descriptor statistics
    cv::Scalar mean, stddev;
    cv::meanStdDev(descriptors, mean, stddev);

    rec_->log("superpoint/descriptor_stats/mean", rerun::Scalar(mean[0]));
    rec_->log("superpoint/descriptor_stats/stddev", rerun::Scalar(stddev[0]));
    rec_->log("superpoint/num_features",
              rerun::Scalar(static_cast<double>(keypoints.size())));
  }
}

void RerunViewer::LogLoopClosure(KeyFrame* pKF1, KeyFrame* pKF2,
                                 const std::vector<cv::DMatch>& matches) {
  if (!pKF1 || !pKF2) return;

  // Log loop closure connection
  cv::Mat pos1 = pKF1->GetCameraCenter();
  cv::Mat pos2 = pKF2->GetCameraCenter();

  std::vector<rerun::Position3D> loop_line = {
      {pos1.at<float>(0), pos1.at<float>(1), pos1.at<float>(2)},
      {pos2.at<float>(0), pos2.at<float>(1), pos2.at<float>(2)}};

  std::string entity_path = "world/loop_closures/" +
                            std::to_string(pKF1->mnId) + "_" +
                            std::to_string(pKF2->mnId);
  rec_->log(entity_path,
            rerun::LineStrips3D({loop_line})
                .with_colors({{255, 0, 255}}));  // Magenta for loop closures

  rec_->log("loop_closure/matches_count",
            rerun::Scalar(static_cast<double>(matches.size())));
}

void RerunViewer::DrawTrajectory() {
  std::unique_lock<std::mutex> lock(mMutexTrajectory);

  if (trajectory_points_.size() < 2) {
    // Log single point if we only have one pose
    if (trajectory_points_.size() == 1) {
      const auto& point = trajectory_points_[0];
      rec_->log("world/trajectory_point", 
                rerun::Points3D({{point.x, point.y, point.z}})
                  .with_colors({{0, 255, 0}})  // Green
                  .with_radii(0.05f));
    }
    return;
  }

  // Draw trajectory as line strips
  std::vector<rerun::Position3D> trajectory;
  for (const auto& point : trajectory_points_) {
    trajectory.emplace_back(point.x, point.y, point.z);
  }

  rec_->log("world/trajectory", 
            rerun::LineStrips3D({trajectory})
              .with_colors({{0, 255, 0}})  // Green trajectory
              .with_radii(0.01f));  // Visible line width
              
  // Also draw current position as a larger point
  if (!trajectory_points_.empty()) {
    const auto& current = trajectory_points_.back();
    rec_->log("world/current_position",
              rerun::Points3D({{current.x, current.y, current.z}})
                .with_colors({{255, 255, 0}})  // Yellow for current
                .with_radii(0.1f));
  }
}

void RerunViewer::DrawMapPoints() {
  if (!mpMap) return;

  std::vector<MapPoint*> map_points = mpMap->GetAllMapPoints();
  LogMapPoints(map_points);
}

void RerunViewer::DrawKeyFrames() {
  if (!mpMap) return;

  std::vector<KeyFrame*> keyframes = mpMap->GetAllKeyFrames();
  for (KeyFrame* pKF : keyframes) {
    if (!pKF || pKF->isBad()) continue;
    LogKeyFrame(pKF);
  }
}

void RerunViewer::DrawCurrentFrame() {
  std::unique_lock<std::mutex> lock(mMutexCurrentFrame);

  if (!current_pose_.empty()) {
    rec_->log("world/current_camera", cvMatToRerunTransform(current_pose_));
  }
}

rerun::Position3D RerunViewer::cvMatToRerunPosition(const cv::Mat& pose) {
  if (pose.rows < 3 || pose.cols < 4) {
    return {0.0f, 0.0f, 0.0f};
  }
  return {pose.at<float>(0, 3), pose.at<float>(1, 3), pose.at<float>(2, 3)};
}

rerun::Transform3D RerunViewer::cvMatToRerunTransform(const cv::Mat& pose) {
  if (pose.rows != 4 || pose.cols != 4) {
    return rerun::Transform3D();
  }

  // Extract rotation and translation
  cv::Mat R = pose.rowRange(0, 3).colRange(0, 3);
  cv::Mat t = pose.rowRange(0, 3).col(3);

  // Convert from camera pose (Tcw) to world pose (Twc)
  cv::Mat Rwc = R.t();
  cv::Mat twc = -Rwc * t;

  // Convert to Rerun format - create translation and rotation matrix
  rerun::datatypes::Vec3D translation{twc.at<float>(0), twc.at<float>(1),
                                      twc.at<float>(2)};

  std::array<float, 9> rot_data = {
      Rwc.at<float>(0, 0), Rwc.at<float>(0, 1), Rwc.at<float>(0, 2),
      Rwc.at<float>(1, 0), Rwc.at<float>(1, 1), Rwc.at<float>(1, 2),
      Rwc.at<float>(2, 0), Rwc.at<float>(2, 1), Rwc.at<float>(2, 2)};
  rerun::datatypes::Mat3x3 rotation(rot_data);

  return rerun::Transform3D(translation, rotation);
}

// Control functions
void RerunViewer::RequestFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

void RerunViewer::RequestStop() {
  std::unique_lock<std::mutex> lock(mMutexStop);
  if (!mbStopped) mbStopRequested = true;
}

bool RerunViewer::isFinished() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  return mbFinished;
}

bool RerunViewer::isStopped() {
  std::unique_lock<std::mutex> lock(mMutexStop);
  return mbStopped;
}

void RerunViewer::Release() {
  RequestFinish();
  if (mptViewer && mptViewer->joinable()) {
    mptViewer->join();
  }
}

bool RerunViewer::CheckFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void RerunViewer::SetFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  mbFinished = true;
}

bool RerunViewer::Stop() {
  std::unique_lock<std::mutex> lock(mMutexStop);
  std::unique_lock<std::mutex> lock2(mMutexFinish);

  if (mbFinishRequested) return false;
  if (mbStopRequested) {
    mbStopped = true;
    mbStopRequested = false;
    return true;
  }
  return false;
}

}  // namespace SuperSLAM