#include "RerunViewer.h"

#include <algorithm>  // For std::sort
#include <chrono>     // For std::chrono
#include <opencv2/opencv.hpp>
#include <string>  // For std::string
#include <thread>  // For std::thread
#include <vector>  // For std::vector

#include "KeyFrame.h"  // For KeyFrame
#include "Logging.h"   // For SLOG_INFO etc.
#include "Map.h"       // For Map
#include "MapPoint.h"  // For MapPoint

namespace SuperSLAM  // Changed from slam to SuperSLAM
{
// Constructor
RerunViewer::RerunViewer()
    : mT(33.0)  // Initialize mT with a default value
{
  rec.spawn().exit_on_failure();

  // World origin
  rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);  // Set an up-axis
  
  // Create blueprint for stereo view
  rec.log_static("world/current_left_camera", rerun::ViewCoordinates::RDF);
  rec.log_static("world/current_right_camera", rerun::ViewCoordinates::RDF);

  /* For keyframe plot their highest similarity score with other
   * keyframes obtained by deep network */
  rec.log_static("plots/loop_deep_score",
                 rerun::SeriesLine()
                     .with_color({255, 0, 0})
                     .with_name("Loop Closure Deep Score")
                     .with_width(2));
  /* For plotting ratio of inlier in total landmarks after frontend optimization
   */
  rec.log_static("plots/frontend_inlier_ratio",
                 rerun::SeriesLine()
                     .with_color({0, 255, 255})
                     .with_name("Frontend landmark inlier ratio")
                     .with_width(2));

  // Configure automatic viewport display for camera streams
  // This ensures camera images are automatically shown in the viewport
  rec.log_static("world/current_camera", rerun::ViewCoordinates::RDF);
  rec.log_static("world/current_left_camera", rerun::ViewCoordinates::RDF);  
  rec.log_static("world/current_right_camera", rerun::ViewCoordinates::RDF);
  
  // Log initial placeholder content to ensure entities are created and visible
  // This helps Rerun automatically select them for display
  std::vector<uint8_t> placeholder_data(640 * 480, 128);  // Gray placeholder
  rerun::WidthHeight placeholder_res(640, 480);
  rec.log("world/current_camera", rerun::Image(rerun::Collection<uint8_t>(placeholder_data), placeholder_res, rerun::datatypes::ColorModel::L));
  rec.log("world/current_left_camera", rerun::Image(rerun::Collection<uint8_t>(placeholder_data), placeholder_res, rerun::datatypes::ColorModel::L));
  rec.log("world/current_right_camera", rerun::Image(rerun::Collection<uint8_t>(placeholder_data), placeholder_res, rerun::datatypes::ColorModel::L));

  // Set custom timing for illustrating data
  rec.set_time_sequence("max_keyframe_id", 0);

  // Set custom timing for illustrating data
  rec.set_time_sequence("currentframe_id", 0);
  SLOG_INFO("RerunViewer initialized");
}

void RerunViewer::SetMap(Map* map) { 
  map_ = map; 
}

void RerunViewer::SetCameras(float fx_left, float fy_left, float cx_left, float cy_left,
                            float fx_right, float fy_right, float cx_right, float cy_right,
                            float baseline) {
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

// void Viewer::SetCameras(Camera::Ptr cameraLeft, Camera::Ptr cameraRight)
// {
//     camera_left_ = cameraLeft;
//     camera_right_ = cameraRight;
// }

void RerunViewer::Close() {
  UpdateMap();
  rec.log("world/log", rerun::TextLog("Finished"));
  SLOG_INFO("Viewer Close requested");
}

void RerunViewer::AddCurrentFrame(Frame* current_frame) {
  std::unique_lock<std::mutex> lck(viewer_data_mutex_);
  current_frame_ = current_frame;
  
  // Log current frame image immediately for real-time visualization (even during initialization)
  if (current_frame_ && !current_frame_->mImGray.empty()) {
    rec.set_time_sequence("currentframe_id", current_frame_->mnId);
    
    std::string entity_name = "world/current_camera";
    cv::Mat img = current_frame_->mImGray;
    
    // Set camera model if available
    if (cameras_set_) {
      rec.log(entity_name, rerun::Pinhole::from_focal_length_and_resolution(
        {fx_left_, fy_left_}, {static_cast<float>(img.cols), static_cast<float>(img.rows)}));
    } else if (!current_frame_->mK.empty()) {
      float fx = current_frame_->mK.at<float>(0, 0);
      float fy = current_frame_->mK.at<float>(1, 1);
      rec.log(entity_name, rerun::Pinhole::from_focal_length_and_resolution(
        {fx, fy}, {static_cast<float>(img.cols), static_cast<float>(img.rows)}));
    }
    
    // Log image
    if (img.channels() == 1) {
      std::vector<uint8_t> img_data(img.ptr(), img.ptr() + img.total());
      rerun::WidthHeight resolution(img.cols, img.rows);
      rec.log(entity_name, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::L));
    } else if (img.channels() == 3) {
      cv::Mat rgb_img;
      cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
      std::vector<uint8_t> img_data(rgb_img.ptr(), rgb_img.ptr() + rgb_img.total());
      rerun::WidthHeight resolution(rgb_img.cols, rgb_img.rows);
      rec.log(entity_name, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::RGB));
    }
    
    // Log keypoints
    if (!current_frame_->mvKeysUn.empty()) {
      std::vector<rerun::Position2D> keypoint_positions;
      for (const auto& kp : current_frame_->mvKeysUn) {
        keypoint_positions.push_back(rerun::Position2D{kp.pt.x, kp.pt.y});
      }
      if (!keypoint_positions.empty()) {
        rec.log(entity_name + "/keypoints", 
               rerun::Points2D(keypoint_positions)
                 .with_colors(rerun::Color{0, 255, 0})  // Green keypoints
                 .with_radii(3.0f));
      }
    }
  }
}

void RerunViewer::AddStereoFrames(Frame* current_frame, const cv::Mat& left_image, const cv::Mat& right_image) {
  std::unique_lock<std::mutex> lck(viewer_data_mutex_);
  current_frame_ = current_frame;
  current_left_image_ = left_image.clone();
  current_right_image_ = right_image.clone();
  has_stereo_images_ = true;
  
  // Log images immediately for real-time visualization (even during initialization)
  if (current_frame_) {
    rec.set_time_sequence("currentframe_id", current_frame_->mnId);
    
    // Log left camera image
    if (!left_image.empty()) {
      std::string left_entity = "world/current_left_camera";
      SLOG_DEBUG("Logging left camera image: {}x{}", left_image.cols, left_image.rows);
      
      // Set camera model
      if (cameras_set_) {
        rec.log(left_entity, rerun::Pinhole::from_focal_length_and_resolution(
          {fx_left_, fy_left_}, {static_cast<float>(left_image.cols), static_cast<float>(left_image.rows)}));
      }
      
      // Log left image
      if (left_image.channels() == 1) {
        std::vector<uint8_t> img_data(left_image.ptr(), left_image.ptr() + left_image.total());
        rerun::WidthHeight resolution(left_image.cols, left_image.rows);
        rec.log(left_entity, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::L));
      } else if (left_image.channels() == 3) {
        cv::Mat rgb_img;
        cv::cvtColor(left_image, rgb_img, cv::COLOR_BGR2RGB);
        std::vector<uint8_t> img_data(rgb_img.ptr(), rgb_img.ptr() + rgb_img.total());
        rerun::WidthHeight resolution(rgb_img.cols, rgb_img.rows);
        rec.log(left_entity, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::RGB));
      }
      
      // Log left keypoints
      if (!current_frame_->mvKeysUn.empty()) {
        std::vector<rerun::Position2D> keypoint_positions;
        for (const auto& kp : current_frame_->mvKeysUn) {
          keypoint_positions.push_back(rerun::Position2D{kp.pt.x, kp.pt.y});
        }
        if (!keypoint_positions.empty()) {
          rec.log(left_entity + "/keypoints", 
                 rerun::Points2D(keypoint_positions)
                   .with_colors(rerun::Color{0, 255, 0})  // Green keypoints
                   .with_radii(3.0f));
          SLOG_DEBUG("Logged {} keypoints on left camera", keypoint_positions.size());
        }
      }
    }
    
    // Log right camera image
    if (!right_image.empty()) {
      std::string right_entity = "world/current_right_camera";
      SLOG_DEBUG("Logging right camera image: {}x{}", right_image.cols, right_image.rows);
      
      // Set camera model
      if (cameras_set_) {
        rec.log(right_entity, rerun::Pinhole::from_focal_length_and_resolution(
          {fx_right_, fy_right_}, {static_cast<float>(right_image.cols), static_cast<float>(right_image.rows)}));
      }
      
      // Log right image
      if (right_image.channels() == 1) {
        std::vector<uint8_t> img_data(right_image.ptr(), right_image.ptr() + right_image.total());
        rerun::WidthHeight resolution(right_image.cols, right_image.rows);
        rec.log(right_entity, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::L));
      } else if (right_image.channels() == 3) {
        cv::Mat rgb_img;
        cv::cvtColor(right_image, rgb_img, cv::COLOR_BGR2RGB);
        std::vector<uint8_t> img_data(rgb_img.ptr(), rgb_img.ptr() + rgb_img.total());
        rerun::WidthHeight resolution(rgb_img.cols, rgb_img.rows);
        rec.log(right_entity, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::RGB));
      }
      
      // Log right keypoints
      if (!current_frame_->mvKeysRight.empty()) {
        std::vector<rerun::Position2D> keypoint_positions;
        for (const auto& kp : current_frame_->mvKeysRight) {
          keypoint_positions.push_back(rerun::Position2D{kp.pt.x, kp.pt.y});
        }
        if (!keypoint_positions.empty()) {
          rec.log(right_entity + "/keypoints", 
                 rerun::Points2D(keypoint_positions)
                   .with_colors(rerun::Color{255, 0, 0})  // Red keypoints for right camera
                   .with_radii(3.0f));
          SLOG_DEBUG("Logged {} keypoints on right camera", keypoint_positions.size());
        }
      }
    }
  }
}

// Helper to convert cv::Mat pose to Rerun Transform3D (world from camera)
rerun::Transform3D CvMatToRerunTransform(const cv::Mat& T_wc) {
  if (T_wc.empty() || T_wc.rows != 4 || T_wc.cols != 4 ||
      T_wc.type() != CV_32F) {
    SLOG_WARN("CvMatToRerunTransform: Invalid pose matrix provided.");
    return rerun::Transform3D();  // Return identity or default
  }

  // Assuming T_wc is world_from_camera (Pose of camera in world frame)
  cv::Mat R_wc = T_wc.rowRange(0, 3).colRange(0, 3);
  cv::Mat t_wc = T_wc.rowRange(0, 3).col(3);

  std::array<float, 9> rot_data;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) rot_data[i * 3 + j] = R_wc.at<float>(i, j);

  return rerun::Transform3D(
      {t_wc.at<float>(0), t_wc.at<float>(1), t_wc.at<float>(2)},
      rerun::datatypes::Mat3x3(rot_data));
}

void RerunViewer::UpdateMap() {
  std::unique_lock<std::mutex> lck(viewer_data_mutex_);
  if (!map_) {
    SLOG_WARN("Viewer::UpdateMap - Map pointer is null.");
    return;
  }

  // Populate all_keyframes_ from map_->GetAllKeyFrames()
  all_keyframes_.clear();
  std::vector<KeyFrame*> vAllKFs = map_->GetAllKeyFrames();
  for (KeyFrame* pKF : vAllKFs) {
    if (pKF) {
      all_keyframes_[pKF->mnId] = pKF;
    }
  }

  // For now, active_keyframes_ is a copy of all_keyframes_
  // TODO: Implement proper active keyframe logic based on SuperSLAM's requirements
  active_keyframes_ = all_keyframes_;

  // Populate active_landmarks_ from map_->GetAllMapPoints()
  active_landmarks_.clear();
  std::vector<MapPoint*> vAllMPs = map_->GetAllMapPoints();
  for (MapPoint* pMP : vAllMPs) {
    if (pMP && !pMP->isBad()) {
      active_landmarks_[pMP->mnId] = pMP;
    }
  }

  if (active_keyframes_.empty()) {
    SLOG_INFO("Viewer::UpdateMap - No active keyframes to display.");
    return;
  }

  // Order active keyframes according to id in decreasing order
  // The 0-index is most recent active keyframe
  std::vector<std::pair<unsigned long, KeyFrame*>> kf_sort;
  for (auto const& [id, kf_ptr] : active_keyframes_) {
    if (kf_ptr) {
      kf_sort.push_back({id, kf_ptr});
    }
  }
  std::sort(kf_sort.begin(), kf_sort.end(), 
            [](const std::pair<unsigned long, KeyFrame*>& a, 
               const std::pair<unsigned long, KeyFrame*>& b) {
              return a.first > b.first;
            });

  if (kf_sort.empty()) {
    SLOG_INFO("Viewer::UpdateMap - Sorted active keyframes list is empty.");
    return;
  }

  // Use most recent active keyframe id as sequence number for visualization
  rec.set_time_sequence("max_keyframe_id", kf_sort[0].first);

  // Get pose of most recent keyframe for coordinate transformation
  cv::Mat T_wc0 = kf_sort[0].second->GetPose(); // Camera to world transform
  cv::Mat T_cw0 = T_wc0.inv(); // World to camera transform (inverse)
  
  // Draw all active keyframes in coordinate of most recent active keyframe
  for (size_t i = 0; i < kf_sort.size(); ++i) {
    KeyFrame* kf_i = kf_sort[i].second;
    if (!kf_i) continue;
    
    std::string entity_name = "world/stereosys" + std::to_string(i) + "/cam_left";
    
    if (i != 0) {
      // Transform i'th keyframe to coordinate of most recent active frame
      cv::Mat T_wci = kf_i->GetPose(); // Camera to world transform for keyframe i
      cv::Mat T_cic0 = T_cw0 * T_wci; // Transform from keyframe i to keyframe 0
      
      // Extract translation and rotation
      cv::Mat t_cic0 = T_cic0.rowRange(0, 3).col(3);
      cv::Mat R_cic0 = T_cic0.rowRange(0, 3).colRange(0, 3);
      
      const Eigen::Vector3f camera_position(t_cic0.at<float>(0), t_cic0.at<float>(1), t_cic0.at<float>(2));
      Eigen::Matrix3f camera_orientation;
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          camera_orientation(row, col) = R_cic0.at<float>(row, col);
        }
      }
      
      rec.log(entity_name, rerun::Transform3D(
        rerun::Vec3D(camera_position.data()),
        rerun::Mat3x3(camera_orientation.data()), true));
    }

    // Left camera pinhole model
    float fx, fy, cx, cy;
    if (cameras_set_) {
      fx = fx_left_;
      fy = fy_left_;
      cx = cx_left_;
      cy = cy_left_;
    } else if (!kf_i->mK.empty()) {
      fx = kf_i->mK.at<float>(0, 0);
      fy = kf_i->mK.at<float>(1, 1);
      cx = kf_i->mK.at<float>(0, 2);
      cy = kf_i->mK.at<float>(1, 2);
    } else {
      // Default values if no camera parameters available
      fx = fy = 500.0f;
      cx = cy = 320.0f;
    }
    
    // Estimate image dimensions (use defaults if not available)
    float img_num_cols = 640.0f;
    float img_num_rows = 480.0f;
    
    rec.log(entity_name, rerun::Pinhole::from_focal_length_and_resolution(
      {fx, fy}, {img_num_cols, img_num_rows}));

    // For newest active keyframe, display camera images
    if (i == 0) {
      // Left camera image
      cv::Mat left_img;
      if (has_stereo_images_ && !current_left_image_.empty()) {
        left_img = current_left_image_;
      } else if (current_frame_ && !current_frame_->mImGray.empty()) {
        left_img = current_frame_->mImGray;
      }
      
      if (!left_img.empty()) {
        if (left_img.channels() == 1) {
          // Convert grayscale to rerun format
          std::vector<uint8_t> img_data(left_img.ptr(), left_img.ptr() + left_img.total());
          rerun::WidthHeight resolution(left_img.cols, left_img.rows);
          rec.log(entity_name, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::L));
        } else if (left_img.channels() == 3) {
          cv::Mat rgb_img;
          cv::cvtColor(left_img, rgb_img, cv::COLOR_BGR2RGB);
          std::vector<uint8_t> img_data(rgb_img.ptr(), rgb_img.ptr() + rgb_img.total());
          rerun::WidthHeight resolution(rgb_img.cols, rgb_img.rows);
          rec.log(entity_name, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::RGB));
        }
        
        // Add keypoints overlay on left camera
        if (current_frame_ && !current_frame_->mvKeysUn.empty()) {
          std::vector<rerun::Position2D> keypoint_positions;
          for (const auto& kp : current_frame_->mvKeysUn) {
            keypoint_positions.push_back(rerun::Position2D{kp.pt.x, kp.pt.y});
          }
          if (!keypoint_positions.empty()) {
            rec.log(entity_name + "/keypoints", 
                   rerun::Points2D(keypoint_positions)
                     .with_colors(rerun::Color{0, 255, 0})  // Green keypoints
                     .with_radii(3.0f));
            SLOG_INFO("Logged {} keypoints on left camera", keypoint_positions.size());
          }
        }
      }
      
      // Right camera image (for stereo)
      if (has_stereo_images_ && !current_right_image_.empty()) {
        std::string right_entity_name = "world/stereosys" + std::to_string(i) + "/cam_right";
        
        // Right camera pinhole model (same as left but different cx)
        if (cameras_set_) {
          rec.log(right_entity_name, rerun::Pinhole::from_focal_length_and_resolution(
            {fx_right_, fy_right_}, {img_num_cols, img_num_rows}));
        }
        
        cv::Mat right_img = current_right_image_;
        if (right_img.channels() == 1) {
          // Convert grayscale to rerun format
          std::vector<uint8_t> img_data(right_img.ptr(), right_img.ptr() + right_img.total());
          rerun::WidthHeight resolution(right_img.cols, right_img.rows);
          rec.log(right_entity_name, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::L));
        } else if (right_img.channels() == 3) {
          cv::Mat rgb_img;
          cv::cvtColor(right_img, rgb_img, cv::COLOR_BGR2RGB);
          std::vector<uint8_t> img_data(rgb_img.ptr(), rgb_img.ptr() + rgb_img.total());
          rerun::WidthHeight resolution(rgb_img.cols, rgb_img.rows);
          rec.log(right_entity_name, rerun::Image(rerun::Collection<uint8_t>(img_data), resolution, rerun::datatypes::ColorModel::RGB));
        }
        
        // Add keypoints overlay on right camera  
        if (current_frame_ && !current_frame_->mvKeysRight.empty()) {
          std::vector<rerun::Position2D> keypoint_positions;
          for (const auto& kp : current_frame_->mvKeysRight) {
            keypoint_positions.push_back(rerun::Position2D{kp.pt.x, kp.pt.y});
          }
          if (!keypoint_positions.empty()) {
            rec.log(right_entity_name + "/keypoints", 
                   rerun::Points2D(keypoint_positions)
                     .with_colors(rerun::Color{255, 0, 0})  // Red keypoints for right camera
                     .with_radii(3.0f));
            SLOG_INFO("Logged {} keypoints on right camera", keypoint_positions.size());
          }
        }
      }
    }
  }

  // Draw active landmarks (map points) in coordinate of newest active keyframe
  if (!kf_sort.empty()) {
    cv::Mat T_wc0 = kf_sort[0].second->GetPose();
    cv::Mat T_cw0 = T_wc0.inv();
    cv::Mat t_cw0 = T_cw0.rowRange(0, 3).col(3);
    cv::Mat R_cw0 = T_cw0.rowRange(0, 3).colRange(0, 3);
    
    const Eigen::Vector3f camera_position(t_cw0.at<float>(0), t_cw0.at<float>(1), t_cw0.at<float>(2));
    Eigen::Matrix3f camera_orientation;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        camera_orientation(row, col) = R_cw0.at<float>(row, col);
      }
    }
    
    rec.log("world/landmarks", rerun::Transform3D(
      rerun::Vec3D(camera_position.data()),
      rerun::Mat3x3(camera_orientation.data()), true));
    
    std::vector<rerun::Position3D> points3d_vector;
    for (auto const& [id, mp] : active_landmarks_) {
      if (mp && !mp->isBad()) {
        cv::Mat world_pos = mp->GetWorldPos();
        points3d_vector.push_back(rerun::Position3D(
          world_pos.at<float>(0), world_pos.at<float>(1), world_pos.at<float>(2)));
      }
    }
    if (!points3d_vector.empty()) {
      rec.log("world/landmarks", rerun::Points3D(points3d_vector));
    }

    // Draw trajectory in world coordinates (no transform needed)
    std::vector<rerun::datatypes::Vec3D> path;
    
    // Collect all keyframes and sort by ID for proper trajectory order
    std::vector<std::pair<unsigned long, KeyFrame*>> all_kf_sorted;
    for (auto const& [id, kf] : all_keyframes_) {
      if (kf) {
        all_kf_sorted.push_back({id, kf});
      }
    }
    std::sort(all_kf_sorted.begin(), all_kf_sorted.end(),
              [](const std::pair<unsigned long, KeyFrame*>& a, 
                 const std::pair<unsigned long, KeyFrame*>& b) {
                return a.first < b.first;
              });
    
    // Build trajectory from keyframe poses
    for (const auto& [id, kf] : all_kf_sorted) {
      cv::Mat T_wc = kf->GetPose();
      if (!T_wc.empty() && T_wc.rows == 4 && T_wc.cols == 4) {
        cv::Mat t_wc = T_wc.rowRange(0, 3).col(3);
        path.emplace_back(rerun::datatypes::Vec3D{
          t_wc.at<float>(0), t_wc.at<float>(1), t_wc.at<float>(2)
        });
      }
    }
    
    if (path.size() > 1) {
      rec.log("world/trajectory", rerun::LineStrips3D(rerun::LineStrip3D(path)));
      SLOG_INFO("Logged trajectory with {} points", path.size());
    }
  }
}

void RerunViewer::LogInfo(std::string msg, std::string log_type) {
  if (current_frame_) {
    rec.set_time_sequence("currentframe_id", current_frame_->mnId);
  } else {
    rec.set_time_sequence("currentframe_id", 0);
  }

  auto it = log_color.find(log_type);
  if (it != log_color.end()) {
    rec.log("world/log", rerun::TextLog(msg).with_color(it->second));
  } else {
    rec.log("world/log", rerun::TextLog(msg));
  }
}

void RerunViewer::LogInfoMKF(std::string msg, unsigned long maxkeyframe_id,
                             std::string log_type) {
  if (current_frame_) {
    rec.set_time_sequence("currentframe_id", current_frame_->mnId);
  } else {
    rec.set_time_sequence("currentframe_id", 0);
  }
  rec.set_time_sequence("max_keyframe_id", maxkeyframe_id);

  auto it = log_color.find(log_type);
  if (it != log_color.end()) {
    rec.log("world/log", rerun::TextLog(msg).with_color(it->second));
  } else {
    rec.log("world/log", rerun::TextLog(msg));
  }
}

void RerunViewer::Plot(std::string plot_name, double value,
                       unsigned long maxkeyframe_id) {
  if (current_frame_) {
    rec.set_time_sequence("currentframe_id", current_frame_->mnId);
  } else {
    rec.set_time_sequence("currentframe_id", 0);
  }
  rec.set_time_sequence("max_keyframe_id", maxkeyframe_id);
  rec.log(plot_name, rerun::Scalar(value));
}

void RerunViewer::LogKeyFrame(KeyFrame* keyframe) {
  if (!keyframe) return;
  
  std::unique_lock<std::mutex> lck(viewer_data_mutex_);
  
  // Add keyframe to our tracking
  all_keyframes_[keyframe->mnId] = keyframe;
  active_keyframes_[keyframe->mnId] = keyframe;
  
  // Set time sequence for this keyframe
  rec.set_time_sequence("max_keyframe_id", keyframe->mnId);
  
  // Log keyframe creation
  std::string msg = "New KeyFrame created with ID: " + std::to_string(keyframe->mnId);
  rec.log("world/log", rerun::TextLog(msg).with_color(log_color["vo"]));
  
  // Log keyframe pose
  cv::Mat T_wc = keyframe->GetPose();
  std::string entity_path = "world/keyframes/" + std::to_string(keyframe->mnId);
  rec.log(entity_path + "/pose", CvMatToRerunTransform(T_wc));
  
  // Log camera model if intrinsics are available
  if (!keyframe->mK.empty()) {
    float fx = keyframe->mK.at<float>(0, 0);
    float fy = keyframe->mK.at<float>(1, 1);
    float cx = keyframe->mK.at<float>(0, 2);
    float cy = keyframe->mK.at<float>(1, 2);
    
    std::array<float, 9> intrinsics_flat_array = {fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f};
    rec.log(
        entity_path + "/left_camera_model",
        rerun::archetypes::Pinhole(rerun::components::PinholeProjection(
                                       rerun::Mat3x3(intrinsics_flat_array)))
            .with_resolution(rerun::components::Resolution(640, 480))
    );
  }
}

// Implementation for methods from old RerunViewer for system integration
void RerunViewer::Run() {
  mbFinished = false;
  mbStopped = false;
  viewer_running_ =
      true;  // Align with example's variable if used, or stick to SuperSLAM's

  SLOG_INFO("RerunViewer: Starting viewer thread");

  while (viewer_running_) {     // Or use !mbFinishRequested
    if (this->CheckFinish()) {  // CheckFinish needs to be implemented
      break;
    }
    // if (Stop()) { // Stop needs to be implemented
    //     while (isStopped() && !CheckFinish()) {
    //         std::this_thread::sleep_for(std::chrono::milliseconds(3));
    //     }
    // }

    UpdateMap();  // Periodically update the map visualization

    // Sleep to maintain frame rate (or handle updates differently)
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(mT)));  // mT is from old viewer
  }
  this->SetFinish();  // SetFinish needs to be implemented
  SLOG_INFO("RerunViewer: Exiting viewer thread");
}

void RerunViewer::RequestFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  mbFinishRequested = true;
  viewer_running_ = false;  // To break Run() loop
}

bool RerunViewer::isFinished() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  return mbFinished;
}

void RerunViewer::RequestStop() {
  std::unique_lock<std::mutex> lock(mMutexStop);
  if (!mbStopped) {
    mbStopRequested = true;
  }
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
  delete mptViewer;
  mptViewer = nullptr;
  SLOG_INFO("RerunViewer: Released");
}

// Helper function definitions (needs to be part of the class or static)
bool RerunViewer::CheckFinish() {  // Example implementation
  std::unique_lock<std::mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void RerunViewer::SetFinish() {  // Example implementation
  std::unique_lock<std::mutex> lock(mMutexFinish);
  mbFinished = true;
}

bool RerunViewer::Stop() {  // Example implementation
  std::unique_lock<std::mutex> lock(mMutexStop);
  std::unique_lock<std::mutex> lock2(mMutexFinish);

  if (mbFinishRequested) {
    return false;
  }
  if (mbStopRequested) {
    mbStopped = true;
    mbStopRequested = false;
    return true;
  }
  return false;
}

}  // namespace SuperSLAM