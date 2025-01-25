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

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/core/core.hpp>

#include "../../../include/System.h"

using namespace std::chrono_literals;

class ImageGrabber : public rclcpp::Node {
public:
  ImageGrabber(const ImageGrabber &) = default;
  ImageGrabber(ImageGrabber &&) = default;
  ImageGrabber &operator=(const ImageGrabber &) = default;
  ImageGrabber &operator=(ImageGrabber &&) = default;
  ImageGrabber(std::shared_ptr<SuperSLAM::System> pSLAM, bool do_rectify)
      : Node("image_grabber"), mpSLAM(std::move(pSLAM)),
        do_rectify(do_rectify) {
    if (do_rectify) {
      // Load settings related to stereo calibration
      cv::FileStorage fsSettings(
          this->get_parameter("path_to_settings").as_string(),
          cv::FileStorage::READ);
      if (!fsSettings.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "ERROR: Wrong path to settings");
        throw std::runtime_error("Wrong path to settings");
      }

      cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
      fsSettings["LEFT.K"] >> K_l;
      fsSettings["RIGHT.K"] >> K_r;

      fsSettings["LEFT.P"] >> P_l;
      fsSettings["RIGHT.P"] >> P_r;

      fsSettings["LEFT.R"] >> R_l;
      fsSettings["RIGHT.R"] >> R_r;

      fsSettings["LEFT.D"] >> D_l;
      fsSettings["RIGHT.D"] >> D_r;

      int rows_l = fsSettings["LEFT.height"];
      int cols_l = fsSettings["LEFT.width"];
      int rows_r = fsSettings["RIGHT.height"];
      int cols_r = fsSettings["RIGHT.width"];

      if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() ||
          R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
          rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
        RCLCPP_ERROR(
            this->get_logger(),
            "ERROR: Calibration parameters to rectify stereo are missing!");
        throw std::runtime_error(
            "Calibration parameters to rectify stereo are missing!");
      }

      cv::initUndistortRectifyMap(K_l, D_l, R_l,
                                  P_l.rowRange(0, 3).colRange(0, 3),
                                  cv::Size(cols_l, rows_l), CV_32F, M1l, M2l);
      cv::initUndistortRectifyMap(K_r, D_r, R_r,
                                  P_r.rowRange(0, 3).colRange(0, 3),
                                  cv::Size(cols_r, rows_r), CV_32F, M1r, M2r);
    }

    left_sub_.subscribe(this, "/camera/left/image_raw");
    right_sub_.subscribe(this, "/camera/right/image_raw");

    sync_ = std::make_shared<message_filters::Synchronizer<sync_pol>>(
        sync_pol(10), left_sub_, right_sub_);
    sync_->registerCallback(std::bind(&ImageGrabber::GrabStereo, this,
                                      std::placeholders::_1,
                                      std::placeholders::_2));
  }

private:
  void GrabStereo(const sensor_msgs::msg::Image::ConstSharedPtr &msgLeft,
                  const sensor_msgs::msg::Image::ConstSharedPtr &msgRight);

  std::shared_ptr<SuperSLAM::System> mpSLAM;
  bool do_rectify;
  cv::Mat M1l, M2l, M1r, M2r;
  message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
  using sync_pol =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                      sensor_msgs::msg::Image>;
  std::shared_ptr<message_filters::Synchronizer<sync_pol>> sync_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  if (argc != 4) {
    std::cerr << "Usage: ros2 run SuperSLAM Stereo path_to_vocabulary "
                 "path_to_settings do_rectify"
              << std::endl;
    return 1;
  }

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  auto SLAM = std::make_shared<SuperSLAM::System>(
      argv[1], argv[2], SuperSLAM::System::STEREO, true);

  bool do_rectify;
  std::istringstream(argv[3]) >> std::boolalpha >> do_rectify;

  auto node = std::make_shared<ImageGrabber>(SLAM, do_rectify);

  rclcpp::spin(node);

  // Stop all threads
  SLAM->Shutdown();

  // Save camera trajectory
  SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM_Format.txt");
  SLAM->SaveTrajectoryTUM("FrameTrajectory_TUM_Format.txt");
  SLAM->SaveTrajectoryKITTI("FrameTrajectory_KITTI_Format.txt");

  rclcpp::shutdown();

  return 0;
}

void ImageGrabber::GrabStereo(
    const sensor_msgs::msg::Image::ConstSharedPtr &msgLeft,
    const sensor_msgs::msg::Image::ConstSharedPtr &msgRight) {
  // Copy the ROS image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptrLeft;
  try {
    cv_ptrLeft =
        cv_bridge::toCvShare(msgLeft, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptrRight;
  try {
    cv_ptrRight =
        cv_bridge::toCvShare(msgRight, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  if (do_rectify) {
    cv::Mat imLeft, imRight;
    cv::remap(cv_ptrLeft->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
    cv::remap(cv_ptrRight->image, imRight, M1r, M2r, cv::INTER_LINEAR);
    mpSLAM->TrackStereo(imLeft, imRight, cv_ptrLeft->header.stamp.sec);
  } else {
    mpSLAM->TrackStereo(cv_ptrLeft->image, cv_ptrRight->image,
                        cv_ptrLeft->header.stamp.sec);
  }
}