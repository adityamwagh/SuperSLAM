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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/core/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "../../../include/System.h"

using namespace std::chrono_literals;

class ImageGrabber : public rclcpp::Node {
public:
  ImageGrabber(std::shared_ptr<SuperSLAM::System> pSLAM)
      : Node("image_grabber"), mpSLAM(std::move(pSLAM)) {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10,
        std::bind(&ImageGrabber::GrabImage, this, std::placeholders::_1));
  }

private:
  void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg);

  std::shared_ptr<SuperSLAM::System> mpSLAM;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  if (argc != 3) {
    std::cerr
        << "Usage: ros2 run SuperSLAM Mono path_to_vocabulary path_to_settings"
        << "\n";
    return 1;
  }

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  auto SLAM = std::make_shared<SuperSLAM::System>(
      argv[1], argv[2], SuperSLAM::System::MONOCULAR, true);

  auto node = std::make_shared<ImageGrabber>(SLAM);

  rclcpp::spin(node);

  // Stop all threads
  SLAM->Shutdown();

  // Save camera trajectory
  SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  rclcpp::shutdown();

  return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::msg::Image::SharedPtr msg) {
  // Copy the ROS image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  mpSLAM->TrackMonocular(cv_ptr->image, cv_ptr->header.stamp.sec);
}