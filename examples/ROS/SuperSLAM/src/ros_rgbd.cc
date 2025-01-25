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
    rgb_sub_.subscribe(this, "/camera/rgb/image_raw");
    depth_sub_.subscribe(this, "/camera/depth_registered/image_raw");

    sync_ = std::make_shared<message_filters::Synchronizer<sync_pol>>(
        sync_pol(10), rgb_sub_, depth_sub_);
    sync_->registerCallback(std::bind(&ImageGrabber::GrabRGBD, this,
                                      std::placeholders::_1,
                                      std::placeholders::_2));
  }

private:
  void GrabRGBD(const sensor_msgs::msg::Image::ConstSharedPtr &msgRGB,
                const sensor_msgs::msg::Image::ConstSharedPtr &msgD);

  std::shared_ptr<SuperSLAM::System> mpSLAM;
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  using sync_pol =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                      sensor_msgs::msg::Image>;
  std::shared_ptr<message_filters::Synchronizer<sync_pol>> sync_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  if (argc != 3) {
    std::cerr
        << "Usage: ros2 run SuperSLAM RGBD path_to_vocabulary path_to_settings"
        << std::endl;
    return 1;
  }

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  auto SLAM = std::make_shared<SuperSLAM::System>(
      argv[1], argv[2], SuperSLAM::System::RGBD, true);

  auto node = std::make_shared<ImageGrabber>(SLAM);

  rclcpp::spin(node);

  // Stop all threads
  SLAM->Shutdown();

  // Save camera trajectory
  SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  rclcpp::shutdown();

  return 0;
}

void ImageGrabber::GrabRGBD(
    const sensor_msgs::msg::Image::ConstSharedPtr &msgRGB,
    const sensor_msgs::msg::Image::ConstSharedPtr &msgD) {
  // Copy the ROS image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptrRGB;
  try {
    cv_ptrRGB =
        cv_bridge::toCvShare(msgRGB, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptrD;
  try {
    cv_ptrD =
        cv_bridge::toCvShare(msgD, sensor_msgs::image_encodings::TYPE_32FC1);
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  mpSLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image,
                    cv_ptrRGB->header.stamp.sec);
}