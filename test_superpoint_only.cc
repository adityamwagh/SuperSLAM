#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Logging.h"
#include "ReadConfig.h"
#include "SuperPointTRT.h"

void drawKeypoints(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints, cv::Mat& output,
                   const cv::Scalar& color = cv::Scalar(0, 255, 0)) {
  output = image.clone();
  if (output.channels() == 1) {
    cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
  }

  for (const auto& kp : keypoints) {
    cv::circle(output, kp.pt, 3, color, -1);
    cv::circle(output, kp.pt, 5, color, 1);
  }
}

int main(int argc, char** argv) {
  // Initialize logging
  SuperSLAM::Logger::initialize();

  // Paths
  std::string config_path = "/home/aditya/Projects/SuperSLAM/utils/config.yaml";
  std::string weights_dir = "/home/aditya/Projects/SuperSLAM/weights";
  std::string image_path =
      "/home/aditya/Downloads/data_odometry_gray/dataset/sequences/00/image_0/"
      "000000.png";

  // Parse command line arguments
  if (argc >= 2) {
    image_path = argv[1];
  }

  SLOG_INFO("Testing SuperPoint with image: {}", image_path);

  try {
    // Load configuration
    Configs configs(config_path, weights_dir);

    // Initialize SuperPoint
    SLOG_INFO("Initializing SuperPoint...");
    auto superpoint =
        std::make_shared<SuperPointTRT>(configs.superpoint_config);
    if (!superpoint->initialize()) {
      SLOG_ERROR("Failed to initialize SuperPoint!");
      return -1;
    }

    // Load image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
      SLOG_ERROR("Failed to load image!");
      return -1;
    }

    SLOG_INFO("Image size: {}x{}", image.cols, image.rows);

    // Extract features
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    SLOG_INFO("Extracting features...");
    auto start = std::chrono::high_resolution_clock::now();
    if (!superpoint->infer(image, keypoints, descriptors)) {
      SLOG_ERROR("Failed to extract features!");
      return -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    SLOG_INFO("SuperPoint inference time: {} ms", duration.count());
    SLOG_INFO("Detected {} keypoints", keypoints.size());
    SLOG_INFO("Descriptor shape: {} x {}", descriptors.rows, descriptors.cols);
    SLOG_INFO("Descriptor type: {}", descriptors.type());

    // Log some keypoint details
    for (int i = 0; i < std::min(5, (int)keypoints.size()); i++) {
      SLOG_INFO("Keypoint {}: x={:.2f}, y={:.2f}, score={:.4f}", i,
                keypoints[i].pt.x, keypoints[i].pt.y, keypoints[i].response);
    }

    // Visualize results
    cv::Mat keypoints_vis;
    drawKeypoints(image, keypoints, keypoints_vis);

    // Save result
    cv::imwrite("keypoints_test.jpg", keypoints_vis);
    SLOG_INFO("Result saved to keypoints_test.jpg");

    // Display result
    cv::namedWindow("Keypoints", cv::WINDOW_NORMAL);
    cv::imshow("Keypoints", keypoints_vis);

    SLOG_INFO("Press any key to exit...");
    cv::waitKey(0);

  } catch (const std::exception& e) {
    SLOG_ERROR("Exception: {}", e.what());
    return -1;
  }

  return 0;
}