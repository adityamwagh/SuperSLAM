/**
 * Simple test to verify keypoint visualization in RerunViewer
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include "System.h"

int main(int argc, char **argv) {
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  if (argc != 3) {
    spdlog::error("Usage: ./test_simple_visualization path_to_settings path_to_image");
    return 1;
  }

  // Create SLAM system in monocular mode with viewer enabled
  SuperSLAM::System SLAM("", argv[1], SuperSLAM::System::MONOCULAR, true);

  // Load test image
  cv::Mat im = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  if (im.empty()) {
    spdlog::error("Failed to load image: {}", argv[2]);
    return 1;
  }

  spdlog::info("Loaded image: {}x{}", im.cols, im.rows);

  // Process a few frames with the same image to test visualization
  for (int i = 0; i < 10; i++) {
    spdlog::info("Processing frame {}", i);
    cv::Mat Tcw = SLAM.TrackMonocular(im, i * 0.1);

    // Wait a bit to see the visualization
    usleep(500000);  // 0.5 seconds
  }

  // Keep viewer open for inspection
  spdlog::info("Press Enter to exit...");
  std::cin.get();

  SLAM.Shutdown();
  return 0;
}