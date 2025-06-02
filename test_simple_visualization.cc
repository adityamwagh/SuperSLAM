/**
 * Simple test to verify keypoint visualization in RerunViewer
 */

#include <iostream>
#include <opencv2/opencv.hpp>

#include "System.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: ./test_simple_visualization path_to_settings path_to_image"
        << std::endl;
    return 1;
  }

  // Create SLAM system in monocular mode with viewer enabled
  SuperSLAM::System SLAM("", argv[1], SuperSLAM::System::MONOCULAR, true);

  // Load test image
  cv::Mat im = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  if (im.empty()) {
    std::cerr << "Failed to load image: " << argv[2] << std::endl;
    return 1;
  }

  std::cout << "Loaded image: " << im.cols << "x" << im.rows << std::endl;

  // Process a few frames with the same image to test visualization
  for (int i = 0; i < 10; i++) {
    std::cout << "Processing frame " << i << std::endl;
    cv::Mat Tcw = SLAM.TrackMonocular(im, i * 0.1);

    // Wait a bit to see the visualization
    usleep(500000);  // 0.5 seconds
  }

  // Keep viewer open for inspection
  std::cout << "Press Enter to exit..." << std::endl;
  std::cin.get();

  SLAM.Shutdown();
  return 0;
}