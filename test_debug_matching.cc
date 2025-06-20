#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Logging.h"
#include "ReadConfig.h"
#include "SuperGlueTRT.h"
#include "SuperPointTRT.h"

// Simple brute force matcher for comparison
void bruteForceMatcher(const cv::Mat& desc1, const cv::Mat& desc2,
                       std::vector<cv::DMatch>& matches) {
  matches.clear();

  for (int i = 0; i < desc1.rows; i++) {
    float minDist = std::numeric_limits<float>::max();
    int bestIdx = -1;

    for (int j = 0; j < desc2.rows; j++) {
      float dist = cv::norm(desc1.row(i), desc2.row(j), cv::NORM_L2);
      if (dist < minDist) {
        minDist = dist;
        bestIdx = j;
      }
    }

    if (bestIdx >= 0) {
      cv::DMatch match;
      match.queryIdx = i;
      match.trainIdx = bestIdx;
      match.distance = minDist;
      matches.push_back(match);
    }
  }
}

int main(int argc, char** argv) {
  // Initialize logging
  SuperSLAM::Logger::initialize();

  // Paths
  std::string config_path = "/home/aditya/Projects/SuperSLAM/utils/config.yaml";
  std::string weights_dir = "/home/aditya/Projects/SuperSLAM/weights";
  std::string image1_path =
      "/home/aditya/Downloads/data_odometry_gray/dataset/sequences/00/image_0/"
      "000000.png";
  std::string image2_path =
      "/home/aditya/Downloads/data_odometry_gray/dataset/sequences/00/image_0/"
      "000001.png";

  SLOG_INFO("Debug matching test");

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

    // Load images
    cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);

    if (image1.empty() || image2.empty()) {
      SLOG_ERROR("Failed to load images!");
      return -1;
    }

    // Extract features
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    SLOG_INFO("Extracting features from image 1...");
    if (!superpoint->infer(image1, keypoints1, descriptors1)) {
      SLOG_ERROR("Failed to extract features from image 1!");
      return -1;
    }

    SLOG_INFO("Extracting features from image 2...");
    if (!superpoint->infer(image2, keypoints2, descriptors2)) {
      SLOG_ERROR("Failed to extract features from image 2!");
      return -1;
    }

    SLOG_INFO("Image 1: {} keypoints, descriptors: {}x{}", keypoints1.size(),
              descriptors1.rows, descriptors1.cols);
    SLOG_INFO("Image 2: {} keypoints, descriptors: {}x{}", keypoints2.size(),
              descriptors2.rows, descriptors2.cols);

    // Test brute force matching first
    std::vector<cv::DMatch> bf_matches;
    bruteForceMatcher(descriptors1, descriptors2, bf_matches);
    SLOG_INFO("Brute force matches: {}", bf_matches.size());

    // Sort matches by distance and show top 5
    std::sort(bf_matches.begin(), bf_matches.end(),
              [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
              });

    for (int i = 0; i < std::min(5, (int)bf_matches.size()); i++) {
      SLOG_INFO("BF Match {}: idx1={}, idx2={}, dist={:.3f}", i,
                bf_matches[i].queryIdx, bf_matches[i].trainIdx,
                bf_matches[i].distance);
    }

    // Now test SuperGlue
    SLOG_INFO("\nInitializing SuperGlue...");
    auto superglue = std::make_shared<SuperGlueTRT>(configs.superglue_config);
    if (!superglue->initialize()) {
      SLOG_ERROR("Failed to initialize SuperGlue!");
      return -1;
    }

    SLOG_INFO("Testing SuperGlue matching...");
    MatchResult match_result;
    if (!superglue->match(keypoints1, descriptors1, keypoints2, descriptors2,
                          match_result)) {
      SLOG_ERROR("Failed to match features!");
      return -1;
    }

    SLOG_INFO("SuperGlue matches: {}", match_result.matches.size());

    // Show SuperGlue matches
    for (int i = 0; i < std::min(5, (int)match_result.matches.size()); i++) {
      SLOG_INFO("SG Match {}: idx1={}, idx2={}, dist={:.3f}", i,
                match_result.matches[i].queryIdx,
                match_result.matches[i].trainIdx,
                match_result.matches[i].distance);
    }

    // Analyze score matrix if available
    if (!match_result.scores.empty()) {
      cv::Mat scores = match_result.scores;
      double minVal, maxVal;
      cv::minMaxLoc(scores, &minVal, &maxVal);
      SLOG_INFO("Score matrix: {}x{}, min={:.3f}, max={:.3f}", scores.rows,
                scores.cols, minVal, maxVal);
    }

  } catch (const std::exception& e) {
    SLOG_ERROR("Exception: {}", e.what());
    return -1;
  }

  return 0;
}