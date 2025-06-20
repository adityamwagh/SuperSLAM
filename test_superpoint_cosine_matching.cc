#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Logging.h"
#include "ReadConfig.h"
#include "SuperPointTRT.h"

// Draw keypoints on image
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

// Compute cosine similarity between two descriptors
float cosineSimilarity(const cv::Mat& desc1, const cv::Mat& desc2) {
  // desc1 and desc2 are 1x256 vectors
  float dot = desc1.dot(desc2);
  float norm1 = cv::norm(desc1);
  float norm2 = cv::norm(desc2);
  return dot /
         (norm1 * norm2 + 1e-8);  // Add small epsilon to avoid division by zero
}

// Match descriptors using cosine similarity
void cosineMatch(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                 std::vector<cv::DMatch>& matches, float threshold = 0.8) {
  matches.clear();

  // For each descriptor in image 1
  for (int i = 0; i < descriptors1.rows; i++) {
    float bestSim = -1.0f;
    float secondBestSim = -1.0f;
    int bestIdx = -1;

    // Find best match in image 2
    for (int j = 0; j < descriptors2.rows; j++) {
      float sim = cosineSimilarity(descriptors1.row(i), descriptors2.row(j));

      if (sim > bestSim) {
        secondBestSim = bestSim;
        bestSim = sim;
        bestIdx = j;
      } else if (sim > secondBestSim) {
        secondBestSim = sim;
      }
    }

    // Apply threshold and ratio test
    if (bestIdx >= 0 && bestSim > threshold) {
      // Optional: Lowe's ratio test
      float ratio = (secondBestSim > 0) ? bestSim / secondBestSim : 2.0f;
      if (ratio >
          1.2f) {  // Good match should be significantly better than second best
        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = bestIdx;
        match.distance = 1.0f - bestSim;  // Convert similarity to distance
        matches.push_back(match);
      }
    }
  }
}

void drawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kpts1,
                 const cv::Mat& img2, const std::vector<cv::KeyPoint>& kpts2,
                 const std::vector<cv::DMatch>& matches, cv::Mat& output) {
  cv::Mat img1_color, img2_color;

  if (img1.channels() == 1) {
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
  } else {
    img1_color = img1.clone();
  }

  if (img2.channels() == 1) {
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
  } else {
    img2_color = img2.clone();
  }

  // Create side-by-side image
  int width = img1_color.cols + img2_color.cols;
  int height = std::max(img1_color.rows, img2_color.rows);
  output = cv::Mat::zeros(height, width, CV_8UC3);

  img1_color.copyTo(output(cv::Rect(0, 0, img1_color.cols, img1_color.rows)));
  img2_color.copyTo(
      output(cv::Rect(img1_color.cols, 0, img2_color.cols, img2_color.rows)));

  // Draw keypoints
  for (const auto& kp : kpts1) {
    cv::circle(output, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
  }

  for (const auto& kp : kpts2) {
    cv::Point2f pt = kp.pt;
    pt.x += img1_color.cols;
    cv::circle(output, pt, 3, cv::Scalar(0, 255, 0), -1);
  }

  // Draw matches with different colors based on quality
  for (const auto& match : matches) {
    float similarity = 1.0f - match.distance;
    cv::Scalar color;

    // Color based on match quality (red=bad, yellow=medium, green=good)
    if (similarity > 0.95f) {
      color = cv::Scalar(0, 255, 0);  // Green for excellent matches
    } else if (similarity > 0.9f) {
      color = cv::Scalar(0, 255, 255);  // Yellow for good matches
    } else {
      color = cv::Scalar(0, 165, 255);  // Orange for acceptable matches
    }

    cv::Point2f pt1 = kpts1[match.queryIdx].pt;
    cv::Point2f pt2 = kpts2[match.trainIdx].pt;
    pt2.x += img1_color.cols;

    cv::line(output, pt1, pt2, color, 1);
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

  // Parse command line arguments
  if (argc >= 3) {
    image1_path = argv[1];
    image2_path = argv[2];
  }

  SLOG_INFO("Testing SuperPoint with cosine similarity matching");
  SLOG_INFO("  Image 1: {}", image1_path);
  SLOG_INFO("  Image 2: {}", image2_path);

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

    SLOG_INFO("Image 1 size: {}x{}", image1.cols, image1.rows);
    SLOG_INFO("Image 2 size: {}x{}", image2.cols, image2.rows);

    // Extract features from both images
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    SLOG_INFO("Extracting features from image 1...");
    auto start = std::chrono::high_resolution_clock::now();
    if (!superpoint->infer(image1, keypoints1, descriptors1)) {
      SLOG_ERROR("Failed to extract features from image 1!");
      return -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    SLOG_INFO("SuperPoint inference time (image 1): {} ms", duration.count());
    SLOG_INFO("Detected {} keypoints in image 1", keypoints1.size());

    SLOG_INFO("Extracting features from image 2...");
    start = std::chrono::high_resolution_clock::now();
    if (!superpoint->infer(image2, keypoints2, descriptors2)) {
      SLOG_ERROR("Failed to extract features from image 2!");
      return -1;
    }
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    SLOG_INFO("SuperPoint inference time (image 2): {} ms", duration.count());
    SLOG_INFO("Detected {} keypoints in image 2", keypoints2.size());

    // Normalize descriptors (L2 normalization)
    SLOG_INFO("Normalizing descriptors...");
    for (int i = 0; i < descriptors1.rows; i++) {
      cv::normalize(descriptors1.row(i), descriptors1.row(i));
    }
    for (int i = 0; i < descriptors2.rows; i++) {
      cv::normalize(descriptors2.row(i), descriptors2.row(i));
    }

    // Match using cosine similarity
    std::vector<cv::DMatch> matches;
    SLOG_INFO("Matching with cosine similarity...");
    start = std::chrono::high_resolution_clock::now();
    cosineMatch(descriptors1, descriptors2, matches, 0.8f);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    SLOG_INFO("Matching time: {} ms", duration.count());
    SLOG_INFO("Found {} matches", matches.size());

    // Sort matches by similarity
    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
              });

    // Show top matches
    SLOG_INFO("\nTop 10 matches:");
    for (int i = 0; i < std::min(10, (int)matches.size()); i++) {
      float similarity = 1.0f - matches[i].distance;
      cv::Point2f pt1 = keypoints1[matches[i].queryIdx].pt;
      cv::Point2f pt2 = keypoints2[matches[i].trainIdx].pt;
      float dx = pt2.x - pt1.x;
      float dy = pt2.y - pt1.y;
      SLOG_INFO(
          "Match {}: similarity={:.3f}, pt1=({:.1f},{:.1f}), "
          "pt2=({:.1f},{:.1f}), delta=({:.1f},{:.1f})",
          i, similarity, pt1.x, pt1.y, pt2.x, pt2.y, dx, dy);
    }

    // Try different thresholds
    std::vector<float> thresholds = {0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f};
    SLOG_INFO("\nMatches at different thresholds:");
    for (float thresh : thresholds) {
      std::vector<cv::DMatch> thresh_matches;
      cosineMatch(descriptors1, descriptors2, thresh_matches, thresh);
      SLOG_INFO("Threshold {:.2f}: {} matches", thresh, thresh_matches.size());
    }

    // Visualize results
    cv::Mat matches_vis;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, matches_vis);

    // Save results
    cv::imwrite("cosine_matches.jpg", matches_vis);
    SLOG_INFO("\nResults saved to cosine_matches.jpg");

    // Also try with OpenCV's brute force matcher for comparison
    cv::BFMatcher bf_matcher(cv::NORM_L2);
    std::vector<cv::DMatch> bf_matches;
    bf_matcher.match(descriptors1, descriptors2, bf_matches);
    SLOG_INFO("\nOpenCV BFMatcher (L2): {} matches", bf_matches.size());

    // Create additional visualizations

    // 1. Individual keypoint visualizations
    cv::Mat kp1_vis, kp2_vis;
    drawKeypoints(image1, keypoints1, kp1_vis, cv::Scalar(0, 255, 0));
    drawKeypoints(image2, keypoints2, kp2_vis, cv::Scalar(0, 255, 0));

    // 2. Create a grid visualization showing different match qualities
    std::vector<cv::DMatch> excellent_matches, good_matches, fair_matches;
    for (const auto& match : matches) {
      float similarity = 1.0f - match.distance;
      if (similarity > 0.95f) {
        excellent_matches.push_back(match);
      } else if (similarity > 0.9f) {
        good_matches.push_back(match);
      } else {
        fair_matches.push_back(match);
      }
    }

    cv::Mat excellent_vis, good_vis, fair_vis;
    drawMatches(image1, keypoints1, image2, keypoints2, excellent_matches,
                excellent_vis);
    drawMatches(image1, keypoints1, image2, keypoints2, good_matches, good_vis);
    drawMatches(image1, keypoints1, image2, keypoints2, fair_matches, fair_vis);

    // 3. Create motion visualization (show displacement vectors)
    cv::Mat motion_vis = image1.clone();
    if (motion_vis.channels() == 1) {
      cv::cvtColor(motion_vis, motion_vis, cv::COLOR_GRAY2BGR);
    }

    for (const auto& match : matches) {
      cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
      cv::Point2f pt2 = keypoints2[match.trainIdx].pt;

      // Draw arrow from pt1 to pt2
      cv::arrowedLine(motion_vis, pt1, pt2, cv::Scalar(0, 255, 0), 1,
                      cv::LINE_AA, 0, 0.1);
      cv::circle(motion_vis, pt1, 2, cv::Scalar(0, 0, 255), -1);
    }

    // 4. Create match distribution heatmap
    cv::Mat heatmap = cv::Mat::zeros(image1.rows, image1.cols, CV_32F);
    for (const auto& match : matches) {
      cv::Point2f pt = keypoints1[match.queryIdx].pt;
      cv::circle(heatmap, pt, 20, cv::Scalar(1.0), -1);
    }
    cv::GaussianBlur(heatmap, heatmap, cv::Size(31, 31), 10);
    double minVal, maxVal;
    cv::minMaxLoc(heatmap, &minVal, &maxVal);
    heatmap = heatmap / maxVal;
    cv::Mat heatmap_color;
    cv::Mat heatmap_8u;
    heatmap.convertTo(heatmap_8u, CV_8U, 255.0);
    cv::applyColorMap(heatmap_8u, heatmap_color, cv::COLORMAP_JET);
    cv::Mat heatmap_overlay;
    cv::Mat image1_color;
    if (image1.channels() == 1) {
      cv::cvtColor(image1, image1_color, cv::COLOR_GRAY2BGR);
    } else {
      image1_color = image1;
    }
    cv::addWeighted(image1_color, 0.7, heatmap_color, 0.3, 0, heatmap_overlay);

    // 5. Create combined visualization
    int pad = 10;
    int single_width = image1.cols;
    int single_height = image1.rows;
    int combined_width = single_width * 3 + pad * 2;
    int combined_height = single_height * 3 + pad * 2;

    cv::Mat combined = cv::Mat::zeros(combined_height, combined_width, CV_8UC3);
    combined.setTo(cv::Scalar(50, 50, 50));  // Gray background

    // Top row: Original images with keypoints and main matches
    kp1_vis.copyTo(combined(cv::Rect(0, 0, single_width, single_height)));
    kp2_vis.copyTo(
        combined(cv::Rect(single_width + pad, 0, single_width, single_height)));
    cv::resize(matches_vis, matches_vis, cv::Size(single_width, single_height));
    matches_vis.copyTo(combined(
        cv::Rect(2 * (single_width + pad), 0, single_width, single_height)));

    // Middle row: Motion visualization and heatmap
    motion_vis.copyTo(combined(
        cv::Rect(0, single_height + pad, single_width, single_height)));
    heatmap_overlay.copyTo(combined(cv::Rect(
        single_width + pad, single_height + pad, single_width, single_height)));

    // Bottom row: Different quality matches
    cv::resize(excellent_vis, excellent_vis,
               cv::Size(single_width, single_height));
    excellent_vis.copyTo(
        combined(cv::Rect(2 * (single_width + pad), single_height + pad,
                          single_width, single_height)));

    // Add text labels
    cv::putText(combined, "Image 1 Keypoints", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Image 2 Keypoints",
                cv::Point(single_width + pad + 10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "All Matches (Cosine Similarity)",
                cv::Point(2 * (single_width + pad) + 10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Motion Vectors",
                cv::Point(10, single_height + pad + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Match Distribution Heatmap",
                cv::Point(single_width + pad + 10, single_height + pad + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(
        combined, "Excellent Matches (>0.95)",
        cv::Point(2 * (single_width + pad) + 10, single_height + pad + 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Add statistics
    std::string stats =
        cv::format("Total Matches: %d | Excellent: %d | Good: %d | Fair: %d",
                   (int)matches.size(), (int)excellent_matches.size(),
                   (int)good_matches.size(), (int)fair_matches.size());
    cv::putText(combined, stats, cv::Point(10, combined_height - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    // Save visualizations
    cv::imwrite("cosine_matches.jpg", matches_vis);
    cv::imwrite("cosine_matches_combined.jpg", combined);
    cv::imwrite("cosine_matches_motion.jpg", motion_vis);
    cv::imwrite("cosine_matches_heatmap.jpg", heatmap_overlay);

    SLOG_INFO("\nVisualization files saved:");
    SLOG_INFO("  - cosine_matches.jpg (main matches)");
    SLOG_INFO("  - cosine_matches_combined.jpg (complete visualization)");
    SLOG_INFO("  - cosine_matches_motion.jpg (motion vectors)");
    SLOG_INFO("  - cosine_matches_heatmap.jpg (match distribution)");

    // Display results
    cv::namedWindow("Combined Visualization", cv::WINDOW_NORMAL);
    cv::imshow("Combined Visualization", combined);

    cv::namedWindow("Motion Vectors", cv::WINDOW_NORMAL);
    cv::imshow("Motion Vectors", motion_vis);

    SLOG_INFO("\nPress any key to exit...");
    cv::waitKey(0);

  } catch (const std::exception& e) {
    SLOG_ERROR("Exception: {}", e.what());
    return -1;
  }

  return 0;
}