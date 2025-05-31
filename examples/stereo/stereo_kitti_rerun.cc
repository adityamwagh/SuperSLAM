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

#include <Logging.h>
#include <System.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <thread>

namespace fs = std::filesystem;

struct StereoImages {
  std::vector<std::string> left_filenames;
  std::vector<std::string> right_filenames;
  std::vector<double> timestamps;
};

StereoImages LoadImages(const fs::path& sequence_path) {
  StereoImages images;
  
  // For KITTI stereo, we use image_0 (left) and image_1 (right) directories
  const fs::path left_path = sequence_path / "image_0";
  const fs::path right_path = sequence_path / "image_1";
  const fs::path times_path = sequence_path / "times.txt";

  // Load timestamps
  std::ifstream times_file(times_path);
  if (times_file.is_open()) {
    std::string line;
    while (std::getline(times_file, line)) {
      if (!line.empty()) { 
        images.timestamps.push_back(std::stod(line)); 
      }
    }
  } else {
    // If no times.txt, generate timestamps at 10 Hz
    SLOG_WARN("No times.txt found, generating timestamps at 10 Hz");
  }

  // Check if directories exist
  if (!fs::exists(left_path)) {
    throw std::runtime_error("Left camera directory does not exist: " + left_path.string());
  }
  if (!fs::exists(right_path)) {
    throw std::runtime_error("Right camera directory does not exist: " + right_path.string());
  }

  SLOG_INFO("Loading from left: {}", left_path.string());
  SLOG_INFO("Loading from right: {}", right_path.string());

  // Load image filenames
  std::vector<fs::path> left_files, right_files;

  for (const auto& entry : fs::directory_iterator(left_path)) {
    if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
      left_files.push_back(entry.path());
    }
  }

  for (const auto& entry : fs::directory_iterator(right_path)) {
    if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
      right_files.push_back(entry.path());
    }
  }

  // Sort files by name
  std::sort(left_files.begin(), left_files.end());
  std::sort(right_files.begin(), right_files.end());

  // Ensure we have matching pairs
  size_t n_images = std::min(left_files.size(), right_files.size());

  SLOG_INFO("Found {} left images and {} right images", left_files.size(), right_files.size());

  for (size_t i = 0; i < n_images; ++i) {
    images.left_filenames.push_back(left_files[i].string());
    images.right_filenames.push_back(right_files[i].string());

    // Generate timestamp if not loaded from file
    if (images.timestamps.size() <= i) {
      images.timestamps.push_back(i * 0.1); // 10 Hz
    }
  }

  if (n_images > 0) {
    SLOG_DEBUG("First left image: {}", images.left_filenames[0]);
    SLOG_DEBUG("First right image: {}", images.right_filenames[0]);
  }

  return images;
}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::cerr << "Usage: ./stereo_kitti_rerun [path_to_vocabulary] path_to_settings "
                 "path_to_sequence [--no-viz]\n";
    std::cerr << "       ./stereo_kitti_rerun path_to_settings "
                 "path_to_sequence [--no-viz]  (no vocabulary mode)\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "With ORB vocabulary (for loop closure):\n";
    std::cerr << "./stereo_kitti_rerun vocabulary/ORBvoc.txt "
                 "examples/stereo/KITTI00-02.yaml "
                 "/path/to/KITTI/dataset/sequences/00\n";
    std::cerr << "\nWithout vocabulary (SuperPoint only - no loop closure):\n";
    std::cerr << "./stereo_kitti_rerun "
                 "examples/stereo/KITTI00-02.yaml "
                 "/path/to/KITTI/dataset/sequences/00 --no-viz\n";
    std::cerr << "\nFeatures:\n";
    std::cerr << "- Stereo camera trajectory and poses\n";
    std::cerr << "- 3D map points from stereo triangulation\n";
    std::cerr << "- SuperPoint keypoints on both images\n";
    std::cerr << "- Coordinate system relative to most recent keyframe\n";
    std::cerr << "- Real-time processing statistics\n";
    std::cerr << "- Video-like sequential playback\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "- --no-viz: Disable visualization for maximum framerate\n";
    std::cerr << "\nNOTE: Without vocabulary, loop closure detection is "
                 "disabled but SLAM still works!\n";
    std::cerr << "View at http://localhost:9090\n";
    return 1;
  }

  try {
    // Initialize logging
    SuperSLAM::Logger::initialize();

    // Parse command line arguments
    std::string vocab_path = "";
    std::string settings_path = "";
    std::string sequence_path = "";
    bool enable_visualization = true;

    if (argc == 3 || (argc == 4 && std::string(argv[3]) == "--no-viz")) {
      // No vocabulary mode
      settings_path = argv[1];
      sequence_path = argv[2];
      if (argc == 4) enable_visualization = false;
      SLOG_INFO("Running in no-vocabulary mode (loop closure disabled)");
    } else {
      // With vocabulary mode
      vocab_path = argv[1];
      settings_path = argv[2];
      sequence_path = argv[3];
      if (argc == 5 && std::string(argv[4]) == "--no-viz") {
        enable_visualization = false;
      }
      SLOG_INFO("Running with vocabulary: {}", vocab_path);
    }

    if (!enable_visualization) {
      SLOG_INFO("Visualization disabled for maximum performance");
    }

    // Retrieve paths to images
    const fs::path path_to_sequence(sequence_path);

    SLOG_INFO("Loading KITTI stereo images from: {}", path_to_sequence.string());
    StereoImages images = LoadImages(path_to_sequence);

    int nImages = images.left_filenames.size();
    SLOG_INFO("Loaded {} stereo pairs", nImages);

    // Create SLAM system with or without visualization
    SLOG_INFO("Initializing SuperSLAM stereo system{}...",
              enable_visualization ? " with Rerun visualization"
                                   : " (no visualization)");

    SuperSLAM::System SLAM(vocab_path.empty() ? "" : vocab_path.c_str(),
                           settings_path.c_str(), SuperSLAM::System::STEREO,
                           enable_visualization);

    SLOG_INFO("\n=== KITTI Stereo SLAM {} ===",
              enable_visualization ? "with Rerun" : "(High Performance Mode)");
    SLOG_INFO("Dataset: {}", path_to_sequence.string());
    SLOG_INFO("Stereo pairs: {}", nImages);
    SLOG_INFO("Feature extractor: SuperPoint (ORB-free!)");
    if (vocab_path.empty()) {
      SLOG_INFO("Loop closure: DISABLED (no vocabulary)");
    } else {
      SLOG_INFO("Loop closure: ENABLED with vocabulary");
    }
    if (enable_visualization) {
      SLOG_INFO("Visualization: http://localhost:9090");
    }
    SLOG_INFO("================================\n");

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack(nImages);

    // Performance tracking
    int nLostFrames = 0;

    // Main processing loop
    cv::Mat imLeft, imRight;
    auto start_time = std::chrono::steady_clock::now();

    for (int ni = 0; ni < nImages; ni++) {
      // Read stereo images
      imLeft = cv::imread(images.left_filenames[ni], cv::IMREAD_UNCHANGED);
      imRight = cv::imread(images.right_filenames[ni], cv::IMREAD_UNCHANGED);
      double tframe = images.timestamps[ni];

      if (imLeft.empty() || imRight.empty()) {
        SLOG_ERROR("Failed to load stereo pair: {} | {}", 
                   images.left_filenames[ni], images.right_filenames[ni]);
        continue;
      }

      auto t1 = std::chrono::steady_clock::now();

      // Pass stereo images to SLAM system
      cv::Mat pose = SLAM.TrackStereo(imLeft, imRight, tframe);
      int tracking_state = SLAM.GetTrackingState();

      auto t2 = std::chrono::steady_clock::now();

      double ttrack =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
              .count();

      vTimesTrack[ni] = ttrack;

      // Track performance metrics
      if (tracking_state == 3) {  // LOST state
        nLostFrames++;
      }

      // Print progress every 50 frames for KITTI sequences
      if (ni % 50 == 0 && ni > 0) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
        double fps = ni / elapsed_seconds;

        SLOG_INFO(
            "Frame {}/{} | FPS: {:.1f} | Track: {:.3f}ms | Lost: {} | State: "
            "{}",
            ni, nImages, fps, ttrack * 1000, nLostFrames,
            (tracking_state == -1)  ? "SYSTEM_NOT_READY"
            : (tracking_state == 0) ? "NO_IMAGES_YET"
            : (tracking_state == 1) ? "NOT_INITIALIZED"
            : (tracking_state == 2) ? "OK"
            : (tracking_state == 3) ? "LOST"
                                    : "UNKNOWN");
        
        // Print additional debug info for visualization
        if (tracking_state == 2) {
          SLOG_INFO("  → Tracked keypoints: {}", SLAM.GetTrackedKeyPointsUn().size());
          if (enable_visualization) {
            SLOG_INFO("  → Visualization available at: http://localhost:9090");
          }
        } else if (tracking_state == 1) {
          SLOG_WARN("  → System still initializing - need good stereo matches for triangulation");
        }
      }

      // Maintain real-time processing rate for KITTI (10 Hz)
      if (ttrack < 0.1) {
        std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>((0.1 - ttrack) * 1e6)));
      }
    }

    SLOG_INFO("\nProcessing complete. Finalizing...");

    // Stop all threads
    SLAM.Shutdown();

    // Final statistics
    std::sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime =
        std::accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);

    SLOG_INFO("\n=== Performance Summary ===");
    SLOG_INFO("Total stereo pairs: {}", nImages);
    SLOG_INFO("Median tracking time: {} ms", vTimesTrack[nImages / 2] * 1000);
    SLOG_INFO("Mean tracking time: {} ms", (totaltime / nImages) * 1000);
    SLOG_INFO("Average FPS: {}", nImages / totaltime);
    SLOG_INFO("Lost frames: {} ({}%)", nLostFrames,
              (100.0 * nLostFrames / nImages));

    // Save trajectory
    SLOG_INFO("\nSaving trajectories...");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_KITTI_stereo.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory_KITTI_stereo.txt");
    SLAM.SaveTrajectoryKITTI("CameraTrajectory_KITTI_stereo_format.txt");

    SLOG_INFO("Saved:");
    SLOG_INFO("- KeyFrameTrajectory_KITTI_stereo.txt");
    SLOG_INFO("- CameraTrajectory_KITTI_stereo.txt");
    SLOG_INFO("- CameraTrajectory_KITTI_stereo_format.txt");

  } catch (const std::exception& e) {
    SLOG_ERROR("Error: {}", e.what());
    return 1;
  }

  return 0;
}