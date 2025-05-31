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

StereoImages LoadImages(const fs::path &sequence_path) {
  StereoImages images;
  
  // For KITTI, we load all images from image_0 and image_1 directories
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
    std::cout << "No times.txt found, generating timestamps at 10 Hz\n";
  }

  // Check if directories exist
  if (!fs::exists(left_path)) {
    throw std::runtime_error("Left camera directory does not exist: " + left_path.string());
  }
  if (!fs::exists(right_path)) {
    throw std::runtime_error("Right camera directory does not exist: " + right_path.string());
  }

  std::cout << "Loading from left: " << left_path << std::endl;
  std::cout << "Loading from right: " << right_path << std::endl;

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
  
  std::cout << "Found " << left_files.size() << " left images and " 
            << right_files.size() << " right images" << std::endl;
  
  for (size_t i = 0; i < n_images; ++i) {
    images.left_filenames.push_back(left_files[i].string());
    images.right_filenames.push_back(right_files[i].string());
    
    // Generate timestamp if not loaded from file
    if (images.timestamps.size() <= i) {
      images.timestamps.push_back(i * 0.1);  // 10 Hz
    }
  }

  if (n_images > 0) {
    std::cout << "First left image: " << images.left_filenames[0] << std::endl;
    std::cout << "First right image: " << images.right_filenames[0] << std::endl;
  }

  return images;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: ./stereo_kitti_rerun path_to_vocabulary path_to_settings "
                 "path_to_sequence"
              << "\n";
    std::cerr << "\nExample:\n";
    std::cerr << "./stereo_kitti_rerun vocabulary/ORBvoc.txt "
                 "examples/stereo/KITTI00-02.yaml "
                 "/path/to/KITTI/dataset/sequences/00\n";
    std::cerr << "\nStereo SLAM for KITTI with dual camera visualization.\n";
    std::cerr << "Features:\n";
    std::cerr << "- Left and right camera feeds\n";
    std::cerr << "- Camera trajectory (green line)\n";
    std::cerr << "- Current pose (yellow point)\n";
    std::cerr << "- 3D map points (colored by observations)\n";
    std::cerr << "- SuperPoint keypoints on both images\n";
    std::cerr << "- Stereo depth estimation\n";
    std::cerr << "\nView at http://localhost:9090\n";
    return 1;
  }

  try {
    // Retrieve paths to images
    const fs::path path_to_sequence(argv[3]);

    std::cout << "Loading stereo images from: " << path_to_sequence << "\n";
    StereoImages images = LoadImages(path_to_sequence);

    int nImages = images.left_filenames.size();
    std::cout << "Loaded " << nImages << " stereo pairs\n";

    // Create SLAM system with visualization enabled
    std::cout << "Initializing SuperSLAM stereo system with Rerun visualization...\n";
    SuperSLAM::System SLAM(argv[1], argv[2], SuperSLAM::System::STEREO, true);

    std::cout << "\n=== Stereo KITTI SLAM with Rerun ===\n";
    std::cout << "Dataset: " << path_to_sequence << "\n";
    std::cout << "Stereo pairs: " << nImages << "\n";
    std::cout << "Visualization: http://localhost:9090\n";
    std::cout << "===================================\n\n";

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
        std::cerr << "Failed to load stereo pair " << ni << "\n";
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

      // Print progress every 50 frames
      if (ni % 50 == 0 && ni > 0) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
        double fps = ni / elapsed_seconds;
        
        std::cout << "Frame " << ni << "/" << nImages 
                  << " | FPS: " << std::fixed << std::setprecision(1) << fps
                  << " | Track: " << std::setprecision(3) << ttrack * 1000 << "ms"
                  << " | Lost: " << nLostFrames 
                  << " | State: ";
        
        switch(tracking_state) {
          case -1: std::cout << "SYSTEM_NOT_READY"; break;
          case 0: std::cout << "NO_IMAGES_YET"; break;
          case 1: std::cout << "NOT_INITIALIZED"; break;
          case 2: std::cout << "OK"; break;
          case 3: std::cout << "LOST"; break;
          default: std::cout << "UNKNOWN"; break;
        }
        std::cout << "\n";
      }

      // Maintain real-time processing rate for KITTI (10 Hz)
      if (ttrack < 0.1) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(static_cast<long>((0.1 - ttrack) * 1e6)));
      }
    }

    std::cout << "\nProcessing complete. Finalizing...\n";

    // Stop all threads
    SLAM.Shutdown();

    // Final statistics
    std::sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime =
        std::accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);
    
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << "Total stereo pairs: " << nImages << "\n";
    std::cout << "Median tracking time: " << vTimesTrack[nImages / 2] * 1000 << " ms\n";
    std::cout << "Mean tracking time: " << (totaltime / nImages) * 1000 << " ms\n";
    std::cout << "Average FPS: " << nImages / totaltime << "\n";
    std::cout << "Lost frames: " << nLostFrames << " (" 
              << (100.0 * nLostFrames / nImages) << "%)\n";

    // Save trajectory
    std::cout << "\nSaving trajectories...\n";
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_KITTI.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory_KITTI.txt");
    SLAM.SaveTrajectoryKITTI("CameraTrajectory_KITTI_format.txt");
    
    std::cout << "Saved:\n";
    std::cout << "- KeyFrameTrajectory_KITTI.txt\n";
    std::cout << "- CameraTrajectory_KITTI.txt\n";
    std::cout << "- CameraTrajectory_KITTI_format.txt\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}