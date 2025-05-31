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
#include <map>
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
  
  // For EuRoC, check if we need to add mav0 subdirectory
  fs::path data_path = sequence_path;
  if (fs::exists(sequence_path / "mav0")) {
    data_path = sequence_path / "mav0";
  }
  
  const fs::path left_data = data_path / "cam0" / "data";
  const fs::path right_data = data_path / "cam1" / "data";

  std::cout << "Looking for left images in: " << left_data << std::endl;
  std::cout << "Looking for right images in: " << right_data << std::endl;

  // Get all PNG files from both directories
  std::vector<fs::path> left_files, right_files;
  
  for (const auto& entry : fs::directory_iterator(left_data)) {
    if (entry.path().extension() == ".png") {
      left_files.push_back(entry.path());
    }
  }
  
  for (const auto& entry : fs::directory_iterator(right_data)) {
    if (entry.path().extension() == ".png") {
      right_files.push_back(entry.path());
    }
  }

  // Sort files by filename (which contains timestamp)
  std::sort(left_files.begin(), left_files.end());
  std::sort(right_files.begin(), right_files.end());

  std::cout << "Found " << left_files.size() << " left images and " 
            << right_files.size() << " right images" << std::endl;

  // Take the minimum count and match by filename
  size_t n_images = std::min(left_files.size(), right_files.size());
  
  for (size_t i = 0; i < n_images; ++i) {
    // Extract timestamp from filename (remove .png extension)
    std::string left_stem = left_files[i].stem().string();
    std::string right_stem = right_files[i].stem().string();
    
    // Only add if filenames match (same timestamp)
    if (left_stem == right_stem) {
      images.left_filenames.push_back(left_files[i].string());
      images.right_filenames.push_back(right_files[i].string());
      
      // Convert timestamp from filename (nanoseconds) to seconds
      uint64_t timestamp_ns = std::stoull(left_stem);
      images.timestamps.push_back(timestamp_ns * 1e-9);
    }
  }

  std::cout << "Loaded " << images.timestamps.size() << " synchronized stereo pairs" << std::endl;
  
  if (images.timestamps.size() > 0) {
    std::cout << "First left image: " << images.left_filenames[0] << std::endl;
    std::cout << "First right image: " << images.right_filenames[0] << std::endl;
    std::cout << "First timestamp: " << images.timestamps[0] << std::endl;
  }

  return images;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: ./stereo_euroc_rerun path_to_vocabulary path_to_settings "
                 "path_to_sequence"
              << "\n";
    std::cerr << "\nExample:\n";
    std::cerr << "./stereo_euroc_rerun vocabulary/ORBvoc.txt "
                 "examples/stereo/EuRoC.yaml "
                 "/path/to/EuRoC/MH_01_easy\n";
    std::cerr << "\nStereo SLAM for EuRoC with dual camera visualization.\n";
    std::cerr << "Features:\n";
    std::cerr << "- Left and right camera feeds (cam0 and cam1)\n";
    std::cerr << "- Camera trajectory (green line)\n";
    std::cerr << "- Current pose (yellow point)\n";
    std::cerr << "- 3D map points (colored by observations)\n";
    std::cerr << "- SuperPoint keypoints on both images\n";
    std::cerr << "- High-frequency stereo processing (20 Hz)\n";
    std::cerr << "\nView at http://localhost:9090\n";
    return 1;
  }

  try {
    // Retrieve paths to images
    const fs::path path_to_sequence(argv[3]);

    std::cout << "Loading EuRoC stereo images from: " << path_to_sequence << "\n";
    StereoImages images = LoadImages(path_to_sequence);

    int nImages = images.left_filenames.size();
    std::cout << "Loaded " << nImages << " stereo pairs\n";

    // Create SLAM system with visualization enabled
    std::cout << "Initializing SuperSLAM stereo system with Rerun visualization...\n";
    SuperSLAM::System SLAM(argv[1], argv[2], SuperSLAM::System::STEREO, true);

    std::cout << "\n=== Stereo EuRoC SLAM with Rerun ===\n";
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
        std::cerr << "Left file: " << images.left_filenames[ni] << "\n";
        std::cerr << "Right file: " << images.right_filenames[ni] << "\n";
        std::cerr << "Left exists: " << fs::exists(images.left_filenames[ni]) << "\n";
        std::cerr << "Right exists: " << fs::exists(images.right_filenames[ni]) << "\n";
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

      // Print progress every 100 frames
      if (ni % 100 == 0 && ni > 0) {
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

      // Maintain real-time processing rate
      double T = 0;
      if (ni < nImages - 1)
        T = images.timestamps[ni + 1] - tframe;
      else if (ni > 0)
        T = tframe - images.timestamps[ni - 1];

      if (ttrack < T) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(static_cast<long>((T - ttrack) * 1e6)));
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
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_EuRoC.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory_EuRoC.txt");
    
    std::cout << "Saved:\n";
    std::cout << "- KeyFrameTrajectory_EuRoC.txt\n";
    std::cout << "- CameraTrajectory_EuRoC.txt\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}