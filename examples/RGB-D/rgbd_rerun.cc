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

struct RGBDImages {
  std::vector<std::string> rgb_filenames;
  std::vector<std::string> depth_filenames;
  std::vector<double> timestamps;
};

RGBDImages LoadImages(const fs::path &associations_file) {
  RGBDImages images;
  std::ifstream file(associations_file);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open associations file: " + associations_file.string());
  }

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line[0] != '#') {
      std::stringstream ss(line);
      double rgb_timestamp, depth_timestamp;
      std::string rgb_filename, depth_filename;
      
      ss >> rgb_timestamp >> rgb_filename >> depth_timestamp >> depth_filename;
      
      images.timestamps.push_back(rgb_timestamp);
      images.rgb_filenames.push_back(rgb_filename);
      images.depth_filenames.push_back(depth_filename);
    }
  }

  return images;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: ./rgbd_rerun path_to_vocabulary path_to_settings "
                 "path_to_sequence path_to_associations"
              << "\n";
    std::cerr << "\nExample:\n";
    std::cerr << "./rgbd_rerun vocabulary/ORBvoc.txt "
                 "examples/RGB-D/TUM1.yaml "
                 "/path/to/TUM/dataset/rgbd_dataset_freiburg1_xyz "
                 "examples/RGB-D/associations/fr1_xyz.txt\n";
    std::cerr << "\nRGB-D SLAM with depth-enhanced visualization.\n";
    std::cerr << "Features:\n";
    std::cerr << "- RGB camera feed with keypoints\n";
    std::cerr << "- Depth image visualization\n";
    std::cerr << "- Camera trajectory (green line)\n";
    std::cerr << "- Current pose (yellow point)\n";
    std::cerr << "- Dense 3D map points with depth info\n";
    std::cerr << "- SuperPoint keypoints with depth values\n";
    std::cerr << "- Real-time depth-based mapping\n";
    std::cerr << "\nView at http://localhost:9090\n";
    return 1;
  }

  try {
    // Retrieve paths
    const fs::path path_to_sequence(argv[3]);
    const fs::path associations_file(argv[4]);

    std::cout << "Loading RGB-D images from: " << path_to_sequence << "\n";
    std::cout << "Using associations: " << associations_file << "\n";
    RGBDImages images = LoadImages(associations_file);

    int nImages = images.rgb_filenames.size();
    std::cout << "Loaded " << nImages << " RGB-D pairs\n";

    // Create SLAM system with visualization enabled
    std::cout << "Initializing SuperSLAM RGB-D system with Rerun visualization...\n";
    SuperSLAM::System SLAM(argv[1], argv[2], SuperSLAM::System::RGBD, true);

    std::cout << "\n=== RGB-D SLAM with Rerun ===\n";
    std::cout << "Dataset: " << path_to_sequence << "\n";
    std::cout << "RGB-D pairs: " << nImages << "\n";
    std::cout << "Visualization: http://localhost:9090\n";
    std::cout << "=============================\n\n";

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack(nImages);

    // Performance tracking
    int nLostFrames = 0;

    // Main processing loop
    cv::Mat imRGB, imD;
    auto start_time = std::chrono::steady_clock::now();
    
    for (int ni = 0; ni < nImages; ni++) {
      // Read RGB and depth images
      std::string rgb_path = (path_to_sequence / images.rgb_filenames[ni]).string();
      std::string depth_path = (path_to_sequence / images.depth_filenames[ni]).string();
      
      imRGB = cv::imread(rgb_path, cv::IMREAD_UNCHANGED);
      imD = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
      double tframe = images.timestamps[ni];

      if (imRGB.empty() || imD.empty()) {
        std::cerr << "Failed to load RGB-D pair " << ni << "\n";
        continue;
      }

      auto t1 = std::chrono::steady_clock::now();

      // Pass RGB-D images to SLAM system
      cv::Mat pose = SLAM.TrackRGBD(imRGB, imD, tframe);
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
    std::cout << "Total RGB-D pairs: " << nImages << "\n";
    std::cout << "Median tracking time: " << vTimesTrack[nImages / 2] * 1000 << " ms\n";
    std::cout << "Mean tracking time: " << (totaltime / nImages) * 1000 << " ms\n";
    std::cout << "Average FPS: " << nImages / totaltime << "\n";
    std::cout << "Lost frames: " << nLostFrames << " (" 
              << (100.0 * nLostFrames / nImages) << "%)\n";

    // Save trajectory
    std::cout << "\nSaving trajectories...\n";
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_RGBD.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory_RGBD.txt");
    
    std::cout << "Saved:\n";
    std::cout << "- KeyFrameTrajectory_RGBD.txt\n";
    std::cout << "- CameraTrajectory_RGBD.txt\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}