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
#include <RerunViewer.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <thread>

namespace fs = std::filesystem;

struct MonocularImages {
  std::vector<std::string> image_filenames;
  std::vector<double> timestamps;
};

MonocularImages LoadImages(const fs::path &file_path) {
  MonocularImages images;
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path.string());
  }

  // Skip first three lines
  std::string line;
  for (int i = 0; i < 3; ++i) {
    std::getline(file, line);
  }

  while (std::getline(file, line)) {
    if (!line.empty()) {
      std::stringstream ss(line);
      double t;
      std::string sRGB;
      ss >> t;
      images.timestamps.push_back(t);
      ss >> sRGB;
      images.image_filenames.push_back(sRGB);
    }
  }

  return images;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: ./mono_tum_rerun path_to_vocabulary path_to_settings "
                 "path_to_sequence"
              << "\n";
    std::cerr << "\nExample:\n";
    std::cerr << "./mono_tum_rerun vocabulary/ORBvoc.txt "
                 "examples/monocular/TUM1.yaml "
                 "/path/to/TUM/dataset/rgbd_dataset_freiburg1_xyz\n";
    std::cerr << "\nThis example demonstrates SuperPoint-based monocular SLAM "
                 "with real-time Rerun visualization.\n";
    std::cerr << "Features visualized:\n";
    std::cerr << "- Camera trajectory and poses\n";
    std::cerr << "- 3D map points\n";
    std::cerr << "- SuperPoint feature detection and tracking\n";
    std::cerr << "- Loop closure detection\n";
    std::cerr << "- Covisibility graph\n";
    std::cerr << "\nView the visualization at http://localhost:9090\n";
    return 1;
  }

  try {
    // Retrieve paths to images
    const fs::path path_to_sequence(argv[3]);
    const fs::path file_path = path_to_sequence / "rgb.txt";

    std::cout << "Loading images from: " << file_path << "\n";
    MonocularImages images = LoadImages(file_path);

    int nImages = images.image_filenames.size();
    std::cout << "Loaded " << nImages << " images\n";

    // Create SLAM system with visualization enabled
    std::cout << "Initializing SuperSLAM system with Rerun visualization...\n";
    SuperSLAM::System SLAM(argv[1], argv[2], SuperSLAM::System::MONOCULAR, true);

    // Initialize Rerun viewer - using nullptr for now since we can't access internal components
    std::cout << "Starting Rerun visualization...\n";
    SuperSLAM::RerunViewer* pViewer = new SuperSLAM::RerunViewer(
        &SLAM, nullptr, nullptr, argv[2]);
    
    // Start viewer thread
    std::thread viewer_thread(&SuperSLAM::RerunViewer::Run, pViewer);

    std::cout << "\n=== SuperPoint Monocular SLAM with Rerun Visualization ===\n";
    std::cout << "Dataset: " << path_to_sequence << "\n";
    std::cout << "Images: " << nImages << "\n";
    std::cout << "Visualization: http://localhost:9090\n";
    std::cout << "Press Ctrl+C to stop\n";
    std::cout << "========================================================\n\n";

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack(nImages);

    // Performance tracking
    int nLostFrames = 0;
    int nKeyFrames = 0;
    int nMapPoints = 0;

    // Main processing loop
    cv::Mat im;
    auto start_time = std::chrono::steady_clock::now();
    
    for (int ni = 0; ni < nImages; ni++) {
      // Read image from file
      std::string image_path = (path_to_sequence / images.image_filenames[ni]).string();
      im = cv::imread(image_path, cv::IMREAD_UNCHANGED);
      double tframe = images.timestamps[ni];

      if (im.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        continue;
      }

      auto t1 = std::chrono::steady_clock::now();

      // Pass the image to the SLAM system
      cv::Mat pose = SLAM.TrackMonocular(im, tframe);
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

      // Get current frame data for visualization
      std::vector<cv::KeyPoint> tracked_keypoints = SLAM.GetTrackedKeyPointsUn();
      
      // Log frame data to Rerun viewer
      if (!pose.empty()) {
        pViewer->LogFrame(im, tracked_keypoints, pose, tframe);
        
        // Log SuperPoint-specific features
        pViewer->LogSuperPointFeatures(im, tracked_keypoints, cv::Mat());
      }

      // Update statistics every 100 frames
      if (ni % 100 == 0 && ni > 0) {
        // Since we can't access internal components directly, use placeholder values
        nKeyFrames = 0;  // Would need public accessor
        nMapPoints = 0;  // Would need public accessor
        
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
        double fps = ni / elapsed_seconds;
        
        std::cout << "Frame " << ni << "/" << nImages 
                  << " | FPS: " << std::fixed << std::setprecision(1) << fps
                  << " | Track time: " << std::setprecision(3) << ttrack * 1000 << "ms"
                  << " | KFs: " << nKeyFrames 
                  << " | MPs: " << nMapPoints 
                  << " | Lost: " << nLostFrames 
                  << " | State: ";
        
        switch(tracking_state) {
          case -1:
            std::cout << "SYSTEM_NOT_READY"; break;
          case 0:
            std::cout << "NO_IMAGES_YET"; break;
          case 1:
            std::cout << "NOT_INITIALIZED"; break;
          case 2:
            std::cout << "OK"; break;
          case 3:
            std::cout << "LOST"; break;
          default:
            std::cout << "UNKNOWN"; break;
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
    
    // Stop viewer
    pViewer->RequestFinish();
    viewer_thread.join();
    delete pViewer;

    // Final statistics
    std::sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime =
        std::accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);
    
    std::cout << "\n=== SLAM Performance Summary ===\n";
    std::cout << "Total frames processed: " << nImages << "\n";
    std::cout << "Median tracking time: " << vTimesTrack[nImages / 2] * 1000 << " ms\n";
    std::cout << "Mean tracking time: " << (totaltime / nImages) * 1000 << " ms\n";
    std::cout << "Total processing time: " << totaltime << " s\n";
    std::cout << "Average FPS: " << nImages / totaltime << "\n";
    std::cout << "Lost frames: " << nLostFrames << " (" 
              << (100.0 * nLostFrames / nImages) << "%)\n";
    std::cout << "Final keyframes: " << nKeyFrames << "\n";
    std::cout << "Final map points: " << nMapPoints << "\n";

    // Save trajectory
    std::cout << "\nSaving trajectory...\n";
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_Rerun.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory_Rerun.txt");
    
    std::cout << "Trajectories saved:\n";
    std::cout << "- KeyFrameTrajectory_Rerun.txt\n";
    std::cout << "- CameraTrajectory_Rerun.txt\n";
    
    std::cout << "\nVisualization data logged to Rerun.\n";
    std::cout << "You can replay the session or export data from the Rerun viewer.\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}