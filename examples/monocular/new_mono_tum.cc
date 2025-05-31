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

struct TUMSequence {
  std::vector<std::string> image_filenames;
  std::vector<double> timestamps;

  void LoadFromFile(const fs::path& rgb_txt_path) {
    std::ifstream file(rgb_txt_path);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open TUM rgb.txt file: " +
                               rgb_txt_path.string());
    }

    std::string line;
    // Skip header lines that start with '#'
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::stringstream ss(line);
      double timestamp;
      std::string image_path;

      if (ss >> timestamp >> image_path) {
        timestamps.push_back(timestamp);
        image_filenames.push_back(image_path);
      }
    }

    std::cout << "Loaded " << image_filenames.size()
              << " images from TUM sequence" << "\n";
  }
};

void PrintUsage() {
  std::cout
      << "Usage: ./new_mono_tum vocabulary_path settings_path sequence_path"
      << "\n";
  std::cout << "  vocabulary_path: Path to SuperPoint vocabulary file"
            << "\n";
  std::cout << "  settings_path:   Path to camera configuration file"
            << "\n";
  std::cout << "  sequence_path:   Path to TUM dataset sequence directory"
            << "\n";
  std::cout << "\n";
  std::cout << "Example:" << "\n";
  std::cout
      << "  ./new_mono_tum vocabulary/ORBvoc.txt examples/monocular/TUM1.yaml "
         "/path/to/rgbd_dataset_freiburg1_xyz"
      << "\n";
  std::cout << "\n";
  std::cout
      << "The Rerun viewer will automatically open at http://localhost:9090"
      << "\n";
  std::cout << "You can view:" << "\n";
  std::cout << "  - Live camera feed with detected keypoints" << "\n";
  std::cout << "  - Real-time camera trajectory in 3D" << "\n";
  std::cout << "  - 3D map points" << "\n";
  std::cout << "  - KeyFrame poses and connections" << "\n";
}

int main(int argc, char** argv) {
  if (argc != 4) {
    PrintUsage();
    return 1;
  }

  try {
    const fs::path vocabulary_path(argv[1]);
    const fs::path settings_path(argv[2]);
    const fs::path sequence_path(argv[3]);
    const fs::path rgb_txt_path = sequence_path / "rgb.txt";

    // Validate input paths
    if (!fs::exists(vocabulary_path)) {
      throw std::runtime_error("Vocabulary file not found: " +
                               vocabulary_path.string());
    }
    if (!fs::exists(settings_path)) {
      throw std::runtime_error("Settings file not found: " +
                               settings_path.string());
    }
    if (!fs::exists(sequence_path)) {
      throw std::runtime_error("Sequence directory not found: " +
                               sequence_path.string());
    }
    if (!fs::exists(rgb_txt_path)) {
      throw std::runtime_error("rgb.txt file not found: " +
                               rgb_txt_path.string());
    }

    // Load TUM sequence
    TUMSequence sequence;
    sequence.LoadFromFile(rgb_txt_path);

    if (sequence.image_filenames.empty()) {
      throw std::runtime_error("No images found in sequence");
    }

    // Create SLAM system with Rerun viewer enabled
    std::cout << "\nInitializing SuperSLAM system with Rerun visualization..."
              << "\n";
    SuperSLAM::System SLAM(vocabulary_path.string(), settings_path.string(),
                           SuperSLAM::System::MONOCULAR, true);

    // Give the viewer a moment to initialize
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Vector for tracking time statistics
    std::vector<float> tracking_times;
    tracking_times.reserve(sequence.image_filenames.size());

    std::cout << "\n=== Starting SLAM Processing ===" << "\n";
    std::cout << "Sequence: " << sequence_path.filename() << "\n";
    std::cout << "Images: " << sequence.image_filenames.size() << "\n";
    std::cout << "Rerun Viewer: http://localhost:9090" << "\n";
    std::cout << "=================================" << "\n";
    std::cout << "\nVisualization features:" << "\n";
    std::cout << "• camera/image - Live camera feed" << "\n";
    std::cout << "• camera/keypoints - Detected SuperPoint features"
              << "\n";
    std::cout << "• world/camera/pose - Current camera pose" << "\n";
    std::cout << "• world/trajectory - Camera trajectory path" << "\n";
    std::cout << "• world/map_points - 3D map points" << "\n";
    std::cout << "• world/keyframes - KeyFrame poses" << "\n";
    std::cout << "=================================" << "\n";

    // Main processing loop
    cv::Mat image;
    int tracking_state = -1;
    int frames_processed = 0;

    for (size_t i = 0; i < sequence.image_filenames.size(); i++) {
      // Construct full image path
      fs::path image_path = sequence_path / sequence.image_filenames[i];

      // Load image
      image = cv::imread(image_path.string(), cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        continue;
      }

      double timestamp = sequence.timestamps[i];

      // Process frame with timing
      auto start_time = std::chrono::steady_clock::now();

      cv::Mat camera_pose = SLAM.TrackMonocular(image, timestamp);

      auto end_time = std::chrono::steady_clock::now();

      double processing_time =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                    start_time)
              .count();
      tracking_times.push_back(processing_time);

      // Get tracking state for status reporting
      int current_state = SLAM.GetTrackingState();
      if (current_state != tracking_state) {
        tracking_state = current_state;
        std::string state_name;
        switch (tracking_state) {
          case 0:
            state_name = "NO_IMAGES_YET";
            break;
          case 1:
            state_name = "NOT_INITIALIZED";
            break;
          case 2:
            state_name = "OK";
            break;
          case 3:
            state_name = "LOST";
            break;
          default:
            state_name = "UNKNOWN";
            break;
        }
        std::cout << "Tracking state changed to: " << state_name << "\n";
      }

      // Count successfully tracked frames
      if (current_state == 2) {  // OK
        frames_processed++;
      }

      // Print progress
      if ((i + 1) % 50 == 0 || i == 0) {
        std::cout << "Frame " << std::setw(4) << (i + 1) << "/"
                  << sequence.image_filenames.size()
                  << " | Time: " << std::fixed << std::setprecision(3)
                  << processing_time << "s"
                  << " | Tracked: " << std::setw(4) << frames_processed
                  << " | State: " << tracking_state << "\n";
      }

      // Maintain reasonable processing speed
      double target_time = 0.033;  // ~30 FPS
      if (i < sequence.image_filenames.size() - 1) {
        double time_to_next = sequence.timestamps[i + 1] - timestamp;
        target_time = std::min(time_to_next, 0.1);  // Cap at 100ms
      }

      if (processing_time < target_time) {
        std::this_thread::sleep_for(std::chrono::microseconds(
            static_cast<long>((target_time - processing_time) * 1e6)));
      }
    }

    // Give time for final visualization updates
    std::cout << "\nProcessing complete. Allowing time for final visualization "
                 "updates..."
              << "\n";
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Shutdown SLAM system
    std::cout << "Shutting down SLAM system..." << "\n";
    SLAM.Shutdown();

    // Print statistics
    std::sort(tracking_times.begin(), tracking_times.end());
    float total_time =
        std::accumulate(tracking_times.begin(), tracking_times.end(), 0.0f);

    std::cout << "\n=== Processing Statistics ===" << "\n";
    std::cout << "Total frames: " << tracking_times.size() << "\n";
    std::cout << "Successfully tracked: " << frames_processed << "\n";
    std::cout << "Success rate: " << std::fixed << std::setprecision(1)
              << (100.0 * frames_processed / tracking_times.size()) << "%"
              << "\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(2)
              << total_time << "s" << "\n";
    std::cout << "Average time per frame: " << std::fixed
              << std::setprecision(4) << total_time / tracking_times.size()
              << "s" << "\n";
    std::cout << "Median time per frame: " << std::fixed << std::setprecision(4)
              << tracking_times[tracking_times.size() / 2] << "s" << "\n";
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2)
              << tracking_times.size() / total_time << "\n";
    std::cout << "=============================" << "\n";

    // Save trajectory if we have successful tracking
    if (frames_processed > 0) {
      std::cout << "\nSaving trajectory..." << "\n";
      SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
      std::cout << "Trajectory saved to KeyFrameTrajectory.txt" << "\n";
    } else {
      std::cout
          << "\nNo trajectory to save (no frames were successfully tracked)"
          << "\n";
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  std::cout << "\nSLAM processing completed!" << "\n";
  std::cout << "The Rerun viewer will continue running at http://localhost:9090"
            << "\n";
  std::cout << "You can explore the visualization and press Ctrl+C to exit."
            << "\n";

  return 0;
}