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

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "System.h"

namespace {

constexpr char kUsageMessage[] = R"(
Usage: ./mono_kitti_rerun vocabulary_path settings_file_path sequence_path

Arguments:
  vocabulary_path    - Path to the ORB vocabulary file
  settings_file_path - Path to the settings file  
  sequence_path      - Path to the KITTI sequence directory

Example:
  ./mono_kitti_rerun vocabulary/ORBvoc.txt examples/monocular/KITTI00-02.yaml /path/to/kitti/sequences/00
)";

std::vector<std::string> LoadImages(const std::string& strFile,
                                    std::vector<double>& vTimestamps) {
  std::ifstream f(strFile.c_str());
  if (!f.is_open()) {
    spdlog::error("Failed to open timestamp file: {}", strFile);
    return {};
  }

  std::vector<std::string> vstrImageFilenames;
  std::string s;
  while (std::getline(f, s) && !s.empty()) {
    std::stringstream ss(s);
    std::string timestamp_str;
    if (std::getline(ss, timestamp_str)) {
      vTimestamps.emplace_back(std::stod(timestamp_str));

      // Construct image filename with proper zero-padding
      std::stringstream filename_ss;
      filename_ss << std::setfill('0') << std::setw(6) << vTimestamps.size() - 1
                  << ".png";
      vstrImageFilenames.emplace_back(filename_ss.str());
    }
  }

  spdlog::info("Loaded {} images from timestamp file",
               vstrImageFilenames.size());
  return vstrImageFilenames;
}

bool ValidateInputs(const std::string& vocab_path,
                    const std::string& settings_path,
                    const std::string& sequence_path) {
  // Check vocabulary file
  if (!std::ifstream(vocab_path).good()) {
    spdlog::error("Cannot open vocabulary file: {}", vocab_path);
    return false;
  }

  // Check settings file
  if (!std::ifstream(settings_path).good()) {
    spdlog::error("Cannot open settings file: {}", settings_path);
    return false;
  }

  // Check if sequence directory exists
  const auto times_file = sequence_path + "/times.txt";
  if (!std::ifstream(times_file).good()) {
    spdlog::error("Cannot find times.txt in sequence directory: {}",
                  times_file);
    return false;
  }

  // Check if image_0 directory exists by trying to read a sample image
  const auto image_dir = sequence_path + "/image_0";
  const auto sample_image = image_dir + "/000000.png";
  if (!std::ifstream(sample_image).good()) {
    spdlog::error("Cannot find image_0 directory or sample image: {}",
                  sample_image);
    return false;
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  // Configure spdlog
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  if (argc != 4) {
    spdlog::error("Wrong number of arguments");
    spdlog::info("{}", kUsageMessage);
    return 1;
  }

  const std::string vocab_path = argv[1];
  const std::string settings_file = argv[2];
  const std::string sequence_path = argv[3];

  spdlog::info(
      "Starting SuperSLAM Monocular KITTI Example with Rerun Visualization");
  spdlog::info("Vocabulary: {}", vocab_path);
  spdlog::info("Settings: {}", settings_file);
  spdlog::info("Sequence: {}", sequence_path);

  if (!ValidateInputs(vocab_path, settings_file, sequence_path)) {
    return 1;
  }

  // Retrieve paths to images
  std::vector<std::string> vstrImageFilenames;
  std::vector<double> vTimestamps;
  const std::string times_file = sequence_path + "/times.txt";

  vstrImageFilenames = LoadImages(times_file, vTimestamps);
  const std::size_t nImages = vstrImageFilenames.size();

  if (nImages == 0) {
    spdlog::error("No images found in sequence");
    return 1;
  }

  spdlog::info("Found {} images in sequence", nImages);

  // Initialize the SLAM system with Rerun viewer enabled
  spdlog::info("Initializing SLAM system (this may take a moment)...");
  auto slam_system = std::make_unique<SuperSLAM::System>(
      vocab_path, settings_file, SuperSLAM::System::MONOCULAR, true);

  if (!slam_system) {
    spdlog::error("Failed to initialize SLAM system");
    return 1;
  }

  spdlog::info("SLAM system initialized successfully");

  // Vector for tracking results
  std::vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  // Main loop
  cv::Mat im;
  spdlog::info("Starting main tracking loop");

  for (std::size_t ni = 0; ni < nImages; ++ni) {
    // Read image from file
    const std::string image_path =
        sequence_path + "/image_0/" + vstrImageFilenames[ni];
    im = cv::imread(image_path, cv::IMREAD_UNCHANGED);

    if (im.empty()) {
      spdlog::error("Failed to load image: {}", image_path);
      continue;
    }

    const double tframe = vTimestamps[ni];

    // Start timing
    const auto t1 = std::chrono::steady_clock::now();

    // Pass the image to the SLAM system
    cv::Mat Tcw = slam_system->TrackMonocular(im, tframe);

    const auto t2 = std::chrono::steady_clock::now();

    const auto ttrack =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    vTimesTrack[ni] = ttrack / 1000.0f;  // Convert to milliseconds

    // Log progress
    if ((ni + 1) % 50 == 0 || ni == 0) {
      spdlog::info("Processed frame {}/{} - Tracking time: {:.2f}ms", ni + 1,
                   nImages, vTimesTrack[ni]);
    }

    // Small delay to allow processing and reduce Rerun logging frequency
    if (ni < nImages - 1) {
      const double T = vTimestamps[ni + 1] - tframe;
      const auto min_delay =
          std::chrono::milliseconds(10);  // Minimum 10ms delay
      if (ttrack < T * 1000000.0) {
        const auto target_delay = std::chrono::microseconds(
            static_cast<long long>(T * 1000000.0 - ttrack));
        std::this_thread::sleep_for(std::max(
            min_delay, std::chrono::duration_cast<std::chrono::milliseconds>(
                           target_delay)));
      } else {
        std::this_thread::sleep_for(min_delay);
      }
    }
  }

  spdlog::info("Main tracking loop completed");

  // Stop all threads
  slam_system->Shutdown();

  // Tracking time statistics
  std::sort(vTimesTrack.begin(), vTimesTrack.end());
  const float totaltime =
      std::accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);

  spdlog::info("=== Tracking Performance Statistics ===");
  spdlog::info("Number of processed frames: {}", nImages);
  spdlog::info("Mean tracking time: {:.3f}ms", totaltime / nImages);
  spdlog::info("Median tracking time: {:.3f}ms", vTimesTrack[nImages / 2]);
  spdlog::info("Total tracking time: {:.3f}s", totaltime / 1000.0f);

  // Save trajectory in KITTI format
  const std::string trajectory_file = "KeyFrameTrajectory_KITTI_mono.txt";
  slam_system->SaveKeyFrameTrajectoryTUM(trajectory_file);
  spdlog::info("Trajectory saved to: {}", trajectory_file);

  spdlog::info("SuperSLAM Monocular KITTI Example completed successfully");
  return 0;
}