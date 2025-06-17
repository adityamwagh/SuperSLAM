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
Usage: ./stereo_kitti_no_viewer vocabulary_path settings_file_path sequence_path

Arguments:
  vocabulary_path    - Path to the ORB vocabulary file
  settings_file_path - Path to the settings file  
  sequence_path      - Path to the KITTI sequence directory

Example:
  ./stereo_kitti_no_viewer vocabulary/ORBvoc.txt examples/stereo/KITTI00-02.yaml /path/to/kitti/sequences/00
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

  // Check if both image directories exist by trying to read sample images
  const auto image_left_dir = sequence_path + "/image_0";
  const auto image_right_dir = sequence_path + "/image_1";
  const auto sample_left = image_left_dir + "/000000.png";
  const auto sample_right = image_right_dir + "/000000.png";

  if (!std::ifstream(sample_left).good()) {
    spdlog::error("Cannot find left image directory or sample image: {}",
                  sample_left);
    return false;
  }

  if (!std::ifstream(sample_right).good()) {
    spdlog::error("Cannot find right image directory or sample image: {}",
                  sample_right);
    return false;
  }

  return true;
}

std::pair<cv::Mat, cv::Mat> LoadStereoImages(const std::string& sequence_path,
                                             const std::string& filename) {
  const std::string left_path = sequence_path + "/image_0/" + filename;
  const std::string right_path = sequence_path + "/image_1/" + filename;

  cv::Mat left_img = cv::imread(left_path, cv::IMREAD_UNCHANGED);
  cv::Mat right_img = cv::imread(right_path, cv::IMREAD_UNCHANGED);

  if (left_img.empty()) {
    spdlog::warn("Failed to load left image: {}", left_path);
  }
  if (right_img.empty()) {
    spdlog::warn("Failed to load right image: {}", right_path);
  }

  return std::make_pair(std::move(left_img), std::move(right_img));
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

  spdlog::info("Starting SuperSLAM Stereo KITTI Example (No Viewer)");
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

  spdlog::info("Found {} stereo image pairs in sequence", nImages);

  // Initialize the SLAM system WITHOUT viewer to test core functionality
  spdlog::info(
      "Initializing SLAM system WITHOUT viewer (this may take a moment)...");
  auto slam_system = std::make_unique<SuperSLAM::System>(
      vocab_path, settings_file, SuperSLAM::System::STEREO, false);

  if (!slam_system) {
    spdlog::error("Failed to initialize SLAM system");
    return 1;
  }

  spdlog::info("SLAM system initialized successfully");

  // Vector for tracking results
  std::vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  // Main loop
  spdlog::info("Starting main tracking loop");

  for (std::size_t ni = 0; ni < nImages; ++ni) {
    // Read stereo images from file
    auto [imLeft, imRight] =
        LoadStereoImages(sequence_path, vstrImageFilenames[ni]);

    if (imLeft.empty() || imRight.empty()) {
      spdlog::error("Failed to load stereo pair for frame {}: {}", ni,
                    vstrImageFilenames[ni]);
      continue;
    }

    const double tframe = vTimestamps[ni];

    // Start timing
    const auto t1 = std::chrono::steady_clock::now();

    // Pass the stereo images to the SLAM system
    cv::Mat Tcw = slam_system->TrackStereo(imLeft, imRight, tframe);

    const auto t2 = std::chrono::steady_clock::now();

    const auto ttrack =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    vTimesTrack[ni] = ttrack / 1000.0f;  // Convert to milliseconds

    // Log progress every 10 frames to reduce noise
    if ((ni + 1) % 10 == 0 || ni == 0) {
      spdlog::info("Processed stereo frame {}/{} - Tracking time: {:.2f}ms",
                   ni + 1, nImages, vTimesTrack[ni]);
    }

    // Small delay to allow processing
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
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
  const std::string trajectory_file =
      "KeyFrameTrajectory_KITTI_stereo_no_viewer.txt";
  slam_system->SaveKeyFrameTrajectoryTUM(trajectory_file);
  spdlog::info("Trajectory saved to: {}", trajectory_file);

  // Save KITTI format trajectory (for stereo/RGBD datasets)
  const std::string kitti_trajectory_file =
      "CameraTrajectory_KITTI_stereo_no_viewer.txt";
  slam_system->SaveTrajectoryKITTI(kitti_trajectory_file);
  spdlog::info("KITTI format trajectory saved to: {}", kitti_trajectory_file);

  spdlog::info(
      "SuperSLAM Stereo KITTI Example (No Viewer) completed successfully");
  return 0;
}