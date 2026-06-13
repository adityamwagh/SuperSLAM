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

// KITTI stereo example. Rerun visualization (cameras, map points, trajectory)
// is on by default; pass --no-viewer for headless runs.

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "SuperSLAM.h"
#include "example_common.h"

namespace {

std::vector<std::string> load_images(const std::string& strFile, std::vector<double>& vTimestamps) {
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
      std::stringstream filename_ss;
      filename_ss << std::setfill('0') << std::setw(6) << vTimestamps.size() - 1 << ".png";
      vstrImageFilenames.emplace_back(filename_ss.str());
    }
  }

  spdlog::info("Loaded {} images from timestamp file", vstrImageFilenames.size());
  return vstrImageFilenames;
}

} // namespace

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  superslam::ExampleArgs args;
  if (!superslam::parse_example_args(argc, argv, "kitti", args)) {
    return 1;
  }

  const std::string times_file = args.sequence + "/times.txt";
  const std::string left_dir = args.sequence + "/image_0/";
  const std::string right_dir = args.sequence + "/image_1/";
  if (!std::ifstream(times_file).good() || !std::ifstream(left_dir + "000000.png").good()) {
    spdlog::error("Not a KITTI sequence dir (need times.txt + image_0/000000.png): {}",
                  args.sequence);
    return 1;
  }

  std::vector<double> vTimestamps;
  const std::vector<std::string> vstrImageFilenames = load_images(times_file, vTimestamps);
  const std::size_t nImages = vstrImageFilenames.size();
  if (nImages == 0) {
    spdlog::error("No images found in sequence");
    return 1;
  }
  spdlog::info("Found {} stereo image pairs", nImages);

  superslam::SuperSLAM slam(args.settings, args.use_viewer);

  std::vector<float> vTimesTrack(nImages, 0.0f);
  for (std::size_t ni = 0; ni < nImages; ++ni) {
    cv::Mat imLeft = cv::imread(left_dir + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
    cv::Mat imRight = cv::imread(right_dir + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
    if (imLeft.empty() || imRight.empty()) {
      spdlog::error("Failed to load stereo pair {}: {}", ni, vstrImageFilenames[ni]);
      continue;
    }

    const double tframe = vTimestamps[ni];
    const auto t1 = std::chrono::steady_clock::now();
    slam.track_stereo(imLeft, imRight, tframe);
    const auto t2 = std::chrono::steady_clock::now();
    vTimesTrack[ni] =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;

    if ((ni + 1) % 50 == 0 || ni == 0) {
      spdlog::info("Processed frame {}/{} - {:.2f}ms", ni + 1, nImages, vTimesTrack[ni]);
    }

    // When visualizing, pace to the dataset rate so the viewer is watchable.
    if (args.use_viewer && ni + 1 < nImages) {
      superslam::pace_to_timestamp(vTimesTrack[ni], vTimestamps[ni + 1] - tframe);
    }
  }

  slam.shutdown();
  superslam::report_timing(vTimesTrack);
  slam.save_trajectory_kitti("CameraTrajectory_kitti.txt");
  spdlog::info("Trajectory saved: CameraTrajectory_kitti.txt (KITTI)");
  return 0;
}
