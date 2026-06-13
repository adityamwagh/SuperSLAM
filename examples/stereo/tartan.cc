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

// TartanAir / TartanGround stereo example. The sequence dir is a trajectory
// folder with image_lcam_front/ + image_rcam_front/. Rerun is on by default;
// pass --no-viewer.

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "SuperSLAM.h"
#include "example_common.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  superslam::ExampleArgs args;
  if (!superslam::parse_example_args(argc, argv, "tartan", args))
    return 1;

  const std::string left_dir = args.sequence + "/image_lcam_front/";
  if (!fs::exists(left_dir)) {
    spdlog::error("Not a TartanAir/TartanGround trajectory dir (need "
                  "image_lcam_front/): {}",
                  args.sequence);
    return 1;
  }

  std::vector<std::string> left_files;
  for (const auto& e : fs::directory_iterator(left_dir))
    if (e.path().extension() == ".png")
      left_files.push_back(e.path().filename().string());
  std::sort(left_files.begin(), left_files.end());
  if (left_files.empty()) {
    spdlog::error("No images in {}", left_dir);
    return 1;
  }
  spdlog::info("Found {} stereo frames", left_files.size());

  superslam::SuperSLAM slam(args.settings, args.use_viewer);

  std::vector<float> vTimesTrack;
  vTimesTrack.reserve(left_files.size());
  for (std::size_t ni = 0; ni < left_files.size(); ++ni) {
    std::string right_name = left_files[ni];
    right_name.replace(right_name.find("lcam"), 4, "rcam");
    const cv::Mat imLeft = cv::imread(left_dir + left_files[ni], cv::IMREAD_UNCHANGED);
    const cv::Mat imRight =
        cv::imread(args.sequence + "/image_rcam_front/" + right_name, cv::IMREAD_UNCHANGED);
    if (imLeft.empty() || imRight.empty()) {
      spdlog::error("Failed to load stereo pair {}: {}", ni, left_files[ni]);
      continue;
    }

    const double tframe = 0.1 * static_cast<double>(ni);
    const auto t1 = std::chrono::steady_clock::now();
    slam.track_stereo(imLeft, imRight, tframe);
    const auto t2 = std::chrono::steady_clock::now();
    const float ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
    vTimesTrack.push_back(ms);

    if ((ni + 1) % 50 == 0 || ni == 0)
      spdlog::info("Processed frame {}/{} - {:.2f}ms", ni + 1, left_files.size(), ms);
    if (args.use_viewer && ni + 1 < left_files.size())
      superslam::pace_to_timestamp(ms, 0.1);
  }

  slam.shutdown();
  superslam::report_timing(vTimesTrack);
  slam.save_trajectory_kitti("CameraTrajectory_tartan.txt");
  spdlog::info("Trajectory saved: CameraTrajectory_tartan.txt");
  return 0;
}
