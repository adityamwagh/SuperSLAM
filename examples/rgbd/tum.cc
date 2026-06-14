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

// TUM RGB-D example. Rerun visualization is on by default; pass --no-viewer
// for headless runs.

#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "SuperSLAM.h"
#include "example_common.h"

namespace {

bool load_list(const std::string& file, std::vector<double>& ts, std::vector<std::string>& paths) {
  std::ifstream f(file.c_str());
  if (!f.is_open())
    return false;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::stringstream ss(line);
    double t;
    std::string p;
    if (ss >> t >> p) {
      ts.push_back(t);
      paths.push_back(p);
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  superslam::ExampleArgs args;
  if (!superslam::parse_example_args(argc, argv, "tum", args))
    return 1;

  std::vector<double> tRgb, tDepth;
  std::vector<std::string> rgbPaths, depthPaths;
  if (!load_list(args.sequence + "/rgb.txt", tRgb, rgbPaths) ||
      !load_list(args.sequence + "/depth.txt", tDepth, depthPaths)) {
    spdlog::error("Not a TUM RGB-D dir (need rgb.txt + depth.txt): {}", args.sequence);
    return 1;
  }

  // Associate each RGB frame with the nearest depth frame within 20 ms.
  const double kMaxDt = 0.02;
  std::vector<std::size_t> depthFor(rgbPaths.size(), std::numeric_limits<std::size_t>::max());
  for (std::size_t i = 0; i < tRgb.size(); ++i) {
    double best = kMaxDt;
    for (std::size_t k = 0; k < tDepth.size(); ++k) {
      const double dt = std::fabs(tRgb[i] - tDepth[k]);
      if (dt < best) {
        best = dt;
        depthFor[i] = k;
      }
    }
  }

  superslam::SuperSLAM slam(args.settings, args.use_viewer);

  std::vector<float> vTimesTrack;
  vTimesTrack.reserve(rgbPaths.size());
  for (std::size_t i = 0; i < rgbPaths.size(); ++i) {
    if (depthFor[i] == std::numeric_limits<std::size_t>::max())
      continue;
    const cv::Mat rgb = cv::imread(args.sequence + "/" + rgbPaths[i], cv::IMREAD_COLOR);
    const cv::Mat depth =
        cv::imread(args.sequence + "/" + depthPaths[depthFor[i]], cv::IMREAD_UNCHANGED);
    if (rgb.empty() || depth.empty()) {
      spdlog::error("Failed to load RGB-D frame {}: {}", i, rgbPaths[i]);
      continue;
    }

    const double tframe = tRgb[i];
    const auto t1 = std::chrono::steady_clock::now();
    slam.track_rgbd(rgb, depth, tframe);
    const auto t2 = std::chrono::steady_clock::now();
    const float ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
    vTimesTrack.push_back(ms);

    if (vTimesTrack.size() % 50 == 0 || i == 0)
      spdlog::info("Processed frame {}/{} - {:.2f}ms", vTimesTrack.size(), rgbPaths.size(), ms);

    if (args.use_viewer && i + 1 < rgbPaths.size())
      superslam::pace_to_timestamp(ms, tRgb[i + 1] - tframe);
  }

  slam.shutdown();
  superslam::report_timing(vTimesTrack);
  slam.save_trajectory("CameraTrajectory_tum.txt", superslam::SuperSLAM::TrajectoryFormat::TUM);
  spdlog::info("Trajectory saved: CameraTrajectory_tum.txt");
  return 0;
}
