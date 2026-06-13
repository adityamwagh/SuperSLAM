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

// EuRoC stereo example. Images are raw/distorted, so they are rectified from
// the LEFT.*/RIGHT.* matrices in the settings yaml before tracking. Rerun
// visualization is on by default; pass --no-viewer for headless runs.

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <cstring>

#include <fstream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "SuperSLAM.h"
#include "example_common.h"

namespace {

// Read EuRoC cam0/data.csv: "#timestamp [ns],filename". Returns image
// basenames (shared by cam0/cam1, hardware-synced) and timestamps in seconds.
std::vector<std::string> load_images(const std::string& csv_file,
                                     std::vector<double>& vTimestamps) {
  std::ifstream f(csv_file.c_str());
  if (!f.is_open()) {
    spdlog::error("Failed to open EuRoC csv: {}", csv_file);
    return {};
  }

  std::vector<std::string> vstrImageFilenames;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream ss(line);
    std::string ts_str, fname;
    if (std::getline(ss, ts_str, ',') && std::getline(ss, fname)) {
      while (!fname.empty() && (fname.back() == '\r' || fname.back() == ' ')) {
        fname.pop_back();
      }
      vTimestamps.emplace_back(std::stod(ts_str) / 1e9);
      vstrImageFilenames.emplace_back(fname);
    }
  }
  spdlog::info("Loaded {} timestamps from {}", vstrImageFilenames.size(), csv_file);
  return vstrImageFilenames;
}

// Parse a {rows, cols, data: [...]} node into a CV_64F cv::Mat (returns empty
// on a missing/malformed node, matching the old FileStorage >> behaviour).
cv::Mat yaml_to_mat(const YAML::Node& n) {
  if (!n || !n["rows"] || !n["cols"] || !n["data"])
    return cv::Mat();
  const int rows = n["rows"].as<int>(), cols = n["cols"].as<int>();
  const auto data = n["data"].as<std::vector<double>>();
  if (static_cast<int>(data.size()) != rows * cols)
    return cv::Mat();
  cv::Mat m(rows, cols, CV_64F);
  std::memcpy(m.data, data.data(), data.size() * sizeof(double));
  return m;
}

// Build left/right rectification remap tables from LEFT.*/RIGHT.* in settings.
bool build_rectify_maps(const std::string& settings_path,
                        cv::Mat& M1l,
                        cv::Mat& M2l,
                        cv::Mat& M1r,
                        cv::Mat& M2r) {
  YAML::Node fs;
  try {
    fs = YAML::LoadFile(settings_path);
  } catch (const YAML::Exception& e) {
    spdlog::error("Failed to open settings for rectification: {} ({})", settings_path, e.what());
    return false;
  }

  cv::Mat K_l = yaml_to_mat(fs["LEFT.K"]), K_r = yaml_to_mat(fs["RIGHT.K"]);
  cv::Mat D_l = yaml_to_mat(fs["LEFT.D"]), D_r = yaml_to_mat(fs["RIGHT.D"]);
  cv::Mat R_l = yaml_to_mat(fs["LEFT.R"]), R_r = yaml_to_mat(fs["RIGHT.R"]);
  cv::Mat P_l = yaml_to_mat(fs["LEFT.P"]), P_r = yaml_to_mat(fs["RIGHT.P"]);
  int rows_l = fs["LEFT.height"].as<int>(0), cols_l = fs["LEFT.width"].as<int>(0);
  int rows_r = fs["RIGHT.height"].as<int>(0), cols_r = fs["RIGHT.width"].as<int>(0);

  if (K_l.empty() || K_r.empty() || D_l.empty() || D_r.empty() || R_l.empty() || R_r.empty() ||
      P_l.empty() || P_r.empty() || rows_l == 0 || cols_l == 0 || rows_r == 0 || cols_r == 0) {
    spdlog::error("Rectification matrices (LEFT.*/RIGHT.*) missing or empty "
                  "in settings");
    return false;
  }

  cv::initUndistortRectifyMap(K_l,
                              D_l,
                              R_l,
                              P_l.rowRange(0, 3).colRange(0, 3),
                              cv::Size(cols_l, rows_l),
                              CV_32F,
                              M1l,
                              M2l);
  cv::initUndistortRectifyMap(K_r,
                              D_r,
                              R_r,
                              P_r.rowRange(0, 3).colRange(0, 3),
                              cv::Size(cols_r, rows_r),
                              CV_32F,
                              M1r,
                              M2r);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  superslam::ExampleArgs args;
  if (!superslam::parse_example_args(argc, argv, "euroc", args)) {
    return 1;
  }

  cv::Mat M1l, M2l, M1r, M2r;
  if (!build_rectify_maps(args.settings, M1l, M2l, M1r, M2r)) {
    return 1;
  }

  const std::string csv_file = args.sequence + "/mav0/cam0/data.csv";
  std::vector<double> vTimestamps;
  const std::vector<std::string> vstrImageFilenames = load_images(csv_file, vTimestamps);
  const std::size_t nImages = vstrImageFilenames.size();
  if (nImages == 0) {
    spdlog::error("No images found (need {}/mav0/cam0/data.csv)", args.sequence);
    return 1;
  }
  spdlog::info("Found {} stereo image pairs", nImages);

  const std::string left_dir = args.sequence + "/mav0/cam0/data/";
  const std::string right_dir = args.sequence + "/mav0/cam1/data/";

  superslam::SuperSLAM slam(args.settings, args.use_viewer);

  std::vector<float> vTimesTrack(nImages, 0.0f);
  cv::Mat imLeftRect, imRightRect;
  for (std::size_t ni = 0; ni < nImages; ++ni) {
    cv::Mat imLeft = cv::imread(left_dir + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
    cv::Mat imRight = cv::imread(right_dir + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
    if (imLeft.empty() || imRight.empty()) {
      spdlog::error("Failed to load stereo pair {}: {}", ni, vstrImageFilenames[ni]);
      continue;
    }

    cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
    cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);

    const double tframe = vTimestamps[ni];
    const auto t1 = std::chrono::steady_clock::now();
    slam.track_stereo(imLeftRect, imRightRect, tframe);
    const auto t2 = std::chrono::steady_clock::now();
    vTimesTrack[ni] =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;

    if ((ni + 1) % 50 == 0 || ni == 0) {
      spdlog::info("Processed frame {}/{} - {:.2f}ms", ni + 1, nImages, vTimesTrack[ni]);
    }

    if (args.use_viewer && ni + 1 < nImages) {
      superslam::pace_to_timestamp(vTimesTrack[ni], vTimestamps[ni + 1] - tframe);
    }
  }

  slam.shutdown();
  superslam::report_timing(vTimesTrack);
  slam.save_trajectory("CameraTrajectory_euroc.txt", superslam::SuperSLAM::TrajectoryFormat::TUM);
  spdlog::info("Trajectory saved: CameraTrajectory_euroc.txt");
  return 0;
}
