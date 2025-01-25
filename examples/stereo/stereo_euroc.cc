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

struct StereoImages {
  std::vector<std::string> left_images;
  std::vector<std::string> right_images;
  std::vector<double> timestamps;
};

StereoImages LoadImages(const fs::path &path_left, const fs::path &path_right,
                        const fs::path &path_times) {
  StereoImages images;
  std::ifstream fTimes(path_times);
  if (!fTimes.is_open()) {
    throw std::runtime_error("Could not open timestamps file: " +
                             path_times.string());
  }

  std::string line;
  while (std::getline(fTimes, line)) {
    if (!line.empty()) {
      std::stringstream ss(line);
      std::string timestamp;
      ss >> timestamp;
      images.left_images.push_back((path_left / (timestamp + ".png")).string());
      images.right_images.push_back(
          (path_right / (timestamp + ".png")).string());
      double t;
      ss >> t;
      images.timestamps.push_back(t / 1e9);
    }
  }

  return images;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "Usage: ./stereo_euroc path_to_vocabulary path_to_settings "
                 "path_to_left_folder path_to_right_folder path_to_times_file"
              << std::endl;
    return 1;
  }

  try {
    // Retrieve paths to images
    const fs::path path_left(argv[3]);
    const fs::path path_right(argv[4]);
    const fs::path path_times(argv[5]);

    StereoImages images = LoadImages(path_left, path_right, path_times);

    if (images.left_images.empty() || images.right_images.empty()) {
      throw std::runtime_error("ERROR: No images in provided path.");
    }

    if (images.left_images.size() != images.right_images.size()) {
      throw std::runtime_error(
          "ERROR: Different number of left and right images.");
    }

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      throw std::runtime_error("ERROR: Wrong path to settings");
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;
    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;
    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;
    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() ||
        R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
        rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
      throw std::runtime_error(
          "ERROR: Calibration parameters to rectify stereo are missing!");
    }

    cv::Mat M1l, M2l, M1r, M2r;
    cv::initUndistortRectifyMap(K_l, D_l, R_l,
                                P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l), CV_32F, M1l, M2l);
    cv::initUndistortRectifyMap(K_r, D_r, R_r,
                                P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r), CV_32F, M1r, M2r);

    const int nImages = images.left_images.size();

    // Create SLAM system. It initializes all system threads and gets ready to
    // process frames.
    SuperSLAM::System SLAM(argv[1], argv[2], SuperSLAM::System::STEREO, true);

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack(nImages);

    std::cout << "\n-------\n";
    std::cout << "Start processing sequence ...\n";
    std::cout << "Images in the sequence: " << nImages << "\n\n";

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    for (int ni = 0; ni < nImages; ni++) {
      // Read left and right images from file
      imLeft = cv::imread(images.left_images[ni], cv::IMREAD_UNCHANGED);
      imRight = cv::imread(images.right_images[ni], cv::IMREAD_UNCHANGED);

      if (imLeft.empty()) {
        throw std::runtime_error("Failed to load image at: " +
                                 images.left_images[ni]);
      }

      if (imRight.empty()) {
        throw std::runtime_error("Failed to load image at: " +
                                 images.right_images[ni]);
      }

      cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
      cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);

      double tframe = images.timestamps[ni];

      auto t1 = std::chrono::steady_clock::now();

      // Pass the images to the SLAM system
      SLAM.TrackStereo(imLeftRect, imRightRect, tframe);

      auto t2 = std::chrono::steady_clock::now();

      double ttrack =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
              .count();

      vTimesTrack[ni] = ttrack;

      // Wait to load the next frame
      double T = 0;
      if (ni < nImages - 1)
        T = images.timestamps[ni + 1] - tframe;
      else if (ni > 0)
        T = tframe - images.timestamps[ni - 1];

      if (ttrack < T)
        std::this_thread::sleep_for(
            std::chrono::microseconds(static_cast<long>((T - ttrack) * 1e6)));
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    std::sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime =
        std::accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);
    std::cout << "-------\n\n";
    std::cout << "median tracking time: " << vTimesTrack[nImages / 2] << "\n";
    std::cout << "mean tracking time: " << totaltime / nImages << "\n";

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}