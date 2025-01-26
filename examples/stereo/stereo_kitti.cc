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

StereoImages LoadImages(const fs::path &path_to_sequence) {
  StereoImages images;
  fs::path path_time_file = path_to_sequence / "times.txt";
  std::ifstream fTimes(path_time_file);
  if (!fTimes.is_open()) {
    throw std::runtime_error("Could not open timestamps file: " +
                             path_time_file.string());
  }

  std::string line;
  while (std::getline(fTimes, line)) {
    if (!line.empty()) {
      std::stringstream ss(line);
      double t;
      ss >> t;
      images.timestamps.push_back(t);
    }
  }

  fs::path prefix_left = path_to_sequence / "image_0";
  fs::path prefix_right = path_to_sequence / "image_1";

  const int nTimes = images.timestamps.size();
  images.left_images.resize(nTimes);
  images.right_images.resize(nTimes);

  for (int i = 0; i < nTimes; i++) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << i;
    images.left_images[i] = (prefix_left / (ss.str() + ".png")).string();
    images.right_images[i] = (prefix_right / (ss.str() + ".png")).string();
  }

  return images;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings "
                 "path_to_sequence"
              << "\n";
    return 1;
  }

  try {
    // Retrieve paths to images
    const fs::path path_to_sequence(argv[3]);
    StereoImages images = LoadImages(path_to_sequence);

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
    cv::Mat imLeft, imRight;
    for (int ni = 0; ni < nImages; ni++) {
      // Read left and right images from file
      imLeft = cv::imread(images.left_images[ni], cv::IMREAD_UNCHANGED);
      imRight = cv::imread(images.right_images[ni], cv::IMREAD_UNCHANGED);
      double tframe = images.timestamps[ni];

      if (imLeft.empty()) {
        throw std::runtime_error("Failed to load image at: " +
                                 images.left_images[ni]);
      }

      auto t1 = std::chrono::steady_clock::now();

      // Pass the images to the SLAM system
      SLAM.TrackStereo(imLeft, imRight, tframe);

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
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }

  return 0;
}