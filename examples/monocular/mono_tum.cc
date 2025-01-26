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
    std::cerr << "Usage: ./mono_tum path_to_vocabulary path_to_settings "
                 "path_to_sequence"
              << "\n";
    return 1;
  }

  try {
    // Retrieve paths to images
    const fs::path path_to_sequence(argv[3]);
    const fs::path file_path = path_to_sequence / "rgb.txt";

    MonocularImages images = LoadImages(file_path);

    int nImages = images.image_filenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to
    // process frames.
    SuperSLAM::System SLAM(argv[1], argv[2], SuperSLAM::System::MONOCULAR,
                           true);

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack(nImages);

    std::cout << "\n-------\n";
    std::cout << "Start processing sequence ...\n";
    std::cout << "Images in the sequence: " << nImages << "\n\n";

    // Main loop
    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++) {
      // Read image from file
      im = cv::imread((path_to_sequence / images.image_filenames[ni]).string(),
                      cv::IMREAD_UNCHANGED);
      double tframe = images.timestamps[ni];

      if (im.empty()) {
        throw std::runtime_error(
            "Failed to load image at: " +
            (path_to_sequence / images.image_filenames[ni]).string());
      }

      auto t1 = std::chrono::steady_clock::now();

      // Pass the image to the SLAM system
      SLAM.TrackMonocular(im, tframe);

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
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }

  return 0;
}