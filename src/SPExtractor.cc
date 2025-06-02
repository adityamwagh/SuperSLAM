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

#include "SPExtractor.h"

#include <algorithm>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "Logging.h"

using namespace cv;
using namespace std;

namespace SuperSLAM {

const int EDGE_THRESHOLD = 19;  // For border handling

// Constructor for SPextractor
SPextractor::SPextractor(int _nfeatures, const std::string &engine_file,
                         int max_keypoints, double keypoint_threshold,
                         int remove_borders)
    : nfeatures(_nfeatures) {
  // Initialize SuperPointTRT model
  model = std::make_shared<SuperPointTRT>(engine_file, max_keypoints,
                                          keypoint_threshold, remove_borders);
  if (!model->initialize()) {
    SLOG_ERROR(
        "Error in SuperPoint building engine. Please check your engine file "
        "and TensorRT installation.");
    throw std::runtime_error("Failed to build SuperPoint TensorRT engine");
  }
  SLOG_INFO("SPExtractor initialized with SuperPoint - target features: {}",
            nfeatures);
}

// Main operator to extract features
void SPextractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                             std::vector<cv::KeyPoint> &_keypoints,
                             cv::OutputArray _descriptors) {
  if (_image.empty()) {
    return;
  }

  cv::Mat image = _image.getMat();
  assert(image.type() == CV_8UC1);

  std::vector<cv::KeyPoint> allExtractedKeypoints;
  cv::Mat allExtractedDescriptors;

  // Use SuperPointTRT to extract features from the original image
  if (!model->infer(image, allExtractedKeypoints, allExtractedDescriptors)) {
    SLOG_ERROR("SuperPoint inference failed for original image");
    _descriptors.release();
    _keypoints.clear();
    return;
  }
  SLOG_INFO("SuperPoint extracted {} raw keypoints",
            allExtractedKeypoints.size());

  SLOG_INFO("Converted to OpenCV: {} keypoints, {} descriptors",
            allExtractedKeypoints.size(), allExtractedDescriptors.rows);

  _keypoints = allExtractedKeypoints;
  if (!allExtractedDescriptors.empty()) {
    allExtractedDescriptors.copyTo(_descriptors);
  } else {
    _descriptors.release();
  }

  if (_keypoints.empty()) {
    SLOG_INFO("Extractor returning 0 keypoints and 0 descriptors");
  } else {
    SLOG_INFO("Extractor returning {} keypoints and {} descriptors",
              _keypoints.size(), _descriptors.getMat().rows);
  }
}

void SPextractor::ComputeNoMoreKeyPoints(std::list<cv::KeyPoint> &kpts) {
  // ... existing code ...
}

}  // namespace SuperSLAM
