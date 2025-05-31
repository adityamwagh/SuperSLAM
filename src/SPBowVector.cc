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

#include "SPBowVector.h"

#include <algorithm>
#include <iostream>

namespace SuperSLAM {

void SPBowVector::convertSuperPointToBow(
    const cv::Mat& superpoint_descriptors,
    std::vector<cv::Mat>& bow_descriptors) {
  bow_descriptors.clear();

  if (superpoint_descriptors.empty()) {
    return;
  }

  // SuperPoint descriptors are 256 x N, we need each descriptor as a separate
  // Mat DBoW3 expects each descriptor as a row vector
  int num_descriptors = superpoint_descriptors.cols;
  bow_descriptors.reserve(num_descriptors);

  for (int i = 0; i < num_descriptors; ++i) {
    // Extract column i as a row vector (1 x 256)
    cv::Mat descriptor;
    cv::Mat col = superpoint_descriptors.col(i);
    cv::Mat col_t = col.t();
    descriptor = col_t.clone();

    // Ensure CV_32F type for DBoW3
    if (descriptor.type() != CV_32F) {
      descriptor.convertTo(descriptor, CV_32F);
    }

    // L2 normalize the descriptor
    cv::normalize(descriptor, descriptor, 1.0, 0.0, cv::NORM_L2);

    bow_descriptors.push_back(descriptor);
  }
}

void SPBowVector::computeBoW(const DBoW3::Vocabulary& vocabulary,
                             const cv::Mat& superpoint_descriptors,
                             DBoW3::BowVector& bow_vector,
                             DBoW3::FeatureVector& feature_vector) {
  // Convert SuperPoint descriptors to DBoW3 format
  std::vector<cv::Mat> bow_descriptors;
  convertSuperPointToBow(superpoint_descriptors, bow_descriptors);

  if (bow_descriptors.empty()) {
    bow_vector.clear();
    feature_vector.clear();
    return;
  }

  // Compute BoW representation using DBoW3 vocabulary (4-argument version)
  vocabulary.transform(bow_descriptors, bow_vector, feature_vector, 4);
}

double SPBowVector::computeSimilarity(const DBoW3::BowVector& bow1,
                                      const DBoW3::BowVector& bow2) {
  // Use simple dot product scoring for BowVector similarity
  double score = 0.0;
  DBoW3::BowVector::const_iterator v1_it, v2_it;
  const DBoW3::BowVector::const_iterator v1_end = bow1.end();
  const DBoW3::BowVector::const_iterator v2_end = bow2.end();

  v1_it = bow1.begin();
  v2_it = bow2.begin();

  while (v1_it != v1_end && v2_it != v2_end) {
    if (v1_it->first == v2_it->first) {
      score += v1_it->second * v2_it->second;
      ++v1_it;
      ++v2_it;
    } else if (v1_it->first < v2_it->first) {
      ++v1_it;
    } else {
      ++v2_it;
    }
  }

  return score;
}

void SPBowVector::filterDescriptorsByResponse(
    const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
    cv::Mat& filtered_descriptors, float min_response) {
  if (keypoints.empty() || descriptors.empty()) {
    filtered_descriptors = cv::Mat();
    return;
  }

  std::vector<int> good_indices;
  good_indices.reserve(keypoints.size());

  // Find keypoints with sufficient response
  for (size_t i = 0; i < keypoints.size(); ++i) {
    if (keypoints[i].response >= min_response) {
      good_indices.push_back(static_cast<int>(i));
    }
  }

  if (good_indices.empty()) {
    filtered_descriptors = cv::Mat();
    return;
  }

  // Extract filtered descriptors
  filtered_descriptors =
      cv::Mat(descriptors.rows, static_cast<int>(good_indices.size()),
              descriptors.type());

  for (size_t i = 0; i < good_indices.size(); ++i) {
    descriptors.col(good_indices[i])
        .copyTo(filtered_descriptors.col(static_cast<int>(i)));
  }
}

void SPBowVector::normalizeDescriptors(cv::Mat& descriptors) {
  if (descriptors.empty()) {
    return;
  }

  // Normalize each column (descriptor) to unit length
  for (int i = 0; i < descriptors.cols; ++i) {
    cv::Mat descriptor = descriptors.col(i);
    cv::normalize(descriptor, descriptor, 1.0, 0.0, cv::NORM_L2);
  }
}

void SPBowVector::prepareTrainingData(
    const std::vector<std::vector<cv::KeyPoint>>& keypoints_list,
    const std::vector<cv::Mat>& descriptors_list,
    std::vector<cv::Mat>& training_descriptors, int max_features_per_image) {
  training_descriptors.clear();

  if (keypoints_list.size() != descriptors_list.size()) {
    std::cerr << "SPBowVector: Keypoints and descriptors list sizes don't match"
              << "\n";
    return;
  }

  for (size_t img_idx = 0; img_idx < descriptors_list.size(); ++img_idx) {
    const auto& keypoints = keypoints_list[img_idx];
    const auto& descriptors = descriptors_list[img_idx];

    if (descriptors.empty()) {
      continue;
    }

    // Filter by response and limit number of features
    cv::Mat filtered_descriptors;
    filterDescriptorsByResponse(keypoints, descriptors, filtered_descriptors,
                                0.01f);

    if (filtered_descriptors.empty()) {
      continue;
    }

    // Limit number of features per image
    int num_features =
        std::min(max_features_per_image, filtered_descriptors.cols);

    // Sort by response and take top features
    std::vector<std::pair<float, int>> response_indices;
    response_indices.reserve(keypoints.size());

    for (size_t i = 0; i < keypoints.size() &&
                       i < static_cast<size_t>(filtered_descriptors.cols);
         ++i) {
      response_indices.emplace_back(keypoints[i].response, static_cast<int>(i));
    }

    // Sort by response (highest first)
    std::sort(response_indices.begin(), response_indices.end(),
              [](const std::pair<float, int>& a,
                 const std::pair<float, int>& b) { return a.first > b.first; });

    // Take top features
    for (int i = 0;
         i < num_features && i < static_cast<int>(response_indices.size());
         ++i) {
      int desc_idx = response_indices[i].second;
      if (desc_idx < filtered_descriptors.cols) {
        cv::Mat descriptor;
        cv::Mat col = filtered_descriptors.col(desc_idx);
        cv::Mat col_t = col.t();
        descriptor = col_t.clone();  // Convert to row vector

        // Ensure CV_32F type
        if (descriptor.type() != CV_32F) {
          descriptor.convertTo(descriptor, CV_32F);
        }

        // L2 normalize
        cv::normalize(descriptor, descriptor, 1.0, 0.0, cv::NORM_L2);

        training_descriptors.push_back(descriptor);
      }
    }
  }

  std::cout << "SPBowVector: Prepared " << training_descriptors.size()
            << " training descriptors from " << descriptors_list.size()
            << " images" << "\n";
}

}  // namespace SuperSLAM