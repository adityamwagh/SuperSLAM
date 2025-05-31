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

#ifndef SPBOWVECTOR_H
#define SPBOWVECTOR_H

#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include "thirdparty/DBoW3/src/DBoW3.h"

namespace SuperSLAM {

/**
 * @brief SuperPoint Bag-of-Words utilities for loop closure detection
 * 
 * This class provides functionality to convert SuperPoint descriptors into
 * bag-of-words representations using DBoW3 vocabulary for loop closure detection.
 * Uses L2-normalized SuperPoint descriptors directly with cosine similarity.
 */
class SPBowVector {
public:
    /**
     * @brief Convert SuperPoint descriptors to DBoW3 format
     * @param superpoint_descriptors SuperPoint descriptors (256 x N matrix)
     * @param bow_descriptors Output vector of descriptors for DBoW3 (each row as separate Mat)
     */
    static void convertSuperPointToBow(const cv::Mat& superpoint_descriptors,
                                     std::vector<cv::Mat>& bow_descriptors);

    /**
     * @brief Compute BoW representation from SuperPoint descriptors
     * @param vocabulary DBoW3 vocabulary
     * @param superpoint_descriptors SuperPoint descriptors (256 x N matrix, CV_32F)
     * @param bow_vector Output bag-of-words vector
     * @param feature_vector Output feature vector for matching
     */
    static void computeBoW(const DBoW3::Vocabulary& vocabulary,
                          const cv::Mat& superpoint_descriptors,
                          DBoW3::BowVector& bow_vector,
                          DBoW3::FeatureVector& feature_vector);

    /**
     * @brief Compute similarity between two SuperPoint BoW vectors
     * @param bow1 First BoW vector
     * @param bow2 Second BoW vector  
     * @return Similarity score [0, 1]
     */
    static double computeSimilarity(const DBoW3::BowVector& bow1,
                                  const DBoW3::BowVector& bow2);

    /**
     * @brief Filter SuperPoint descriptors by response threshold
     * @param keypoints SuperPoint keypoints with response values
     * @param descriptors SuperPoint descriptors (256 x N matrix)
     * @param filtered_descriptors Output filtered descriptors
     * @param min_response Minimum response threshold (default: 0.01)
     */
    static void filterDescriptorsByResponse(const std::vector<cv::KeyPoint>& keypoints,
                                          const cv::Mat& descriptors,
                                          cv::Mat& filtered_descriptors,
                                          float min_response = 0.01f);

    /**
     * @brief Normalize SuperPoint descriptors to unit length (L2 normalization)
     * @param descriptors Input/output descriptors matrix (256 x N)
     */
    static void normalizeDescriptors(cv::Mat& descriptors);

    /**
     * @brief Create training data for vocabulary generation
     * @param keypoints_list Vector of keypoint sets from multiple images
     * @param descriptors_list Vector of descriptor matrices from multiple images  
     * @param training_descriptors Output training descriptors for vocabulary
     * @param max_features_per_image Maximum features to use per image (default: 500)
     */
    static void prepareTrainingData(const std::vector<std::vector<cv::KeyPoint>>& keypoints_list,
                                  const std::vector<cv::Mat>& descriptors_list,
                                  std::vector<cv::Mat>& training_descriptors,
                                  int max_features_per_image = 500);
};

} // namespace SuperSLAM

#endif // SPBOWVECTOR_H