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

#ifndef SPEXTRACTOR_H
#define SPEXTRACTOR_H

#include <list>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#include "ReadConfig.h"
#include "SuperPointTRT.h"

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

namespace SuperSLAM {

class ExtractorNode {
 public:
  ExtractorNode() : bNoMore(false) {}

  void DivideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3,
                  ExtractorNode& n4);

  std::vector<cv::KeyPoint> vKeys;
  cv::Point2i UL, UR, BL, BR;
  std::list<ExtractorNode>::iterator lit;
  bool bNoMore;
};

class SPextractor {
 public:
  SPextractor(int nfeatures,  // Only number of features and config needed
              const SuperPointConfig& config);

  ~SPextractor() {}

  // Compute the SuperPoint features and descriptors on an image.
  // Keypoints are dispersed on the image using an octree if more than nfeatures
  // are found. Mask is ignored in the current implementation.
  void operator()(cv::InputArray image, cv::InputArray mask,
                  std::vector<cv::KeyPoint>& keypoints,
                  cv::OutputArray descriptors);

  // TODO: Check if needed
  static void ComputeNoMoreKeyPoints(std::list<cv::KeyPoint>& kpts);

 protected:
  int nfeatures;
  std::shared_ptr<SuperPointTRT> model;
};

// typedef SPextractor ORBextractor;  // Removed: No longer needed after refactoring

}  // namespace SuperSLAM

#endif
