/**
 * This file is part of SuperSLAM.
 *
 * Copyright (C)  Aditya Wagh <adityamwagh at outlook dot com> (New York
University) For more information see
 * <https://github.com/adityamwagh/SuperSLAM>
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

#ifndef SPMATCHER_H
#define SPMATCHER_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

namespace SuperSLAM {

class SPmatcher {
 public:
  SPmatcher(float nnratio = 0.6, bool checkOri = true);

  // Computes the Hamming distance between two SP descriptors
  static float DescriptorDistance(const cv::Mat& a, const cv::Mat& b);

  // Search matches between Frame keypoints and projected MapPoints. Returns
  // number of matches Used to track the local map (Tracking)
  int SearchByProjection(Frame& F, const std::vector<MapPoint*>& vpMapPoints,
                         const float th = 3);

  // Project MapPoints tracked in last frame into the current frame and search
  // matches. Used to track from previous frame (Tracking)
  int SearchByProjection(Frame& CurrentFrame, const Frame& LastFrame,
                         const float th, const bool bMono);

  // Project MapPoints seen in KeyFrame into the Frame and search matches.
  // Used in relocalisation (Tracking)
  int SearchByProjection(Frame& CurrentFrame, KeyFrame* pKF,
                         const std::set<MapPoint*>& sAlreadyFound,
                         const float th, const int SPdist);

  // Project MapPoints using a Similarity Transformation and search matches.
  // Used in loop detection (Loop Closing)
  int SearchByProjection(KeyFrame* pKF, cv::Mat Scw,
                         const std::vector<MapPoint*>& vpPoints,
                         std::vector<MapPoint*>& vpMatched, int th);

  // Search matches between MapPoints in a KeyFrame and SP in a Frame.
  // Brute force constrained to SP that belong to the same vocabulary node (at a
  // certain level) Used in Relocalisation and Loop Detection
  int SearchByBoW(KeyFrame* pKF, Frame& F,
                  std::vector<MapPoint*>& vpMapPointMatches);
  int SearchByBoW(KeyFrame* pKF1, KeyFrame* pKF2,
                  std::vector<MapPoint*>& vpMatches12);

  int SearchByNN(KeyFrame* pKF, Frame& F,
                 std::vector<MapPoint*>& vpMapPointMatches);
  int SearchByNN(Frame& CurrentFrame, const Frame& LastFrame);
  int SearchByNN(Frame& F, const std::vector<MapPoint*>& vpMapPoints);

  // Matching for the Map Initialization (only used in the monocular case)
  int SearchForInitialization(Frame& F1, Frame& F2,
                              std::vector<cv::Point2f>& vbPrevMatched,
                              std::vector<int>& vnMatches12,
                              int windowSize = 10);

  // Matching to triangulate new MapPoints. Check Epipolar Constraint.
  int SearchForTriangulation(
      KeyFrame* pKF1, KeyFrame* pKF2, cv::Mat F12,
      std::vector<std::pair<size_t, size_t>>& vMatchedPairs,
      const bool bOnlyStereo);

  // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3
  // [s12*R12|t12] In the stereo and RGB-D case, s12=1
  int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2,
                   std::vector<MapPoint*>& vpMatches12, const float& s12,
                   const cv::Mat& R12, const cv::Mat& t12, const float th);

  // Project MapPoints into KeyFrame and search for duplicated MapPoints.
  int Fuse(KeyFrame* pKF, const std::vector<MapPoint*>& vpMapPoints,
           const float th = 3.0);

  // Project MapPoints into KeyFrame using a given Sim3 and search for
  // duplicated MapPoints.
  int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*>& vpPoints,
           float th, std::vector<MapPoint*>& vpReplacePoint);

 public:
  static const float TH_LOW;
  static const float TH_HIGH;
  static const int HISTO_LENGTH;

 protected:
  bool CheckDistEpipolarLine(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                             const cv::Mat& F12, const KeyFrame* pKF);

  float RadiusByViewingCos(const float& viewCos);

  void ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1,
                          int& ind2, int& ind3);

  float mfNNratio;
  bool mbCheckOrientation;
};

typedef SPmatcher ORBmatcher;  // alias for compatible

}  // namespace SuperSLAM

#endif  // SPMATCHER_H
