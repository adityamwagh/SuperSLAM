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

#include "Frame.h"

#include <thread>

#include "Converter.h"
#include "Logging.h"
#include "SPMatcher.h"

namespace SuperSLAM {

long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame() {}

// Copy Constructor
Frame::Frame(const Frame& frame)
    : mpORBvocabulary(frame.mpORBvocabulary),
      mpSPextractorLeft(frame.mpSPextractorLeft),
      mpSPextractorRight(frame.mpSPextractorRight),
      mTimeStamp(frame.mTimeStamp),
      mImGray(frame.mImGray.clone()),
      mImRight(frame.mImRight.clone()),
      mK(frame.mK.clone()),
      mDistCoef(frame.mDistCoef.clone()),
      mbf(frame.mbf),
      mb(frame.mb),
      mThDepth(frame.mThDepth),
      N(frame.N),
      mvKeys(frame.mvKeys),
      mvKeysRight(frame.mvKeysRight),
      mvKeysUn(frame.mvKeysUn),
      mvuRight(frame.mvuRight),
      mvDepth(frame.mvDepth),
      mBowVec(frame.mBowVec),
      mFeatVec(frame.mFeatVec),
      mDescriptors(frame.mDescriptors.clone()),
      mDescriptorsRight(frame.mDescriptorsRight.clone()),
      mvpMapPoints(frame.mvpMapPoints),
      mvbOutlier(frame.mvbOutlier),
      mnId(frame.mnId),
      mpReferenceKF(frame.mpReferenceKF),
      mnScaleLevels(frame.mnScaleLevels),
      mfScaleFactor(frame.mfScaleFactor),
      mfLogScaleFactor(frame.mfLogScaleFactor),
      mvScaleFactors(frame.mvScaleFactors),
      mvInvScaleFactors(frame.mvInvScaleFactors),
      mvLevelSigma2(frame.mvLevelSigma2),
      mvInvLevelSigma2(frame.mvInvLevelSigma2) {
  for (int i = 0; i < FRAME_GRID_COLS; i++) {
    for (int j = 0; j < FRAME_GRID_ROWS; j++) {
      mGrid[i][j] = frame.mGrid[i][j];
    }
  }

  if (!frame.mTcw.empty()) {
    SetPose(frame.mTcw);
  }
}

Frame::Frame(const cv::Mat& imLeft, const cv::Mat& imRight,
             const double& timeStamp, SPextractor* extractorLeft,
             SPextractor* extractorRight, ORBVocabulary* voc, cv::Mat& K,
             cv::Mat& distCoef, const float& bf, const float& thDepth,
             std::shared_ptr<SuperGlueTRT> superglue)
    : mpORBvocabulary(voc),
      mpSPextractorLeft(extractorLeft),
      mpSPextractorRight(extractorRight),
      mTimeStamp(timeStamp),
      mImGray(imLeft.clone()),
      mImRight(imRight.clone()),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mbf(bf),
      mThDepth(thDepth),
      mpReferenceKF(static_cast<KeyFrame*>(NULL)),
      mpSuperGlue(superglue) {
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info (Simplified for single-scale SuperPoint)
  mnScaleLevels = 1;
  mfScaleFactor = 1.0f;
  mfLogScaleFactor = std::log(mfScaleFactor);  // log(1.0) = 0.0
  mvScaleFactors = {1.0f};
  mvInvScaleFactors = {1.0f};
  mvLevelSigma2 = {1.0f};  // Base sigma square, SuperPoint uses 1.0*1.0
  mvInvLevelSigma2 = {1.0f};

  // SuperPoint feature extraction
  std::thread threadLeft(&Frame::ExtractSP, this, 0, imLeft);
  std::thread threadRight(&Frame::ExtractSP, this, 1, imRight);
  threadLeft.join();
  threadRight.join();

  N = mvKeys.size();

  if (mvKeys.empty()) {
    return;
  }

  UndistortKeyPoints();

  ComputeStereoMatches();

  mvpMapPoints = std::vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));
  mvbOutlier = std::vector<bool>(N, false);

  // This is done only for the first Frame (or after a change in the
  // calibration)
  if (mbInitialComputations) {
    ComputeImageBounds(imLeft);

    mfGridElementWidthInv =
        static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
    mfGridElementHeightInv =
        static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat& imGray, const cv::Mat& imDepth,
             const double& timeStamp, SPextractor* extractor,
             ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef, const float& bf,
             const float& thDepth)
    : mpORBvocabulary(voc),
      mpSPextractorLeft(extractor),
      mpSPextractorRight(static_cast<SPextractor*>(NULL)),
      mTimeStamp(timeStamp),
      mImGray(imGray.clone()),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mbf(bf),
      mThDepth(thDepth) {
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info (Simplified for single-scale SuperPoint)
  mnScaleLevels = 1;
  mfScaleFactor = 1.0f;
  mfLogScaleFactor = std::log(mfScaleFactor);
  mvScaleFactors = {1.0f};
  mvInvScaleFactors = {1.0f};
  mvLevelSigma2 = {1.0f};
  mvInvLevelSigma2 = {1.0f};

  // SuperPoint feature extraction
  ExtractSP(0, imGray);

  N = mvKeys.size();

  if (mvKeys.empty()) {
    return;
  }

  UndistortKeyPoints();

  ComputeStereoFromRGBD(imDepth);

  mvpMapPoints = std::vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));
  mvbOutlier = std::vector<bool>(N, false);

  // This is done only for the first Frame (or after a change in the
  // calibration)
  if (mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                            static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                             static_cast<float>(mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat& imGray, const double& timeStamp,
             SPextractor* extractor, ORBVocabulary* voc, cv::Mat& K,
             cv::Mat& distCoef, const float& bf, const float& thDepth)
    : mpORBvocabulary(voc),
      mpSPextractorLeft(extractor),
      mpSPextractorRight(static_cast<SPextractor*>(NULL)),
      mTimeStamp(timeStamp),
      mImGray(imGray.clone()),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mbf(bf),
      mThDepth(thDepth) {
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info (Simplified for single-scale SuperPoint)
  mnScaleLevels = 1;
  mfScaleFactor = 1.0f;
  mfLogScaleFactor = std::log(mfScaleFactor);
  mvScaleFactors = {1.0f};
  mvInvScaleFactors = {1.0f};
  mvLevelSigma2 = {1.0f};
  mvInvLevelSigma2 = {1.0f};

  // SuperPoint feature extraction
  ExtractSP(0, imGray);

  N = mvKeys.size();

  if (mvKeys.empty()) {
    return;
  }

  UndistortKeyPoints();

  // Set no stereo information
  mvuRight = std::vector<float>(N, -1);
  mvDepth = std::vector<float>(N, -1);

  mvpMapPoints = std::vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));
  mvbOutlier = std::vector<bool>(N, false);

  // This is done only for the first Frame (or after a change in the
  // calibration)
  if (mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                            static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                             static_cast<float>(mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid() {
  int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++) {
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
      mGrid[i][j].reserve(nReserve);
    }
  }

  for (int i = 0; i < N; i++) {
    const cv::KeyPoint& kp = mvKeysUn[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY)) {
      mGrid[nGridPosX][nGridPosY].push_back(i);
    }
  }
}

void Frame::ExtractSP(int flag, const cv::Mat& im) {
  if (flag == 0) {
    (*mpSPextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
  } else {
    (*mpSPextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
  }
}

void Frame::SetPose(cv::Mat Tcw) {
  mTcw = Tcw.clone();
  UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices() {
  mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
  mRwc = mRcw.t();
  mtcw = mTcw.rowRange(0, 3).col(3);
  mOw = -mRcw.t() * mtcw;
}

bool Frame::isInFrustum(MapPoint* pMP, float viewingCosLimit) {
  pMP->mbTrackInView = false;

  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos();

  // 3D in camera coordinates
  const cv::Mat Pc = mRcw * P + mtcw;
  const float& PcX = Pc.at<float>(0);
  const float& PcY = Pc.at<float>(1);
  const float& PcZ = Pc.at<float>(2);

  // Check positive depth
  if (PcZ < 0.0f) {
    return false;
  }

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  const float u = fx * PcX * invz + cx;
  const float v = fy * PcY * invz + cy;

  if (u < mnMinX || u > mnMaxX) {
    return false;
  }
  if (v < mnMinY || v > mnMaxY) {
    return false;
  }

  // Check distance is in the scale invariance region of the MapPoint
  const float maxDistance = pMP->GetMaxDistanceInvariance();
  const float minDistance = pMP->GetMinDistanceInvariance();
  const cv::Mat PO = P - mOw;
  const float dist = cv::norm(PO);

  if (dist < minDistance || dist > maxDistance) {
    return false;
  }

  // Check viewing angle
  cv::Mat Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit) {
    return false;
  }

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  pMP->mbTrackInView = true;
  pMP->mTrackProjX = u;
  pMP->mTrackProjXR = u - mbf * invz;
  pMP->mTrackProjY = v;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

std::vector<size_t> Frame::GetFeaturesInArea(const float& x, const float& y,
                                             const float& r, const int minLevel,
                                             const int maxLevel) const {
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX =
      std::max(0, (int)std::floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS) {
    return vIndices;
  }

  const int nMaxCellX =
      std::min((int)FRAME_GRID_COLS - 1,
               (int)std::ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0) {
    return vIndices;
  }

  const int nMinCellY =
      std::max(0, (int)std::floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS) {
    return vIndices;
  }

  const int nMaxCellY =
      std::min((int)FRAME_GRID_ROWS - 1,
               (int)std::ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0) {
    return vIndices;
  }

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const std::vector<size_t> vCell = mGrid[ix][iy];
      if (vCell.empty()) {
        continue;
      }

      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint& kpUn = mvKeysUn[vCell[j]];
        if (bCheckLevels) {
          if (kpUn.octave < minLevel) {
            continue;
          }
          if (maxLevel >= 0) {
            if (kpUn.octave > maxLevel) {
              continue;
            }
          }
        }

        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r) {
          vIndices.push_back(vCell[j]);
        }
      }
    }
  }

  return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY) {
  posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
  posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

  // Keypoint's coordinates are undistorted, which could cause to go out of the
  // image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 ||
      posY >= FRAME_GRID_ROWS) {
    return false;
  }

  return true;
}

void Frame::ComputeBoW() {
  if (mBowVec.empty()) {
    std::vector<cv::Mat> vCurrentDesc =
        Converter::toDescriptorVector(mDescriptors);
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

void Frame::UndistortKeyPoints() {
  if (mDistCoef.at<float>(0) == 0.0) {
    mvKeysUn = mvKeys;
    return;
  }

  // Fill matrix with points
  cv::Mat mat(N, 2, CV_32F);
  for (int i = 0; i < N; i++) {
    mat.at<float>(i, 0) = mvKeys[i].pt.x;
    mat.at<float>(i, 1) = mvKeys[i].pt.y;
  }

  // Undistort points
  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
  mat = mat.reshape(1);

  // Fill undistorted keypoint vector
  mvKeysUn.resize(N);
  for (int i = 0; i < N; i++) {
    cv::KeyPoint kp = mvKeys[i];
    kp.pt.x = mat.at<float>(i, 0);
    kp.pt.y = mat.at<float>(i, 1);
    mvKeysUn[i] = kp;
  }
}

void Frame::ComputeImageBounds(const cv::Mat& imLeft) {
  if (mDistCoef.at<float>(0) != 0.0) {
    cv::Mat mat(4, 2, CV_32F);
    mat.at<float>(0, 0) = 0.0;
    mat.at<float>(0, 1) = 0.0;
    mat.at<float>(1, 0) = imLeft.cols;
    mat.at<float>(1, 1) = 0.0;
    mat.at<float>(2, 0) = 0.0;
    mat.at<float>(2, 1) = imLeft.rows;
    mat.at<float>(3, 0) = imLeft.cols;
    mat.at<float>(3, 1) = imLeft.rows;

    // Undistort corners
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    mnMinX = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
    mnMaxX = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
    mnMinY = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
    mnMaxY = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));

  } else {
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
  }
}

void Frame::ComputeStereoMatches() {
  mvuRight = std::vector<float>(N, -1.0f);
  mvDepth = std::vector<float>(N, -1.0f);

  // If SuperGlue is available, use it for stereo matching
  if (mpSuperGlue && !mvKeys.empty() && !mvKeysRight.empty()) {
    SLOG_INFO("Using SuperGlue for stereo matching");

    // Use SuperGlue to match features between left and right images
    MatchResult result;
    if (mpSuperGlue->match(mvKeys, mDescriptors, mvKeysRight, mDescriptorsRight,
                           result)) {
      SLOG_INFO("SuperGlue found {} matches between left and right images",
                result.matches.size());

      // Process SuperGlue matches
      int nEpipolarValid = 0;
      int nDisparityValid = 0;
      for (const auto& match : result.matches) {
        int iL = match.queryIdx;  // Left image keypoint index
        int iR = match.trainIdx;  // Right image keypoint index

        if (iL >= 0 && iL < N && iR >= 0 && iR < mvKeysRight.size()) {
          const cv::KeyPoint& kpL = mvKeys[iL];
          const cv::KeyPoint& kpR = mvKeysRight[iR];

          // Check epipolar constraint (keypoints should be on same row within
          // tolerance)
          const float deltaV = std::abs(kpL.pt.y - kpR.pt.y);
          if (deltaV < 3.0f) {  // 3 pixel tolerance
            nEpipolarValid++;
            float disparity = kpL.pt.x - kpR.pt.x;

            // Check disparity is positive and within reasonable range
            // For KITTI stereo, typical disparities range from 1 to ~100 pixels
            if (disparity > 0.1 && disparity < 200.0) {
              nDisparityValid++;
              mvDepth[iL] = mbf / disparity;
              mvuRight[iL] = kpR.pt.x;
            }
          }
        }
      }

      SLOG_INFO(
          "SuperGlue matching stats: {} epipolar valid, {} disparity valid out "
          "of {} matches",
          nEpipolarValid, nDisparityValid, result.matches.size());

      // Count valid depth values
      int nValidDepth = 0;
      for (int i = 0; i < N; i++) {
        if (mvDepth[i] > 0) {
          nValidDepth++;
        }
      }
      SLOG_INFO(
          "SuperGlue stereo matching: {} keypoints, {} with valid depth (need "
          ">500 for init)",
          N, nValidDepth);
      return;
    } else {
      SLOG_WARN(
          "SuperGlue matching failed, falling back to descriptor-based "
          "matching");
    }
  }

  // Fallback to descriptor-based matching
  const float thOrbDist = (SPmatcher::TH_HIGH + SPmatcher::TH_LOW) / 2;

  const int nRows = mImGray.rows;

  // Assign keypoints to row table
  std::vector<std::vector<size_t>> vRowIndices(nRows, std::vector<size_t>());

  for (int i = 0; i < nRows; i++) {
    vRowIndices[i].reserve(200);
  }

  const int Nr = mvKeysRight.size();

  for (int iR = 0; iR < Nr; iR++) {
    const cv::KeyPoint& kp = mvKeysRight[iR];
    const float& kpY = kp.pt.y;
    const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
    const int maxr = ceil(kpY + r);
    const int minr = floor(kpY - r);

    for (int yi = minr; yi <= maxr; yi++) {
      vRowIndices[yi].push_back(iR);
    }
  }

  // Set limits for search
  const float minZ = mb;
  const float maxZ = mb * mThDepth;

  // For each left keypoint search a match in the right image
  std::vector<std::pair<int, int>> vDistIdx;
  vDistIdx.resize(N);

  for (int iL = 0; iL < N; iL++) {
    const cv::KeyPoint& kpL = mvKeys[iL];
    const int& levelL = kpL.octave;
    const float& vL = kpL.pt.y;
    const float& uL = kpL.pt.x;

    const std::vector<size_t>& vCandidates = vRowIndices[vL];

    if (vCandidates.empty()) {
      continue;
    }

    const float minU = uL - maxZ;
    const float maxU = uL - minZ;

    if (maxU < 0) {
      continue;
    }

    float bestDist = SPmatcher::TH_HIGH;
    size_t bestIdxR = 0;

    const cv::Mat& dL = mDescriptors.row(iL);

    // Compare descriptor to right keypoints
    for (size_t iC = 0; iC < vCandidates.size(); iC++) {
      const size_t iR = vCandidates[iC];
      const cv::KeyPoint& kpR = mvKeysRight[iR];

      if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1) {
        continue;
      }

      const float& uR = kpR.pt.x;

      if (uR >= minU && uR <= maxU) {
        const cv::Mat& dR = mDescriptorsRight.row(iR);
        const float dist = SPmatcher::DescriptorDistance(dL, dR);

        if (dist < bestDist) {
          bestDist = dist;
          bestIdxR = iR;
        } else if (dist == bestDist &&
                   mvKeysRight[iR].response > mvKeysRight[bestIdxR].response) {
          bestIdxR = iR;
        }
      }
    }

    // Subpixel match by correlation
    if (bestDist < thOrbDist) {
      // coordinates in image pyramid at keypoint scale
      const float uR0 = mvKeysRight[bestIdxR].pt.x;
      const float scaleFactor = mvInvScaleFactors[kpL.octave];
      const float scaleduL = round(kpL.pt.x * scaleFactor);
      const float scaledvL = round(kpL.pt.y * scaleFactor);
      const float scaleduR0 = round(uR0 * scaleFactor);

      // sliding window search
      const int w = 5;
      cv::Mat IL = mImGray.rowRange(scaledvL - w, scaledvL + w + 1)
                       .colRange(scaleduL - w, scaleduL + w + 1);
      IL.convertTo(IL, CV_32F);
      IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

      int bestDist = INT_MAX;
      int bestincR = 0;
      const int L = 5;
      std::vector<float> vDists;
      vDists.resize(2 * L + 1);

      const float iniu = scaleduR0 + L - w;
      const float endu = scaleduR0 + L + w + 1;
      if (iniu < 0 || endu >= mImRight.cols) {
        continue;
      }

      for (int incR = -L; incR <= +L; incR++) {
        cv::Mat IR =
            mImRight.rowRange(scaledvL - w, scaledvL + w + 1)
                .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
        IR.convertTo(IR, CV_32F);
        IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

        float dist = cv::norm(IL, IR, cv::NORM_L1);
        if (dist < bestDist) {
          bestDist = dist;
          bestincR = incR;
        }

        vDists[L + incR] = dist;
      }

      if (bestincR == -L || bestincR == L) {
        continue;
      }

      // Sub-pixel match (Parabola fitting)
      const float dist1 = vDists[L + bestincR - 1];
      const float dist2 = vDists[L + bestincR];
      const float dist3 = vDists[L + bestincR + 1];

      const float deltaR =
          (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

      if (deltaR < -1 || deltaR > 1) {
        continue;
      }

      // Re-scaled coordinate
      float bestuR = mvScaleFactors[kpL.octave] *
                     ((float)scaleduR0 + (float)bestincR + deltaR);

      float disparity = (uL - bestuR);

      if (disparity >= minZ && disparity < maxZ) {
        if (disparity <= 0) {
          disparity = 0.01;
          bestuR = uL - 0.01;
        }
        mvDepth[iL] = mbf / disparity;
        mvuRight[iL] = bestuR;
        vDistIdx.push_back(std::pair<int, int>(bestDist, iL));
      }
    }
  }

  sort(vDistIdx.begin(), vDistIdx.end());
  const float median = vDistIdx[vDistIdx.size() / 2].first;
  const float thDist = 1.5f * 1.4f * median;

  for (int i = vDistIdx.size() - 1; i >= 0; i--) {
    if (vDistIdx[i].first < thDist) {
      break;
    } else {
      mvuRight[vDistIdx[i].second] = -1;
      mvDepth[vDistIdx[i].second] = -1;
    }
  }

  // Debug: Count valid depth values for stereo initialization
  int nValidDepth = 0;
  for (int i = 0; i < N; i++) {
    if (mvDepth[i] > 0) {
      nValidDepth++;
    }
  }
  SLOG_INFO(
      "Stereo matching: {} keypoints, {} with valid depth (need >500 for init)",
      N, nValidDepth);
}

void Frame::ComputeStereoFromRGBD(const cv::Mat& imDepth) {
  mvuRight = std::vector<float>(N, -1);
  mvDepth = std::vector<float>(N, -1);

  for (int i = 0; i < N; i++) {
    const cv::KeyPoint& kp = mvKeys[i];
    const cv::KeyPoint& kpU = mvKeysUn[i];

    const float& v = kp.pt.y;
    const float& u = kp.pt.x;

    const float d = imDepth.at<float>(v, u);

    if (d > 0) {
      mvDepth[i] = d;
      mvuRight[i] = kpU.pt.x - mbf / d;
    }
  }
}

cv::Mat Frame::UnprojectStereo(const int& i) {
  const float z = mvDepth[i];
  if (z > 0) {
    const float u = mvKeysUn[i].pt.x;
    const float v = mvKeysUn[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
    return mRwc * x3Dc + mOw;
  } else {
    return cv::Mat();
  }
}

}  // namespace SuperSLAM
