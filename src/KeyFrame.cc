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

#include "KeyFrame.h"

#include <mutex>

#include "Converter.h"
#include "Logging.h"
#include "SPMatcher.h"

namespace SuperSLAM {

long unsigned int KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB)
    : mnFrameId(F.mnId),
      mTimeStamp(F.mTimeStamp),
      mnGridCols(FRAME_GRID_COLS),
      mnGridRows(FRAME_GRID_ROWS),
      mfGridElementWidthInv(F.mfGridElementWidthInv),
      mfGridElementHeightInv(F.mfGridElementHeightInv),
      mnTrackReferenceForFrame(0),
      mnFuseTargetForKF(0),
      mnBALocalForKF(0),
      mnBAFixedForKF(0),
      mnLoopQuery(0),
      mnLoopWords(0),
      mnRelocQuery(0),
      mnRelocWords(0),
      mnBAGlobalForKF(0),
      fx(F.fx),
      fy(F.fy),
      cx(F.cx),
      cy(F.cy),
      invfx(F.invfx),
      invfy(F.invfy),
      mbf(F.mbf),
      mb(F.mb),
      mThDepth(F.mThDepth),
      N(F.N),
      mvKeys(F.mvKeys),
      mvKeysUn(F.mvKeysUn),
      mvuRight(F.mvuRight),
      mvDepth(F.mvDepth),
      mDescriptors(F.mDescriptors.clone()),
      mBowVec(F.mBowVec),
      mFeatVec(F.mFeatVec),
      mnScaleLevels(F.mnScaleLevels),
      mfScaleFactor(F.mfScaleFactor),
      mfLogScaleFactor(F.mfLogScaleFactor),
      mvScaleFactors(F.mvScaleFactors),
      mvLevelSigma2(F.mvLevelSigma2),
      mvInvLevelSigma2(F.mvInvLevelSigma2),
      mnMinX(F.mnMinX),
      mnMinY(F.mnMinY),
      mnMaxX(F.mnMaxX),
      mnMaxY(F.mnMaxY),
      mK(F.mK),
      mvpMapPoints(F.mvpMapPoints),
      mpKeyFrameDB(pKFDB),
      mpORBvocabulary(F.mpORBvocabulary),
      mbFirstConnection(true),
      mpParent(NULL),
      mbNotErase(false),
      mbToBeErased(false),
      mbBad(false),
      mHalfBaseline(F.mb / 2),
      mpMap(pMap) {
  mnId = nNextId++;

  mGrid.resize(mnGridCols);
  for (int i = 0; i < mnGridCols; i++) {
    mGrid[i].resize(mnGridRows);
    for (int j = 0; j < mnGridRows; j++) mGrid[i][j] = F.mGrid[i][j];
  }

  SetPose(F.mTcw);
}

void KeyFrame::ComputeBoW() {
  if (mBowVec.empty() || mFeatVec.empty()) {
    SLOG_DEBUG("ComputeBoW: Starting BoW computation...");

    // Check if descriptors are SuperPoint (256-dim float) or ORB (32-dim
    // binary)
    if (!mDescriptors.empty()) {
      SLOG_DEBUG("ComputeBoW: Descriptor type={}, size={}x{}",
                 mDescriptors.type(), mDescriptors.rows, mDescriptors.cols);

      // SuperPoint descriptors are CV_32F with 256 columns
      if (mDescriptors.type() == CV_32F && mDescriptors.cols == 256) {
        SLOG_DEBUG(
            "ComputeBoW: Detected SuperPoint descriptors, skipping BoW "
            "computation");
        // For now, create empty BoW vectors to avoid crashes
        // TODO: Implement proper SuperPoint vocabulary
        mBowVec.clear();
        mFeatVec.clear();
        return;
      }
    }

    std::vector<cv::Mat> vCurrentDesc =
        Converter::toDescriptorVector(mDescriptors);
    SLOG_DEBUG("ComputeBoW: Converting to descriptor vector, count={}",
               vCurrentDesc.size());

    // Feature vector associate features with nodes in the 4th level (from
    // leaves up) We assume the vocabulary tree has 6 levels, change the 4
    // otherwise
    SLOG_DEBUG("ComputeBoW: Calling vocabulary transform...");
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    SLOG_DEBUG("ComputeBoW: BoW computation complete");
  }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_) {
  std::unique_lock<std::mutex> lock(mMutexPose);
  Tcw_.copyTo(Tcw);
  cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
  cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
  cv::Mat Rwc = Rcw.t();
  Ow = -Rwc * tcw;

  Twc = cv::Mat::eye(4, 4, Tcw.type());
  Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
  Ow.copyTo(Twc.rowRange(0, 3).col(3));
  cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
  Cw = Twc * center;
}

cv::Mat KeyFrame::GetPose() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Cw.clone();
}

cv::Mat KeyFrame::GetRotation() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Tcw.rowRange(0, 3).colRange(0, 3).clone();
}

cv::Mat KeyFrame::GetTranslation() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Tcw.rowRange(0, 3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight) {
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (!mConnectedKeyFrameWeights.count(pKF))
      mConnectedKeyFrameWeights[pKF] = weight;
    else if (mConnectedKeyFrameWeights[pKF] != weight)
      mConnectedKeyFrameWeights[pKF] = weight;
    else
      return;
  }

  UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  std::vector<std::pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(mConnectedKeyFrameWeights.size());
  for (std::map<KeyFrame *, int>::iterator
           mit = mConnectedKeyFrameWeights.begin(),
           mend = mConnectedKeyFrameWeights.end();
       mit != mend; mit++)
    vPairs.push_back(std::make_pair(mit->second, mit->first));

  sort(vPairs.begin(), vPairs.end());
  std::list<KeyFrame *> lKFs;
  std::list<int> lWs;
  for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  mvpOrderedConnectedKeyFrames =
      std::vector<KeyFrame *>(lKFs.begin(), lKFs.end());
  mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
}

std::set<KeyFrame *> KeyFrame::GetConnectedKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  std::set<KeyFrame *> s;
  for (std::map<KeyFrame *, int>::iterator mit =
           mConnectedKeyFrameWeights.begin();
       mit != mConnectedKeyFrameWeights.end(); mit++)
    s.insert(mit->first);
  return s;
}

std::vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mvpOrderedConnectedKeyFrames;
}

std::vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  if ((int)mvpOrderedConnectedKeyFrames.size() < N)
    return mvpOrderedConnectedKeyFrames;
  else
    return std::vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(),
                                   mvpOrderedConnectedKeyFrames.begin() + N);
}

std::vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w) {
  std::unique_lock<std::mutex> lock(mMutexConnections);

  if (mvpOrderedConnectedKeyFrames.empty()) return std::vector<KeyFrame *>();

  std::vector<int>::iterator it =
      upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w,
                  KeyFrame::weightComp);
  if (it == mvOrderedWeights.end())
    return std::vector<KeyFrame *>();
  else {
    int n = it - mvOrderedWeights.begin();
    return std::vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(),
                                   mvpOrderedConnectedKeyFrames.begin() + n);
  }
}

int KeyFrame::GetWeight(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  if (mConnectedKeyFrameWeights.count(pKF))
    return mConnectedKeyFrameWeights[pKF];
  else
    return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint *pMP) {
  int idx = pMP->GetIndexInKeyFrame(this);
  if (idx >= 0) mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP) {
  mvpMapPoints[idx] = pMP;
}

std::set<MapPoint *> KeyFrame::GetMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  std::set<MapPoint *> s;
  for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++) {
    if (!mvpMapPoints[i]) continue;
    MapPoint *pMP = mvpMapPoints[i];
    if (!pMP->isBad()) s.insert(pMP);
  }
  return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);

  int nPoints = 0;
  const bool bCheckObs = minObs > 0;
  for (int i = 0; i < N; i++) {
    MapPoint *pMP = mvpMapPoints[i];
    if (pMP) {
      if (!pMP->isBad()) {
        if (bCheckObs) {
          if (mvpMapPoints[i]->Observations() >= minObs) nPoints++;
        } else
          nPoints++;
      }
    }
  }

  return nPoints;
}

std::vector<MapPoint *> KeyFrame::GetMapPointMatches() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mvpMapPoints;
}

MapPoint *KeyFrame::GetMapPoint(const size_t &idx) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections() {
  std::map<KeyFrame *, int> KFcounter;

  std::vector<MapPoint *> vpMP;

  {
    std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
    vpMP = mvpMapPoints;
  }

  // For all map points in keyframe check in which other keyframes are they seen
  // Increase counter for those keyframes
  for (std::vector<MapPoint *>::iterator vit = vpMP.begin(), vend = vpMP.end();
       vit != vend; vit++) {
    MapPoint *pMP = *vit;

    if (!pMP) continue;

    if (pMP->isBad()) continue;

    std::map<KeyFrame *, size_t> observations = pMP->GetObservations();

    for (std::map<KeyFrame *, size_t>::iterator mit = observations.begin(),
                                                mend = observations.end();
         mit != mend; mit++) {
      if (mit->first->mnId == mnId) continue;
      KFcounter[mit->first]++;
    }
  }

  // This should not happen
  if (KFcounter.empty()) return;

  // If the counter is greater than threshold add connection
  // In case no keyframe counter is over threshold add the one with maximum
  // counter
  int nmax = 0;
  KeyFrame *pKFmax = NULL;
  int th = 15;

  std::vector<std::pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(KFcounter.size());
  for (std::map<KeyFrame *, int>::iterator mit = KFcounter.begin(),
                                           mend = KFcounter.end();
       mit != mend; mit++) {
    if (mit->second > nmax) {
      nmax = mit->second;
      pKFmax = mit->first;
    }
    if (mit->second >= th) {
      vPairs.push_back(std::make_pair(mit->second, mit->first));
      (mit->first)->AddConnection(this, mit->second);
    }
  }

  if (vPairs.empty()) {
    vPairs.push_back(std::make_pair(nmax, pKFmax));
    pKFmax->AddConnection(this, nmax);
  }

  sort(vPairs.begin(), vPairs.end());
  std::list<KeyFrame *> lKFs;
  std::list<int> lWs;
  for (size_t i = 0; i < vPairs.size(); i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  {
    std::unique_lock<std::mutex> lockCon(mMutexConnections);

    // mspConnectedKeyFrames = spConnectedKeyFrames;
    mConnectedKeyFrameWeights = KFcounter;
    mvpOrderedConnectedKeyFrames =
        std::vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

    if (mbFirstConnection && mnId != 0) {
      mpParent = mvpOrderedConnectedKeyFrames.front();
      mpParent->AddChild(this);
      mbFirstConnection = false;
    }
  }
}

void KeyFrame::AddChild(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  mpParent = pKF;
  pKF->AddChild(this);
}

std::set<KeyFrame *> KeyFrame::GetChilds() {
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  return mspChildrens;
}

KeyFrame *KeyFrame::GetParent() {
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  mbNotErase = true;
  mspLoopEdges.insert(pKF);
}

std::set<KeyFrame *> KeyFrame::GetLoopEdges() {
  std::unique_lock<std::mutex> lockCon(mMutexConnections);
  return mspLoopEdges;
}

void KeyFrame::SetNotErase() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  mbNotErase = true;
}

void KeyFrame::SetErase() {
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mspLoopEdges.empty()) {
      mbNotErase = false;
    }
  }

  if (mbToBeErased) {
    SetBadFlag();
  }
}

void KeyFrame::SetBadFlag() {
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mnId == 0)
      return;
    else if (mbNotErase) {
      mbToBeErased = true;
      return;
    }
  }

  for (std::map<KeyFrame *, int>::iterator
           mit = mConnectedKeyFrameWeights.begin(),
           mend = mConnectedKeyFrameWeights.end();
       mit != mend; mit++)
    mit->first->EraseConnection(this);

  for (size_t i = 0; i < mvpMapPoints.size(); i++)
    if (mvpMapPoints[i]) mvpMapPoints[i]->EraseObservation(this);
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    std::unique_lock<std::mutex> lock1(mMutexFeatures);

    mConnectedKeyFrameWeights.clear();
    mvpOrderedConnectedKeyFrames.clear();

    // Update Spanning Tree
    std::set<KeyFrame *> sParentCandidates;
    sParentCandidates.insert(mpParent);

    // Assign at each iteration one children with a parent (the pair with
    // highest covisibility weight) Include that children as new parent
    // candidate for the rest
    while (!mspChildrens.empty()) {
      bool bContinue = false;

      int max = -1;
      KeyFrame *pC;
      KeyFrame *pP;

      for (std::set<KeyFrame *>::iterator sit = mspChildrens.begin(),
                                          send = mspChildrens.end();
           sit != send; sit++) {
        KeyFrame *pKF = *sit;
        if (pKF->isBad()) continue;

        // Check if a parent candidate is connected to the keyframe
        std::vector<KeyFrame *> vpConnected =
            pKF->GetVectorCovisibleKeyFrames();
        for (size_t i = 0, iend = vpConnected.size(); i < iend; i++) {
          for (std::set<KeyFrame *>::iterator spcit = sParentCandidates.begin(),
                                              spcend = sParentCandidates.end();
               spcit != spcend; spcit++) {
            if (vpConnected[i]->mnId == (*spcit)->mnId) {
              int w = pKF->GetWeight(vpConnected[i]);
              if (w > max) {
                pC = pKF;
                pP = vpConnected[i];
                max = w;
                bContinue = true;
              }
            }
          }
        }
      }

      if (bContinue) {
        pC->ChangeParent(pP);
        sParentCandidates.insert(pC);
        mspChildrens.erase(pC);
      } else
        break;
    }

    // If a children has no covisibility links with any parent candidate, assign
    // to the original parent of this KF
    if (!mspChildrens.empty())
      for (std::set<KeyFrame *>::iterator sit = mspChildrens.begin();
           sit != mspChildrens.end(); sit++) {
        (*sit)->ChangeParent(mpParent);
      }

    mpParent->EraseChild(this);
    mTcp = Tcw * mpParent->GetPoseInverse();
    mbBad = true;
  }

  mpMap->EraseKeyFrame(this);
  mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame *pKF) {
  bool bUpdate = false;
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF)) {
      mConnectedKeyFrameWeights.erase(pKF);
      bUpdate = true;
    }
  }

  if (bUpdate) UpdateBestCovisibles();
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y,
                                                const float &r) const {
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX =
      std::max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= mnGridCols) return vIndices;

  const int nMaxCellX =
      std::min((int)mnGridCols - 1,
               (int)std::ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0) return vIndices;

  const int nMinCellY =
      std::max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= mnGridRows) return vIndices;

  const int nMaxCellY =
      std::min((int)mnGridRows - 1,
               (int)std::ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0) return vIndices;

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const std::vector<size_t> vCell = mGrid[ix][iy];
      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r) vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const {
  return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i) {
  const float z = mvDepth[i];
  if (z > 0) {
    const float u = mvKeys[i].pt.x;
    const float v = mvKeys[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

    std::unique_lock<std::mutex> lock(mMutexPose);
    return Twc.rowRange(0, 3).colRange(0, 3) * x3Dc + Twc.rowRange(0, 3).col(3);
  } else
    return cv::Mat();
}

float KeyFrame::ComputeSceneMedianDepth(const int q) {
  std::vector<MapPoint *> vpMapPoints;
  cv::Mat Tcw_;
  {
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPose);
    vpMapPoints = mvpMapPoints;
    Tcw_ = Tcw.clone();
  }

  std::vector<float> vDepths;
  vDepths.reserve(N);
  cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
  Rcw2 = Rcw2.t();
  float zcw = Tcw_.at<float>(2, 3);
  for (int i = 0; i < N; i++) {
    if (mvpMapPoints[i]) {
      MapPoint *pMP = mvpMapPoints[i];
      cv::Mat x3Dw = pMP->GetWorldPos();
      float z = Rcw2.dot(x3Dw) + zcw;
      vDepths.push_back(z);
    }
  }

  sort(vDepths.begin(), vDepths.end());

  return vDepths[(vDepths.size() - 1) / q];
}

}  // namespace SuperSLAM
