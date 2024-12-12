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

#include "Map.h"

#include <mutex>

namespace SuperSLAM {

Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0) {}

void Map::AddKeyFrame(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mspKeyFrames.insert(pKF);
  if (pKF->mnId > mnMaxKFid) mnMaxKFid = pKF->mnId;
}

void Map::AddMapPoint(MapPoint* pMP) {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint* pMP) {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mspMapPoints.erase(pMP);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mspKeyFrames.erase(pKF);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const std::vector<MapPoint*>& vpMPs) {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mnBigChangeIdx;
}

std::vector<KeyFrame*> Map::GetAllKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return std::vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
}

std::vector<MapPoint*> Map::GetAllMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mspKeyFrames.size();
}

std::vector<MapPoint*> Map::GetReferenceMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mnMaxKFid;
}

void Map::clear() {
  for (std::set<MapPoint*>::iterator sit = mspMapPoints.begin(),
                                     send = mspMapPoints.end();
       sit != send; sit++)
    delete *sit;

  for (std::set<KeyFrame*>::iterator sit = mspKeyFrames.begin(),
                                     send = mspKeyFrames.end();
       sit != send; sit++)
    delete *sit;

  mspMapPoints.clear();
  mspKeyFrames.clear();
  mnMaxKFid = 0;
  mvpReferenceMapPoints.clear();
  mvpKeyFrameOrigins.clear();
}

}  // namespace SuperSLAM
