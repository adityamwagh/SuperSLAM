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

#ifndef MAP_H
#define MAP_H

#include <mutex>
#include <set>

#include "KeyFrame.h"
#include "MapPoint.h"

namespace SuperSLAM {

class MapPoint;
class KeyFrame;

class Map {
 public:
  Map();

  void AddKeyFrame(KeyFrame* pKF);
  void AddMapPoint(MapPoint* pMP);
  void EraseMapPoint(MapPoint* pMP);
  void EraseKeyFrame(KeyFrame* pKF);
  void SetReferenceMapPoints(const std::vector<MapPoint*>& vpMPs);
  void InformNewBigChange();
  int GetLastBigChangeIdx();

  std::vector<KeyFrame*> GetAllKeyFrames();
  std::vector<MapPoint*> GetAllMapPoints();
  std::vector<MapPoint*> GetReferenceMapPoints();

  long unsigned int MapPointsInMap();
  long unsigned KeyFramesInMap();

  long unsigned int GetMaxKFid();

  void clear();

  std::vector<KeyFrame*> mvpKeyFrameOrigins;

  std::mutex mMutexMapUpdate;

  // This avoid that two points are created simultaneously in separate threads
  // (id conflict)
  std::mutex mMutexPointCreation;

 protected:
  std::set<MapPoint*> mspMapPoints;
  std::set<KeyFrame*> mspKeyFrames;

  std::vector<MapPoint*> mvpReferenceMapPoints;

  long unsigned int mnMaxKFid;

  // Index related to a big change in the map (loop closure, global BA)
  int mnBigChangeIdx;

  std::mutex mMutexMap;
};

}  // namespace SuperSLAM

#endif  // MAP_H
