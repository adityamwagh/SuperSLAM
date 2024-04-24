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

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Map.h"
#include "MapPoint.h"
#include "Tracking.h"

namespace SuperSLAM {

class Tracking;
class Viewer;

class FrameDrawer {
 public:
  FrameDrawer(Map* pMap);

  // Update info from the last processed frame.
  void Update(Tracking* pTracker);

  // Draw last processed frame.
  cv::Mat DrawFrame();

 protected:
  void DrawTextInfo(cv::Mat& im, int nState, cv::Mat& imText);

  // Info of the frame to be drawn
  cv::Mat mIm;
  int N;
  std::vector<cv::KeyPoint> mvCurrentKeys;
  std::vector<bool> mvbMap, mvbVO;
  bool mbOnlyTracking;
  int mnTracked, mnTrackedVO;
  std::vector<cv::KeyPoint> mvIniKeys;
  std::vector<int> mvIniMatches;
  int mState;

  Map* mpMap;

  std::mutex mMutex;
};

}  // namespace SuperSLAM

#endif  // FRAMEDRAWER_H
