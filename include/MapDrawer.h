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

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include <pangolin/pangolin.h>

#include <mutex>

#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"

namespace SuperSLAM {

class MapDrawer {
 public:
  MapDrawer(Map* pMap, const std::string& strSettingPath);

  Map* mpMap;

  void DrawMapPoints();
  void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
  void DrawCurrentCamera(pangolin::OpenGlMatrix& Twc);
  void SetCurrentCameraPose(const cv::Mat& Tcw);
  void SetReferenceKeyFrame(KeyFrame* pKF);
  void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M);

 private:
  float mKeyFrameSize;
  float mKeyFrameLineWidth;
  float mGraphLineWidth;
  float mPointSize;
  float mCameraSize;
  float mCameraLineWidth;

  cv::Mat mCameraPose;

  std::mutex mMutexCamera;
};

}  // namespace SuperSLAM

#endif  // MAPDRAWER_H
