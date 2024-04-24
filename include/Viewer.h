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

#ifndef VIEWER_H
#define VIEWER_H

#include <mutex>

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "System.h"
#include "Tracking.h"

namespace SuperSLAM {

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;

class Viewer {
 public:
  Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer,
         Tracking* pTracking, const std::string& strSettingPath);

  // Main thread function. Draw points, keyframes, the current camera pose and
  // the last processed frame. Drawing is refreshed according to the camera fps.
  // We use Pangolin.
  void Run();

  void RequestFinish();

  void RequestStop();

  bool isFinished();

  bool isStopped();

  void Release();

 private:
  bool Stop();

  System* mpSystem;
  FrameDrawer* mpFrameDrawer;
  MapDrawer* mpMapDrawer;
  Tracking* mpTracker;

  // 1/fps in ms
  double mT;
  float mImageWidth, mImageHeight;

  float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

  bool CheckFinish();
  void SetFinish();
  bool mbFinishRequested;
  bool mbFinished;
  std::mutex mMutexFinish;

  bool mbStopped;
  bool mbStopRequested;
  std::mutex mMutexStop;
};

}  // namespace SuperSLAM

#endif  // VIEWER_H
