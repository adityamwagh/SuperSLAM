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

#ifndef TRACKING_H
#define TRACKING_H

#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Frame.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapDrawer.h"
#include "SPVocabulary.h"
#include "SPextractor.h"
#include "System.h"
#include "Viewer.h"

namespace SuperSLAM {

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking {
 public:
  Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer,
           MapDrawer* pMapDrawer, Map* pMap, KeyFrameDatabase* pKFDB,
           const std::string& strSettingPath, const int sensor);

  // Preprocess the input and call Track(). Extract features and performs stereo
  // matching.
  cv::Mat GrabImageStereo(const cv::Mat& imRectLeft, const cv::Mat& imRectRight,
                          const double& timestamp);
  cv::Mat GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                        const double& timestamp);
  cv::Mat GrabImageMonocular(const cv::Mat& im, const double& timestamp);

  void SetLocalMapper(LocalMapping* pLocalMapper);
  void SetLoopClosing(LoopClosing* pLoopClosing);
  void SetViewer(Viewer* pViewer);

  // Load new settings
  // The focal lenght should be similar or scale prediction will fail when
  // projecting points
  // TODO: Modify MapPoint::PredictScale to take into account focal lenght
  void ChangeCalibration(const std::string& strSettingPath);

  // Use this function if you have deactivated local mapping and you only want
  // to localize the camera.
  void InformOnlyTracking(const bool& flag);

 public:
  // Tracking states
  enum eTrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    LOST = 3
  };

  eTrackingState mState;
  eTrackingState mLastProcessedState;

  // Input sensor
  int mSensor;

  // Current Frame
  Frame mCurrentFrame;
  cv::Mat mImGray;

  // Initialization Variables (Monocular)
  std::vector<int> mvIniLastMatches;
  std::vector<int> mvIniMatches;
  std::vector<cv::Point2f> mvbPrevMatched;
  std::vector<cv::Point3f> mvIniP3D;
  Frame mInitialFrame;

  // Lists used to recover the full camera trajectory at the end of the
  // execution. Basically we store the reference keyframe for each frame and its
  // relative transformation
  std::list<cv::Mat> mlRelativeFramePoses;
  std::list<KeyFrame*> mlpReferences;
  std::list<double> mlFrameTimes;
  std::list<bool> mlbLost;

  // True if local mapping is deactivated and we are performing only
  // localization
  bool mbOnlyTracking;

  void Reset();

 protected:
  // Main tracking function. It is independent of the input sensor.
  void Track();

  // Map initialization for stereo and RGB-D
  void StereoInitialization();

  // Map initialization for monocular
  void MonocularInitialization();
  void CreateInitialMapMonocular();

  void CheckReplacedInLastFrame();
  bool TrackReferenceKeyFrame();
  void UpdateLastFrame();
  bool TrackWithMotionModel();

  bool Relocalization();

  void UpdateLocalMap();
  void UpdateLocalPoints();
  void UpdateLocalKeyFrames();

  bool TrackLocalMap();
  void SearchLocalPoints();

  bool NeedNewKeyFrame();
  void CreateNewKeyFrame();

  // In case of performing only localization, this flag is true when there are
  // no matches to points in the map. Still tracking will continue if there are
  // enough matches with temporal points. In that case we are doing visual
  // odometry. The system will try to do relocalization to recover "zero-drift"
  // localization to the map.
  bool mbVO;

  // Other Thread Pointers
  LocalMapping* mpLocalMapper;
  LoopClosing* mpLoopClosing;

  // ORB
  ORBextractor *mpORBextractorLeft, *mpORBextractorRight;
  ORBextractor* mpIniORBextractor;

  // BoW
  ORBVocabulary* mpORBVocabulary;
  KeyFrameDatabase* mpKeyFrameDB;

  // Initalization (only for monocular)
  Initializer* mpInitializer;

  // Local Map
  KeyFrame* mpReferenceKF;
  std::vector<KeyFrame*> mvpLocalKeyFrames;
  std::vector<MapPoint*> mvpLocalMapPoints;

  // System
  System* mpSystem;

  // Drawers
  Viewer* mpViewer;
  FrameDrawer* mpFrameDrawer;
  MapDrawer* mpMapDrawer;

  // Map
  Map* mpMap;

  // Calibration matrix
  cv::Mat mK;
  cv::Mat mDistCoef;
  float mbf;

  // New KeyFrame rules (according to fps)
  int mMinFrames;
  int mMaxFrames;

  // Threshold close/far points
  // Points seen as close by the stereo/RGBD sensor are considered reliable
  // and inserted from just one frame. Far points requiere a match in two
  // keyframes.
  float mThDepth;

  // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are
  // scaled.
  float mDepthMapFactor;

  // Current matches in frame
  int mnMatchesInliers;

  // Last Frame, KeyFrame and Relocalisation Info
  KeyFrame* mpLastKeyFrame;
  Frame mLastFrame;
  unsigned int mnLastKeyFrameId;
  unsigned int mnLastRelocFrameId;

  // Motion Model
  cv::Mat mVelocity;

  // Color order (true RGB, false BGR, ignored if grayscale)
  bool mbRGB;

  std::list<MapPoint*> mlpTemporalPoints;
};

}  // namespace SuperSLAM

#endif  // TRACKING_H
