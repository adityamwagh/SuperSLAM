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

#include "Tracking.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <mutex>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>

#include "Converter.h"
#include "Initializer.h"
#include "Logging.h"
#include "Map.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include "SPMatcher.h"

namespace SuperSLAM {

Tracking::Tracking(System* pSys, ORBVocabulary* pVoc, Map* pMap,
                   KeyFrameDatabase* pKFDB, const std::string& strSettingPath,
                   const int sensor)
    : mState(NO_IMAGES_YET),
      mSensor(sensor),
      mbOnlyTracking(false),
      mbVO(false),
      mpORBVocabulary(pVoc),
      mpKeyFrameDB(pKFDB),
      mpInitializer(nullptr),
      mpSystem(pSys),
      mpViewer(NULL),
      mpMap(pMap),
      mnLastRelocFrameId(0) {
  // Load camera parameters from settings file

  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  mbf = fSettings["Camera.bf"];

  float fps = fSettings["Camera.fps"];
  if (fps == 0) {
    fps = 30;
  }

  // Max/Min Frames to insert keyframes and to check relocalisation
  mMinFrames = 0;
  mMaxFrames = fps;

  SLOG_INFO("Camera Parameters:");
  SLOG_INFO("- fx: {}", fx);
  SLOG_INFO("- fy: {}", fy);
  SLOG_INFO("- cx: {}", cx);
  SLOG_INFO("- cy: {}", cy);
  SLOG_INFO("- k1: {}", DistCoef.at<float>(0));
  SLOG_INFO("- k2: {}", DistCoef.at<float>(1));
  if (DistCoef.rows == 5) {
    SLOG_INFO("- k3: {}", DistCoef.at<float>(4));
  }
  SLOG_INFO("- p1: {}", DistCoef.at<float>(2));
  SLOG_INFO("- p2: {}", DistCoef.at<float>(3));
  SLOG_INFO("- fps: {}", fps);

  int nRGB = fSettings["Camera.RGB"];
  mbRGB = nRGB;

  if (mbRGB) {
    SLOG_INFO("- color order: RGB (ignored if grayscale)");
  } else {
    SLOG_INFO("- color order: BGR (ignored if grayscale)");
  }

  // Load SuperPoint parameters

  int nFeatures = fSettings["ORBextractor.nFeatures"];
  float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
  int nLevels = fSettings["ORBextractor.nLevels"];
  int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
  int fMinThFAST = fSettings["ORBextractor.minThFAST"];

  // Load SuperPoint/SuperGlue configuration directly from settings
  std::string modelDir = fSettings["SuperPoint.model_dir"];

  // Read SuperPoint parameters
  int sp_max_keypoints = fSettings["superpoint"]["max_keypoints"];
  double sp_keypoint_threshold = fSettings["superpoint"]["keypoint_threshold"];
  int sp_remove_borders = fSettings["superpoint"]["remove_borders"];
  std::string sp_engine_file =
      modelDir + "/" + std::string(fSettings["superpoint"]["engine_file"]);

  mpSPextractorLeft = std::make_unique<SuperSLAM::SPextractor>(
      nFeatures, sp_engine_file, sp_max_keypoints, sp_keypoint_threshold,
      sp_remove_borders);

  if (sensor == System::STEREO) {
    mpSPextractorRight = std::make_unique<SuperSLAM::SPextractor>(
        nFeatures, sp_engine_file, sp_max_keypoints, sp_keypoint_threshold,
        sp_remove_borders);

    // Initialize SuperGlue for stereo matching
    int sg_image_width = fSettings["superglue"]["image_width"];
    int sg_image_height = fSettings["superglue"]["image_height"];
    std::string sg_engine_file =
        modelDir + "/" + std::string(fSettings["superglue"]["engine_file"]);

    mpSuperGlueStereo = std::make_shared<SuperGlueTRT>(
        sg_engine_file, sg_image_width, sg_image_height);
    if (!mpSuperGlueStereo->initialize()) {
      SLOG_ERROR("Failed to initialize SuperGlue for stereo matching!");
      mpSuperGlueStereo.reset();
    } else {
      SLOG_INFO("SuperGlue initialized successfully for stereo matching");
    }
  }

  if (sensor == System::MONOCULAR) {
    mpIniSPextractor = std::make_unique<SuperSLAM::SPextractor>(
        2 * nFeatures, sp_engine_file, sp_max_keypoints, sp_keypoint_threshold,
        sp_remove_borders);
  }

  SLOG_INFO("SuperPoint Extractor Parameters:");
  SLOG_INFO("- Number of Features: {}", nFeatures);
  SLOG_INFO("- Scale Levels: {}", nLevels);
  SLOG_INFO("- Scale Factor: {}", fScaleFactor);
  SLOG_INFO("- Initial Fast Threshold: {}", fIniThFAST);
  SLOG_INFO("- Minimum Fast Threshold: {}", fMinThFAST);

  if (sensor == System::STEREO || sensor == System::RGBD) {
    mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
    SLOG_INFO("\nDepth Threshold (Close/Far Points): {}", mThDepth);
  }

  if (sensor == System::RGBD) {
    mDepthMapFactor = fSettings["DepthMapFactor"];
    if (fabs(mDepthMapFactor - 1.0f) < 1e-5) {
      mDepthMapFactor = 1;
    } else {
      mDepthMapFactor = 1.0f / mDepthMapFactor;
    }
  }

  SLOG_INFO("Loading SuperPoint Vocabulary. This could take a while...");
}

void Tracking::SetLocalMapper(LocalMapping* pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing* pLoopClosing) {
  mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(RerunViewer* pViewer) { mpViewer = pViewer; }

cv::Mat Tracking::GrabImageStereo(const cv::Mat& imRectLeft,
                                  const cv::Mat& imRectRight,
                                  const double& timestamp) {
  mImGray = imRectLeft;
  cv::Mat imGrayRight = imRectRight;

  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
    }
  }

  // Store right image for visualization
  mImGrayRight = imGrayRight.clone();

  mCurrentFrame =
      Frame(mImGray, imGrayRight, timestamp, mpSPextractorLeft.get(),
            mpSPextractorRight.get(), mpORBVocabulary, mK, mDistCoef, mbf,
            mThDepth, mpSuperGlueStereo);

  Track();

  return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                                const double& timestamp) {
  mImGray = imRGB;
  cv::Mat imDepth = imD;

  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }
  }

  if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F) {
    imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);
  }

  mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpSPextractorLeft.get(),
                        mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

  Track();

  return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im,
                                     const double& timestamp) {
  mImGray = im;

  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }
  }

  if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) {
    mCurrentFrame = Frame(mImGray, timestamp, mpIniSPextractor.get(),
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
  } else {
    mCurrentFrame = Frame(mImGray, timestamp, mpSPextractorLeft.get(),
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
  }

  Track();

  return mCurrentFrame.mTcw.clone();
}

void Tracking::Track() {
  if (mState == NO_IMAGES_YET) {
    mState = NOT_INITIALIZED;
  }

  mLastProcessedState = mState;

  // Get Map std::mutex -> Map cannot be changed
  std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

  if (mState == NOT_INITIALIZED) {
    if (mSensor == System::STEREO || mSensor == System::RGBD) {
      StereoInitialization();
    } else {
      MonocularInitialization();
    }

    // Log frame data to RerunViewer
    if (mpViewer) {
      SLOG_INFO("Tracking: State={}, Keypoints={}, Pose valid={}", mState,
                mCurrentFrame.mvKeysUn.size(), (!mCurrentFrame.mTcw.empty()));

      // Use stereo logging for stereo sensor
      if (mSensor == System::STEREO && !mImGrayRight.empty()) {
        mpViewer->AddCurrentFrame(&mCurrentFrame);
      } else {
        mpViewer->AddCurrentFrame(&mCurrentFrame);
      }
    }

    if (mState != OK) {
      return;
    }
  } else {
    // System is initialized. Track Frame.
    bool bOK;

    // Initial camera pose estimation using motion model or relocalization (if
    // tracking is lost)
    if (!mbOnlyTracking) {
      // Local Mapping is activated. This is the normal behaviour, unless
      // you explicitly activate the "only tracking" mode.

      if (mState == OK) {
        // Local Mapping might have changed some MapPoints tracked in last frame
        CheckReplacedInLastFrame();

        if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
          bOK = TrackReferenceKeyFrame();
        } else {
          bOK = TrackWithMotionModel();
          if (!bOK) {
            bOK = TrackReferenceKeyFrame();
          }
        }
      } else {
        bOK = Relocalization();
      }
    } else {
      // Localization Mode: Local Mapping is deactivated

      if (mState == LOST) {
        bOK = Relocalization();
      } else {
        if (!mbVO) {
          // In last frame we tracked enough MapPoints in the map

          if (!mVelocity.empty()) {
            bOK = TrackWithMotionModel();
          } else {
            bOK = TrackReferenceKeyFrame();
          }
        } else {
          // In last frame we tracked mainly "visual odometry" points.

          // We compute two camera poses, one from motion model and one doing
          // relocalization. If relocalization is sucessfull we choose that
          // solution, otherwise we retain the "visual odometry" solution.

          bool bOKMM = false;
          bool bOKReloc = false;
          std::vector<MapPoint*> vpMPsMM;
          std::vector<bool> vbOutMM;
          cv::Mat TcwMM;
          if (!mVelocity.empty()) {
            bOKMM = TrackWithMotionModel();
            vpMPsMM = mCurrentFrame.mvpMapPoints;
            vbOutMM = mCurrentFrame.mvbOutlier;
            TcwMM = mCurrentFrame.mTcw.clone();
          }
          bOKReloc = Relocalization();

          if (bOKMM && !bOKReloc) {
            mCurrentFrame.SetPose(TcwMM);
            mCurrentFrame.mvpMapPoints = vpMPsMM;
            mCurrentFrame.mvbOutlier = vbOutMM;

            if (mbVO) {
              for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvpMapPoints[i] &&
                    !mCurrentFrame.mvbOutlier[i]) {
                  mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                }
              }
            }
          } else if (bOKReloc) {
            mbVO = false;
          }

          bOK = bOKReloc || bOKMM;
        }
      }
    }

    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    // If we have an initial estimation of the camera pose and matching. Track
    // the local map.
    if (!mbOnlyTracking) {
      if (bOK) {
        bOK = TrackLocalMap();
      }
    } else {
      // mbVO true means that there are few matches to MapPoints in the map. We
      // cannot retrieve a local map and therefore we do not perform
      // TrackLocalMap(). Once the system relocalizes the camera we will use the
      // local map again.
      if (bOK && !mbVO) {
        bOK = TrackLocalMap();
      }
    }

    if (bOK) {
      mState = OK;
    } else {
      mState = LOST;
    }

    // Update drawer
    // Log frame data to RerunViewer
    if (mpViewer) {
      SLOG_INFO("Tracking: State={}, Keypoints={}, Pose valid={}", mState,
                mCurrentFrame.mvKeysUn.size(), (!mCurrentFrame.mTcw.empty()));

      // Use stereo logging for stereo sensor
      if (mSensor == System::STEREO && !mImGrayRight.empty()) {
        mpViewer->AddCurrentFrame(&mCurrentFrame);
      } else {
        mpViewer->AddCurrentFrame(&mCurrentFrame);
      }
    }

    // If tracking were good, check if we insert a keyframe
    if (bOK) {
      // Update motion model
      if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        mLastFrame.GetRotationInverse().copyTo(
            LastTwc.rowRange(0, 3).colRange(0, 3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
        mVelocity = mCurrentFrame.mTcw * LastTwc;
      } else {
        mVelocity = cv::Mat();
      }

      // Log camera pose to RerunViewer
      if (mpViewer) {
        mpViewer->AddCurrentFrame(&mCurrentFrame);
      }

      // Clean VO matches
      for (int i = 0; i < mCurrentFrame.N; i++) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP) {
          if (pMP->Observations() < 1) {
            mCurrentFrame.mvbOutlier[i] = false;
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
          }
        }
      }

      // Delete temporal MapPoints
      for (std::list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(),
                                          lend = mlpTemporalPoints.end();
           lit != lend; lit++) {
        MapPoint* pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();

      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrame()) {
        CreateNewKeyFrame();
      }

      // We allow points with high innovation (considererd outliers by the Huber
      // Function) pass to the new keyframe, so that bundle adjustment will
      // finally decide if they are outliers or not. We don't want next frame to
      // estimate its position with those points so we discard them in the
      // frame.
      for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) {
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
      }
    }

    // Reset if the camera get lost soon after initialization
    if (mState == LOST) {
      if (mpMap->KeyFramesInMap() <= 5) {
        SLOG_INFO("Track lost soon after initialisation, reseting...");
        mpSystem->Reset();
        return;
      }
    }

    if (!mCurrentFrame.mpReferenceKF) {
      mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }

    mLastFrame = Frame(mCurrentFrame);
  }

  // Store frame pose information to retrieve the complete camera trajectory
  // afterwards.
  if (!mCurrentFrame.mTcw.empty()) {
    cv::Mat Tcr =
        mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
    mlbLost.push_back(mState == LOST);
  } else {
    // This can happen if tracking is lost
    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
    mlpReferences.push_back(mlpReferences.back());
    mlFrameTimes.push_back(mlFrameTimes.back());
    mlbLost.push_back(mState == LOST);
  }
}

void Tracking::StereoInitialization() {
  if (mCurrentFrame.N > 500) {
    // Set Frame pose to the origin
    mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

    // Create KeyFrame
    KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    // Insert KeyFrame in the map
    mpMap->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
        pNewMP->AddObservation(pKFini, i);
        pKFini->AddMapPoint(pNewMP, i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pNewMP);

        mCurrentFrame.mvpMapPoints[i] = pNewMP;
      }
    }

    SLOG_INFO("New map created with {} points", mpMap->MapPointsInMap());

    mpLocalMapper->InsertKeyFrame(pKFini);

    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFini;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    // Log camera pose to RerunViewer
    if (mpViewer) {
      mpViewer->AddCurrentFrame(&mCurrentFrame);
    }

    mState = OK;
  }
}

void Tracking::MonocularInitialization() {
  if (!mpInitializer) {
    // Set Reference Frame
    if (mCurrentFrame.mvKeys.size() > 100) {
      mInitialFrame = Frame(mCurrentFrame);
      mLastFrame = Frame(mCurrentFrame);
      mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++) {
        mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
      }

      mpInitializer = std::make_unique<Initializer>(mCurrentFrame, 1.0, 200);

      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      return;
    }
  } else {
    // Try to initialize
    if ((int)mCurrentFrame.mvKeys.size() <= 100) {
      mpInitializer.reset();
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }

    // Find correspondences
    SPmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(
        mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);
    SLOG_INFO("Matches {}", nmatches);
    // Check if there are enough correspondences
    if (nmatches < 100) {
      mpInitializer.reset();
      return;
    }

    cv::Mat Rcw;  // Current Camera Rotation
    cv::Mat tcw;  // Current Camera Translation
    std::vector<bool>
        vbTriangulated;  // Triangulated Correspondences (mvIniMatches)

    if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw,
                                  mvIniP3D, vbTriangulated)) {
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;
        }
      }

      SLOG_INFO("Numver of Matches After");

      // Set Frame Poses
      mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
      cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
      Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
      tcw.copyTo(Tcw.rowRange(0, 3).col(3));
      mCurrentFrame.SetPose(Tcw);

      CreateInitialMapMonocular();
    }
  }
}

void Tracking::CreateInitialMapMonocular() {
  SLOG_INFO("Setting Monocular Map");

  // Create KeyFrames
  SLOG_INFO("Creating KeyFrames...");
  KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
  KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  SLOG_INFO("Computing BoW for initial frame...");
  pKFini->ComputeBoW();
  SLOG_INFO("Computing BoW for current frame...");
  pKFcur->ComputeBoW();
  SLOG_INFO("BoW computation complete");

  // Insert KFs in the map
  SLOG_INFO("Adding KeyFrames to map...");
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  SLOG_INFO("Creating MapPoints from {} matches...", mvIniMatches.size());
  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    if (mvIniMatches[i] < 0) {
      continue;
    }

    // Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);

    MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

    // Add to Map
    mpMap->AddMapPoint(pMP);
  }

  // Update Connections
  SLOG_INFO("Updating connections...");
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  SLOG_INFO("New Map created with {} points", mpMap->MapPointsInMap());

  SLOG_INFO("Starting Bundle Adjustment...");
  Optimizer::GlobalBundleAdjustemnt(mpMap, 20);
  SLOG_INFO("Bundle Adjustment complete");

  // Set median depth to 1
  SLOG_INFO("Computing scene median depth...");
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;
  SLOG_INFO("Median depth: {}, Tracked points: {}", medianDepth,
            pKFcur->TrackedMapPoints(1));

  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
    SLOG_INFO("Wrong initialization, reseting...");
    Reset();
    return;
  }

  // Scale initial baseline
  cv::Mat Tc2w = pKFcur->GetPose();
  Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
  pKFcur->SetPose(Tc2w);

  // Scale points
  std::vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    if (vpAllMapPoints[iMP]) {
      MapPoint* pMP = vpAllMapPoints[iMP];
      pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }
  }

  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);

  // Log keyframes to RerunViewer
  if (mpViewer) {
    mpViewer->LogKeyFrame(pKFini);
    mpViewer->LogKeyFrame(pKFcur);
  }

  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKFcur;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  // Log keyframe to RerunViewer
  if (mpViewer) {
    mpViewer->LogKeyFrame(pKFcur);
  }

  mpMap->mvpKeyFrameOrigins.push_back(pKFini);

  SLOG_INFO("CreateInitialMapMonocular completed successfully!");
  mState = OK;
}

void Tracking::CheckReplacedInLastFrame() {
  for (int i = 0; i < mLastFrame.N; i++) {
    MapPoint* pMP = mLastFrame.mvpMapPoints[i];

    if (pMP) {
      MapPoint* pRep = pMP->GetReplaced();
      if (pRep) {
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}

bool Tracking::TrackReferenceKeyFrame() {
  // Compute Bag of Words std::vector
  mCurrentFrame.ComputeBoW();

  // We perform first a SuperPoint matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  SPmatcher matcher(0.7, true);
  std::vector<MapPoint*> vpMapPointMatches;

  // int nmatches =
  // matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
  int nmatches =
      matcher.SearchByNN(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

  if (nmatches < 15) {
    return false;
  }

  mCurrentFrame.mvpMapPoints = vpMapPointMatches;
  mCurrentFrame.SetPose(mLastFrame.mTcw);

  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
        nmatchesMap++;
      }
    }
  }

  return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
  // Update pose according to reference keyframe
  KeyFrame* pRef = mLastFrame.mpReferenceKF;
  cv::Mat Tlr = mlRelativeFramePoses.back();

  mLastFrame.SetPose(Tlr * pRef->GetPose());

  if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR ||
      !mbOnlyTracking) {
    return;
  }

  // Create "visual odometry" MapPoints
  // We sort points according to their measured depth by the stereo/RGB-D sensor
  std::vector<std::pair<float, int>> vDepthIdx;
  vDepthIdx.reserve(mLastFrame.N);
  for (int i = 0; i < mLastFrame.N; i++) {
    float z = mLastFrame.mvDepth[i];
    if (z > 0) {
      vDepthIdx.push_back(std::make_pair(z, i));
    }
  }

  if (vDepthIdx.empty()) {
    return;
  }

  sort(vDepthIdx.begin(), vDepthIdx.end());

  // We insert all close points (depth<mThDepth)
  // If less than 100 close points, we insert the 100 closest ones.
  int nPoints = 0;
  for (size_t j = 0; j < vDepthIdx.size(); j++) {
    int i = vDepthIdx[j].second;

    bool bCreateNew = false;

    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
    if (!pMP) {
      bCreateNew = true;
    } else if (pMP->Observations() < 1) {
      bCreateNew = true;
    }

    if (bCreateNew) {
      cv::Mat x3D = mLastFrame.UnprojectStereo(i);
      MapPoint* pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

      mLastFrame.mvpMapPoints[i] = pNewMP;

      mlpTemporalPoints.push_back(pNewMP);
      nPoints++;
    } else {
      nPoints++;
    }

    if (vDepthIdx[j].first > mThDepth && nPoints > 100) {
      break;
    }
  }
}

bool Tracking::TrackWithMotionModel() {
  SPmatcher matcher(0.9, true);

  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  UpdateLastFrame();

  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint*>(NULL));

  // Project points seen in previous frame
  // int th;
  // if(mSensor!=System::STEREO)
  //     th=15;
  // else
  //     th=7;
  int th = 15;
  int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th,
                                            mSensor == System::MONOCULAR);

  // If few matches, uses a wider window search
  if (nmatches < 20) {
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));
    nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th,
                                          mSensor == System::MONOCULAR);
  }

  if (nmatches < 20) {
    return false;
  }

  // Optimize frame pose with all matches
  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
        nmatchesMap++;
      }
    }
  }

  if (mbOnlyTracking) {
    mbVO = nmatchesMap < 10;
    return nmatches > 20;
  }

  return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the
  // frame. We retrieve the local map and try to find matches to points in the
  // local map.

  UpdateLocalMap();

  SearchLocalPoints();

  // Optimize Pose
  Optimizer::PoseOptimization(&mCurrentFrame);
  mnMatchesInliers = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (!mCurrentFrame.mvbOutlier[i]) {
        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
            mnMatchesInliers++;
          }
        } else {
          mnMatchesInliers++;
        }
      } else if (mSensor == System::STEREO) {
        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
      }
    }
  }

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalisation recently
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
      mnMatchesInliers < 50) {
    return false;
  }

  if (mnMatchesInliers < 10) {
    return false;
  } else {
    return true;
  }
}

bool Tracking::NeedNewKeyFrame() {
  if (mbOnlyTracking) {
    return false;
  }

  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
    return false;
  }

  const int nKFs = mpMap->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last
  // relocalisation
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
      nKFs > mMaxFrames) {
    return false;
  }

  // Tracked MapPoints in the reference keyframe
  int nMinObs = 3;
  if (nKFs <= 2) {
    nMinObs = 2;
  }
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be
  // potentially created.
  int nNonTrackedClose = 0;
  int nTrackedClose = 0;
  if (mSensor != System::MONOCULAR) {
    for (int i = 0; i < mCurrentFrame.N; i++) {
      if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
          nTrackedClose++;
        } else {
          nNonTrackedClose++;
        }
      }
    }
  }

  bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

  // Thresholds
  float thRefRatio = 0.75f;
  if (nKFs < 2) {
    thRefRatio = 0.4f;
  }

  if (mSensor == System::MONOCULAR) {
    thRefRatio = 0.9f;
  }

  // Condition 1a: More than "MaxFrames" have passed from last keyframe
  // insertion
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames &&
                    bLocalMappingIdle);
  // Condition 1c: tracking is weak
  const bool c1c =
      mSensor != System::MONOCULAR &&
      (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
  // Condition 2: Few tracked points compared to reference keyframe. Lots of
  // visual odometry compared to map matches.
  const bool c2 =
      ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
       mnMatchesInliers > 15);

  if ((c1a || c1b || c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle) {
      return true;
    } else {
      mpLocalMapper->InterruptBA();
      if (mSensor != System::MONOCULAR) {
        if (mpLocalMapper->KeyframesInQueue() < 3) {
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    }
  } else {
    return false;
  }
}

void Tracking::CreateNewKeyFrame() {
  if (!mpLocalMapper->SetNotStop(true)) {
    return;
  }

  KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  if (mSensor != System::MONOCULAR) {
    mCurrentFrame.UpdatePoseMatrices();

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        vDepthIdx.push_back(std::make_pair(z, i));
      }
    }

    if (!vDepthIdx.empty()) {
      sort(vDepthIdx.begin(), vDepthIdx.end());

      int nPoints = 0;
      for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP) {
          bCreateNew = true;
        } else if (pMP->Observations() < 1) {
          bCreateNew = true;
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }

        if (bCreateNew) {
          cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
          MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
          pNewMP->AddObservation(pKF, i);
          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          mpMap->AddMapPoint(pNewMP);

          mCurrentFrame.mvpMapPoints[i] = pNewMP;
          nPoints++;
        } else {
          nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100) {
          break;
        }
      }
    }
  }

  mpLocalMapper->InsertKeyFrame(pKF);

  // Log keyframe to RerunViewer
  if (mpViewer) {
    mpViewer->LogKeyFrame(pKF);
  }

  mpLocalMapper->SetNotStop(false);

  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints() {
  // Do not search map points already matched
  for (std::vector<MapPoint*>::iterator
           vit = mCurrentFrame.mvpMapPoints.begin(),
           vend = mCurrentFrame.mvpMapPoints.end();
       vit != vend; vit++) {
    MapPoint* pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        *vit = static_cast<MapPoint*>(NULL);
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        pMP->mbTrackInView = false;
      }
    }
  }

  int nToMatch = 0;

  // Project points in frame and check its visibility
  for (std::vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(),
                                        vend = mvpLocalMapPoints.end();
       vit != vend; vit++) {
    MapPoint* pMP = *vit;
    if (pMP->mnLastFrameSeen == mCurrentFrame.mnId) {
      continue;
    }
    if (pMP->isBad()) {
      continue;
    }
    // Project (this fills MapPoint variables for matching)
    if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
      pMP->IncreaseVisible();
      nToMatch++;
    }
  }

  if (nToMatch > 0) {
    SPmatcher matcher(0.8);
    // int th = 1;
    // if(mSensor==System::RGBD)
    //     th=3;
    // // If the camera has been relocalised recently, perform a coarser search
    // if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
    //     th=5;
    // matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    // NN only matching
    matcher.SearchByNN(mCurrentFrame, mvpLocalMapPoints);
  }
}

void Tracking::UpdateLocalMap() {
  // This is for visualization
  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
  mvpLocalMapPoints.clear();

  for (std::vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                              itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    KeyFrame* pKF = *itKF;
    const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for (std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin(),
                                                itEndMP = vpMPs.end();
         itMP != itEndMP; itMP++) {
      MapPoint* pMP = *itMP;
      if (!pMP) {
        continue;
      }
      if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) {
        continue;
      }
      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      }
    }
  }
}

void Tracking::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  std::map<KeyFrame*, int> keyframeCounter;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
      if (!pMP->isBad()) {
        const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
        for (std::map<KeyFrame*, size_t>::const_iterator
                 it = observations.begin(),
                 itend = observations.end();
             it != itend; it++) {
          keyframeCounter[it->first]++;
        }
      } else {
        mCurrentFrame.mvpMapPoints[i] = NULL;
      }
    }
  }

  if (keyframeCounter.empty()) {
    return;
  }

  int max = 0;
  KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map. Also
  // check which keyframe shares most points
  for (std::map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(),
                                                itEnd = keyframeCounter.end();
       it != itEnd; it++) {
    KeyFrame* pKF = it->first;

    if (pKF->isBad()) {
      continue;
    }

    if (it->second > max) {
      max = it->second;
      pKFmax = pKF;
    }

    mvpLocalKeyFrames.push_back(it->first);
    pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }

  // Include also some not-already-included keyframes that are neighbors to
  // already-included keyframes
  for (std::vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                              itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    // Limit the number of keyframes
    if (mvpLocalKeyFrames.size() > 80) {
      break;
    }

    KeyFrame* pKF = *itKF;

    const std::vector<KeyFrame*> vNeighs =
        pKF->GetBestCovisibilityKeyFrames(10);

    for (std::vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(),
                                                itEndNeighKF = vNeighs.end();
         itNeighKF != itEndNeighKF; itNeighKF++) {
      KeyFrame* pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad()) {
        if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pNeighKF);
          pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    const std::set<KeyFrame*> spChilds = pKF->GetChilds();
    for (std::set<KeyFrame*>::const_iterator sit = spChilds.begin(),
                                             send = spChilds.end();
         sit != send; sit++) {
      KeyFrame* pChildKF = *sit;
      if (!pChildKF->isBad()) {
        if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pChildKF);
          pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    KeyFrame* pParent = pKF->GetParent();
    if (pParent) {
      if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pParent);
        pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }
  }

  if (pKFmax) {
    mpReferenceKF = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}

bool Tracking::Relocalization() {
  // Compute Bag of Words std::vector
  mCurrentFrame.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for
  // relocalisation
  std::vector<KeyFrame*> vpCandidateKFs =
      mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

  if (vpCandidateKFs.empty()) {
    return false;
  }

  const int nKFs = vpCandidateKFs.size();

  // We perform first a SuperPoint matching with each candidate
  // If enough matches are found we setup a PnP solver
  SPmatcher matcher(0.75, true);

  std::vector<std::unique_ptr<PnPsolver>> vpPnPsolvers;
  vpPnPsolvers.resize(nKFs);

  std::vector<std::vector<MapPoint*>> vvpMapPointMatches;
  vvpMapPointMatches.resize(nKFs);

  std::vector<bool> vbDiscarded;
  vbDiscarded.resize(nKFs);

  int nCandidates = 0;

  for (int i = 0; i < nKFs; i++) {
    KeyFrame* pKF = vpCandidateKFs[i];
    if (pKF->isBad()) {
      vbDiscarded[i] = true;
    } else {
      int nmatches =
          matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
      if (nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        auto pSolver =
            std::make_unique<PnPsolver>(mCurrentFrame, vvpMapPointMatches[i]);
        pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
        vpPnPsolvers[i] = std::move(pSolver);
        nCandidates++;
      }
    }
  }

  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool bMatch = false;
  SPmatcher matcher2(0.9, true);

  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nKFs; i++) {
      if (vbDiscarded[i]) {
        continue;
      }

      // Perform 5 Ransac Iterations
      std::vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;

      PnPsolver* pSolver = vpPnPsolvers[i].get();
      cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

      // If Ransac reachs max. iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        nCandidates--;
      }

      // If a Camera Pose is computed, optimize
      if (!Tcw.empty()) {
        Tcw.copyTo(mCurrentFrame.mTcw);

        std::set<MapPoint*> sFound;

        const int np = vbInliers.size();

        for (int j = 0; j < np; j++) {
          if (vbInliers[j]) {
            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
            sFound.insert(vvpMapPointMatches[i][j]);
          } else {
            mCurrentFrame.mvpMapPoints[j] = NULL;
          }
        }

        int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

        if (nGood < 10) {
          continue;
        }

        for (int io = 0; io < mCurrentFrame.N; io++) {
          if (mCurrentFrame.mvbOutlier[io]) {
            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);
          }
        }

        // If few inliers, search by projection in a coarse window and optimize
        // again
        if (nGood < 50) {
          int nadditional = matcher2.SearchByProjection(
              mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

          if (nadditional + nGood >= 50) {
            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

            // If many inliers but still not enough, search by projection again
            // in a narrower window the camera has been already optimized with
            // many points
            if (nGood > 30 && nGood < 50) {
              sFound.clear();
              for (int ip = 0; ip < mCurrentFrame.N; ip++) {
                if (mCurrentFrame.mvpMapPoints[ip]) {
                  sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                }
              }
              nadditional = matcher2.SearchByProjection(
                  mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

              // Final optimization
              if (nGood + nadditional >= 50) {
                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                for (int io = 0; io < mCurrentFrame.N; io++) {
                  if (mCurrentFrame.mvbOutlier[io]) {
                    mCurrentFrame.mvpMapPoints[io] = NULL;
                  }
                }
              }
            }
          }
        }

        // If the pose is supported by enough inliers stop ransacs and continue
        if (nGood >= 50) {
          bMatch = true;
          break;
        }
      }
    }
  }

  if (!bMatch) {
    return false;
  } else {
    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
  }
}

void Tracking::Reset() {
  SLOG_INFO("System Reseting");
  if (mpViewer) {
    mpViewer->RequestStop();
    while (!mpViewer->isStopped()) {
      std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }
  }

  // Reset Local Mapping
  SLOG_INFO("Reseting Local Mapper...");
  mpLocalMapper->RequestReset();
  SLOG_INFO(" done");

  // Reset Loop Closing
  SLOG_INFO("Reseting Loop Closing...");
  mpLoopClosing->RequestReset();
  SLOG_INFO(" done");

  // Clear BoW Database
  SLOG_INFO("Reseting Database...");
  mpKeyFrameDB->clear();
  SLOG_INFO(" done");

  // Clear Map (this erase MapPoints and KeyFrames)
  mpMap->clear();

  KeyFrame::nNextId = 0;
  Frame::nNextId = 0;
  mState = NO_IMAGES_YET;

  mpInitializer.reset();

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();

  if (mpViewer) {
    mpViewer->Release();
  }
}

void Tracking::ChangeCalibration(const std::string& strSettingPath) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  mbf = fSettings["Camera.bf"];

  Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool& flag) { mbOnlyTracking = flag; }

}  // namespace SuperSLAM
