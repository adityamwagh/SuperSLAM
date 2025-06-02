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

#include "System.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include "Converter.h"
#include "Logging.h"

namespace SuperSLAM {

System::System(const std::string& strVocFile,
               const std::string& strSettingsFile, const eSensor sensor,
               const bool bUseViewer)
    : mSensor(sensor),
      mbReset(false),
      mbActivateLocalizationMode(false),
      mbDeactivateLocalizationMode(false) {
  // Initialize logging system
  SuperSLAM::Logger::initialize();

  // Output welcome message
  SLOG_INFO("SuperSLAM Copyright (C) Aditya Wagh (adityamwagh@outlook.com)");
  SLOG_INFO("This program comes with ABSOLUTELY NO WARRANTY;");
  SLOG_INFO("This is free software, and you are welcome to redistribute it");
  SLOG_INFO("under certain conditions. See LICENSE.txt.");

  if (mSensor == MONOCULAR) {
    SLOG_INFO("Input sensor was set to: Monocular");
  } else if (mSensor == STEREO) {
    SLOG_INFO("Input sensor was set to: Stereo");
  } else if (mSensor == RGBD) {
    SLOG_INFO("Input sensor was set to: RGB-D");
  }

  // Check settings file
  cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    SLOG_ERROR("Failed to open settings file at: {}", strSettingsFile);
    exit(-1);
  }

  // Load SuperPoint Vocabulary (optional for ORB-free operation)
  mpVocabulary = std::make_unique<ORBVocabulary>();
  bool use_vocabulary = !strVocFile.empty();

  if (use_vocabulary) {
    SLOG_INFO("Loading SuperPoint Vocabulary. This could take a while...");
    try {
      mpVocabulary->load(strVocFile);
      SLOG_INFO("Vocabulary loaded successfully! Loop closure enabled.");
    } catch (const std::exception& e) {
      SLOG_WARN("Failed to load vocabulary: {}", e.what());
      SLOG_WARN("Continuing without vocabulary - loop closure disabled.");
      use_vocabulary = false;
    }
  } else {
    SLOG_INFO(
        "No vocabulary provided - running in ORB-free mode (loop closure "
        "disabled)");
  }

  // Create KeyFrame Database
  mpKeyFrameDatabase = std::make_unique<KeyFrameDatabase>(*mpVocabulary);

  // Create the Map
  mpMap = std::make_unique<Map>();

  // Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this
  // constructor)
  mpTracker = std::make_unique<Tracking>(this, mpVocabulary.get(), mpMap.get(),
                                         mpKeyFrameDatabase.get(),
                                         strSettingsFile, mSensor);

  // Initialize the Local Mapping thread and launch
  mpLocalMapper =
      std::make_unique<LocalMapping>(mpMap.get(), mSensor == MONOCULAR);
  mptLocalMapping = std::make_unique<std::thread>(&SuperSLAM::LocalMapping::Run,
                                                  mpLocalMapper.get());

  // Initialize the Loop Closing thread and launch (only if vocabulary is
  // available)
  if (use_vocabulary) {
    mpLoopCloser =
        std::make_unique<LoopClosing>(mpMap.get(), mpKeyFrameDatabase.get(),
                                      mpVocabulary.get(), mSensor != MONOCULAR);
    mptLoopClosing = std::make_unique<std::thread>(&SuperSLAM::LoopClosing::Run,
                                                   mpLoopCloser.get());
  } else {
    // mpLoopCloser and mptLoopClosing remain nullptr (unique_ptr default)
    SLOG_INFO("Loop closing thread disabled (no vocabulary)");
  }

  // Initialize the Viewer thread and launch
  if (bUseViewer) {
    mpViewer = std::make_unique<RerunViewer>();
    if (mpMap && mpViewer) {
      mpViewer->SetMap(mpMap.get());
    }

    // Set camera parameters for the viewer
    if (mpViewer) {
      float fx = fsSettings["Camera.fx"];
      float fy = fsSettings["Camera.fy"];
      float cx = fsSettings["Camera.cx"];
      float cy = fsSettings["Camera.cy"];

      if (mSensor == STEREO) {
        // For stereo, calculate baseline from Camera.bf
        float bf = fsSettings["Camera.bf"];
        float baseline = bf / fx;  // baseline = bf / fx

        // Right camera has same intrinsics but shifted cx
        float cx_right = cx - baseline * fx;
        mpViewer->SetCameras(fx, fy, cx, cy, fx, fy, cx_right, cy, baseline);
      } else {
        // For monocular, use same parameters for both cameras (second camera
        // won't be used)
        mpViewer->SetCameras(fx, fy, cx, cy, fx, fy, cx, cy, 0.0f);
      }
    }

    mptViewer =
        std::make_unique<std::thread>(&RerunViewer::Run, mpViewer.get());
    if (mpTracker && mpViewer) {
      mpTracker->SetViewer(mpViewer.get());
    }
  }

  // Set pointers between threads
  mpTracker->SetLocalMapper(mpLocalMapper.get());
  if (mpLoopCloser) {
    mpTracker->SetLoopClosing(mpLoopCloser.get());
    mpLocalMapper->SetLoopCloser(mpLoopCloser.get());
  }

  mpLocalMapper->SetTracker(mpTracker.get());
}

System::~System() {
  // Shutdown the system first
  Shutdown();

  // Wait for threads to finish
  if (mptLocalMapping && mptLocalMapping->joinable()) {
    mptLocalMapping->join();
  }
  if (mptLoopClosing && mptLoopClosing->joinable()) {
    mptLoopClosing->join();
  }
  if (mptViewer && mptViewer->joinable()) {
    mptViewer->join();
  }

  // Smart pointers will automatically clean up the objects
}

cv::Mat System::TrackStereo(const cv::Mat& imLeft, const cv::Mat& imRight,
                            const double& timestamp) {
  if (mSensor != STEREO) {
    std::cerr << "ERROR: you called TrackStereo but input sensor was not set "
                 "to STEREO."
              << "\n";
    exit(-1);
  }

  // Check mode change
  {
    std::unique_lock<std::mutex> lock(mMutexMode);
    if (mbActivateLocalizationMode) {
      mpLocalMapper->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!mpLocalMapper->isStopped()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
      }

      mpTracker->InformOnlyTracking(true);
      mbActivateLocalizationMode = false;
    }
    if (mbDeactivateLocalizationMode) {
      mpTracker->InformOnlyTracking(false);
      mpLocalMapper->Release();
      mbDeactivateLocalizationMode = false;
    }
  }

  // Check reset
  {
    std::unique_lock<std::mutex> lock(mMutexReset);
    if (mbReset) {
      mpTracker->Reset();
      mbReset = false;
    }
  }

  cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

  // Update viewer with stereo images if available
  if (mpViewer) {
    mpViewer->AddStereoFrames(&mpTracker->mCurrentFrame, imLeft, imRight);
  }

  std::unique_lock<std::mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat& im, const cv::Mat& depthmap,
                          const double& timestamp) {
  if (mSensor != RGBD) {
    SLOG_ERROR(
        "ERROR: you called TrackRGBD but input sensor was not set to RGBD.");
    exit(-1);
  }

  // Check mode change
  {
    std::unique_lock<std::mutex> lock(mMutexMode);
    if (mbActivateLocalizationMode) {
      mpLocalMapper->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!mpLocalMapper->isStopped()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
      }

      mpTracker->InformOnlyTracking(true);
      mbActivateLocalizationMode = false;
    }
    if (mbDeactivateLocalizationMode) {
      mpTracker->InformOnlyTracking(false);
      mpLocalMapper->Release();
      mbDeactivateLocalizationMode = false;
    }
  }

  // Check reset
  {
    std::unique_lock<std::mutex> lock(mMutexReset);
    if (mbReset) {
      mpTracker->Reset();
      mbReset = false;
    }
  }

  cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp);

  std::unique_lock<std::mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat& im, const double& timestamp) {
  if (mSensor != MONOCULAR) {
    SLOG_ERROR(
        "ERROR: you called TrackMonocular but input sensor was not set to "
        "Monocular.");
    exit(-1);
  }

  // Check mode change
  {
    std::unique_lock<std::mutex> lock(mMutexMode);
    if (mbActivateLocalizationMode) {
      mpLocalMapper->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!mpLocalMapper->isStopped()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
      }

      mpTracker->InformOnlyTracking(true);
      mbActivateLocalizationMode = false;
    }
    if (mbDeactivateLocalizationMode) {
      mpTracker->InformOnlyTracking(false);
      mpLocalMapper->Release();
      mbDeactivateLocalizationMode = false;
    }
  }

  // Check reset
  {
    std::unique_lock<std::mutex> lock(mMutexReset);
    if (mbReset) {
      mpTracker->Reset();
      mbReset = false;
    }
  }

  cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

  // Update viewer with monocular image if available
  if (mpViewer) {
    mpViewer->AddCurrentFrame(&mpTracker->mCurrentFrame);
  }

  std::unique_lock<std::mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

  return Tcw;
}

void System::ActivateLocalizationMode() {
  std::unique_lock<std::mutex> lock(mMutexMode);
  mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode() {
  std::unique_lock<std::mutex> lock(mMutexMode);
  mbDeactivateLocalizationMode = true;
}

bool System::MapChanged() {
  static int n = 0;
  int curn = mpMap->GetLastBigChangeIdx();
  if (n < curn) {
    n = curn;
    return true;
  } else {
    return false;
  }
}

void System::Reset() {
  std::unique_lock<std::mutex> lock(mMutexReset);
  mbReset = true;
}

void System::Shutdown() {
  mpLocalMapper->RequestFinish();
  if (mpLoopCloser) {
    mpLoopCloser->RequestFinish();
  }
  if (mpViewer) {
    mpViewer->RequestFinish();
    while (!mpViewer->isFinished()) {
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }
  }

  // Wait until all thread have effectively stopped
  if (mpLoopCloser) {
    while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() ||
           mpLoopCloser->isRunningGBA()) {
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }
  } else {
    while (!mpLocalMapper->isFinished()) {
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }
  }

  if (mpViewer) {
    mpViewer->Release();
  }
}

void System::SaveTrajectoryTUM(const std::string& filename) {
  SLOG_INFO("Saving camera trajectory to {} ...", filename);
  if (mSensor == MONOCULAR) {
    SLOG_ERROR("ERROR: SaveTrajectoryTUM cannot be used for monocular.");
    return;
  }

  std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  std::ofstream f;
  f.open(filename.c_str());
  f << std::fixed;

  // Frame pose is stored relative to its reference keyframe (which is optimized
  // by BA and pose graph). We need to get first the keyframe pose and then
  // concatenate the relative transformation. Frames not localized (tracking
  // failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and
  // a flag which is true when tracking failed (lbL).
  std::list<SuperSLAM::KeyFrame*>::iterator lRit =
      mpTracker->mlpReferences.begin();
  std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  std::list<bool>::iterator lbL = mpTracker->mlbLost.begin();
  for (std::list<cv::Mat>::iterator
           lit = mpTracker->mlRelativeFramePoses.begin(),
           lend = mpTracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++, lbL++) {
    if (*lbL) {
      continue;
    }

    KeyFrame* pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    // If the reference keyframe was culled, traverse the spanning tree to get a
    // suitable keyframe.
    while (pKF->isBad()) {
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    std::vector<float> q = Converter::toQuaternion(Rwc);

    f << std::setprecision(6) << *lT << " " << std::setprecision(9)
      << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2)
      << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
  }
  f.close();
  SLOG_INFO("trajectory saved!");
}

void System::SaveKeyFrameTrajectoryTUM(const std::string& filename) {
  SLOG_INFO("Saving keyframe trajectory to {} ...", filename);

  std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  // cv::Mat Two = vpKFs[0]->GetPoseInverse();

  std::ofstream f;
  f.open(filename.c_str());
  f << std::fixed;

  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];

    // pKF->SetPose(pKF->GetPose()*Two);

    if (pKF->isBad()) {
      continue;
    }

    cv::Mat R = pKF->GetRotation().t();
    std::vector<float> q = Converter::toQuaternion(R);
    cv::Mat t = pKF->GetCameraCenter();
    f << std::setprecision(6) << pKF->mTimeStamp << std::setprecision(7) << " "
      << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " "
      << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
  }

  f.close();
  SLOG_INFO("trajectory saved!");
}

void System::SaveTrajectoryKITTI(const std::string& filename) {
  SLOG_INFO("Saving camera trajectory to {} ...", filename);
  if (mSensor == MONOCULAR) {
    SLOG_ERROR("ERROR: SaveTrajectoryKITTI cannot be used for monocular.");
    return;
  }

  std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  std::ofstream f;
  f.open(filename.c_str());
  f << std::fixed;

  // Frame pose is stored relative to its reference keyframe (which is optimized
  // by BA and pose graph). We need to get first the keyframe pose and then
  // concatenate the relative transformation. Frames not localized (tracking
  // failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and
  // a flag which is true when tracking failed (lbL).
  std::list<SuperSLAM::KeyFrame*>::iterator lRit =
      mpTracker->mlpReferences.begin();
  std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  for (std::list<cv::Mat>::iterator
           lit = mpTracker->mlRelativeFramePoses.begin(),
           lend = mpTracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++) {
    SuperSLAM::KeyFrame* pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    while (pKF->isBad()) {
      //  SLOG_WARN("bad parent");
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    f << std::setprecision(9) << Rwc.at<float>(0, 0) << " "
      << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " "
      << twc.at<float>(0) << " " << Rwc.at<float>(1, 0) << " "
      << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " "
      << twc.at<float>(1) << " " << Rwc.at<float>(2, 0) << " "
      << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " "
      << twc.at<float>(2) << "\n";
  }
  f.close();
  SLOG_INFO("trajectory saved!");
}

int System::GetTrackingState() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackingState;
}

std::vector<MapPoint*> System::GetTrackedMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackedMapPoints;
}

std::vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackedKeyPointsUn;
}

}  // namespace SuperSLAM
