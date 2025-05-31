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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iomanip>
#include <thread>

#include "Converter.h"

namespace SuperSLAM {

System::System(const std::string &strVocFile,
               const std::string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer)
    : mSensor(sensor),
      mpViewer(static_cast<RerunViewer *>(NULL)),
      mbReset(false),
      mbActivateLocalizationMode(false),
      mbDeactivateLocalizationMode(false) {
  // Output welcome message
  std::cout << "\n"
            << "SuperSLAM Copyright (C) Aditya Wagh (adityamwagh@outlook.com)"
            << "\n"
            << "This program comes with ABSOLUTELY NO WARRANTY;" << "\n"
            << "This is free software, and you are welcome to redistribute it"
            << "\n"
            << "under certain conditions. See LICENSE.txt." << "\n"
            << "\n";

  std::cout << "Input sensor was set to: ";

  if (mSensor == MONOCULAR)
    std::cout << "Monocular" << "\n";
  else if (mSensor == STEREO)
    std::cout << "Stereo" << "\n";
  else if (mSensor == RGBD)
    std::cout << "RGB-D" << "\n";

  // Check settings file
  cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << strSettingsFile << "\n";
    exit(-1);
  }

  // Load SuperPoint Vocabulary
  std::cout << "\n"
            << "Loading SuperPoint Vocabulary. This could take a while..."
            << "\n";

  mpVocabulary = new ORBVocabulary();
  // bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
  // if(!bVocLoad)
  //{
  //    std::cerr << "Wrong path to vocabulary. " << "\n";
  //    std::cerr << "Falied to open at: " << strVocFile << "\n";
  //    exit(-1);
  //}
  // std::cout << "Vocabulary loaded!" << "\n" << "\n";
  mpVocabulary->load(strVocFile);

  // Create KeyFrame Database
  mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

  // Create the Map
  mpMap = new Map();

  // Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this
  // constructor)
  mpTracker = new Tracking(this, mpVocabulary, mpMap, mpKeyFrameDatabase,
                           strSettingsFile, mSensor);

  // Initialize the Local Mapping thread and launch
  mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);
  mptLocalMapping =
      new std::thread(&SuperSLAM::LocalMapping::Run, mpLocalMapper);

  // Initialize the Loop Closing thread and launch
  mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary,
                                 mSensor != MONOCULAR);
  mptLoopClosing = new std::thread(&SuperSLAM::LoopClosing::Run, mpLoopCloser);

  // Initialize the Viewer thread and launch
  if (bUseViewer) {
    mpViewer = new RerunViewer(this, mpTracker, mpMap, strSettingsFile);
    mptViewer = new std::thread(&RerunViewer::Run, mpViewer);
    mpTracker->SetViewer(mpViewer);
  }

  // Set pointers between threads
  mpTracker->SetLocalMapper(mpLocalMapper);
  mpTracker->SetLoopClosing(mpLoopCloser);

  mpLocalMapper->SetTracker(mpTracker);
  mpLocalMapper->SetLoopCloser(mpLoopCloser);

  mpLoopCloser->SetTracker(mpTracker);
  mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                            const double &timestamp) {
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

  std::unique_lock<std::mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap,
                          const double &timestamp) {
  if (mSensor != RGBD) {
    std::cerr
        << "ERROR: you called TrackRGBD but input sensor was not set to RGBD."
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

  cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp);

  std::unique_lock<std::mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp) {
  if (mSensor != MONOCULAR) {
    std::cerr << "ERROR: you called TrackMonocular but input sensor was not "
                 "set to Monocular."
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

  cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

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
  } else
    return false;
}

void System::Reset() {
  std::unique_lock<std::mutex> lock(mMutexReset);
  mbReset = true;
}

void System::Shutdown() {
  mpLocalMapper->RequestFinish();
  mpLoopCloser->RequestFinish();
  if (mpViewer) {
    mpViewer->RequestFinish();
    while (!mpViewer->isFinished())
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
  }

  // Wait until all thread have effectively stopped
  while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() ||
         mpLoopCloser->isRunningGBA()) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
  }

  if (mpViewer) mpViewer->Release();
}

void System::SaveTrajectoryTUM(const std::string &filename) {
  std::cout << "\n"
            << "Saving camera trajectory to " << filename << " ..."
            << "\n";
  if (mSensor == MONOCULAR) {
    std::cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular."
              << "\n";
    return;
  }

  std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
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
  std::list<SuperSLAM::KeyFrame *>::iterator lRit =
      mpTracker->mlpReferences.begin();
  std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  std::list<bool>::iterator lbL = mpTracker->mlbLost.begin();
  for (std::list<cv::Mat>::iterator
           lit = mpTracker->mlRelativeFramePoses.begin(),
           lend = mpTracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++, lbL++) {
    if (*lbL) continue;

    KeyFrame *pKF = *lRit;

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
  std::cout << "\n" << "trajectory saved!" << "\n";
}

void System::SaveKeyFrameTrajectoryTUM(const std::string &filename) {
  std::cout << "\n"
            << "Saving keyframe trajectory to " << filename << " ..."
            << "\n";

  std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  // cv::Mat Two = vpKFs[0]->GetPoseInverse();

  std::ofstream f;
  f.open(filename.c_str());
  f << std::fixed;

  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame *pKF = vpKFs[i];

    // pKF->SetPose(pKF->GetPose()*Two);

    if (pKF->isBad()) continue;

    cv::Mat R = pKF->GetRotation().t();
    std::vector<float> q = Converter::toQuaternion(R);
    cv::Mat t = pKF->GetCameraCenter();
    f << std::setprecision(6) << pKF->mTimeStamp << std::setprecision(7) << " "
      << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " "
      << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
  }

  f.close();
  std::cout << "\n" << "trajectory saved!" << "\n";
}

void System::SaveTrajectoryKITTI(const std::string &filename) {
  std::cout << "\n"
            << "Saving camera trajectory to " << filename << " ..."
            << "\n";
  if (mSensor == MONOCULAR) {
    std::cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular."
              << "\n";
    return;
  }

  std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
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
  std::list<SuperSLAM::KeyFrame *>::iterator lRit =
      mpTracker->mlpReferences.begin();
  std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  for (std::list<cv::Mat>::iterator
           lit = mpTracker->mlRelativeFramePoses.begin(),
           lend = mpTracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++) {
    SuperSLAM::KeyFrame *pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    while (pKF->isBad()) {
      //  std::cout << "bad parent" << "\n";
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
  std::cout << "\n" << "trajectory saved!" << "\n";
}

int System::GetTrackingState() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackingState;
}

std::vector<MapPoint *> System::GetTrackedMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackedMapPoints;
}

std::vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackedKeyPointsUn;
}

}  // namespace SuperSLAM
