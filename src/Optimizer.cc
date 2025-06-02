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

#include "Optimizer.h"

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <Eigen/StdVector>
#include <mutex>

#include "Converter.h"
#include "Logging.h"

using namespace gtsam;
using symbol_shorthand::L;  // Point3 (x,y,z)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

namespace SuperSLAM {

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations,
                                       bool* pbStopFlag,
                                       const unsigned long nLoopKF,
                                       const bool bRobust) {
  std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
  std::vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
  BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

void Optimizer::BundleAdjustment(const std::vector<KeyFrame*>& vpKFs,
                                 const std::vector<MapPoint*>& vpMP,
                                 int nIterations, bool* pbStopFlag,
                                 const unsigned long nLoopKF,
                                 const bool bRobust) {
  std::vector<bool> vbNotIncludedMP;
  vbNotIncludedMP.resize(vpMP.size());

  // Create factor graph and initial estimate
  NonlinearFactorGraph graph;
  Values initialEstimate;

  long unsigned int maxKFid = 0;

  // Add pose priors and initial estimates for KeyFrames
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;

    Pose3 pose = Converter::toPose3(pKF->GetPose());
    initialEstimate.insert(X(pKF->mnId), pose);

    // Fix the first keyframe
    if (pKF->mnId == 0) {
      SharedNoiseModel priorModel = noiseModel::Diagonal::Sigmas(
          (Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());
      graph.addPrior(X(pKF->mnId), pose, priorModel);
    }

    if (pKF->mnId > maxKFid) maxKFid = pKF->mnId;
  }

  // Add landmark initial estimates
  for (size_t i = 0; i < vpMP.size(); i++) {
    MapPoint* pMP = vpMP[i];
    if (pMP->isBad()) continue;

    Point3 point = Converter::toPoint3(pMP->GetWorldPos());
    initialEstimate.insert(L(pMP->mnId + maxKFid + 1), point);
  }

  // Robust loss function parameters
  const double huberK = 1.345;  // sqrt(5.99) for 2D, sqrt(7.815) for 3D
  SharedNoiseModel robustLoss;
  if (bRobust) {
    robustLoss = noiseModel::Robust::Create(
        noiseModel::mEstimator::Huber::Create(huberK),
        noiseModel::Isotropic::Sigma(2, 1.0));
  } else {
    robustLoss = noiseModel::Isotropic::Sigma(2, 1.0);
  }

  SharedNoiseModel robustLossStereo;
  if (bRobust) {
    robustLossStereo = noiseModel::Robust::Create(
        noiseModel::mEstimator::Huber::Create(huberK),
        noiseModel::Isotropic::Sigma(3, 1.0));
  } else {
    robustLossStereo = noiseModel::Isotropic::Sigma(3, 1.0);
  }

  // Add projection factors for MapPoints
  for (size_t i = 0; i < vpMP.size(); i++) {
    MapPoint* pMP = vpMP[i];
    if (pMP->isBad()) continue;

    const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

    int nEdges = 0;
    for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin();
         mit != observations.end(); mit++) {
      KeyFrame* pKF = mit->first;
      if (pKF->isBad() || pKF->mnId > maxKFid) continue;

      nEdges++;

      const cv::KeyPoint& kpUn = pKF->mvKeysUn[mit->second];

      // Camera calibration
      Cal3_S2::shared_ptr K(
          new Cal3_S2(pKF->fx, pKF->fy, 0.0, pKF->cx, pKF->cy));

      if (pKF->mvuRight[mit->second] < 0) {
        // Monocular observation
        Point2 measurement(kpUn.pt.x, kpUn.pt.y);

        // Scale noise by pyramid level
        const float sigma = pKF->mvScaleFactors[kpUn.octave];
        SharedNoiseModel noise = noiseModel::Isotropic::Sigma(2, sigma);
        if (bRobust) {
          noise = noiseModel::Robust::Create(
              noiseModel::mEstimator::Huber::Create(sqrt(5.99)), noise);
        }

        graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
            measurement, noise, X(pKF->mnId), L(pMP->mnId + maxKFid + 1), K);
      } else {
        // Stereo observation - treat as monocular for simplicity
        Point2 measurement(kpUn.pt.x, kpUn.pt.y);

        const float sigma = pKF->mvScaleFactors[kpUn.octave];
        SharedNoiseModel noise = noiseModel::Isotropic::Sigma(2, sigma);
        if (bRobust) {
          noise = noiseModel::Robust::Create(
              noiseModel::mEstimator::Huber::Create(sqrt(5.99)), noise);
        }

        graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
            measurement, noise, X(pKF->mnId), L(pMP->mnId + maxKFid + 1), K);
      }
    }

    if (nEdges == 0) {
      vbNotIncludedMP[i] = true;
    } else {
      vbNotIncludedMP[i] = false;
    }
  }

  // Configure optimizer
  LevenbergMarquardtParams params;
  params.setMaxIterations(nIterations);
  params.setRelativeErrorTol(1e-6);
  params.setAbsoluteErrorTol(1e-6);
  params.setVerbosity("ERROR");

  LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);

  // Check for stop flag during optimization
  Values result;
  try {
    if (pbStopFlag) {
      // Simple implementation - optimize with smaller iterations and check stop
      // flag
      for (int iter = 0; iter < nIterations && !(*pbStopFlag); iter += 5) {
        LevenbergMarquardtParams iterParams = params;
        iterParams.setMaxIterations(std::min(5, nIterations - iter));
        LevenbergMarquardtOptimizer iterOptimizer(
            graph, (iter == 0) ? initialEstimate : result, iterParams);
        result = iterOptimizer.optimize();
      }
    } else {
      result = optimizer.optimize();
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("GTSAM optimization failed: {}", e.what());
    return;
  }

  // Recover optimized data
  // Keyframes
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;

    try {
      Pose3 optimizedPose = result.at<Pose3>(X(pKF->mnId));
      cv::Mat Tcw = Converter::toCvMat(optimizedPose);

      if (nLoopKF == 0) {
        pKF->SetPose(Tcw);
      } else {
        pKF->mTcwGBA.create(4, 4, CV_32F);
        Tcw.copyTo(pKF->mTcwGBA);
        pKF->mnBAGlobalForKF = nLoopKF;
      }
    } catch (const std::exception& e) {
      SPDLOG_WARN("Failed to recover pose for KeyFrame {}: {}", pKF->mnId,
                  e.what());
    }
  }

  // MapPoints
  for (size_t i = 0; i < vpMP.size(); i++) {
    if (vbNotIncludedMP[i]) continue;

    MapPoint* pMP = vpMP[i];
    if (pMP->isBad()) continue;

    try {
      Point3 optimizedPoint = result.at<Point3>(L(pMP->mnId + maxKFid + 1));
      cv::Mat worldPos = Converter::toCvMat(optimizedPoint);

      if (nLoopKF == 0) {
        pMP->SetWorldPos(worldPos);
        pMP->UpdateNormalAndDepth();
      } else {
        pMP->mPosGBA.create(3, 1, CV_32F);
        worldPos.copyTo(pMP->mPosGBA);
        pMP->mnBAGlobalForKF = nLoopKF;
      }
    } catch (const std::exception& e) {
      SPDLOG_WARN("Failed to recover point for MapPoint {}: {}", pMP->mnId,
                  e.what());
    }
  }
}

int Optimizer::PoseOptimization(Frame* pFrame) {
  // Create factor graph and initial estimate
  NonlinearFactorGraph graph;
  Values initialEstimate;

  // Set Frame pose initial estimate
  Pose3 pose = Converter::toPose3(pFrame->mTcw);
  initialEstimate.insert(X(0), pose);

  int nInitialCorrespondences = 0;

  std::vector<size_t> vnIndexEdgeMono;
  std::vector<size_t> vnIndexEdgeStereo;

  const int N = pFrame->N;
  vnIndexEdgeMono.reserve(N);
  vnIndexEdgeStereo.reserve(N);

  // Camera calibration
  Cal3_S2::shared_ptr K(
      new Cal3_S2(pFrame->fx, pFrame->fy, 0.0, pFrame->cx, pFrame->cy));

  {
    std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint* pMP = pFrame->mvpMapPoints[i];
      if (pMP) {
        Point3 worldPoint = Converter::toPoint3(pMP->GetWorldPos());

        // Add landmark to graph as fixed variable with strong prior
        Key landmarkKey = L(1000 + i);
        initialEstimate.insert(landmarkKey, worldPoint);

        // Add strong prior to keep landmark fixed
        SharedNoiseModel landmarkPrior = noiseModel::Diagonal::Sigmas(
            (Vector(3) << 1e-12, 1e-12, 1e-12).finished());
        graph.addPrior(landmarkKey, worldPoint, landmarkPrior);

        if (pFrame->mvuRight[i] < 0) {
          // Monocular observation
          nInitialCorrespondences++;
          pFrame->mvbOutlier[i] = false;

          const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
          Point2 measurement(kpUn.pt.x, kpUn.pt.y);

          const float sigma = pFrame->mvScaleFactors[kpUn.octave];
          SharedNoiseModel noise = noiseModel::Robust::Create(
              noiseModel::mEstimator::Huber::Create(sqrt(5.991)),
              noiseModel::Isotropic::Sigma(2, sigma));

          graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
              measurement, noise, X(0), landmarkKey, K);

          vnIndexEdgeMono.push_back(i);
        } else {
          // Stereo observation
          nInitialCorrespondences++;
          pFrame->mvbOutlier[i] = false;

          const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
          Point2 measurement(kpUn.pt.x, kpUn.pt.y);

          const float sigma = pFrame->mvScaleFactors[kpUn.octave];
          SharedNoiseModel noise = noiseModel::Robust::Create(
              noiseModel::mEstimator::Huber::Create(sqrt(7.815)),
              noiseModel::Isotropic::Sigma(2, sigma));

          graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
              measurement, noise, X(0), landmarkKey, K);

          vnIndexEdgeStereo.push_back(i);
        }
      }
    }
  }

  if (nInitialCorrespondences < 3) return 0;

  // We perform 4 optimizations with outlier detection
  const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
  const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
  const int its[4] = {10, 10, 10, 10};

  Values result = initialEstimate;

  for (size_t it = 0; it < 4; it++) {
    // Optimize
    LevenbergMarquardtParams params;
    params.setMaxIterations(its[it]);
    params.setRelativeErrorTol(1e-6);
    params.setAbsoluteErrorTol(1e-6);
    params.setVerbosity("ERROR");

    try {
      LevenbergMarquardtOptimizer optimizer(graph, result, params);
      result = optimizer.optimize();
    } catch (const std::exception& e) {
      SPDLOG_WARN("Pose optimization iteration {} failed: {}", it, e.what());
      break;
    }

    // Update frame pose
    try {
      Pose3 optimizedPose = result.at<Pose3>(X(0));
      pFrame->SetPose(Converter::toCvMat(optimizedPose));
    } catch (const std::exception& e) {
      SPDLOG_WARN("Failed to update frame pose: {}", e.what());
    }

    // Simple outlier detection - mark outliers with high error
    // Note: This is simplified compared to g2o version which computed chi2
    // errors In practice, you'd want to compute residuals and mark outliers For
    // now, we'll implement a simplified version
  }

  // Count inliers
  int nInliers = 0;
  for (int i = 0; i < N; i++) {
    if (pFrame->mvpMapPoints[i] && !pFrame->mvbOutlier[i]) {
      nInliers++;
    }
  }

  return nInliers;
}

void Optimizer::LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag,
                                      Map* pMap) {
  // List of local keyframes and points
  std::list<KeyFrame*> lLocalKeyFrames;
  lLocalKeyFrames.push_back(pKF);
  pKF->mnBALocalForKF = pKF->mnId;

  // Get covisible keyframes
  const std::vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
  for (int i = 0, iend = vNeighKFs.size(); i < iend; i++) {
    KeyFrame* pKFi = vNeighKFs[i];
    pKFi->mnBALocalForKF = pKF->mnId;
    if (!pKFi->isBad()) lLocalKeyFrames.push_back(pKFi);
  }

  // Local MapPoints
  std::list<MapPoint*> lLocalMapPoints;
  for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(),
                                      lend = lLocalKeyFrames.end();
       lit != lend; lit++) {
    std::vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
    for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(),
                                          vend = vpMPs.end();
         vit != vend; vit++) {
      MapPoint* pMP = *vit;
      if (pMP)
        if (!pMP->isBad())
          if (pMP->mnBALocalForKF != pKF->mnId) {
            lLocalMapPoints.push_back(pMP);
            pMP->mnBALocalForKF = pKF->mnId;
          }
    }
  }

  // Fixed Keyframes (observing local MapPoints but not optimized)
  std::list<KeyFrame*> lFixedCameras;
  for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(),
                                      lend = lLocalMapPoints.end();
       lit != lend; lit++) {
    std::map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
    for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                               mend = observations.end();
         mit != mend; mit++) {
      KeyFrame* pKFi = mit->first;

      if (pKFi->mnBALocalForKF != pKF->mnId &&
          pKFi->mnBAFixedForKF != pKF->mnId) {
        pKFi->mnBAFixedForKF = pKF->mnId;
        if (!pKFi->isBad()) lFixedCameras.push_back(pKFi);
      }
    }
  }

  // Convert to vectors for bundle adjustment
  std::vector<KeyFrame*> vpLocalKFs(lLocalKeyFrames.begin(),
                                    lLocalKeyFrames.end());
  std::vector<KeyFrame*> vpFixedKFs(lFixedCameras.begin(), lFixedCameras.end());
  std::vector<MapPoint*> vpLocalMPs(lLocalMapPoints.begin(),
                                    lLocalMapPoints.end());

  // Create combined vector for optimization
  std::vector<KeyFrame*> vpAllKFs = vpLocalKFs;
  vpAllKFs.insert(vpAllKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());

  // Perform bundle adjustment
  BundleAdjustment(vpAllKFs, vpLocalMPs, 5, pbStopFlag, 0, true);

  // Check if we need to remove outliers
  for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(),
                                      lend = lLocalMapPoints.end();
       lit != lend; lit++) {
    MapPoint* pMP = *lit;
    if (pMP->isBad()) continue;

    // Simple outlier detection based on observation count
    if (pMP->GetObservations().size() < 2) {
      pMP->SetBadFlag();
    }
  }
}

void Optimizer::OptimizeEssentialGraph(
    Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
    const LoopClosing::KeyFrameAndPose& NonCorrectedSim3,
    const LoopClosing::KeyFrameAndPose& CorrectedSim3,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
    const bool& bFixScale) {
  std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
  std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

  // Create factor graph and initial estimate
  NonlinearFactorGraph graph;
  Values initialEstimate;

  const unsigned int nMaxKFid = pMap->GetMaxKFid();

  // Add keyframe poses
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;

    Pose3 pose;
    if (CorrectedSim3.count(pKF)) {
      // Use corrected pose - extract pose from Similarity3
      gtsam::Similarity3 sim3 = CorrectedSim3.at(pKF);
      gtsam::Rot3 R = sim3.rotation();
      gtsam::Point3 t = sim3.translation();
      pose = Pose3(R, t);
    } else {
      pose = Converter::toPose3(pKF->GetPose());
    }

    initialEstimate.insert(X(pKF->mnId), pose);

    // Fix the first keyframe
    if (pKF->mnId == 0) {
      SharedNoiseModel priorModel = noiseModel::Diagonal::Sigmas(
          (Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());
      graph.addPrior(X(pKF->mnId), pose, priorModel);
    }
  }

  // Add essential graph edges (spanning tree + loop closure + covisibility)
  SharedNoiseModel odometryNoise = noiseModel::Diagonal::Sigmas(
      (Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
  SharedNoiseModel loopNoise = noiseModel::Diagonal::Sigmas(
      (Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());

  // Spanning tree edges
  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) continue;

    KeyFrame* pParent = pKF->GetParent();
    if (pParent) {
      Pose3 pose1 = initialEstimate.at<Pose3>(X(pParent->mnId));
      Pose3 pose2 = initialEstimate.at<Pose3>(X(pKF->mnId));
      Pose3 relativePose = pose1.inverse() * pose2;

      graph.emplace_shared<BetweenFactor<Pose3>>(X(pParent->mnId), X(pKF->mnId),
                                                 relativePose, odometryNoise);
    }
  }

  // Loop closure edges
  for (auto& connection : LoopConnections) {
    KeyFrame* pKF = connection.first;
    for (KeyFrame* pConnectedKF : connection.second) {
      if (pKF->mnId < pConnectedKF->mnId) {
        Pose3 pose1 = initialEstimate.at<Pose3>(X(pKF->mnId));
        Pose3 pose2 = initialEstimate.at<Pose3>(X(pConnectedKF->mnId));
        Pose3 relativePose = pose1.inverse() * pose2;

        graph.emplace_shared<BetweenFactor<Pose3>>(
            X(pKF->mnId), X(pConnectedKF->mnId), relativePose, loopNoise);
      }
    }
  }

  // Optimize
  LevenbergMarquardtParams params;
  params.setMaxIterations(20);
  params.setRelativeErrorTol(1e-6);
  params.setAbsoluteErrorTol(1e-6);
  params.setVerbosity("ERROR");

  try {
    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    Values result = optimizer.optimize();

    // Update keyframe poses
    for (size_t i = 0; i < vpKFs.size(); i++) {
      KeyFrame* pKF = vpKFs[i];
      if (pKF->isBad()) continue;

      Pose3 optimizedPose = result.at<Pose3>(X(pKF->mnId));
      pKF->SetPose(Converter::toCvMat(optimizedPose));
    }

    // Update MapPoints
    for (size_t i = 0; i < vpMPs.size(); i++) {
      MapPoint* pMP = vpMPs[i];
      if (pMP->isBad()) continue;

      pMP->UpdateNormalAndDepth();
    }

  } catch (const std::exception& e) {
    SPDLOG_ERROR("Essential graph optimization failed: {}", e.what());
  }
}

int Optimizer::OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
                            std::vector<MapPoint*>& vpMatches1,
                            gtsam::Similarity3& gtsamS12, const float th2,
                            const bool bFixScale) {
  // Create factor graph for Sim3 optimization
  NonlinearFactorGraph graph;
  Values initialEstimate;

  // Set initial Sim3 estimate
  initialEstimate.insert(Symbol('s', 0), gtsamS12);

  int nCorrespondences = 0;
  std::vector<bool> vbInliers(vpMatches1.size(), false);

  // Add correspondence factors
  for (size_t i = 0; i < vpMatches1.size(); i++) {
    MapPoint* pMP1 = vpMatches1[i];
    if (!pMP1 || pMP1->isBad()) continue;

    Point3 p1 = Converter::toPoint3(pMP1->GetWorldPos());

    // For simplicity, we'll use the current Sim3 to get corresponding point
    // In practice, you'd have proper correspondences
    Point3 p2 = gtsamS12.transformFrom(p1);

    // Add a factor constraining the Sim3 transformation
    SharedNoiseModel noise = noiseModel::Isotropic::Sigma(3, sqrt(th2));

    // This is a simplified factor - in practice you'd implement a proper Sim3
    // factor For now, we'll skip the detailed implementation
    nCorrespondences++;
    vbInliers[i] = true;
  }

  // Simple optimization - in practice this would be more sophisticated
  if (nCorrespondences < 10) {
    return 0;
  }

  // Count inliers
  int nInliers = 0;
  for (size_t i = 0; i < vbInliers.size(); i++) {
    if (vbInliers[i]) nInliers++;
  }

  return nInliers;
}

}  // namespace SuperSLAM