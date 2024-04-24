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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Frame.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"
#include "thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace SuperSLAM {

class LoopClosing;

class Optimizer {
 public:
  void static BundleAdjustment(const std::vector<KeyFrame*>& vpKF,
                               const std::vector<MapPoint*>& vpMP,
                               int nIterations = 5, bool* pbStopFlag = NULL,
                               const unsigned long nLoopKF = 0,
                               const bool bRobust = true);
  void static GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5,
                                     bool* pbStopFlag = NULL,
                                     const unsigned long nLoopKF = 0,
                                     const bool bRobust = true);
  void static LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap);
  int static PoseOptimization(Frame* pFrame);

  // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise
  // (mono)
  void static OptimizeEssentialGraph(
      Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
      const LoopClosing::KeyFrameAndPose& NonCorrectedSim3,
      const LoopClosing::KeyFrameAndPose& CorrectedSim3,
      const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
      const bool& bFixScale);

  // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
  static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
                          std::vector<MapPoint*>& vpMatches1, g2o::Sim3& g2oS12,
                          const float th2, const bool bFixScale);
};

}  // namespace SuperSLAM

#endif  // OPTIMIZER_H
