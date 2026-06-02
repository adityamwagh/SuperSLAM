#include "FrameTracker.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <cmath>

#include "PoseOptimizationFactors.h" // PoseOnlyStereoFactor, stereo_diag_noise

using gtsam::symbol_shorthand::X;

namespace superslam {

gtsam::Pose3 FrameTracker::track(const gtsam::Pose3& initial_guess,
                                 const std::vector<PointObs>& matches) {
  gtsam::NonlinearFactorGraph graph;
  const double mbf = K_->fx() * K_->baseline();
  for (const PointObs& m : matches) {
    const double disparity = m.meas.uL() - m.meas.uR();
    auto noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(std::sqrt(7.815)),
        stereo_diag_noise(10.0, disparity, mbf));
    graph.emplace_shared<PoseOnlyStereoFactor>(m.meas, noise, X(0), m.Xw, K_);
  }
  gtsam::Values values;
  values.insert(X(0), initial_guess);
  const gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, values).optimize();
  return result.at<gtsam::Pose3>(X(0));
}

} // namespace superslam
