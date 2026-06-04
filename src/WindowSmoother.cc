#include "WindowSmoother.h"

#include "Profiling.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include <cstdlib>
#include <exception>

#include "Logging.h"

using gtsam::symbol_shorthand::X;

namespace superslam {

WindowSmoother::WindowSmoother(const gtsam::Cal3_S2Stereo::shared_ptr& K, size_t window_size)
    : K_(K), window_size_(window_size) {}

void WindowSmoother::add_keyframe(size_t keyframe_id,
                                  const gtsam::Pose3& initial_pose,
                                  const std::vector<StereoObs>& obs) {
  poses_[keyframe_id] = initial_pose;
  obs_by_kf_[keyframe_id] = obs;
  window_.push_back(keyframe_id);
  while (window_.size() > window_size_) { // Fixed-lag: drop the oldest.
    const size_t old = window_.front();
    window_.pop_front();
    poses_.erase(old);
    obs_by_kf_.erase(old);
  }
}

void WindowSmoother::optimize() {
  if (window_.size() < 2)
    return; // Need parallax.

  // Per-stage timing under SUPERSLAM_PROFILE: rebuild versus solve.
  auto prof_t = std::chrono::steady_clock::now();
  auto prof_mark = [&prof_t](const char* label) {
    if (superslam::Profiler::enabled())
      superslam::Profiler::instance().add(
          label,
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prof_t)
              .count());
    prof_t = std::chrono::steady_clock::now();
  };

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  for (size_t keyframe_id : window_)
    values.insert(X(keyframe_id), poses_.at(keyframe_id));

  // Gauge anchor: pin the oldest window keyframe with a strong prior.
  const size_t anchor = window_.front();
  graph.addPrior(X(anchor), poses_.at(anchor), gtsam::noiseModel::Isotropic::Sigma(6, 1e-4));

  // Smart stereo factors require an ISOTROPIC measurement noise (one pixel
  // sigma for uL/uR/v). Scale is pinned by multi-view parallax across the
  // window, not by per-observation disparity weighting, which lives in the
  // per-frame tracker's diagonal stereo_diag_noise.
  const char* sEnv = std::getenv("SUPERSLAM_SMART_SIGMA_PX");
  const double sigmaPx = sEnv ? std::atof(sEnv) : 1.0;
  const auto measNoise = gtsam::noiseModel::Isotropic::Sigma(3, sigmaPx);

  // One smart stereo factor per landmark seen in at least 2 window keyframes.
  std::map<size_t, gtsam::SmartStereoProjectionPoseFactor::shared_ptr> smart;
  gtsam::SmartStereoProjectionParams params;
  params.setDegeneracyMode(gtsam::DegeneracyMode::ZERO_ON_DEGENERACY);
  // Smart-factor outlier rejection: drop observations whose reprojection error
  // exceeds the threshold.
  params.setDynamicOutlierRejectionThreshold(3.0);
  for (size_t keyframe_id : window_) {
    for (const StereoObs& o : obs_by_kf_.at(keyframe_id)) {
      auto& f = smart[o.landmark_id];
      if (!f)
        f = boost::make_shared<gtsam::SmartStereoProjectionPoseFactor>(measNoise, params);
      f->add(o.meas, X(keyframe_id), K_);
    }
  }
  for (auto& kv : smart)
    if (kv.second->keys().size() >= 2)
      graph.add(kv.second);
  prof_mark("ws_rebuild"); // Build Values and smart stereo factors.

  // Batch LM over the window. Degenerate windows can make the linear solve
  // ill-posed; keep the seeded poses on failure. Iteration cap and early-exit
  // tolerances; SUPERSLAM_WS_MAX_ITERS overrides.
  gtsam::LevenbergMarquardtParams lm_params;
  const char* iterEnv = std::getenv("SUPERSLAM_WS_MAX_ITERS");
  lm_params.maxIterations = iterEnv ? std::atoi(iterEnv) : 4;
  lm_params.relativeErrorTol = 1e-3;
  lm_params.absoluteErrorTol = 1e-3;
  lm_params.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
  lm_params.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
  try {
    const gtsam::Values result =
        gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();
    for (size_t keyframe_id : window_) {
      if (!result.exists(X(keyframe_id)))
        continue;
      const gtsam::Pose3& p = result.at<gtsam::Pose3>(X(keyframe_id));
      if (!p.matrix().allFinite() || p.translation().norm() > 1e6)
        return; // LM diverged (non-finite or exploded); keep all previous poses.
    }
    for (size_t keyframe_id : window_)
      if (result.exists(X(keyframe_id)))
        poses_[keyframe_id] = result.at<gtsam::Pose3>(X(keyframe_id));
  } catch (const std::exception& e) {
    SLOG_WARN("WindowSmoother: window optimize failed ({}); keeping previous poses",
              std::string(e.what()).substr(0, 64));
  }
  prof_mark("ws_solve"); // LM optimize and result write-back.
}

gtsam::Pose3 WindowSmoother::pose_of(size_t keyframe_id) const {
  return poses_.at(keyframe_id);
}
size_t WindowSmoother::window_count() const {
  return window_.size();
}
bool WindowSmoother::in_window(size_t keyframe_id) const {
  return poses_.count(keyframe_id) != 0;
}

} // namespace superslam
