#include "GlobalPoseGraph.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "Logging.h"

using gtsam::symbol_shorthand::X;

namespace superslam {

namespace {
bool finite(const gtsam::Pose3& p) {
  return p.matrix().allFinite();
}
} // namespace

GlobalPoseGraph::GlobalPoseGraph() {}

void GlobalPoseGraph::add_keyframe(size_t keyframe_id, const gtsam::Pose3& initial, bool is_first) {
  if (nodes_.count(keyframe_id))
    return;
  const gtsam::Pose3 init = finite(initial) ? initial : gtsam::Pose3();
  seeds_.insert(X(keyframe_id), init);
  nodes_.insert(keyframe_id);
  if (is_first)
    backbone_.addPrior(X(keyframe_id), init, gtsam::noiseModel::Isotropic::Sigma(6, 1e-4));
}

void GlobalPoseGraph::add_odometry(size_t from,
                                   size_t to,
                                   const gtsam::Pose3& rel,
                                   const gtsam::SharedNoiseModel& noise) {
  const gtsam::Pose3 r = finite(rel) ? rel : gtsam::Pose3();
  backbone_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(from), X(to), r, noise);
}

void GlobalPoseGraph::add_loop(size_t from,
                               size_t to,
                               const gtsam::Pose3& rel,
                               const gtsam::SharedNoiseModel& noise) {
  if (!finite(rel))
    return;
  loops_.push_back(gtsam::NonlinearFactor::shared_ptr(
      new gtsam::BetweenFactor<gtsam::Pose3>(X(from), X(to), rel, noise)));
}

void GlobalPoseGraph::sync_seeds() {
  for (const auto& key : estimate_.keys())
    seeds_.update(key, estimate_.at<gtsam::Pose3>(key));
}

namespace {
bool sane(const gtsam::Values& v) {
  for (const auto& key : v.keys()) {
    const gtsam::Pose3& p = v.at<gtsam::Pose3>(key);
    if (!p.matrix().allFinite() || p.translation().norm() > 1e6)
      return false; // LM diverged on an inconsistent loop edge (non-finite or
                    // exploded).
  }
  return true;
}
} // namespace

std::map<size_t, gtsam::Pose3> GlobalPoseGraph::optimize_and_get_all() {
  last_loop_rejected_ = false;
  while (true) {
    gtsam::NonlinearFactorGraph g = backbone_;
    for (const auto& l : loops_)
      g.push_back(l);
    gtsam::Values result;
    bool ok = false;
    try {
      result = gtsam::LevenbergMarquardtOptimizer(g, seeds_).optimize();
      ok = sane(result);
    } catch (const gtsam::IndeterminantLinearSystemException&) {
      ok = false;
    }
    if (ok) {
      estimate_ = result;
      sync_seeds();
      break;
    }
    if (loops_.empty()) {
      SLOG_ERROR("GlobalPoseGraph: pose graph unsolvable; keeping last estimate");
      break;
    }
    loops_.pop_back();
    last_loop_rejected_ = true;
  }
  std::map<size_t, gtsam::Pose3> out;
  for (size_t id : nodes_)
    out[id] = pose_of(id);
  return out;
}

gtsam::Pose3 GlobalPoseGraph::pose_of(size_t keyframe_id) const {
  if (estimate_.exists(X(keyframe_id)))
    return estimate_.at<gtsam::Pose3>(X(keyframe_id));
  return seeds_.at<gtsam::Pose3>(X(keyframe_id));
}

} // namespace superslam
