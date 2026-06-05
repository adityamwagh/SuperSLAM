#pragma once
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <cstddef>
#include <map>
#include <set>
#include <vector>

namespace superslam {

// Tier-2 backend: a batch pose graph over keyframe poses. Nodes are 6-DOF
// Pose3 keyframes (symbol X(keyframe_id)); edges are BetweenFactor<Pose3>
// odometry constraints from the active window plus loop-closure constraints.
// optimize_and_get_all() runs a batch Levenberg-Marquardt solve. Pose keys
// share the X(keyframe_id) symbol space with WindowSmoother.
class GlobalPoseGraph {
public:
  GlobalPoseGraph();

  // Insert a keyframe node with its initial pose estimate. Anchor the first
  // keyframe with a strong prior to fix the global gauge.
  void add_keyframe(size_t keyframe_id, const gtsam::Pose3& initial, bool is_first);

  // Add an odometry edge between consecutive keyframes, sourced from the
  // jointly-optimized window (rel = pose_of(from).between(pose_of(to))).
  void add_odometry(size_t from,
                    size_t to,
                    const gtsam::Pose3& rel,
                    const gtsam::SharedNoiseModel& noise);

  // Add a loop-closure edge between a revisited keyframe and the current one.
  // Use a robust (Huber) noise model.
  void
  add_loop(size_t from, size_t to, const gtsam::Pose3& rel, const gtsam::SharedNoiseModel& noise);

  // Run a batch Levenberg-Marquardt solve over the pose graph and return all
  // keyframe poses. Call after a batch of add*() calls (e.g. once per
  // keyframe, and after a loop edge).
  std::map<size_t, gtsam::Pose3> optimize_and_get_all();

  // Return the current best estimate for one keyframe (from the last optimize).
  gtsam::Pose3 pose_of(size_t keyframe_id) const;

  size_t size() const { return nodes_.size(); }
  bool has(size_t keyframe_id) const { return nodes_.count(keyframe_id) != 0; }

  bool last_loop_rejected() const { return last_loop_rejected_; }

private:
  void sync_seeds();

  gtsam::NonlinearFactorGraph backbone_;
  std::vector<gtsam::NonlinearFactor::shared_ptr> loops_;
  gtsam::Values seeds_;
  gtsam::Values estimate_;
  std::set<size_t> nodes_;
  bool last_loop_rejected_ = false;
};

} // namespace superslam
