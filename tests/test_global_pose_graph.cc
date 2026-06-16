#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include <vector>

#include "GlobalPoseGraph.h"

using namespace gtsam;

namespace {
SharedNoiseModel odomNoise() {
  return noiseModel::Diagonal::Sigmas((Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
}
} // namespace

// An exact odometry chain (seeds perturbed) must recover the ground-truth
// poses.
TEST(GlobalPoseGraph, RecoversChainFromOdometry) {
  std::vector<Pose3> gt = {Pose3(),
                           Pose3(Rot3(), Point3(1, 0, 0)),
                           Pose3(Rot3(), Point3(2, 0, 0)),
                           Pose3(Rot3(), Point3(3, 0, 0))};
  superslam::GlobalPoseGraph g;
  g.add_keyframe(0, gt[0], /*is_first=*/true);
  for (size_t k = 1; k < gt.size(); ++k) {
    const Pose3 perturbed = gt[k] * Pose3(Rot3::Rz(0.05), Point3(0.2, -0.1, 0.0));
    g.add_keyframe(k, perturbed, /*is_first=*/false);
    g.add_odometry(k - 1, k, gt[k - 1].between(gt[k]), odomNoise());
  }
  auto poses = g.optimize_and_get_all();
  for (size_t k = 0; k < gt.size(); ++k)
    EXPECT_TRUE(assert_equal(gt[k], poses.at(k), 1e-3));
}

// A loop closure must pull a drifted trajectory back toward ground truth.
TEST(GlobalPoseGraph, LoopClosureCorrectsDrift) {
  // Ground-truth square loop (planar, rotating 90deg at each corner),
  // returning to the start. 8 keyframes around the perimeter.
  const int N = 8;
  std::vector<Pose3> gt;
  {
    Pose3 p; // origin
    for (int i = 0; i < N; ++i) {
      gt.push_back(p);
      // step forward 1 m in local x, turn left 45deg -> closes an octagon back
      // to start
      p = p * Pose3(Rot3::Rz(2 * M_PI / N), Point3(1, 0, 0));
    }
  }

  // Odometry with a small consistent yaw bias -> dead reckoning drifts.
  const Pose3 bias(Rot3::Rz(0.04), Point3(0, 0, 0));

  superslam::GlobalPoseGraph g;
  g.add_keyframe(0, gt[0], /*is_first=*/true);
  Pose3 deadReckon = gt[0];
  for (int k = 1; k < N; ++k) {
    const Pose3 odo = gt[k - 1].between(gt[k]) * bias; // biased measurement
    deadReckon = deadReckon * odo;                     // integrated seed (drifts)
    g.add_keyframe(k, deadReckon, /*is_first=*/false);
    g.add_odometry(k - 1, k, odo, odomNoise());
  }
  auto before = g.optimize_and_get_all();
  const double driftBefore = (before.at(N - 1).translation() - gt[N - 1].translation()).norm();
  EXPECT_GT(driftBefore, 0.05); // odometry-only actually drifts

  // Place recognition detects KF_{N-1} revisits KF0; the verified relative
  // pose is the ground-truth constraint. Looser noise than odometry but
  // robust.
  const Pose3 loopRel = gt[N - 1].between(gt[0]);
  g.add_loop(N - 1, 0, loopRel, odomNoise());
  auto after = g.optimize_and_get_all();
  const double driftAfter = (after.at(N - 1).translation() - gt[N - 1].translation()).norm();

  EXPECT_LT(driftAfter, driftBefore); // loop closure reduced drift
  EXPECT_LT(driftAfter, 0.5 * driftBefore);
}
