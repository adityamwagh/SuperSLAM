#include <gtest/gtest.h>

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>
#include <vector>

#include "WindowSmoother.h"

using namespace gtsam;

TEST(WindowSmoother, RecoversKnownPosesAndMetricScale) {
  auto K = boost::make_shared<Cal3_S2Stereo>(500, 500, 0, 320, 240,
                                             0.5); // fx,fy,s,cx,cy,baseline
  // Ground-truth: camera translates +1 m along x at each of 4 keyframes.
  std::vector<Pose3> gt = {Pose3(),
                           Pose3(Rot3(), Point3(1, 0, 0)),
                           Pose3(Rot3(), Point3(2, 0, 0)),
                           Pose3(Rot3(), Point3(3, 0, 0))};
  std::vector<Point3> lms =
      {{0, 0, 8}, {2, 1, 10}, {-1, -1, 7}, {3, 2, 12}, {1, -2, 9}, {-2, 1, 11}};

  superslam::WindowSmoother sm(K, /*window_size=*/4);
  for (size_t k = 0; k < gt.size(); ++k) {
    std::vector<superslam::StereoObs> obs;
    for (size_t l = 0; l < lms.size(); ++l) {
      const StereoPoint2 z = StereoCamera(gt[k], K).project(lms[l]);
      obs.push_back({l, z});
    }
    // KF0 is the gauge anchor (= origin, exact); later keyframes are seeded
    // with a perturbed guess the smoother must correct.
    const Pose3 guess = (k == 0) ? gt[0] : gt[k] * Pose3(Rot3::Rz(0.02), Point3(0.1, -0.05, 0.08));
    sm.add_keyframe(k, guess, obs);
  }
  sm.optimize();

  ASSERT_EQ(sm.window_count(), 4);
  // Metric scale: distance KF0->KF3 must be ~3 m (not drifted).
  const double d = (sm.pose_of(3).translation() - sm.pose_of(0).translation()).norm();
  EXPECT_LT(std::abs(d - 3.0), 0.05); // scale pinned
  for (size_t k = 0; k < gt.size(); ++k)
    EXPECT_TRUE(assert_equal(gt[k], sm.pose_of(k), 0.05)); // poses recovered
}
