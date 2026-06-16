#include <gtest/gtest.h>

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>

#include "StereoFrame.h"

using namespace gtsam;

TEST(StereoFrame, BackprojectReturnsMetricWorldPoint) {
  auto K = boost::make_shared<Cal3_S2Stereo>(500, 500, 0, 320, 240, 0.5);
  const Pose3 camInWorld(Rot3::Rz(0.1), Point3(2, -1, 0.5)); // Twc
  const Point3 worldPt(3, 0.5, 9);
  const StereoPoint2 z = StereoCamera(camInWorld, K).project(worldPt);

  superslam::StereoFrame f;
  f.pose = camInWorld;
  f.stereo = {z};
  f.has_depth = {1};
  EXPECT_TRUE(assert_equal(worldPt, f.backproject(0, *K), 1e-4));
}
