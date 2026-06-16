#include <gtest/gtest.h>

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>
#include <vector>

#include "FrameTracker.h"

using namespace gtsam;

TEST(FrameTracker, RecoversKnownPoseFromMatchedPoints) {
  auto K = boost::make_shared<Cal3_S2Stereo>(500, 500, 0, 320, 240, 0.5);
  const Pose3 truth(Rot3::RzRyRx(0.05, -0.1, 0.03), Point3(0.4, -0.2, 0.1));
  std::vector<Point3> lms = {{0, 0, 8}, {2, 1, 10}, {-1, -1, 7}, {3, 2, 12}, {1, -2, 9}};

  std::vector<superslam::PointObs> matches;
  for (const Point3& Xw : lms)
    matches.push_back({Xw, StereoCamera(truth, K).project(Xw)});

  superslam::FrameTracker tracker(K);
  const Pose3 est = tracker.track(Pose3(), matches); // seed from identity
  EXPECT_TRUE(assert_equal(truth, est, 1e-3));
}
