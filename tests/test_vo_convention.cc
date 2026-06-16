#include <algorithm>

#include <gtest/gtest.h>

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>

#include "InferenceInterfaces.h"
#include "StereoFrame.h"
#include "VoEstimator.h"

using namespace superslam;
using namespace gtsam;

namespace {
// Identity index matcher: left i <-> right i up to min(N, M).
struct IdMatcher : IFeatureMatcher {
  static MatchResult identity(size_t na, size_t nb) {
    MatchResult r;
    const int k = static_cast<int>(std::min(na, nb));
    for (int i = 0; i < k; ++i)
      r.matches.push_back(cv::DMatch(i, i, 0.f));
    return r;
  }
  MatchResult match(const std::vector<cv::KeyPoint>& a,
                    const cv::Mat&,
                    const std::vector<cv::KeyPoint>& b,
                    const cv::Mat&) override {
    return identity(a.size(), b.size());
  }
  MatchResult match(const std::vector<cv::KeyPoint>& a,
                    const DeviceDescriptors&,
                    const std::vector<cv::KeyPoint>& b,
                    const DeviceDescriptors&) override {
    return identity(a.size(), b.size());
  }
  cv::Mat descriptors_to_host(const DeviceDescriptors&) override { return cv::Mat(); }
};
// StereoFrame of `lms` as seen from camera at Twc `pose`.
StereoFrame
make(const std::vector<Point3>& lms, const Pose3& pose, const Cal3_S2Stereo::shared_ptr& K) {
  StereoFrame f;
  for (const auto& X : lms) {
    const StereoPoint2 z = StereoCamera(pose, K).project(X);
    f.keypoints_left.push_back(cv::KeyPoint(z.uL(), z.v(), 1));
    f.stereo.push_back(z);
    f.has_depth.push_back(1);
  }
  // descriptors_left left default-empty; the index matcher ignores descriptor
  // content.
  return f;
}
} // namespace

TEST(VoEstimator, TrajectoryAdvancesUnderForwardMotion) {
  auto K = boost::make_shared<Cal3_S2Stereo>(500, 500, 0, 320, 240, 0.5);
  std::vector<Point3> lms = {{0, 0, 10},
                             {2, 1, 12},
                             {-1, -1, 9},
                             {3, 2, 14},
                             {1, -2, 11},
                             {-2, 1, 13},
                             {0, 2, 10},
                             {2, -1, 12},
                             {-3, 0, 11},
                             {1, 3, 13},
                             {3, -2, 10},
                             {-1, 2, 12},
                             {2, 3, 14},
                             {-2, -2, 9},
                             {0, -3, 11},
                             {3, 0, 13}};
  IdMatcher matcher;
  VoEstimator est(&matcher, *K);

  double prevX = -1.0;
  for (int k = 0; k < 6; ++k) {
    const Pose3 truth(Rot3(), Point3(2.0 * k, 0, 0)); // moving +2 m/step along +x
    StereoFrame f = make(lms, truth, K);
    const Pose3 estPose = est.track(f);
    if (k > 0)
      EXPECT_GT(estPose.translation().x(),
                prevX + 0.5); // MUST advance, not bounce
    prevX = estPose.translation().x();
  }
}

TEST(VoEstimator, AnchorsFirstFrameAtOrigin) {
  auto K = boost::make_shared<Cal3_S2Stereo>(500, 500, 0, 320, 240, 0.5);
  std::vector<Point3> lms = {{0, 0, 10}, {2, 1, 12}, {-1, -1, 9}, {3, 2, 14}};
  IdMatcher matcher;
  VoEstimator est(&matcher, *K);

  StereoFrame f = make(lms, Pose3(), K);
  const Pose3 first = est.track(f);

  EXPECT_TRUE(first.equals(Pose3(),
                           1e-9)); // first stereo frame defines the world frame
  EXPECT_TRUE(f.pose.equals(first,
                            1e-9)); // track() also writes the pose back into the frame
}

TEST(VoEstimator, RecoversMetricTranslationFromStereo) {
  auto K = boost::make_shared<Cal3_S2Stereo>(500, 500, 0, 320, 240, 0.5);
  std::vector<Point3> lms = {{0, 0, 10},
                             {2, 1, 12},
                             {-1, -1, 9},
                             {3, 2, 14},
                             {1, -2, 11},
                             {-2, 1, 13},
                             {0, 2, 10},
                             {2, -1, 12},
                             {-3, 0, 11},
                             {1, 3, 13},
                             {3, -2, 10},
                             {-1, 2, 12},
                             {2, 3, 14},
                             {-2, -2, 9},
                             {0, -3, 11},
                             {3, 0, 13}};
  IdMatcher matcher;
  VoEstimator est(&matcher, *K);

  StereoFrame f0 = make(lms, Pose3(), K);
  est.track(f0);

  const Pose3 truth(Rot3(), Point3(1.7, 0, 0)); // known forward step, metres
  StereoFrame f1 = make(lms, truth, K);
  const Pose3 est1 = est.track(f1);

  // Stereo baseline fixes absolute scale: recovered step matches truth
  // metrically.
  EXPECT_NEAR(est1.translation().x(), 1.7, 0.1);
  EXPECT_LT(est1.translation().norm(), 1.9);
}
