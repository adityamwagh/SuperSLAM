#include <gtest/gtest.h>

#include <gtsam/geometry/Cal3_S2Stereo.h>

#include "InferenceInterfaces.h"
#include "StereoFrontEnd.h"

using namespace superslam;

namespace {
// Extractor that alternates left/right keypoints on successive calls. The
// front-end calls extract() twice per frame (left, then right); the right set
// is shifted by `disparity` px so we can assert the computed uR.
struct AlternatingExtractor : IFeatureExtractor {
  float disparity;
  int call = 0;
  explicit AlternatingExtractor(float d) : disparity(d) {}
  Features extract(const cv::Mat&) override {
    Features f; // descriptors left default-empty; the index matcher ignores them
    const float dx = (call++ % 2 == 0) ? 0.f : disparity; // left first, then right
    f.keypoints = {cv::KeyPoint(100.f - dx, 50.f, 1), cv::KeyPoint(200.f - dx, 80.f, 1)};
    return f;
  }
};
// Identity index matcher: left i <-> right i.
struct IdMatcher : IFeatureMatcher {
  static MatchResult identity(size_t na, size_t nb) {
    MatchResult r;
    for (int i = 0; i < static_cast<int>(std::min(na, nb)); ++i)
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
} // namespace

TEST(StereoFrontEnd, FillsStereoMeasurementsWithCorrectDisparity) {
  AlternatingExtractor ext(/*disparity=*/10.f);
  IdMatcher matcher;
  auto K = gtsam::Cal3_S2Stereo(500, 500, 0, 320, 240, 0.5);
  StereoFrontEnd fe(&ext, &matcher, K, /*minDisparity=*/1.0f);

  StereoFrame f = fe.process(cv::Mat::zeros(480, 640, CV_8U), cv::Mat::zeros(480, 640, CV_8U), 1.0);

  ASSERT_EQ(f.keypoints_left.size(), 2);
  EXPECT_EQ(f.has_depth[0], 1);
  EXPECT_NEAR(f.stereo[0].uL(), 100.0, 1e-6);
  EXPECT_NEAR(f.stereo[0].uR(), 90.0, 1e-6); // disparity 10
  EXPECT_NEAR(f.stereo[0].v(), 50.0, 1e-6);
}

TEST(StereoFrontEnd, MarksBelowFloorDisparityAsNoDepth) {
  AlternatingExtractor ext(
      /*disparity=*/0.f); // left == right -> zero disparity
  IdMatcher matcher;
  auto K = gtsam::Cal3_S2Stereo(500, 500, 0, 320, 240, 0.5);
  StereoFrontEnd fe(&ext, &matcher, K, /*minDisparity=*/1.0f);

  StereoFrame f = fe.process(cv::Mat::zeros(480, 640, CV_8U), cv::Mat::zeros(480, 640, CV_8U), 1.0);
  EXPECT_EQ(f.has_depth[0], 0); // zero disparity rejected
}
