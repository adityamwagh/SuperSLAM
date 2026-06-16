#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>

#include <memory>
#include <vector>

#include "LoopCloser.h"

using namespace superslam;
using namespace gtsam;
using ::testing::_;
using ::testing::Return;

namespace {

// Matcher that returns identity correspondences i->i for the first `n`
// features.
class MockMatcher : public IFeatureMatcher {
public:
  explicit MockMatcher(int n) {
    for (int i = 0; i < n; ++i)
      result_.matches.emplace_back(i, i, 0.0f); // queryIdx, trainIdx, distance
  }
  MOCK_METHOD(MatchResult, doMatch, ());
  MatchResult match(const std::vector<cv::KeyPoint>&,
                    const cv::Mat&,
                    const std::vector<cv::KeyPoint>&,
                    const cv::Mat&) override {
    return result_;
  }
  MatchResult match(const std::vector<cv::KeyPoint>&,
                    const DeviceDescriptors&,
                    const std::vector<cv::KeyPoint>&,
                    const DeviceDescriptors&) override {
    return result_;
  }
  cv::Mat descriptors_to_host(const DeviceDescriptors&) override { return cv::Mat(); }

private:
  MatchResult result_;
};

// Identity place recognizer: descriptor == image; retrieval via the shared
// cosine index.
class StubRecognizer : public IPlaceRecognizer {
public:
  explicit StubRecognizer(float minScore) : minScore_(minScore) {}
  cv::Mat compute_global_descriptor(const cv::Mat& image) override { return image; }
  void add(size_t keyframe_id, const cv::Mat& d) override { idx_.add(keyframe_id, d); }
  std::vector<LoopCandidate> query(const cv::Mat& d, size_t exclude_recent, int top_k) override {
    return idx_.query(d, exclude_recent, top_k, minScore_);
  }

private:
  CosineDescriptorIndex idx_;
  float minScore_;
};

Cal3_S2Stereo::shared_ptr makeK() {
  return boost::make_shared<Cal3_S2Stereo>(500, 500, 0, 320, 240, 0.5);
}

// Build a keyframe record observing `lms` (camera-frame points) from camera
// pose `camInWorld` relative to the candidate frame; here we just project from
// the given stereo camera and tag every observation as stereo-valid.
KeyframeRecord make_keyframe(size_t id,
                             const StereoCamera& cam,
                             const std::vector<Point3>& landmarks_candidate_frame,
                             const cv::Mat& global_descriptor) {
  KeyframeRecord r;
  r.keyframe_id = id;
  r.global_descriptor = global_descriptor;
  for (const Point3& X : landmarks_candidate_frame) {
    r.stereo.push_back(cam.project(X));
    r.has_depth.push_back(1);
    r.keypoints_left.emplace_back(cv::KeyPoint(0.f, 0.f, 1.f));
  }
  return r;
}

std::vector<Point3> makeLandmarks() {
  std::vector<Point3> lms;
  for (int i = 0; i < 40; ++i) {
    const double x = -3 + 0.15 * i, y = -2 + 0.1 * i, z = 6 + 0.1 * (i % 7);
    lms.push_back(Point3(x, y, z));
  }
  return lms;
}
} // namespace

TEST(LoopCloser, VerifyRecoversKnownRelativePose) {
  auto K = makeK();
  const auto lms = makeLandmarks();                         // points in the candidate frame
  const Pose3 T_c_q(Rot3::Rz(0.1), Point3(0.3, -0.1, 0.2)); // query pose in candidate frame

  cv::Mat g = cv::Mat::ones(1, 8, CV_32F);
  KeyframeRecord cand = make_keyframe(0, StereoCamera(Pose3(), K), lms,
                                      g);                                  // candidate at origin
  KeyframeRecord query = make_keyframe(1, StereoCamera(T_c_q, K), lms, g); // query sees same lms

  MockMatcher matcher(static_cast<int>(lms.size()));
  LoopCloser lc(&matcher, K, std::make_unique<StubRecognizer>(0.5f));

  LoopResult r = lc.verify(query, cand);
  ASSERT_TRUE(r.accepted);
  EXPECT_EQ(r.matched_keyframe, 0u);
  EXPECT_GE(r.inliers, 30);
  EXPECT_TRUE(assert_equal(T_c_q, r.relative_pose, 1e-3));
  EXPECT_TRUE(r.noise != nullptr);
}

TEST(LoopCloser, DetectRunsRetrievalAndVerification) {
  auto K = makeK();
  const auto lms = makeLandmarks();
  const Pose3 T_c_q(Rot3::Rz(0.08), Point3(0.2, 0.0, 0.15));

  cv::Mat g = cv::Mat::ones(1, 8, CV_32F);
  KeyframeRecord cand = make_keyframe(0, StereoCamera(Pose3(), K), lms, g);
  KeyframeRecord query = make_keyframe(1, StereoCamera(T_c_q, K), lms, g);

  MockMatcher matcher(static_cast<int>(lms.size()));
  LoopParams params;
  params.required_votes = 1; // accept on first consistent vote for the test
  params.exclude_recent = 0;
  params.min_score = 0.5f;
  LoopCloser lc(&matcher, K, std::make_unique<StubRecognizer>(0.5f), params);

  lc.add_keyframe(cand);
  LoopResult r = lc.detect(query);
  ASSERT_TRUE(r.accepted);
  EXPECT_EQ(r.matched_keyframe, 0u);
  EXPECT_TRUE(assert_equal(T_c_q, r.relative_pose, 1e-3));
}

TEST(LoopCloser, RejectsTooFewInliers) {
  auto K = makeK();
  std::vector<Point3> few = {Point3(0, 0, 8), Point3(1, 1, 9)}; // < minInliers
  cv::Mat g = cv::Mat::ones(1, 8, CV_32F);
  KeyframeRecord cand = make_keyframe(0, StereoCamera(Pose3(), K), few, g);
  KeyframeRecord query =
      make_keyframe(1, StereoCamera(Pose3(Rot3(), Point3(0.1, 0, 0)), K), few, g);

  MockMatcher matcher(static_cast<int>(few.size()));
  LoopCloser lc(&matcher, K, std::make_unique<StubRecognizer>(0.5f));
  EXPECT_FALSE(lc.verify(query, cand).accepted);
}
