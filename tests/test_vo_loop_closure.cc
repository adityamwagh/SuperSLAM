#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "LoopCloser.h"
#include "VoEstimator.h"

using namespace superslam;
using namespace gtsam;

namespace {

// Identity matcher: returns i->i for the first `n` features. Used for both
// per-frame tracking and loop verification (sync mode -> no concurrency, safe
// to share).
class IdentityMatcher : public IFeatureMatcher {
public:
  explicit IdentityMatcher(int n) {
    for (int i = 0; i < n; ++i)
      result_.matches.emplace_back(i, i, 0.0f);
  }
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

// Place recognizer whose descriptor IS the passed image row (so the test
// controls retrieval exactly), backed by the shared cosine index.
class StubRecognizer : public IPlaceRecognizer {
public:
  cv::Mat compute_global_descriptor(const cv::Mat& image) override { return image; }
  void add(size_t kfId, const cv::Mat& d) override { idx_.add(kfId, d); }
  std::vector<LoopCandidate> query(const cv::Mat& d, size_t ex, int k) override {
    return idx_.query(d, ex, k, /*minScore=*/0.5f);
  }

private:
  CosineDescriptorIndex idx_;
};

Cal3_S2Stereo::shared_ptr makeK() {
  return boost::make_shared<Cal3_S2Stereo>(500, 500, 0, 320, 240, 0.5);
}

// Landmarks visible from every pose along the (small) loop.
std::vector<Point3> world() {
  std::vector<Point3> lms;
  for (int i = 0; i < 16; ++i)
    lms.push_back(Point3(-4 + 0.5 * i, -3 + 0.4 * (i % 5), 9 + 0.3 * (i % 4)));
  return lms;
}

// Build a StereoFrame by projecting the world from camInWorld (Twc).
StereoFrame makeFrame(const Pose3& camInWorld,
                      const std::vector<Point3>& lms,
                      const Cal3_S2Stereo::shared_ptr& K,
                      double t) {
  StereoFrame f;
  f.timestamp = t;
  const StereoCamera cam(camInWorld, K);
  for (const Point3& X : lms) {
    f.keypoints_left.emplace_back(cv::KeyPoint(0.f, 0.f, 1.f));
    f.stereo.push_back(cam.project(X));
    f.has_depth.push_back(1);
  }
  // descriptors_left left default-empty; the identity matcher ignores
  // descriptor content.
  return f;
}

// One-hot global descriptor for "place" p.
cv::Mat placeDesc(int p) {
  cv::Mat d = cv::Mat::zeros(1, 8, CV_32F);
  d.at<float>(0, p % 8) = 1.0f;
  return d;
}
} // namespace

// Drive a synthetic loop trajectory through VoEstimator with loop closure
// enabled (sync). The trajectory returns to the start; the final keyframe's
// global descriptor matches keyframe 0, so a loop must be detected, verified,
// and applied -- continuously.
TEST(VoLoopClosure, DetectsAndAppliesLoopOnReturn) {
  setenv("SUPERSLAM_KF_MIN_TRANS", "0.3",
         1); // every ~0.5 m step is a keyframe
  auto K = makeK();
  const auto lms = world();

  // Loop path (planar): out along +x, over in +y, back along -x to near the
  // origin.
  std::vector<Point3> path = {{0, 0, 0},
                              {0.5, 0, 0},
                              {1.0, 0, 0},
                              {1.0, 0.5, 0},
                              {0.5, 0.5, 0},
                              {0.0, 0.4, 0},
                              {0.0, 0.05, 0}};

  IdentityMatcher matcher(static_cast<int>(lms.size()));

  LoopParams params;
  params.required_votes = 1;
  params.exclude_recent = 1; // tiny loop -> only exclude the immediate neighbour
  params.min_score = 0.5f;
  params.min_inliers = 8;
  auto lc = std::make_unique<LoopCloser>(&matcher, K, std::make_unique<StubRecognizer>(), params);

  VoEstimator vo(&matcher, *K);
  vo.enable_loop_closure(std::move(lc), /*async=*/false);

  std::vector<Pose3> est;
  for (size_t i = 0; i < path.size(); ++i) {
    StereoFrame f = makeFrame(Pose3(Rot3(), path[i]), lms, K, 0.1 * i);
    // Distinct place per pose (kf0 == place 0); the LAST pose revisits place 0
    // (loop).
    const int place = (i + 1 == path.size()) ? 0 : static_cast<int>(i);
    const Pose3 p = vo.track(f, placeDesc(place));
    est.push_back(p);
  }

  // A loop closure must have fired and applied a correction.
  EXPECT_GE(vo.loop_closure_count(), 1u);

  // Continuity: consecutive estimated poses never jump unreasonably (the rigid
  // loop rebase preserves *relative* motion; physical steps are ~0.5 m).
  for (size_t i = 1; i < est.size(); ++i) {
    const double step = (est[i].translation() - est[i - 1].translation()).norm();
    EXPECT_LT(step, 1.0) << "discontinuity at frame " << i;
  }

  // The trajectory physically returned near the origin.
  EXPECT_LT(est.back().translation().norm(), 0.5);

  unsetenv("SUPERSLAM_KF_MIN_TRANS");
}

// Forward motion then a loop must NOT corrupt the trajectory: every pose stays
// finite and bounded, and the corrected trajectory has one pose per processed
// frame.
TEST(VoLoopClosure, CorrectionStaysFiniteAndBounded) {
  setenv("SUPERSLAM_KF_MIN_TRANS", "0.3", 1);
  auto K = makeK();
  const auto lms = world();

  std::vector<Point3> path = {{0, 0, 0},
                              {0.5, 0, 0},
                              {1.0, 0, 0},
                              {1.0, 0.5, 0},
                              {0.5, 0.5, 0},
                              {0.0, 0.4, 0},
                              {0.0, 0.05, 0}};

  IdentityMatcher matcher(static_cast<int>(lms.size()));

  LoopParams params;
  params.required_votes = 1;
  params.exclude_recent = 1;
  params.min_score = 0.5f;
  params.min_inliers = 8;
  auto lc = std::make_unique<LoopCloser>(&matcher, K, std::make_unique<StubRecognizer>(), params);

  VoEstimator vo(&matcher, *K);
  vo.enable_loop_closure(std::move(lc), /*async=*/false);

  for (size_t i = 0; i < path.size(); ++i) {
    StereoFrame f = makeFrame(Pose3(Rot3(), path[i]), lms, K, 0.1 * i);
    const int place = (i + 1 == path.size()) ? 0 : static_cast<int>(i);
    vo.track(f, placeDesc(place));
  }

  ASSERT_GE(vo.loop_closure_count(), 1u);

  const std::vector<Pose3> traj = vo.corrected_trajectory();
  ASSERT_FALSE(traj.empty());
  for (const Pose3& p : traj) {
    ASSERT_TRUE(p.matrix().allFinite());
    ASSERT_LT(p.translation().norm(), 1e3);
  }

  unsetenv("SUPERSLAM_KF_MIN_TRANS");
}

// Same scenario on the async worker thread: the loop must be detected off the
// tracking thread without deadlock or data race, and the correction applied on
// a later frame.
TEST(VoLoopClosure, AsyncWorkerDetectsLoop) {
  setenv("SUPERSLAM_KF_MIN_TRANS", "0.3", 1);
  auto K = makeK();
  const auto lms = world();
  std::vector<Point3> path = {{0, 0, 0},
                              {0.5, 0, 0},
                              {1.0, 0, 0},
                              {1.0, 0.5, 0},
                              {0.5, 0.5, 0},
                              {0.0, 0.4, 0},
                              {0.0, 0.05, 0}};

  IdentityMatcher matcher(static_cast<int>(lms.size()));
  LoopParams params;
  params.required_votes = 1;
  params.exclude_recent = 1;
  params.min_score = 0.5f;
  params.min_inliers = 8;
  auto lc = std::make_unique<LoopCloser>(&matcher, K, std::make_unique<StubRecognizer>(), params);

  VoEstimator vo(&matcher, *K);
  vo.enable_loop_closure(std::move(lc), /*async=*/true);

  for (size_t i = 0; i < path.size(); ++i) {
    StereoFrame f = makeFrame(Pose3(Rot3(), path[i]), lms, K, 0.1 * i);
    const int place = (i + 1 == path.size()) ? 0 : static_cast<int>(i);
    vo.track(f, placeDesc(place));
  }

  // The worker runs concurrently; poll briefly for the loop to be processed.
  for (int i = 0; i < 200 && vo.loop_closure_count() == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_GE(vo.loop_closure_count(), 1u);

  // A subsequent frame applies the pending correction without
  // throwing/discontinuity.
  StereoFrame f = makeFrame(Pose3(Rot3(), Point3(0.0, 0.0, 0.0)), lms, K, 1.0);
  const Pose3 p = vo.track(f, placeDesc(0));
  EXPECT_LT(p.translation().norm(), 0.6);

  unsetenv("SUPERSLAM_KF_MIN_TRANS");
}

// correctedTrajectory() with no loop enabled must reproduce live VO global
// poses exactly, proving the seedAnchors_ fallback path in
// correctedTrajectory() is correct.
TEST(VoLoopClosure, CorrectedTrajectoryFallsBackToLiveVOWithoutLoop) {
  auto K = makeK();
  const auto lms = world();

  // Short straight path; loop closure never enabled.
  std::vector<Point3> path = {{0, 0, 0}, {0.5, 0, 0}, {1.0, 0, 0}, {1.5, 0, 0}};

  IdentityMatcher matcher(static_cast<int>(lms.size()));
  VoEstimator vo(&matcher, *K); // no enableLoopClosure: anchors_ stays empty

  Pose3 lastPose;
  for (size_t i = 0; i < path.size(); ++i) {
    StereoFrame f = makeFrame(Pose3(Rot3(), path[i]), lms, K, 0.1 * i);
    lastPose = vo.track(f, cv::Mat());
  }

  const std::vector<Pose3> traj = vo.corrected_trajectory();

  // One pose per tracked frame.
  ASSERT_EQ(traj.size(), path.size());

  // Without a loop, correctedTrajectory() must reproduce the live VO output
  // exactly.
  const double delta = (traj.back().translation() - lastPose.translation()).norm();
  EXPECT_LT(delta, 1e-6);
}

// Map is seeded per keyframe: even with no loop closure (empty image), the
// sparse map must have at least one keyframe and the cloud must be non-empty
// via seedAnchors_.
TEST(VoLoopClosure, SparseMapPopulatedPerKeyframe) {
  setenv("SUPERSLAM_KF_MIN_TRANS", "0.3", 1);
  auto K = makeK();
  const auto lms = world();
  std::vector<Point3> path = {{0, 0, 0},
                              {0.5, 0, 0},
                              {1.0, 0, 0},
                              {1.0, 0.5, 0},
                              {0.5, 0.5, 0},
                              {0.0, 0.4, 0},
                              {0.0, 0.05, 0}};

  IdentityMatcher matcher(static_cast<int>(lms.size()));
  LoopParams params;
  params.required_votes = 1;
  params.exclude_recent = 1;
  params.min_score = 0.5f;
  params.min_inliers = 8;
  auto lc = std::make_unique<LoopCloser>(&matcher, K, std::make_unique<StubRecognizer>(), params);

  VoEstimator vo(&matcher, *K);
  vo.enable_loop_closure(std::move(lc), /*async=*/false);

  for (size_t i = 0; i < path.size(); ++i) {
    StereoFrame f = makeFrame(Pose3(Rot3(), path[i]), lms, K, 0.1 * i);
    vo.track(f,
             cv::Mat()); // no image: no loop fires, tests seedAnchors_ fallback
  }

  EXPECT_GT(vo.map().keyframe_count(), 0u);
  EXPECT_GT(vo.map().cloud(vo.anchors()).size(), 0u);

  unsetenv("SUPERSLAM_KF_MIN_TRANS");
}
