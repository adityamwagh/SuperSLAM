#include <gtest/gtest.h>

#include <opencv4/opencv2/core.hpp>

#include "PlaceRecognizer.h"

using namespace superslam;

namespace {
// Deterministic unit-ish descriptor: a one-hot-ish vector with a small offset
// so two "places" are near-duplicates when seed matches and distinct when it
// doesn't.
cv::Mat desc(int dim, int seed, float jitter = 0.0f) {
  cv::Mat d = cv::Mat::zeros(1, dim, CV_32F);
  d.at<float>(0, seed % dim) = 1.0f;
  d.at<float>(0, (seed + 1) % dim) = 0.5f + jitter;
  return d;
}
} // namespace

TEST(CosineDescriptorIndex, RanksNearDuplicateAboveDistinct) {
  CosineDescriptorIndex idx;
  idx.add(0, desc(16, 3)); // the place we will revisit
  idx.add(1, desc(16, 9)); // an unrelated place
  // Query is a near-duplicate of kf0 (small jitter).
  auto res = idx.query(desc(16, 3, 0.01f),
                       /*excludeRecent=*/0,
                       /*topK=*/5,
                       /*minScore=*/0.0f);
  ASSERT_FALSE(res.empty());
  EXPECT_EQ(res.front().keyframe_id, 0u); // best match is the near-duplicate
  EXPECT_GT(res.front().score, 0.95f);    // high cosine similarity
  if (res.size() > 1)
    EXPECT_LT(res[1].score, res.front().score);
}

TEST(CosineDescriptorIndex, ExcludeRecentSkipsTemporalNeighbours) {
  CosineDescriptorIndex idx;
  for (int i = 0; i < 5; ++i)
    idx.add(i, desc(16, i));
  // Query matches kf4 exactly, but excludeRecent=2 removes kf3,kf4 from
  // candidates.
  auto res = idx.query(desc(16, 4),
                       /*excludeRecent=*/2,
                       /*topK=*/5,
                       /*minScore=*/0.0f);
  for (const auto& c : res)
    EXPECT_LT(c.keyframe_id, 3u); // only kf0..kf2 are eligible
}

TEST(CosineDescriptorIndex, TopKAndMinScoreGate) {
  CosineDescriptorIndex idx;
  for (int i = 0; i < 6; ++i)
    idx.add(i, desc(16, i));
  auto topk = idx.query(desc(16, 0), 0, /*topK=*/2, /*minScore=*/-1.0f);
  EXPECT_LE(topk.size(), 2u);
  // A high gate admits only the (near) exact match.
  auto gated = idx.query(desc(16, 0), 0, /*topK=*/10, /*minScore=*/0.99f);
  for (const auto& c : gated)
    EXPECT_GE(c.score, 0.99f);
}

TEST(CosineDescriptorIndex, EmptyOrAllExcludedReturnsNothing) {
  CosineDescriptorIndex idx;
  EXPECT_TRUE(idx.query(desc(16, 0), 0, 5, 0.0f).empty()); // empty db
  idx.add(0, desc(16, 0));
  EXPECT_TRUE(idx.query(desc(16, 0), /*excludeRecent=*/1, 5, 0.0f).empty()); // all excluded
}

TEST(TemporalConsistencyVoter, RequiresConsecutiveConsistentVotes) {
  TemporalConsistencyVoter voter(/*requiredVotes=*/3, /*idTolerance=*/2);
  LoopCandidate a{10, 0.9f}, b{11, 0.9f}, far{50, 0.9f};
  EXPECT_FALSE(voter.vote(&a)); // 1
  EXPECT_FALSE(voter.vote(&b)); // 2 (within tolerance of 10)
  EXPECT_TRUE(voter.vote(&a));  // 3 -> accept
}

TEST(TemporalConsistencyVoter, ResetsOnGapOrInconsistency) {
  TemporalConsistencyVoter voter(/*requiredVotes=*/2, /*idTolerance=*/1);
  LoopCandidate a{10, 0.9f}, far{99, 0.9f};
  EXPECT_FALSE(voter.vote(&a));
  EXPECT_FALSE(voter.vote(nullptr)); // null breaks the streak
  EXPECT_FALSE(voter.vote(&a));      // streak restarts at 1
  EXPECT_FALSE(voter.vote(&far));    // inconsistent jump -> streak back to 1
  EXPECT_TRUE(voter.vote(&far));     // far is now consistent with itself -> 2 -> accept
}
