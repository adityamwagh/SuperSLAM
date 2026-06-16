#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>

#include "KeyframeDatabase.h"

using namespace superslam;

namespace {
KeyframeRecord makeRec(size_t id) {
  KeyframeRecord r;
  r.keyframe_id = id;
  r.timestamp = 0.1 * id;
  r.pose_at_insert = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(id, 0, 0));
  r.keypoints_left.emplace_back(cv::KeyPoint(1.f, 2.f, 1.f));
  r.descriptors_left = cv::Mat::ones(1, 4, CV_32F) * static_cast<float>(id);
  return r;
}
} // namespace

TEST(KeyframeDatabase, StoreAndLookupRoundTrip) {
  KeyframeDatabase db;
  db.add(makeRec(7));
  db.add(makeRec(42));

  EXPECT_EQ(db.size(), 2u);
  EXPECT_TRUE(db.has(7));
  EXPECT_TRUE(db.has(42));
  EXPECT_FALSE(db.has(0));

  const auto& r = db.get(42);
  EXPECT_EQ(r.keyframe_id, 42u);
  EXPECT_NEAR(r.timestamp, 4.2, 1e-9);
  EXPECT_NEAR(r.pose_at_insert.translation().x(), 42.0, 1e-9);
  EXPECT_FLOAT_EQ(r.descriptors_left.at<float>(0, 0), 42.0f);
}

TEST(KeyframeDatabase, PreservesInsertionOrder) {
  KeyframeDatabase db;
  for (size_t id : {5u, 3u, 9u})
    db.add(makeRec(id));
  const auto& recs = db.records();
  ASSERT_EQ(recs.size(), 3u);
  EXPECT_EQ(recs[0].keyframe_id, 5u);
  EXPECT_EQ(recs[1].keyframe_id, 3u);
  EXPECT_EQ(recs[2].keyframe_id, 9u);
}

TEST(KeyframeDatabase, UnknownIdThrows) {
  KeyframeDatabase db;
  db.add(makeRec(1));
  EXPECT_THROW(db.get(999), std::out_of_range);
}
