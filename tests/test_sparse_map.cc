#include "SparseMap.h"
#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>

using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;

TEST(SparseMap, TransformsCamPointsByAnchor) {
  superslam::SparseMap m;
  m.add_keyframe(0, {Point3(0, 0, 1), Point3(1, 0, 2)});
  m.add_keyframe(1, {Point3(0, 0, 3)});
  EXPECT_EQ(m.keyframe_count(), 2u);

  std::map<size_t, Pose3> anchors;
  anchors[0] = Pose3(Rot3(), Point3(10, 0, 0)); // kf0 shifted +10x
  anchors[1] = Pose3(Rot3(), Point3(0, 5, 0));  // kf1 shifted +5y

  const std::vector<Point3> c = m.cloud(anchors);
  ASSERT_EQ(c.size(), 3u);
  EXPECT_TRUE(c[0].isApprox(Point3(10, 0, 1), 1e-9));
  EXPECT_TRUE(c[1].isApprox(Point3(11, 0, 2), 1e-9));
  EXPECT_TRUE(c[2].isApprox(Point3(0, 5, 3), 1e-9));
}

TEST(SparseMap, SkipsKeyframesMissingFromAnchors) {
  superslam::SparseMap m;
  m.add_keyframe(0, {Point3(0, 0, 1)});
  m.add_keyframe(7, {Point3(0, 0, 1)});
  std::map<size_t, Pose3> anchors;
  anchors[0] = Pose3();
  EXPECT_EQ(m.cloud(anchors).size(), 1u); // kf7 absent -> skipped
}
