#include <gtest/gtest.h>

#include "KeyframeGate.h"

using superslam::should_insert_keyframe;

TEST(KeyframeGate, CovisibilityAndCaps) {
  // High covisibility (280/300 tracked), recent KF -> no new KF.
  EXPECT_FALSE(should_insert_keyframe(/*tracked=*/280,
                                      /*ref=*/300,
                                      /*frames_since_keyframe=*/5));
  // Covisibility dropped below 0.7 (150/300 = 0.5) -> new KF.
  EXPECT_TRUE(should_insert_keyframe(150, 300, 5));
  // Hard match floor breached -> new KF.
  EXPECT_TRUE(should_insert_keyframe(20, 300, 5));
  // Frame cap reached even with high covisibility -> new KF.
  EXPECT_TRUE(should_insert_keyframe(290, 300, 20));
  // Covisibility low but too soon after the last KF -> suppressed.
  EXPECT_FALSE(should_insert_keyframe(150, 300, 1));
}
