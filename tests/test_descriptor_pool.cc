#include <gtest/gtest.h>

#include "DescriptorPool.h"

using superslam::FreeList;

TEST(FreeList, AcquireReleaseRoundTrips) {
  FreeList f(3);
  const int a = f.acquire(), b = f.acquire(), c = f.acquire();
  EXPECT_GE(a, 0);
  EXPECT_GE(b, 0);
  EXPECT_GE(c, 0);
  EXPECT_EQ(f.acquire(), -1); // exhausted
  EXPECT_EQ(f.in_use(), 3);
  f.release(b);
  EXPECT_EQ(f.in_use(), 2);
  EXPECT_EQ(f.acquire(), b); // reuses the freed slot
}

TEST(FreeList, EmptyAndFull) {
  FreeList f(2);
  EXPECT_EQ(f.in_use(), 0);
  const int a = f.acquire();
  const int b = f.acquire();
  EXPECT_EQ(f.in_use(), 2);
  f.release(a);
  f.release(b);
  EXPECT_EQ(f.in_use(), 0);
}
