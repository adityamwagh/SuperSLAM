#pragma once
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoPoint2.h>

#include <cstddef>
#include <deque>
#include <map>
#include <vector>

namespace superslam {

struct StereoObs {
  size_t landmark_id = 0;
  gtsam::StereoPoint2 meas;
};

// Hold a bounded sliding window of the last K keyframe poses, optimized as a
// batch with smart stereo factors (landmarks marginalized). No persisted map.
class WindowSmoother {
public:
  WindowSmoother(const gtsam::Cal3_S2Stereo::shared_ptr& K, size_t window_size);

  void add_keyframe(size_t keyframe_id,
                    const gtsam::Pose3& initial_pose,
                    const std::vector<StereoObs>& obs);
  void optimize();
  gtsam::Pose3 pose_of(size_t keyframe_id) const;
  size_t window_count() const;
  bool in_window(size_t keyframe_id) const;

private:
  gtsam::Cal3_S2Stereo::shared_ptr K_;
  size_t window_size_;
  std::deque<size_t> window_;                          // keyframe_ids in window order
  std::map<size_t, gtsam::Pose3> poses_;               // current estimate per keyframe_id
  std::map<size_t, std::vector<StereoObs>> obs_by_kf_; // observations per keyframe_id
};

} // namespace superslam
