#pragma once
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include <cstddef>
#include <map>
#include <vector>

namespace superslam {

// Sparse point-cloud map for visualization and export. Hold each keyframe's
// depth-valid feature points in the keyframe camera frame; cloud() lifts them
// to the world frame using the loop-corrected keyframe anchors. Output-only;
// not used for tracking.
class SparseMap {
public:
  void add_keyframe(size_t keyframe_id, const std::vector<gtsam::Point3>& camera_points);
  std::vector<gtsam::Point3> cloud(const std::map<size_t, gtsam::Pose3>& anchors) const;
  size_t keyframe_count() const { return points_.size(); }

private:
  std::map<size_t, std::vector<gtsam::Point3>> points_;
};

} // namespace superslam
