#include "SparseMap.h"

namespace superslam {

void SparseMap::add_keyframe(size_t keyframe_id, const std::vector<gtsam::Point3>& camera_points) {
  points_[keyframe_id] = camera_points;
}

std::vector<gtsam::Point3> SparseMap::cloud(const std::map<size_t, gtsam::Pose3>& anchors) const {
  std::vector<gtsam::Point3> out;
  for (const auto& [keyframe_id, pts] : points_) {
    auto it = anchors.find(keyframe_id);
    if (it == anchors.end())
      continue;
    for (const gtsam::Point3& p : pts)
      out.push_back(it->second.transformFrom(p));
  }
  return out;
}

} // namespace superslam
