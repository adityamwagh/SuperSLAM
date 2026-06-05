#pragma once
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <opencv4/opencv2/core.hpp>

#include <cstddef>
#include <map>
#include <vector>

namespace superslam {

// One keyframe's persisted state for loop closure. The authoritative optimized
// pose lives in the GlobalPoseGraph (keyed by keyframe_id); pose_at_insert is a
// cached seed for geometric verification. Keep local features
// (keypoints_left, descriptors_left, stereo); a loop candidate is re-matched
// with LightGlue and back-projected.
struct KeyframeRecord {
  size_t keyframe_id = 0;
  double timestamp = 0.0;
  gtsam::Pose3 pose_at_insert;              // cached seed; not the source of truth
  std::vector<cv::KeyPoint> keypoints_left; // left keypoints
  cv::Mat descriptors_left;                 // [N x D] for LightGlue verification
  std::vector<gtsam::StereoPoint2> stereo;  // (uL, uR, v) per keypoint
  std::vector<char> has_depth;              // 1 if stereo-valid
  cv::Mat global_descriptor;                // [1 x Dg] for place recognition
  std::vector<size_t> covisible;            // keyframe_ids sharing landmarks (candidate pruning)
};

// Insertion-ordered store of keyframe records with O(1) id lookup.
class KeyframeDatabase {
public:
  void add(KeyframeRecord rec);
  const KeyframeRecord& get(size_t keyframe_id) const;
  bool has(size_t keyframe_id) const { return id_to_index_.count(keyframe_id) != 0; }
  size_t size() const { return keyframes_.size(); }

  // Records in insertion order (== keyframe creation order).
  const std::vector<KeyframeRecord>& records() const { return keyframes_; }

private:
  std::vector<KeyframeRecord> keyframes_;
  std::map<size_t, size_t> id_to_index_;
};

} // namespace superslam
