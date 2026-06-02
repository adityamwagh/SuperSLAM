#pragma once
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoPoint2.h>

#include <vector>

namespace superslam {

struct PointObs {
  gtsam::Point3 Xw;         // 3D point (triangulated from the last keyframe's stereo)
  gtsam::StereoPoint2 meas; // its measurement in the current frame
};

// Motion-only pose solve for an ordinary frame against the last keyframe's points.
class FrameTracker {
public:
  explicit FrameTracker(const gtsam::Cal3_S2Stereo::shared_ptr& K) : K_(K) {}
  gtsam::Pose3 track(const gtsam::Pose3& initial_guess, const std::vector<PointObs>& matches);

private:
  gtsam::Cal3_S2Stereo::shared_ptr K_;
};

} // namespace superslam
