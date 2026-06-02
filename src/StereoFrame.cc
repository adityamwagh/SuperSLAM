#include "StereoFrame.h"

namespace superslam {

gtsam::Point3 StereoFrame::backproject(int i, const gtsam::Cal3_S2Stereo& K) const {
  // Backproject the stereo point into the camera frame (the math GTSAM's
  // StereoCamera::backproject performs), then pose (Twc) lifts it to world.
  const double uL = stereo[i].uL(), uR = stereo[i].uR(), v = stereo[i].v();
  const double Z = K.fx() * K.baseline() / (uL - uR); // mbf / disparity
  const double X = (uL - K.px()) * Z / K.fx();
  const double Y = (v - K.py()) * Z / K.fy();
  return pose.transformFrom(gtsam::Point3(X, Y, Z));
}

} // namespace superslam
