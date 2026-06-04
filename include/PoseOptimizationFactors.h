#pragma once

// Motion-only (pose-only) reprojection factors for Tracking::PoseOptimization.
//
// The 3D landmark is captured as a fixed constant; the factor graph has a
// single variable (the 6-DOF camera pose). Each LM step solves a 6x6 system
// rather than the (6+3N)x(6+3N) system of adding every landmark as a free
// variable.
//
// Jacobians are GTSAM's own analytic camera Jacobians (PinholePose::project and
// StereoCamera::project2) restricted to the pose block. The monocular factor is
// convention-identical to gtsam::GenericProjectionFactor with the point fixed.
// The stereo factor additionally constrains the right-image coordinate (uR).

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/PinholePose.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <cmath>
#include <cstdlib>

namespace superslam {

// Pose-only monocular reprojection factor (2 residuals: u, v).
class PoseOnlyProjectionFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;
  gtsam::Point3 Xw_;
  gtsam::Point2 measured_;
  gtsam::Cal3_S2::shared_ptr K_;

public:
  PoseOnlyProjectionFactor(const gtsam::Point2& measured,
                           const gtsam::SharedNoiseModel& model,
                           gtsam::Key pose_key,
                           const gtsam::Point3& Xw,
                           const gtsam::Cal3_S2::shared_ptr& K)
      : Base(model, pose_key), Xw_(Xw), measured_(measured), K_(K) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& pose,
                              boost::optional<gtsam::Matrix&> H = boost::none) const override {
    gtsam::PinholePose<gtsam::Cal3_S2> camera(pose, K_);
    try {
      if (H) {
        gtsam::Matrix26 Dpose;
        const gtsam::Point2 reproj = camera.project(Xw_, Dpose, boost::none, boost::none);
        *H = Dpose;
        return reproj - measured_;
      }
      return camera.project(Xw_) - measured_;
    } catch (const gtsam::CheiralityException&) {
      // Landmark behind the camera: return a large residual and zero Jacobian;
      // the point is culled as an outlier.
      if (H)
        *H = gtsam::Matrix::Zero(2, 6);
      return gtsam::Vector2::Constant(2.0 * K_->fx());
    }
  }
};

// Pose-only stereo reprojection factor (3 residuals: uL, uR, v).
class PoseOnlyStereoFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;
  gtsam::Point3 Xw_;
  gtsam::StereoPoint2 measured_;
  gtsam::Cal3_S2Stereo::shared_ptr K_;

public:
  PoseOnlyStereoFactor(const gtsam::StereoPoint2& measured,
                       const gtsam::SharedNoiseModel& model,
                       gtsam::Key pose_key,
                       const gtsam::Point3& Xw,
                       const gtsam::Cal3_S2Stereo::shared_ptr& K)
      : Base(model, pose_key), Xw_(Xw), measured_(measured), K_(K) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& pose,
                              boost::optional<gtsam::Matrix&> H = boost::none) const override {
    gtsam::StereoCamera camera(pose, K_);
    try {
      if (H) {
        gtsam::Matrix36 Dpose;
        const gtsam::StereoPoint2 reproj = camera.project2(Xw_, Dpose, boost::none);
        *H = Dpose;
        return reproj.vector() - measured_.vector();
      }
      return camera.project(Xw_).vector() - measured_.vector();
    } catch (const gtsam::StereoCheiralityException&) {
      // Landmark behind the camera: return a large residual and zero Jacobian;
      // the point is culled as an outlier.
      if (H)
        *H = gtsam::Matrix::Zero(3, 6);
      return gtsam::Vector3::Constant(2.0 * K_->fx());
    }
  }
};

// Base disparity measurement precision (px). The SuperPoint and LightGlue
// stereo-match disparity floor is approx 8px (not subpixel).
inline double disp_sigma_px() {
  if (const char* e = std::getenv("SUPERSLAM_DISP_SIGMA_PX"))
    return std::atof(e);
  return 8.0;
}

// Depth (m) beyond which stereo depth is smoothly deweighted (small disparity
// is ill-conditioned). Set d_cond = mbf / Z_cond.
inline double stereo_cond_depth_m() {
  if (const char* e = std::getenv("SUPERSLAM_STEREO_COND_DEPTH_M"))
    return std::atof(e);
  return 40.0;
}

// Diagonal stereo measurement noise over (uL, uR, v). uL and v keep the
// octave-scaled reprojection floor sigma_px (the SuperPoint and LightGlue
// matching floor, NaN-safe). uR carries disparity, which is metric depth and
// scale: a tight base sigma_d0 that grows smoothly as disparity goes to 0;
// far points are released before Z flips negative (no cheirality NaN) without a
// hard depth gate.
//   sigma_uR = sigma_d0 * sqrt(1 + (d_cond/d)^2),  d_cond = mbf / Z_cond
inline gtsam::SharedNoiseModel stereo_diag_noise(double sigma_px, double disparity, double mbf) {
  const double sigma_d0 = disp_sigma_px();
  const double d_cond = mbf / stereo_cond_depth_m();
  // Clamp non-positive or tiny disparity; the point is released with a large
  // sigma_uR, not a divide by zero.
  const double d = disparity > 1e-3 ? disparity : 1e-3;
  const double r = d_cond / d;
  const double sigma_uR = sigma_d0 * std::sqrt(1.0 + r * r);
  return gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << sigma_px, sigma_uR, sigma_px).finished());
}

} // namespace superslam
