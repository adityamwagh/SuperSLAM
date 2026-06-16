// Unit tests for the motion-only (pose-only) reprojection factors used by
// Tracking::PoseOptimization. The 3D landmark is a fixed constant; only the
// 6-DOF camera pose is optimized. Jacobians are checked against GTSAM's
// numerical differentiation.
#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/PinholePose.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/StereoFactor.h>

#include <opencv2/core.hpp>

#include "PoseOptimizationFactors.h"

using namespace gtsam;
using superslam::PoseOnlyProjectionFactor;
using superslam::PoseOnlyStereoFactor;
using symbol_shorthand::L;
using symbol_shorthand::X;

namespace {
const Pose3 kTrue(Rot3::RzRyRx(0.10, -0.20, 0.05), Point3(1.0, -0.5, 0.3));
const Point3 kLandmark(2.0, 0.4, 8.0);
// Pose perturbed from the true pose so the residual (and Jacobian) is nonzero.
const Pose3 kPerturbed =
    kTrue.retract((Vector(6) << 0.02, -0.03, 0.01, 0.05, -0.04, 0.06).finished());
} // namespace

TEST(PoseOptimizationFactors, ProjectionReprojectsToZeroResidualAtTruePose) {
  Cal3_S2::shared_ptr K(new Cal3_S2(500.0, 500.0, 0.0, 320.0, 240.0));
  const Point2 measured = PinholePose<Cal3_S2>(kTrue, K).project(kLandmark);
  auto noise = noiseModel::Isotropic::Sigma(2, 1.0);

  PoseOnlyProjectionFactor factor(measured, noise, 0, kLandmark, K);

  EXPECT_LT(factor.evaluateError(kTrue).norm(), 1e-9);
}

TEST(PoseOptimizationFactors, ProjectionAnalyticJacobianMatchesNumerical) {
  Cal3_S2::shared_ptr K(new Cal3_S2(500.0, 500.0, 0.0, 320.0, 240.0));
  const Point2 measured = PinholePose<Cal3_S2>(kTrue, K).project(kLandmark);
  auto noise = noiseModel::Isotropic::Sigma(2, 1.0);

  PoseOnlyProjectionFactor factor(measured, noise, 0, kLandmark, K);

  Matrix Hanalytic;
  factor.evaluateError(kPerturbed, Hanalytic);
  Matrix Hnumeric =
      numericalDerivative11<Vector, Pose3>([&](const Pose3& p) { return factor.evaluateError(p); },
                                           kPerturbed,
                                           1e-6);

  EXPECT_TRUE(assert_equal(Hnumeric, Hanalytic, 1e-5));
}

TEST(PoseOptimizationFactors, StereoReprojectsToZeroResidualAtTruePose) {
  Cal3_S2Stereo::shared_ptr K(new Cal3_S2Stereo(500.0, 500.0, 0.0, 320.0, 240.0, 0.5));
  const StereoPoint2 measured = StereoCamera(kTrue, K).project(kLandmark);
  auto noise = noiseModel::Isotropic::Sigma(3, 1.0);

  PoseOnlyStereoFactor factor(measured, noise, 0, kLandmark, K);

  EXPECT_LT(factor.evaluateError(kTrue).norm(), 1e-9);
}

TEST(PoseOptimizationFactors, StereoAnalyticJacobianMatchesNumerical) {
  Cal3_S2Stereo::shared_ptr K(new Cal3_S2Stereo(500.0, 500.0, 0.0, 320.0, 240.0, 0.5));
  const StereoPoint2 measured = StereoCamera(kTrue, K).project(kLandmark);
  auto noise = noiseModel::Isotropic::Sigma(3, 1.0);

  PoseOnlyStereoFactor factor(measured, noise, 0, kLandmark, K);

  Matrix Hanalytic;
  factor.evaluateError(kPerturbed, Hanalytic);
  Matrix Hnumeric =
      numericalDerivative11<Vector, Pose3>([&](const Pose3& p) { return factor.evaluateError(p); },
                                           kPerturbed,
                                           1e-6);

  EXPECT_TRUE(assert_equal(Hnumeric, Hanalytic, 1e-5));
}

// A landmark behind the camera (Z <= 0) must NOT throw a cheirality exception
// from inside the optimizer (which would terminate the process). Instead it
// returns a large residual so the point is rejected as an outlier.
TEST(PoseOptimizationFactors, StereoHandlesPointBehindCameraWithoutThrowing) {
  Cal3_S2Stereo::shared_ptr K(new Cal3_S2Stereo(500.0, 500.0, 0.0, 320.0, 240.0, 0.5));
  const Point3 behind(0.0, 0.0, -5.0); // 5m behind an identity-pose camera
  const StereoPoint2 measured(300.0, 290.0, 240.0);
  PoseOnlyStereoFactor factor(measured, noiseModel::Isotropic::Sigma(3, 1.0), 0, behind, K);

  Vector e;
  Matrix H;
  ASSERT_NO_THROW(e = factor.evaluateError(Pose3()));
  ASSERT_NO_THROW(factor.evaluateError(Pose3(), H));
  EXPECT_GT(e.norm(), 100.0);
}

TEST(PoseOptimizationFactors, ProjectionHandlesPointBehindCameraWithoutThrowing) {
  Cal3_S2::shared_ptr K(new Cal3_S2(500.0, 500.0, 0.0, 320.0, 240.0));
  const Point3 behind(0.0, 0.0, -5.0);
  const Point2 measured(300.0, 240.0);
  PoseOnlyProjectionFactor factor(measured, noiseModel::Isotropic::Sigma(2, 1.0), 0, behind, K);

  Vector e;
  Matrix H;
  ASSERT_NO_THROW(e = factor.evaluateError(Pose3()));
  ASSERT_NO_THROW(factor.evaluateError(Pose3(), H));
  EXPECT_GT(e.norm(), 100.0);
}

// Map bundle adjustment regression: with only ONE (fixed) keyframe, stereo
// observations must keep every landmark's depth observable. The old map BA
// treated stereo as monocular, so a single view left depth unconstrained ->
// singular system -> GTSAM returned NaN ("Levenberg-Marquardt giving up").
// Stereo factors (which carry uR) make the same single-keyframe problem
// well-posed. This guards the formulation used by Optimizer::BundleAdjustment.
TEST(PoseOptimizationFactors, SingleFixedKeyframeStereoBaRecoversDepth) {
  Cal3_S2Stereo::shared_ptr K(new Cal3_S2Stereo(500.0, 500.0, 0.0, 320.0, 240.0, 0.5));
  const Pose3 kf(Rot3::RzRyRx(0.05, -0.1, 0.02), Point3(0.3, -0.1, 0.0));
  const std::vector<Point3> pts = {{1.0, 0.5, 8.0},
                                   {-1.5, 0.2, 10.0},
                                   {0.4, -0.8, 6.0},
                                   {2.0, 1.0, 12.0},
                                   {-0.7, -0.3, 7.0}};

  NonlinearFactorGraph graph;
  Values init;

  // Fix the single keyframe with a tight prior (the gauge anchor).
  graph.addPrior(
      X(0),
      kf,
      noiseModel::Diagonal::Sigmas((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished()));
  init.insert(X(0), kf);

  auto noise = noiseModel::Isotropic::Sigma(3, 1.0);
  for (size_t i = 0; i < pts.size(); i++) {
    const StereoPoint2 z = StereoCamera(kf, K).project(pts[i]);
    graph.emplace_shared<GenericStereoFactor<Pose3, Point3>>(z, noise, X(0), L(i), K);
    // Perturb the landmark initial estimate (esp. depth) away from truth.
    const Point3 perturbed(pts[i].x() + 0.3, pts[i].y() - 0.2, pts[i].z() + 1.5);
    init.insert(L(i), perturbed);
  }

  const Values result = LevenbergMarquardtOptimizer(graph, init).optimize();

  for (size_t i = 0; i < pts.size(); i++) {
    const Point3 est = result.at<Point3>(L(i));
    ASSERT_TRUE(std::isfinite(est.x()));          // no NaN
    EXPECT_TRUE(assert_equal(pts[i], est, 1e-3)); // depth recovered from one view
  }
}

// Reproduction of the real-data map-BA divergence. The init GlobalBA is a
// two-keyframe stereo problem: KF0 is the gauge anchor, KF1 is free, and the
// landmarks are triangulated from noisy SuperPoint matches (~10 px) spanning
// near and FAR depths (KITTI: up to ~80 m, where stereo disparity is only a
// few pixels and depth is weakly observable). The anchor was pinned with a
// near-zero-sigma prior (1e-9 -> information 1e18). Pairing a near-infinite
// prior eigenvalue with the near-zero far-point depth eigenvalue drives the
// Hessian condition number past double precision -> the LM linear solve
// returns a NaN delta -> "Levenberg-Marquardt giving up". This case sweeps the
// anchor sigma to pin the numerical threshold.
namespace {
// Deterministic pseudo-noise so the regression is reproducible across runs.
double jitter(int seed) {
  // Cheap LCG mapped to roughly [-1, 1].
  unsigned int x = static_cast<unsigned int>(seed) * 1664525u + 1013904223u;
  return (static_cast<double>(x % 20001u) / 10000.0) - 1.0;
}

// Run a 2-KF stereo BA with KITTI-like geometry and ~noisePx measurement
// noise, anchoring KF0 with the given prior sigma. Returns true if every
// optimized landmark and pose is finite (no NaN).
bool TwoViewStereoBAIsFinite(double anchorSigma, double noisePx) {
  // KITTI-04 left/rect calibration and stereo baseline (mbf/fx ~ 0.54 m).
  Cal3_S2Stereo::shared_ptr K(new Cal3_S2Stereo(718.0, 718.0, 0.0, 607.0, 185.0, 0.54));
  const Pose3 kf0;                                                         // identity (world)
  const Pose3 kf1(Rot3::RzRyRx(0.01, 0.02, 0.005), Point3(0.0, 0.0, 1.2)); // forward motion

  // Landmarks spanning near (6 m) to far (80 m); far points have tiny
  // disparity.
  std::vector<Point3> pts;
  for (int i = 0; i < 40; i++) {
    const double z = 6.0 + 1.85 * i; // 6 .. ~78 m
    const double x = 0.6 * jitter(i * 3 + 1) * z * 0.15;
    const double y = 0.5 * jitter(i * 3 + 2) * z * 0.15;
    pts.emplace_back(x, y, z);
  }

  NonlinearFactorGraph graph;
  Values init;

  graph.addPrior(X(0),
                 kf0,
                 noiseModel::Diagonal::Sigmas((Vector(6) << anchorSigma,
                                               anchorSigma,
                                               anchorSigma,
                                               anchorSigma,
                                               anchorSigma,
                                               anchorSigma)
                                                  .finished()));
  init.insert(X(0), kf0);
  init.insert(X(1), kf1);

  auto noise = noiseModel::Isotropic::Sigma(3, 1.0);
  for (size_t i = 0; i < pts.size(); i++) {
    StereoPoint2 z0 = StereoCamera(kf0, K).project(pts[i]);
    StereoPoint2 z1 = StereoCamera(kf1, K).project(pts[i]);
    // Inject inconsistent per-view measurement noise (the real cross-view
    // disagreement that triangulation cannot satisfy exactly).
    const int s = static_cast<int>(i);
    z0 = StereoPoint2(z0.uL() + noisePx * jitter(s + 100),
                      z0.uR() + noisePx * jitter(s + 200),
                      z0.v() + noisePx * jitter(s + 300));
    z1 = StereoPoint2(z1.uL() + noisePx * jitter(s + 400),
                      z1.uR() + noisePx * jitter(s + 500),
                      z1.v() + noisePx * jitter(s + 600));
    graph.emplace_shared<GenericStereoFactor<Pose3, Point3>>(z0, noise, X(0), L(i), K);
    graph.emplace_shared<GenericStereoFactor<Pose3, Point3>>(z1, noise, X(1), L(i), K);

    // Triangulate-from-noisy-z init: midpoint of the two noisy
    // back-projections.
    const Point3 p0 = StereoCamera(kf0, K).backproject(z0);
    init.insert(L(i), p0);
  }

  LevenbergMarquardtParams params;
  params.setMaxIterations(20);
  const Values result = LevenbergMarquardtOptimizer(graph, init, params).optimize();

  bool allFinite = std::isfinite(result.at<Pose3>(X(1)).translation().x());
  for (size_t i = 0; i < pts.size(); i++)
    allFinite = allFinite && std::isfinite(result.at<Point3>(L(i)).x());
  return allFinite;
}
} // namespace

TEST(PoseOptimizationFactors, TwoViewStereoMapBaStaysFiniteWithSaneGaugeAnchor) {
  // A loose anchor (1e-6) keeps the Hessian condition number in range; the
  // optimizer must converge to finite values on noisy, far-point data.
  EXPECT_TRUE(TwoViewStereoBAIsFinite(1e-6, 10.0));
}

// stereo_diag_noise builds the diagonal stereo measurement model that pins
// metric scale: uL and v keep the (loose) reprojection floor, while uR
// (disparity, == depth/scale) gets a tight base sigma that inflates smoothly
// as disparity -> 0 so far/degenerate points are released instead of driving Z
// behind the camera.
namespace {
// Extract the uR (disparity) sigma from a stereo_diag_noise model.
double diagSigmaUR(double sigma_px, double disparity, double mbf) {
  auto d = boost::dynamic_pointer_cast<noiseModel::Diagonal>(
      superslam::stereo_diag_noise(sigma_px, disparity, mbf));
  EXPECT_TRUE(d != nullptr); // must be a Diagonal model
  return d->sigmas()(1);
}
} // namespace

TEST(PoseOptimizationFactors, StereoDiagNoiseLeavesUlVAtFloorAndPinsDisparityNear) {
  const double mbf = 387.0; // KITTI-04-ish (fx*baseline)
  const double sigma_px = 10.0;

  auto d = boost::dynamic_pointer_cast<noiseModel::Diagonal>(
      superslam::stereo_diag_noise(sigma_px, /*disparity=*/100.0, mbf));
  ASSERT_TRUE(d != nullptr);
  const Vector s = d->sigmas();
  EXPECT_LT(std::abs(s(0) - sigma_px), 1e-9); // uL untouched
  EXPECT_LT(std::abs(s(2) - sigma_px), 1e-9); // v  untouched
  EXPECT_LT(std::abs(s(1) - 8.0),
            0.1); // uR ~ sigma_d0 (default 8.0) at large disparity
}

TEST(PoseOptimizationFactors, StereoDiagNoiseInflatesAndStaysFiniteForFarDisparity) {
  const double mbf = 387.0;
  const double sigma_px = 10.0;

  EXPECT_GT(diagSigmaUR(sigma_px, 0.5, mbf), 10.0 * 1.5);                       // far -> inflated
  EXPECT_GT(diagSigmaUR(sigma_px, 5.0, mbf), diagSigmaUR(sigma_px, 50.0, mbf)); // monotonic

  for (double disp : {0.0, -3.0}) { // guard: no div-by-zero
    const double s = diagSigmaUR(sigma_px, disp, mbf);
    EXPECT_TRUE(std::isfinite(s));
    EXPECT_GT(s, 0.0);
  }
}
