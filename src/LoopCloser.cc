#include "LoopCloser.h"

#include <gtsam/geometry/StereoCamera.h>

#include <cmath>
#include <cstdlib>

namespace superslam {

namespace {
double env_double(const char* key, double fallback) {
  const char* v = std::getenv(key);
  return v ? std::atof(v) : fallback;
}

// Stereo backprojection into the camera frame (identity pose). Same math as
// StereoFrame::backproject without the Twc lift; verification works in the
// candidate keyframe's local frame.
gtsam::Point3 backproject_cam(const gtsam::StereoPoint2& s, const gtsam::Cal3_S2Stereo& K) {
  const double Z = K.fx() * K.baseline() / (s.uL() - s.uR());
  const double X = (s.uL() - K.px()) * Z / K.fx();
  const double Y = (s.v() - K.py()) * Z / K.fy();
  return gtsam::Point3(X, Y, Z);
}
} // namespace

LoopCloser::LoopCloser(IFeatureMatcher* matcher,
                       const gtsam::Cal3_S2Stereo::shared_ptr& K,
                       std::unique_ptr<IPlaceRecognizer> recognizer,
                       LoopParams params)
    : matcher_(matcher), K_(K), recognizer_(std::move(recognizer)), params_(params), verifier_(K),
      voter_(params.required_votes, params.id_tolerance) {
  // Apply env overrides.
  params_.min_inliers =
      static_cast<int>(env_double("SUPERSLAM_LOOP_MIN_INLIERS", params_.min_inliers));
  params_.min_score = static_cast<float>(env_double("SUPERSLAM_LOOP_MIN_SCORE", params_.min_score));
}

void LoopCloser::add_keyframe(const KeyframeRecord& keyframe_record) {
  db_.add(keyframe_record);
  recognizer_->add(keyframe_record.keyframe_id, keyframe_record.global_descriptor);
}

LoopResult LoopCloser::verify(const KeyframeRecord& query, const KeyframeRecord& candidate) {
  LoopResult out;
  out.matched_keyframe = candidate.keyframe_id;

  // LightGlue match candidate(left) to query(left): queryIdx=candidate,
  // trainIdx=query.
  const MatchResult m = matcher_->match(candidate.keypoints_left,
                                        candidate.descriptors_left,
                                        query.keypoints_left,
                                        query.descriptors_left);

  std::vector<PointObs> stereo_observations; // candidate-frame 3D point and its
                                             // measurement in query
  for (const cv::DMatch& dm : m.matches) {
    const int ci = dm.queryIdx, qi = dm.trainIdx;
    if (ci < 0 || qi < 0 || ci >= static_cast<int>(candidate.stereo.size()) ||
        qi >= static_cast<int>(query.stereo.size()))
      continue;
    if (!candidate.has_depth[ci] || !query.has_depth[qi])
      continue;
    stereo_observations.push_back({backproject_cam(candidate.stereo[ci], *K_), query.stereo[qi]});
  }
  if (static_cast<int>(stereo_observations.size()) < params_.min_inliers)
    return out; // Too few correspondences to trust a loop.

  // Relative pose: the query camera expressed in the candidate frame
  // (T_candidate_query), recovered with the robust pose-only tracker seeded at
  // identity.
  const gtsam::Pose3 rel = verifier_.track(gtsam::Pose3(), stereo_observations);

  // Count reprojection inliers under the recovered pose.
  int inliers = 0;
  const gtsam::StereoCamera cam(rel, K_);
  for (const PointObs& o : stereo_observations) {
    try {
      const gtsam::StereoPoint2 p = cam.project(o.Xw);
      const double e = std::hypot(p.uL() - o.meas.uL(), p.v() - o.meas.v());
      if (e < params_.inlier_px)
        ++inliers;
    } catch (const std::exception&) {
      // Cheirality (point behind camera), not an inlier.
    }
  }
  out.inliers = inliers;
  if (inliers < params_.min_inliers)
    return out;

  // Edge noise: tighter with more inliers, robustified.
  const double s = params_.noise_base / std::sqrt(static_cast<double>(inliers));
  const double sigR = std::max(s, 0.02);
  const double sigT = std::max(s, 0.20);
  auto diag = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << sigR, sigR, sigR, sigT, sigT, sigT).finished());
  out.noise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(std::sqrt(7.815)),
      diag);
  out.relative_pose = rel;
  out.accepted = true;
  return out;
}

LoopResult LoopCloser::detect(const KeyframeRecord& query) {
  const std::vector<LoopCandidate> cands =
      recognizer_->query(query.global_descriptor, params_.exclude_recent, params_.top_k);

  // Temporal-consistency voting on the best candidate gates one-off false
  // positives.
  const LoopCandidate* best = cands.empty() ? nullptr : &cands.front();
  if (!voter_.vote(best))
    return LoopResult{};

  // Geometrically verify candidates in score order; return the first that
  // survives.
  for (const LoopCandidate& c : cands) {
    if (c.score < params_.min_score)
      break; // Sorted descending; nothing better remains.
    LoopResult r = verify(query, db_.get(c.keyframe_id));
    if (r.accepted)
      return r;
  }
  return LoopResult{};
}

} // namespace superslam
