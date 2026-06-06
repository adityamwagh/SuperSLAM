#pragma once
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include <memory>

#include "FrameTracker.h"
#include "InferenceInterfaces.h"
#include "KeyframeDatabase.h"
#include "PlaceRecognizer.h"

namespace superslam {

// Outcome of a loop-closure attempt for one query keyframe.
struct LoopResult {
  bool accepted = false;
  size_t matched_keyframe = 0;   // database keyframe the loop closes against
  gtsam::Pose3 relative_pose;    // T_matched_query  (== matched.between(query));
                                 // BetweenFactor edge
  gtsam::SharedNoiseModel noise; // robust noise for that edge
  int inliers = 0;
};

// Tunables (env-overridable, matching the repo's SUPERSLAM_* convention).
struct LoopParams {
  float min_score = 0.75f;    // cosine gate before geometric verification
  size_t exclude_recent = 30; // skip temporally-adjacent keyframes
  int top_k = 3;              // candidates handed to verification per query
  int required_votes = 3;     // temporal-consistency streak before accepting
  size_t id_tolerance = 5;    // "same locale" window for the voter
  int min_inliers = 30;       // geometric-verification inlier floor
  double inlier_px = 3.0;     // reprojection inlier threshold (pixels)
  double noise_base = 0.1;    // edge sigma base; scaled by 1/sqrt(inliers)
};

// Pose-graph loop closure with implicit landmarks. Retrieval (place
// recognition) finds a revisited keyframe; LightGlue and the existing
// FrameTracker geometrically verify it and recover the relative pose for a
// BetweenFactor edge in the global pose graph. Own the place recognizer and
// keyframe database. Run on a worker thread.
class LoopCloser {
public:
  LoopCloser(IFeatureMatcher* matcher,
             const gtsam::Cal3_S2Stereo::shared_ptr& K,
             std::unique_ptr<IPlaceRecognizer> recognizer,
             LoopParams params = {});

  // Index a new keyframe (store record and add its global descriptor to
  // retrieval).
  void add_keyframe(const KeyframeRecord& keyframe_record);

  // Close a loop for the query keyframe. Run retrieval, temporal voting, and
  // geometric verification. Return accepted=false until all gates pass.
  LoopResult detect(const KeyframeRecord& query);

  // Geometrically verify a single candidate.
  LoopResult verify(const KeyframeRecord& query, const KeyframeRecord& candidate);

  // Compute a keyframe's global descriptor from its left image (delegates to
  // the place recognizer). Run on the loop worker thread.
  cv::Mat compute_global_descriptor(const cv::Mat& image) {
    return recognizer_->compute_global_descriptor(image);
  }

  const KeyframeDatabase& database() const { return db_; }

private:
  IFeatureMatcher* matcher_;
  gtsam::Cal3_S2Stereo::shared_ptr K_;
  std::unique_ptr<IPlaceRecognizer> recognizer_;
  LoopParams params_;
  KeyframeDatabase db_;
  FrameTracker verifier_;
  TemporalConsistencyVoter voter_;
};

} // namespace superslam
