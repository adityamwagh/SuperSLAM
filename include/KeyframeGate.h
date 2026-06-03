#pragma once
#include <algorithm>

namespace superslam {

// Covisibility-based keyframe gate (scale-invariant): insert a keyframe when
// the fraction of the reference keyframe's features tracked drops below
// covisibility_ratio, when the hard match floor is breached, or after
// max_frames (a parallax cap for slow forward motion). min_frames suppresses
// back-to-back keyframes on a transient covisibility dip.
inline bool should_insert_keyframe(int tracked_matches,
                                   int reference_features,
                                   int frames_since_keyframe,
                                   double covisibility_ratio = 0.7,
                                   int max_frames = 20,
                                   int min_frames = 2,
                                   int min_matches = 30) {
  if (frames_since_keyframe < min_frames)
    return false;
  if (frames_since_keyframe >= max_frames || tracked_matches < min_matches)
    return true;
  const double ratio = static_cast<double>(tracked_matches) / std::max(1, reference_features);
  return ratio < covisibility_ratio;
}

} // namespace superslam
