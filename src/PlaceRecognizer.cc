#include "PlaceRecognizer.h"

#include <algorithm>
#include <cmath>

namespace superslam {

namespace {
// Return a 1xD CV_32F row, L2-normalized.
cv::Mat normalizedRow(const cv::Mat& desc) {
  cv::Mat row = desc.reshape(1, 1);
  if (row.type() != CV_32F)
    row.convertTo(row, CV_32F);
  const double n = cv::norm(row);
  if (n > 1e-12)
    row = row / n;
  return row.clone();
}
} // namespace

void CosineDescriptorIndex::add(size_t keyframe_id, const cv::Mat& global_descriptor) {
  ids_.push_back(keyframe_id);
  db_.push_back(normalizedRow(global_descriptor));
}

std::vector<LoopCandidate> CosineDescriptorIndex::query(const cv::Mat& global_descriptor,
                                                        size_t excludeRecent,
                                                        int topK,
                                                        float minScore) const {
  std::vector<LoopCandidate> out;
  const size_t M = ids_.size();
  if (M == 0 || M <= excludeRecent)
    return out; // nothing old enough to be a loop

  const cv::Mat q = normalizedRow(global_descriptor); // 1 x D
  const size_t limit = M - excludeRecent;             // rows [0, limit) are candidates
  const cv::Mat cand = db_.rowRange(0, limit);        // limit x D
  cv::Mat scores = cand * q.t();                      // limit x 1 cosine similarities

  out.reserve(limit);
  for (size_t i = 0; i < limit; ++i) {
    const float s = scores.at<float>(static_cast<int>(i), 0);
    if (s >= minScore)
      out.push_back({ids_[i], s});
  }
  std::sort(out.begin(), out.end(), [](const LoopCandidate& a, const LoopCandidate& b) {
    return a.score > b.score;
  });
  if (topK > 0 && out.size() > static_cast<size_t>(topK))
    out.resize(topK);
  return out;
}

bool TemporalConsistencyVoter::vote(const LoopCandidate* best) {
  if (!best) {
    streak_ = 0;
    have_last_ = false;
    return false;
  }
  const size_t id = best->keyframe_id;
  const bool consistent = have_last_ && (id >= last_id_ ? id - last_id_ : last_id_ - id) <= tol_;
  streak_ = consistent ? streak_ + 1 : 1;
  last_id_ = id;
  have_last_ = true;
  return streak_ >= required_;
}

} // namespace superslam
