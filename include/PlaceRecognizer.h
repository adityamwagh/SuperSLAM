#pragma once
#include <opencv4/opencv2/core.hpp>

#include <cstddef>
#include <deque>
#include <vector>

namespace superslam {

// Hold a retrieved place-recognition match: a database keyframe and its
// similarity score.
struct LoopCandidate {
  size_t keyframe_id = 0;
  float score = 0.0f; // cosine similarity in [-1, 1]
};

// Pluggable place recognition. The descriptor source (a learned
// global-descriptor model) lives behind this interface; CosineDescriptorIndex
// holds the shared retrieval math. Mirror IFeatureExtractor.
class IPlaceRecognizer {
public:
  virtual ~IPlaceRecognizer() = default;

  // Map an image to an L2-normalized global descriptor [1 x Dg].
  virtual cv::Mat compute_global_descriptor(const cv::Mat& image) = 0;

  // Index a keyframe's descriptor for future queries.
  virtual void add(size_t keyframe_id, const cv::Mat& global_descriptor) = 0;

  // Rank database keyframes by similarity to global_descriptor. excludeRecent
  // skips the most recently added keyframes (temporal neighbours that always
  // self-match); topK caps the returned candidates. Sort returned candidates
  // by descending score.
  virtual std::vector<LoopCandidate>
  query(const cv::Mat& global_descriptor, size_t excludeRecent, int topK) = 0;
};

// Source-agnostic cosine-similarity index over L2-normalized global
// descriptors. Both the real EigenPlaces recognizer and the unit-test stub
// delegate add and query here; tests cover the retrieval logic
// (excludeRecent window, top-K, score) without a GPU. Flat GEMM scan: O(M) per
// query.
class CosineDescriptorIndex {
public:
  // Append a descriptor. Insertion order defines recency (used by
  // excludeRecent).
  void add(size_t keyframe_id, const cv::Mat& global_descriptor);

  // Return the top-K most similar entries, excluding the last excludeRecent
  // insertions and any score below minScore. Descending score.
  std::vector<LoopCandidate>
  query(const cv::Mat& global_descriptor, size_t excludeRecent, int topK, float minScore) const;

  size_t size() const { return ids_.size(); }

private:
  std::vector<size_t> ids_; // keyframe_id per row, insertion order
  cv::Mat db_;              // [M x Dg], one L2-normalized descriptor per row
};

// Debounce loops: accept a place match only after K consecutive queries agree
// on the same locale. Defeat one-off perceptual-aliasing false positives.
// Same locale = matched keyframe ids within idTolerance of each other across
// consecutive votes.
class TemporalConsistencyVoter {
public:
  TemporalConsistencyVoter(int requiredVotes, size_t idTolerance)
      : required_(requiredVotes), tol_(idTolerance) {}

  // Feed the best candidate of the current query (or nullptr if none passed
  // the gate). Return true once `requiredVotes` consecutive,
  // spatially-consistent matches land.
  bool vote(const LoopCandidate* best);

private:
  int required_;
  size_t tol_;
  int streak_ = 0;
  size_t last_id_ = 0;
  bool have_last_ = false;
};

} // namespace superslam
