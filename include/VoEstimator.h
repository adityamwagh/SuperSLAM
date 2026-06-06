#pragma once
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <opencv4/opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "FrameTracker.h"
#include "GlobalPoseGraph.h"
#include "InferenceInterfaces.h"
#include "KeyframeGate.h"
#include "LoopCloser.h"
#include "SparseMap.h"
#include "StereoFrame.h"
#include "WindowSmoother.h"

namespace superslam {

// Orchestrate sliding-window stereo VO, optionally upgraded to SLAM with
// pose-graph loop closure: StereoFrame to per-frame tracker to keyframe gate
// to batch smart-stereo window (Tier 1). When loop closure is enabled, hand
// each keyframe to a batch-LM global pose graph (Tier 2) and a LoopCloser on a
// worker thread. An accepted loop updates the keyframe anchors used to compose
// the live and saved poses. Never rebase tracking or the window. Pose-only, no
// map. Twc throughout.
class VoEstimator {
public:
  // window_size: sliding-window smoother size (0 = use the built-in default).
  // Precedence: env SUPERSLAM_WS_WINDOW, then this arg (if greater than 0), then default.
  VoEstimator(IFeatureMatcher* matcher, const gtsam::Cal3_S2Stereo& K, int window_size = 0);
  ~VoEstimator();

  // Turn on pose-graph loop closure. The LoopCloser must own a DEDICATED
  // matcher, not the tracking matcher. TensorRT contexts are not safe for
  // concurrent use across the tracking and loop threads. With async=false,
  // process loops inline.
  void enable_loop_closure(std::unique_ptr<LoopCloser> loop_closer, bool async = true);

  // Estimate the frame's world pose (Twc) and write it to frame.pose.
  // left_gray is the left image used to compute the keyframe's global
  // descriptor for loop retrieval. Pass it when loop closure is enabled
  // (ignored otherwise).
  gtsam::Pose3 track(StereoFrame& frame, const cv::Mat& left_gray = cv::Mat());

  // Return the number of accepted loop closures so far.
  size_t loop_closure_count() const { return loop_count_.load(); }

  // Drain and join the async loop-closure worker. Idempotent: safe to call
  // more than once. Call before reading corrected_trajectory(), anchors(), or
  // map() from outside the tracking thread.
  void stop_loop_worker();

  // Per-frame corrected trajectory: anchor[ref_keyframe_id] composed with the
  // keyframe-to-frame transform for every tracked frame. Call only after the
  // async worker is joined (for example after stop_loop_worker()).
  std::vector<gtsam::Pose3> corrected_trajectory() const;

  // Return the sparse point-cloud map: camera-frame points indexed by keyframe
  // id. Call only after stop_loop_worker() returns.
  const SparseMap& map() const { return map_; }

  // Return loop-corrected keyframe anchors. Fall back to the local VO poses
  // seeded at insert if no loop has fired yet. Call only after
  // stop_loop_worker() returns.
  std::map<size_t, gtsam::Pose3> anchors() const;

  // Covisibility keyframe gate: insert when the tracked-feature ratio drops
  // below covisibility_ratio, or after max_frames. Scale-invariant.
  void set_keyframe_params(double covisibility_ratio, int max_frames) {
    covisibility_ratio_ = covisibility_ratio;
    max_keyframe_frames_ = max_frames;
  }

private:
  std::vector<StereoObs> collect_stereo_obs(const StereoFrame& frame,
                                            const std::map<int, size_t>& feature_to_landmark);
  std::vector<gtsam::Point3> backproject_stereo(const StereoFrame& frame) const;

  // Loop-closure plumbing.
  struct KeyframeMsg {
    size_t keyframe_id = 0;
    size_t previous_keyframe_id = 0;
    bool has_previous = false;
    gtsam::Pose3 pose;              // window-optimized keyframe pose at creation
    gtsam::Pose3 relative_odometry; // prevKf.between(kf) from the joint window solve
    KeyframeRecord record; // local features and stereo (global_descriptor filled on worker)
    cv::Mat left_gray;     // for the global descriptor (computed on the worker)
  };
  KeyframeMsg
  make_keyframe_msg(size_t keyframe_id, const StereoFrame& frame, const cv::Mat& left_gray) const;
  void submit_keyframe(KeyframeMsg msg);  // enqueue (async) or process inline (sync)
  void process_keyframe(KeyframeMsg msg); // worker body: global graph and loop detect
  void worker_loop();

  IFeatureMatcher* matcher_;
  gtsam::Cal3_S2Stereo::shared_ptr K_;
  WindowSmoother smoother_;
  FrameTracker tracker_;

  bool has_keyframe_ = false;
  size_t last_keyframe_id_ = 0;
  gtsam::Pose3 last_keyframe_pose_;
  gtsam::Pose3 previous_frame_pose_;
  gtsam::Pose3 previous_relative_; // last accepted frame-to-frame motion
                                   // (constant-velocity fallback)
  int frames_since_keyframe_ = 0;
  double covisibility_ratio_ = 0.8;
  int max_keyframe_frames_ = 20;
  StereoFrame last_keyframe_;
  std::map<int, size_t> last_keyframe_feature_to_landmark_; // last KF feature idx to landmark id

  size_t next_keyframe_id_ = 0;
  size_t global_landmark_id_ = 0;

  // Loop closure (Tier 2), worker-thread owned except the
  // correction_mutex_-guarded anchors_.
  bool loop_enabled_ = false;
  bool loop_async_ = true;
  std::unique_ptr<GlobalPoseGraph> global_graph_;
  std::unique_ptr<LoopCloser> loop_closer_;

  std::thread worker_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<KeyframeMsg> queue_;
  bool stop_worker_ = false;

  std::mutex correction_mutex_;
  std::map<size_t, gtsam::Pose3> anchors_; // corrected KF poses (refreshed on each accepted loop)
  std::vector<std::pair<size_t, gtsam::Pose3>>
      frame_records_; // (ref_keyframe_id, keyframe-to-frame transform)
  std::atomic<size_t> loop_count_{0};

  SparseMap map_;                               // camera-frame points per keyframe
  std::map<size_t, gtsam::Pose3> seed_anchors_; // local VO pose at KF insert (fallback for cloud())
};

} // namespace superslam
