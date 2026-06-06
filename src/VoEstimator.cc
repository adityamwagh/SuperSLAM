#include "VoEstimator.h"

#include "Profiling.h"

#include <gtsam/linear/NoiseModel.h>

#include <cstdio>
#include <cstdlib>

namespace superslam {

namespace {
constexpr size_t default_window_size = 8;

double env_double(const char* key, double fallback) {
  const char* v = std::getenv(key);
  return v ? std::atof(v) : fallback;
}

// Resolve the window size: prefer env SUPERSLAM_WS_WINDOW, then YAML cfg when greater than 0,
// otherwise the default.
size_t resolve_window_size(int cfg) {
  const char* v = std::getenv("SUPERSLAM_WS_WINDOW");
  if (v)
    return static_cast<size_t>(std::atoi(v));
  if (cfg > 0)
    return static_cast<size_t>(cfg);
  return default_window_size;
}

// Return odometry edge noise for the global pose graph. Keyframe to keyframe
// relatives come from the jointly optimized window.
gtsam::SharedNoiseModel odometry_noise() {
  const double r = env_double("SUPERSLAM_ODOM_ROT_SIGMA", 0.02);
  const double t = env_double("SUPERSLAM_ODOM_TRANS_SIGMA", 0.05);
  return gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << r, r, r, t, t, t).finished());
}
} // namespace

VoEstimator::VoEstimator(IFeatureMatcher* matcher, const gtsam::Cal3_S2Stereo& K, int window_size)
    : matcher_(matcher), K_(boost::make_shared<gtsam::Cal3_S2Stereo>(K)),
      smoother_(K_, resolve_window_size(window_size)), tracker_(K_) {}

VoEstimator::~VoEstimator() {
  stop_loop_worker();
}

void VoEstimator::stop_loop_worker() {
  if (!worker_.joinable())
    return;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    stop_worker_ = true;
  }
  queue_cv_.notify_all();
  worker_.join();
}

void VoEstimator::enable_loop_closure(std::unique_ptr<LoopCloser> loop_closer, bool async) {
  global_graph_ = std::make_unique<GlobalPoseGraph>();
  loop_closer_ = std::move(loop_closer);
  loop_enabled_ = true;
  loop_async_ = async;
  if (async)
    worker_ = std::thread(&VoEstimator::worker_loop, this);
}

std::vector<StereoObs>
VoEstimator::collect_stereo_obs(const StereoFrame& frame,
                                const std::map<int, size_t>& feature_to_landmark) {
  std::vector<StereoObs> obs;
  const int n = static_cast<int>(frame.keypoints_left.size());
  for (int i = 0; i < n; ++i)
    if (frame.has_depth[i])
      obs.push_back({feature_to_landmark.at(i), frame.stereo[i]});
  return obs;
}

std::vector<gtsam::Point3> VoEstimator::backproject_stereo(const StereoFrame& frame) const {
  std::vector<gtsam::Point3> pts;
  const int n = static_cast<int>(frame.keypoints_left.size());
  for (int i = 0; i < n; ++i) {
    if (!frame.has_depth[i])
      continue;
    const gtsam::StereoPoint2& stereo_point = frame.stereo[i];
    const double Z = K_->fx() * K_->baseline() / (stereo_point.uL() - stereo_point.uR());
    const double X = (stereo_point.uL() - K_->px()) * Z / K_->fx();
    const double Y = (stereo_point.v() - K_->py()) * Z / K_->fy();
    pts.emplace_back(X, Y, Z);
  }
  return pts;
}

VoEstimator::KeyframeMsg VoEstimator::make_keyframe_msg(size_t keyframe_id,
                                                        const StereoFrame& frame,
                                                        const cv::Mat& left_gray) const {
  KeyframeMsg keyframe_msg;
  keyframe_msg.keyframe_id = keyframe_id;
  keyframe_msg.pose = frame.pose;
  KeyframeRecord& r = keyframe_msg.record;
  r.keyframe_id = keyframe_id;
  r.timestamp = frame.timestamp;
  r.pose_at_insert = frame.pose;
  r.keypoints_left = frame.keypoints_left;
  // Convert device descriptors to a host cv::Mat for the host-resident loop database.
  r.descriptors_left = matcher_->descriptors_to_host(frame.descriptors_left);
  r.stereo = frame.stereo;
  r.has_depth = frame.has_depth;
  keyframe_msg.left_gray = left_gray.empty() ? cv::Mat() : left_gray.clone();
  return keyframe_msg;
}

void VoEstimator::submit_keyframe(KeyframeMsg msg) {
  if (loop_async_) {
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      queue_.push_back(std::move(msg));
    }
    queue_cv_.notify_one();
  } else {
    process_keyframe(std::move(msg));
  }
}

void VoEstimator::worker_loop() {
  for (;;) {
    KeyframeMsg keyframe_msg;
    {
      std::unique_lock<std::mutex> lk(queue_mutex_);
      queue_cv_.wait(lk, [this] { return stop_worker_ || !queue_.empty(); });
      if (stop_worker_ && queue_.empty())
        return;
      keyframe_msg = std::move(queue_.front());
      queue_.pop_front();
    }
    process_keyframe(std::move(keyframe_msg));
  }
}

void VoEstimator::process_keyframe(KeyframeMsg msg) {
  // Add a tier-2 node and odometry edge.
  global_graph_->add_keyframe(msg.keyframe_id,
                              msg.pose,
                              /*is_first=*/!msg.has_previous);
  if (msg.has_previous)
    global_graph_->add_odometry(msg.previous_keyframe_id,
                                msg.keyframe_id,
                                msg.relative_odometry,
                                odometry_noise());

  // Index for retrieval. Skip when no image is provided.
  LoopResult loop_result;
  if (!msg.left_gray.empty()) {
    msg.record.global_descriptor = loop_closer_->compute_global_descriptor(msg.left_gray);
    loop_closer_->add_keyframe(msg.record);
    loop_result = loop_closer_->detect(msg.record);
  }

  if (!loop_result.accepted)
    return; // No loop; the odometry edge is recorded.

  global_graph_->add_loop(loop_result.matched_keyframe,
                          msg.keyframe_id,
                          loop_result.relative_pose,
                          loop_result.noise);

  const std::map<size_t, gtsam::Pose3> corrected = global_graph_->optimize_and_get_all();
  if (!global_graph_->last_loop_rejected()) {
    loop_count_.fetch_add(1);
    std::lock_guard<std::mutex> lk(correction_mutex_);
    anchors_ = corrected;
  }
}

std::map<size_t, gtsam::Pose3> VoEstimator::anchors() const {
  // Read only after the async worker is joined, with no concurrent writes to
  // anchors_.
  return anchors_.empty() ? seed_anchors_ : anchors_;
}

std::vector<gtsam::Pose3> VoEstimator::corrected_trajectory() const {
  std::vector<gtsam::Pose3> out;
  out.reserve(frame_records_.size());
  for (const auto& [ref_keyframe_id, rel] : frame_records_) {
    // Priority: loop-corrected anchor, then VO seed anchor at keyframe insert,
    // then identity fallback. seed_anchors_ holds the global VO pose at keyframe
    // creation; anchor*rel reproduces the live VO global pose exactly when no
    // loop has fired.
    gtsam::Pose3 anchor;
    auto it = anchors_.find(ref_keyframe_id);
    if (it != anchors_.end()) {
      anchor = it->second;
    } else {
      auto it2 = seed_anchors_.find(ref_keyframe_id);
      if (it2 != seed_anchors_.end())
        anchor = it2->second;
    }
    out.push_back(anchor * rel);
  }
  return out;
}

gtsam::Pose3 VoEstimator::track(StereoFrame& frame, const cv::Mat& left_gray) {
  SUPERSLAM_PROFILE_SCOPE("vo_track_total");
  const int n = static_cast<int>(frame.keypoints_left.size());

  // First frame: keyframe at origin; stereo sets metric scale.
  if (!has_keyframe_) {
    const gtsam::Pose3 origin;
    frame.pose = origin;
    std::map<int, size_t> feature_to_landmark;
    for (int i = 0; i < n; ++i)
      if (frame.has_depth[i])
        feature_to_landmark[i] = global_landmark_id_++;

    smoother_.add_keyframe(next_keyframe_id_,
                           origin,
                           collect_stereo_obs(frame, feature_to_landmark));
    last_keyframe_id_ = next_keyframe_id_++;
    last_keyframe_pose_ = origin;
    previous_frame_pose_ = origin;
    last_keyframe_ = frame;
    last_keyframe_feature_to_landmark_ = feature_to_landmark;
    has_keyframe_ = true;

    map_.add_keyframe(last_keyframe_id_, backproject_stereo(frame));
    seed_anchors_[last_keyframe_id_] = origin;

    if (loop_enabled_) {
      KeyframeMsg keyframe_msg = make_keyframe_msg(last_keyframe_id_, frame, left_gray);
      keyframe_msg.has_previous = false;
      submit_keyframe(std::move(keyframe_msg));
    }
    frame_records_.emplace_back(last_keyframe_id_, gtsam::Pose3{});
    return origin;
  }

  // Match the current frame to the last keyframe (queryIdx=last_keyframe_,
  // trainIdx=frame).
  MatchResult match_result;
  {
    SUPERSLAM_PROFILE_SCOPE("vo_lg_track_match");
    match_result = matcher_->match(last_keyframe_.keypoints_left,
                                   last_keyframe_.descriptors_left,
                                   frame.keypoints_left,
                                   frame.descriptors_left);
  }

  std::vector<PointObs> matches;                // Input for the per-frame tracker.
  std::map<int, size_t> frame_matched_landmark; // frame feature index to landmark id (from
                                                // last_keyframe_)
  for (const cv::DMatch& m : match_result.matches) {
    const int keyframe_index = m.queryIdx, frame_index = m.trainIdx;
    if (keyframe_index < 0 || frame_index < 0 ||
        keyframe_index >= static_cast<int>(last_keyframe_.keypoints_left.size()) ||
        frame_index >= n)
      continue;
    if (!last_keyframe_.has_depth[keyframe_index])
      continue; // Need a triangulated 3D point.
    if (!frame.has_depth[frame_index])
      continue; // Need a stereo measurement here.
    matches.push_back({last_keyframe_.backproject(keyframe_index, *K_), frame.stereo[frame_index]});
    auto it = last_keyframe_feature_to_landmark_.find(keyframe_index);
    if (it != last_keyframe_feature_to_landmark_.end())
      frame_matched_landmark[frame_index] = it->second; // Carry the landmark id.
  }

  // Per-frame quick pose (seed with the previous frame pose; pose-only LM,
  // CPU).
  gtsam::Pose3 frame_pose = tracker_.track(previous_frame_pose_, matches);

  // Reject a degenerate solve. Too few correspondences (motion blur or tracking
  // loss) leave the pose-only LM unsupported and it can teleport; coast on the
  // last accepted relative motion.
  const int min_matches = static_cast<int>(env_double("SUPERSLAM_TRACK_MIN_MATCHES", 10));
  if (static_cast<int>(matches.size()) < min_matches) {
    frame_pose = previous_frame_pose_ * previous_relative_;
  } else {
    previous_relative_ = previous_frame_pose_.between(frame_pose);
  }

  if (std::getenv("SUPERSLAM_VO_DEBUG")) {
    const gtsam::Pose3 rel_to_keyframe = last_keyframe_pose_.between(frame_pose);
    std::fprintf(stderr,
                 "[trk] nmatch=%zu lastKf|t|=%.2f seed|t|=%.2f res|t|=%.2f "
                 "relKf|t|=%.2f\n",
                 matches.size(),
                 last_keyframe_pose_.translation().norm(),
                 previous_frame_pose_.translation().norm(),
                 frame_pose.translation().norm(),
                 rel_to_keyframe.translation().norm());
  }

  // On a keyframe, insert into the window and re-optimize.
  ++frames_since_keyframe_;
  const double covis = env_double("SUPERSLAM_KF_COVIS", covisibility_ratio_);
  const int reference_features = static_cast<int>(last_keyframe_feature_to_landmark_.size());
  if (should_insert_keyframe(static_cast<int>(matches.size()),
                             reference_features,
                             frames_since_keyframe_,
                             covis,
                             max_keyframe_frames_)) {
    frames_since_keyframe_ = 0;
    const size_t previous_keyframe_id = last_keyframe_id_;
    const size_t keyframe_id = next_keyframe_id_++;
    // Landmark ids: matched features reuse the last keyframe's id; unmatched
    // stereo mint new ids.
    std::map<int, size_t> feature_to_landmark;
    for (int i = 0; i < n; ++i) {
      if (!frame.has_depth[i])
        continue;
      auto it = frame_matched_landmark.find(i);
      feature_to_landmark[i] =
          (it != frame_matched_landmark.end()) ? it->second : global_landmark_id_++;
    }
    smoother_.add_keyframe(keyframe_id, frame_pose, collect_stereo_obs(frame, feature_to_landmark));
    if (!std::getenv("SUPERSLAM_VO_NO_SMOOTHER")) {
      {
        SUPERSLAM_PROFILE_SCOPE("vo_gtsam_optimize");
        smoother_.optimize(); // Batch window solve (CPU), keyframes only.
      }
      frame_pose = smoother_.pose_of(keyframe_id); // Corrected by the window.
    }
    last_keyframe_id_ = keyframe_id;
    last_keyframe_pose_ = frame_pose;
    last_keyframe_feature_to_landmark_ = feature_to_landmark;
    last_keyframe_ = frame;
    last_keyframe_.pose = frame_pose; // For backproject on the next frame (Twc).

    map_.add_keyframe(keyframe_id, backproject_stereo(frame));
    seed_anchors_[keyframe_id] = frame_pose;

    if (loop_enabled_) {
      KeyframeMsg keyframe_msg = make_keyframe_msg(keyframe_id, last_keyframe_, left_gray);
      keyframe_msg.has_previous = true;
      keyframe_msg.previous_keyframe_id = previous_keyframe_id;
      if (smoother_.in_window(previous_keyframe_id) && smoother_.in_window(keyframe_id))
        keyframe_msg.relative_odometry =
            smoother_.pose_of(previous_keyframe_id).between(smoother_.pose_of(keyframe_id));
      submit_keyframe(std::move(keyframe_msg));
    }
  }

  previous_frame_pose_ = frame_pose;
  const gtsam::Pose3 rel_pose = last_keyframe_pose_.inverse() * frame_pose;
  frame_records_.emplace_back(last_keyframe_id_, rel_pose);
  gtsam::Pose3 anchor = last_keyframe_pose_;
  {
    std::lock_guard<std::mutex> lk(correction_mutex_);
    auto it = anchors_.find(last_keyframe_id_);
    if (it != anchors_.end())
      anchor = it->second;
  }
  const gtsam::Pose3 live = anchor * rel_pose;
  frame.pose = live;
  return live;
}

} // namespace superslam
