#include "SuperSLAM.h"

#include <opencv4/opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>

#include "Logging.h"

namespace superslam {

namespace {
// Bridge YAML tuning knobs to the SUPERSLAM_* env vars their components read. setenv with
// overwrite=0 keeps the precedence env, then YAML, then default. Backend.window_size is read at
// VoEstimator construction.
void apply_tuning_overrides(const YAML::Node& config) {
  auto bridge = [](const YAML::Node& node, const char* env) {
    if (node && !std::getenv(env)) {
      const std::string val = node.as<std::string>();
      setenv(env, val.c_str(), /*overwrite=*/0);
      SLOG_INFO("Config: {} = {} (from YAML)", env, val);
    }
  };
  bridge(config["Backend.max_iters"], "SUPERSLAM_WS_MAX_ITERS");
  bridge(config["Backend.smart_sigma_px"], "SUPERSLAM_SMART_SIGMA_PX");
  bridge(config["Backend.odom_rot_sigma"], "SUPERSLAM_ODOM_ROT_SIGMA");
  bridge(config["Backend.odom_trans_sigma"], "SUPERSLAM_ODOM_TRANS_SIGMA");
  bridge(config["Tracking.min_matches"], "SUPERSLAM_TRACK_MIN_MATCHES");
  bridge(config["Tracking.disp_sigma_px"], "SUPERSLAM_DISP_SIGMA_PX");
  bridge(config["Tracking.cond_depth_m"], "SUPERSLAM_STEREO_COND_DEPTH_M");
  if (config["loop"]) {
    bridge(config["loop"]["min_inliers"], "SUPERSLAM_LOOP_MIN_INLIERS");
    bridge(config["loop"]["min_score"], "SUPERSLAM_LOOP_MIN_SCORE");
  }
}

gtsam::Cal3_S2Stereo read_calib(const YAML::Node& config) {
  const float fx = config["Camera.fx"].as<float>(), fy = config["Camera.fy"].as<float>();
  const float cx = config["Camera.cx"].as<float>(), cy = config["Camera.cy"].as<float>();
  const float bf = config["Camera.bf"].as<float>();
  return gtsam::Cal3_S2Stereo(fx, fy, 0.0, cx, cy,
                              bf / fx); // baseline = bf/fx
}

cv::Mat pose_to_cv_tcw(const gtsam::Pose3& Twc) {
  const gtsam::Pose3 Tcw = Twc.inverse();
  cv::Mat out = cv::Mat::eye(4, 4, CV_32F);
  const gtsam::Matrix3 R = Tcw.rotation().matrix();
  const gtsam::Point3& t = Tcw.translation();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c)
      out.at<float>(r, c) = static_cast<float>(R(r, c));
    out.at<float>(r, 3) = static_cast<float>(t(r));
  }
  return out;
}
} // namespace

SuperSLAM::SuperSLAM(const std::string& config_path, bool use_viewer)
    : K_(0, 0, 0, 0, 0, 0), use_viewer_(use_viewer) {
  const YAML::Node config = YAML::LoadFile(config_path);
  apply_tuning_overrides(config); // Bridge YAML tuning knobs to env (env overrides).
  K_ = read_calib(config);

  const std::string model_dir = config["SuperPoint.model_dir"].as<std::string>();
  const int sp_max_kp = config["superpoint"]["max_keypoints"].as<int>();
  const double sp_thresh = config["superpoint"]["keypoint_threshold"].as<double>();
  const int sp_borders = config["superpoint"]["remove_borders"].as<int>();
  const std::string sp_engine =
      model_dir + "/" + config["superpoint"]["engine_file"].as<std::string>();

  const int lg_w = config["lightglue"]["image_width"].as<int>();
  const int lg_h = config["lightglue"]["image_height"].as<int>();
  const std::string lg_engine =
      model_dir + "/" + config["lightglue"]["engine_file"].as<std::string>();

  // Backend factory. One extractor for left and right; one matcher shared by
  // the front-end and estimator.
  extractor_ = std::make_shared<SuperPoint>(sp_engine, sp_max_kp, sp_thresh, sp_borders);
  if (!extractor_->initialize())
    SLOG_ERROR("SuperPoint init failed");
  matcher_ = std::make_shared<LightGlue>(lg_engine, lg_w, lg_h);
  if (!matcher_->initialize())
    SLOG_ERROR("LightGlue init failed");

  const bool rgbd = static_cast<bool>(config["DepthMapFactor"]);
  if (rgbd) {
    const double depth_factor = config["DepthMapFactor"].as<double>();
    const double max_depth = config["ThDepth"].as<double>(40.0) * K_.baseline();
    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << K_.fx(), 0, K_.px(), 0, K_.fy(), K_.py(), 0, 0, 1);
    cv::Mat dist = (cv::Mat_<double>(1, 5) << config["Camera.k1"].as<double>(0.0),
                    config["Camera.k2"].as<double>(0.0),
                    config["Camera.p1"].as<double>(0.0),
                    config["Camera.p2"].as<double>(0.0),
                    config["Camera.k3"].as<double>(0.0));
    rgbd_frontend_ = std::make_unique<RgbdFrontEnd>(extractor_.get(),
                                                    K_,
                                                    depth_factor,
                                                    max_depth,
                                                    camera_matrix,
                                                    dist);
  } else {
    frontend_ = std::make_unique<StereoFrontEnd>(extractor_.get(), matcher_.get(), K_);
  }
  // Backend.window_size: sliding-window smoother size. 0 or absent uses the built-in default; env
  // SUPERSLAM_WS_WINDOW overrides.
  const int window_size = config["Backend.window_size"].as<int>(0);
  estimator_ = std::make_unique<VoEstimator>(matcher_.get(), K_, window_size);
  estimator_->set_keyframe_params(config["KeyFrame.covis_ratio"].as<double>(0.7),
                                  config["KeyFrame.max_frames"].as<int>(20));

  // Optional pose-graph loop closure (EigenPlaces retrieval and LightGlue
  // verification on a worker thread). Enabled by SUPERSLAM_ENABLE_LOOP=1 and a
  // `loop:` config block.
  const bool want_loop = std::getenv("SUPERSLAM_ENABLE_LOOP") && config["loop"];
  if (want_loop) {
    const std::string ep_engine = model_dir + "/" + config["loop"]["engine_file"].as<std::string>();
    const int ep_w = config["loop"]["image_width"].as<int>(640);
    const int ep_h = config["loop"]["image_height"].as<int>(480);

    auto recognizer = std::make_unique<EigenPlaces>(ep_engine, ep_w, ep_h);
    if (!recognizer->initialize()) {
      SLOG_ERROR("EigenPlaces init failed; loop closure disabled");
    } else {
      // Loop-thread matcher: share the deserialized engine with the tracking matcher (the
      // ICudaEngine is immutable and thread-safe) but keep a separate execution context
      // (execution contexts are not safe to share across threads).
      loop_matcher_ = std::make_shared<LightGlue>(matcher_->shared_engine(), lg_w, lg_h);
      if (!loop_matcher_->initialize()) {
        SLOG_ERROR("Loop matcher init failed; loop closure disabled");
      } else {
        auto K = boost::make_shared<gtsam::Cal3_S2Stereo>(K_);
        auto loop_closer =
            std::make_unique<LoopCloser>(loop_matcher_.get(), K, std::move(recognizer));
        estimator_->enable_loop_closure(std::move(loop_closer),
                                        /*async=*/true);
        loop_enabled_ = true;
      }
    }
  }

  if (use_viewer_)
    viewer_ = std::make_unique<RerunViewer>();
  const char* sensor = rgbd ? "RGB-D" : "stereo";
  SLOG_INFO("SuperSLAM ready ({} {}; baseline {:.4f} m)",
            sensor,
            loop_enabled_ ? "SLAM + loop closure" : "VO",
            K_.baseline());
}

cv::Mat SuperSLAM::track_stereo(const cv::Mat& left, const cv::Mat& right, double timestamp) {
  cv::Mat gray_left = left, gray_right = right;
  if (left.channels() == 3)
    cv::cvtColor(left, gray_left, cv::COLOR_BGR2GRAY);
  if (right.channels() == 3)
    cv::cvtColor(right, gray_right, cv::COLOR_BGR2GRAY);

  StereoFrame f = frontend_->process(gray_left, gray_right, timestamp);
  // Pass the left image for the keyframe EigenPlaces global descriptor
  // (ignored when loop closure is disabled).
  const gtsam::Pose3 Twc = estimator_->track(f, loop_enabled_ ? gray_left : cv::Mat());
  trajectory_.push_back(Twc);
  timestamps_.push_back(timestamp);
  if (viewer_)
    viewer_->draw_frame(f, trajectory_, K_);
  return pose_to_cv_tcw(Twc);
}

cv::Mat SuperSLAM::track_rgbd(const cv::Mat& rgb, const cv::Mat& depth, double timestamp) {
  cv::Mat gray = rgb;
  if (rgb.channels() == 3)
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

  StereoFrame f = rgbd_frontend_->process(gray, depth, timestamp);
  const gtsam::Pose3 Twc = estimator_->track(f, loop_enabled_ ? gray : cv::Mat());
  trajectory_.push_back(Twc);
  timestamps_.push_back(timestamp);
  if (viewer_)
    viewer_->draw_frame(f, trajectory_, K_);
  return pose_to_cv_tcw(Twc);
}

size_t SuperSLAM::loop_closure_count() const {
  return estimator_ ? estimator_->loop_closure_count() : 0;
}

void SuperSLAM::save_trajectory(const std::string& path, TrajectoryFormat format) const {
  if (!estimator_)
    return;
  estimator_->stop_loop_worker();
  const std::vector<gtsam::Pose3> corrected = estimator_->corrected_trajectory();

  std::ofstream f(path);
  f << std::fixed << std::setprecision(9);
  if (format == TrajectoryFormat::KITTI) {
    // Camera-to-world 3x4 (Rwc | twc) = Twc, row-major, one pose per line.
    for (const gtsam::Pose3& Twc : corrected) {
      const gtsam::Matrix3 R = Twc.rotation().matrix();
      const gtsam::Point3& t = Twc.translation();
      f << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << t(0) << " " << R(1, 0) << " "
        << R(1, 1) << " " << R(1, 2) << " " << t(1) << " " << R(2, 0) << " " << R(2, 1) << " "
        << R(2, 2) << " " << t(2) << "\n";
    }
  } else {
    // TUM: timestamp tx ty tz qx qy qz qw (Twc).
    for (size_t i = 0; i < corrected.size(); ++i) {
      const gtsam::Point3& t = corrected[i].translation();
      const Eigen::Quaternion<double> q = corrected[i].rotation().toQuaternion();
      const double ts = i < timestamps_.size() ? timestamps_[i] : static_cast<double>(i);
      f << ts << " " << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " "
        << q.z() << " " << q.w() << "\n";
    }
  }
  SLOG_INFO("trajectory saved! ({} poses)", corrected.size());
}

void SuperSLAM::save_map(const std::string& path) const {
  if (!estimator_)
    return;
  estimator_->stop_loop_worker();
  const auto pts = estimator_->map().cloud(estimator_->anchors());

  std::ofstream f(path);
  if (!f) {
    SLOG_ERROR("save_map: cannot open {}", path);
    return;
  }
  f << std::fixed << std::setprecision(6);
  for (const gtsam::Point3& p : pts)
    f << p.x() << " " << p.y() << " " << p.z() << "\n";
  SLOG_INFO("map saved! ({} points)", pts.size());
}

void SuperSLAM::save_trajectory_kitti(const std::string& path) const {
  save_trajectory(path, TrajectoryFormat::KITTI);
}

} // namespace superslam
