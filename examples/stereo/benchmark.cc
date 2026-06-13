/**
 * This file is part of SuperSLAM.
 *
 * Copyright (C) Aditya Wagh <adityamwagh at outlook dot com>
 * For more information see <https://github.com/adityamwagh/SuperSLAM>
 */

// Real-time benchmark: runs the full stereo SLAM pipeline over a KITTI
// sequence and reports per-frame latency (mean/median/p95/max) + sustained
// FPS, plus the number of loop closures. The source of truth for "does it hold
// KITTI's 10 fps" -- planning latencies were estimates; measure here on the
// target GPU.
//
// Loop closure is on when SUPERSLAM_ENABLE_LOOP=1 and the config has a `loop:`
// block.
//   examples/benchmark --sequence ~/datasets/kitti/dataset/sequences/00
//   --settings <cfg>.yaml

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "SuperSLAM.h"
#include "example_common.h"

namespace {
float percentile(std::vector<float> v, double p) {
  if (v.empty())
    return 0.0f;
  std::sort(v.begin(), v.end());
  const size_t idx = std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1)));
  return v[idx];
}
} // namespace

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);

  superslam::ExampleArgs args;
  if (!superslam::parse_example_args(argc, argv, "benchmark", args))
    return 1;
  args.use_viewer = false; // benchmark never paces / draws

  const std::string times_file = args.sequence + "/times.txt";
  const std::string left_dir = args.sequence + "/image_0/";
  const std::string right_dir = args.sequence + "/image_1/";

  std::vector<double> ts;
  {
    std::ifstream f(times_file);
    std::string s;
    while (std::getline(f, s) && !s.empty())
      ts.push_back(std::stod(s));
  }
  if (ts.empty()) {
    spdlog::error("No timestamps in {}", times_file);
    return 1;
  }

  superslam::SuperSLAM slam(args.settings, /*use_viewer=*/false);

  std::vector<float> ms;
  ms.reserve(ts.size());
  const auto wall0 = std::chrono::steady_clock::now();
  for (size_t ni = 0; ni < ts.size(); ++ni) {
    std::ostringstream name;
    name << std::setfill('0') << std::setw(6) << ni << ".png";
    cv::Mat l = cv::imread(left_dir + name.str(), cv::IMREAD_UNCHANGED);
    cv::Mat r = cv::imread(right_dir + name.str(), cv::IMREAD_UNCHANGED);
    if (l.empty() || r.empty())
      continue;

    const auto t1 = std::chrono::steady_clock::now();
    slam.track_stereo(l, r, ts[ni]);
    const auto t2 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f);
  }
  const double wall_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - wall0)
                              .count() /
                          1000.0;

  slam.shutdown();
  slam.save_trajectory_kitti("CameraTrajectory_benchmark.txt");

  const double mean = ms.empty() ? 0.0 : std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
  spdlog::info("=========== SuperSLAM benchmark ===========");
  spdlog::info("frames           : {}", ms.size());
  spdlog::info("per-frame ms      mean={:.2f} p50={:.2f} p95={:.2f} max={:.2f}",
               mean,
               percentile(ms, 0.50),
               percentile(ms, 0.95),
               percentile(ms, 1.0));
  spdlog::info("throughput        : {:.2f} fps over {:.1f}s wall", ms.size() / wall_sec, wall_sec);
  spdlog::info("real-time (>=10fps): {}", mean > 0 && (1000.0 / mean) >= 10.0 ? "YES" : "NO");
  spdlog::info("loop closures     : {}", slam.loop_closure_count());
  spdlog::info("===========================================");
  return 0;
}
