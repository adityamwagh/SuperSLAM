/**
 * This file is part of SuperSLAM.
 *
 * Copyright (C) Aditya Wagh <adityamwagh at outlook dot com>
 * For more information see <https://github.com/adityamwagh/SuperSLAM>
 *
 * SuperSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SuperSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SuperSLAM. If not, see <http://www.gnu.org/licenses/>.
 */

// Shared CLI/timing helpers for the dataset examples (kitti, euroc, tum). Each
// example takes the same form:
//
//   ./<dataset> <settings.yaml> <sequence_dir> [--no-viewer]
//
// The Rerun viewer (cameras, map points, trajectory) is on by default; pass
// --no-viewer for a headless run.

#ifndef SUPERSLAM_EXAMPLE_COMMON_H
#define SUPERSLAM_EXAMPLE_COMMON_H

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace superslam {

struct ExampleArgs {
  std::string settings;
  std::string sequence;
  bool use_viewer = true;
};

// Parse `<settings.yaml> <sequence_dir> [--no-viewer|--viewer]`. Returns false
// (and prints usage) on malformed input.
inline bool
parse_example_args(int argc, char** argv, const std::string& dataset, ExampleArgs& out) {
  std::vector<std::string> positionals;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--no-viewer") {
      out.use_viewer = false;
    } else if (a == "--viewer") {
      out.use_viewer = true;
    } else {
      positionals.push_back(a);
    }
  }

  if (positionals.size() != 2) {
    spdlog::error("Usage: ./{} <settings.yaml> <sequence_dir> [--no-viewer]", dataset);
    return false;
  }

  out.settings = positionals[0];
  out.sequence = positionals[1];
  if (!std::ifstream(out.settings).good()) {
    spdlog::error("Cannot open settings file: {}", out.settings);
    return false;
  }

  spdlog::info("[{}] settings={} sequence={} viewer={}",
               dataset,
               out.settings,
               out.sequence,
               out.use_viewer ? "on" : "off");
  return true;
}

// Sleep so a visualized run plays back near the dataset frame rate. `dt` is
// the inter-frame time (s); `track_ms` is the time already spent tracking this
// frame.
inline void pace_to_timestamp(float track_ms, double dt) {
  const auto min_delay = std::chrono::milliseconds(10);
  const double remain_ms = dt * 1000.0 - track_ms;
  if (remain_ms > 10.0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long long>(remain_ms)));
  } else {
    std::this_thread::sleep_for(min_delay);
  }
}

inline void report_timing(std::vector<float> times) {
  if (times.empty())
    return;
  std::sort(times.begin(), times.end());
  const float total = std::accumulate(times.begin(), times.end(), 0.0f);
  spdlog::info("=== Tracking time: mean {:.2f}ms  median {:.2f}ms  over {} frames ===",
               total / times.size(),
               times[times.size() / 2],
               times.size());
}

} // namespace superslam

#endif // SUPERSLAM_EXAMPLE_COMMON_H
