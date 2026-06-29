#pragma once
// Env-gated per-stage profiler. Enable with SUPERSLAM_PROFILE=1. Accumulate wall-clock per
// named scope and dump mean and total at program exit.
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

#include "Logging.h"

namespace superslam {

class Profiler {
public:
  static Profiler& instance() {
    static Profiler p;
    return p;
  }
  static bool enabled() {
    static const bool e = std::getenv("SUPERSLAM_PROFILE") != nullptr;
    return e;
  }
  void add(const std::string& key, double ms) {
    std::lock_guard<std::mutex> g(m_);
    auto& s = stats_[key];
    s.total_ms += ms;
    ++s.n;
  }
  void dump() {
    std::lock_guard<std::mutex> g(m_);
    SLOG_INFO("[profile] ---- per-stage (SUPERSLAM_PROFILE) ----");
    for (const auto& [key, s] : stats_)
      SLOG_INFO("[profile] {:22s} mean={:7.3f}ms  n={:6d}  total={:8.1f}ms",
                key,
                s.n ? s.total_ms / s.n : 0.0,
                s.n,
                s.total_ms);
  }
  ~Profiler() {
    if (enabled())
      dump();
  }

private:
  struct Stat {
    double total_ms = 0.0;
    long n = 0;
  };
  std::map<std::string, Stat> stats_;
  std::mutex m_;
};

class ScopedTimer {
public:
  explicit ScopedTimer(const char* label) : label_(label), t0_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    if (!Profiler::enabled())
      return;
    const auto t1 = std::chrono::steady_clock::now();
    Profiler::instance().add(label_, std::chrono::duration<double, std::milli>(t1 - t0_).count());
  }

private:
  const char* label_;
  std::chrono::steady_clock::time_point t0_;
};

} // namespace superslam

#define SUPERSLAM_PROF_CONCAT_(a, b) a##b
#define SUPERSLAM_PROF_NAME_(line) SUPERSLAM_PROF_CONCAT_(_prof_, line)
#define SUPERSLAM_PROFILE_SCOPE(label)                                                             \
  ::superslam::ScopedTimer SUPERSLAM_PROF_NAME_(__LINE__)(label)
