#ifndef SUPERSLAM_LOGGING_H
#define SUPERSLAM_LOGGING_H

#include <spdlog/spdlog.h>

namespace SuperSLAM {

class Logger {
 public:
  static void initialize();
  static std::shared_ptr<spdlog::logger> getLogger();

 private:
  static std::shared_ptr<spdlog::logger> logger_;
  static bool initialized_;
};

}  // namespace SuperSLAM

// Convenience macros for logging
#define SLOG_TRACE(...) SuperSLAM::Logger::getLogger()->trace(__VA_ARGS__)
#define SLOG_DEBUG(...) SuperSLAM::Logger::getLogger()->debug(__VA_ARGS__)
#define SLOG_INFO(...) SuperSLAM::Logger::getLogger()->info(__VA_ARGS__)
#define SLOG_WARN(...) SuperSLAM::Logger::getLogger()->warn(__VA_ARGS__)
#define SLOG_ERROR(...) SuperSLAM::Logger::getLogger()->error(__VA_ARGS__)
#define SLOG_CRITICAL(...) SuperSLAM::Logger::getLogger()->critical(__VA_ARGS__)

#endif  // SUPERSLAM_LOGGING_H
