#ifndef SUPERSLAM_LOGGING_H
#define SUPERSLAM_LOGGING_H

#include <spdlog/spdlog.h>

namespace superslam {

class Logger {
public:
  static void initialize();
  static std::shared_ptr<spdlog::logger> getLogger();

private:
  static std::shared_ptr<spdlog::logger> logger_;
  static bool initialized_;
};

} // namespace superslam

// Convenience macros for logging
#define SLOG_TRACE(...) superslam::Logger::getLogger()->trace(__VA_ARGS__)
#define SLOG_DEBUG(...) superslam::Logger::getLogger()->debug(__VA_ARGS__)
#define SLOG_INFO(...) superslam::Logger::getLogger()->info(__VA_ARGS__)
#define SLOG_WARN(...) superslam::Logger::getLogger()->warn(__VA_ARGS__)
#define SLOG_ERROR(...) superslam::Logger::getLogger()->error(__VA_ARGS__)
#define SLOG_CRITICAL(...) superslam::Logger::getLogger()->critical(__VA_ARGS__)

#endif // SUPERSLAM_LOGGING_H
