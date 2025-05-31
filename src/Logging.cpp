#include "Logging.h"
#include <iostream>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace SuperSLAM {

  std::shared_ptr<spdlog::logger> Logger::logger_      = nullptr;
  bool                            Logger::initialized_ = false;

  void Logger::initialize() {
    if (initialized_) { return; }

    try {
      // Create console sink with colors
      auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      console_sink->set_level(spdlog::level::info);
      console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

      // Create file sink
      auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("superslam.log", true);
      file_sink->set_level(spdlog::level::trace);
      file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");

      // Create logger with both sinks
      std::vector<spdlog::sink_ptr> sinks { console_sink, file_sink };
      logger_ = std::make_shared<spdlog::logger>("SuperSLAM", sinks.begin(), sinks.end());

      // Set the global log level
      logger_->set_level(spdlog::level::trace);
      logger_->flush_on(spdlog::level::info);

      // Register the logger
      spdlog::register_logger(logger_);

      initialized_ = true;

      logger_->info("SuperSLAM logging system initialized");
    } catch (const spdlog::spdlog_ex& ex) {
      std::cerr << "Log initialization failed: " << ex.what() << std::endl;
      // Fallback to default logger
      logger_      = spdlog::default_logger();
      initialized_ = true;
    }
  }

  std::shared_ptr<spdlog::logger> Logger::getLogger() {
    if (!initialized_) { initialize(); }
    return logger_;
  }

} // namespace SuperSLAM
