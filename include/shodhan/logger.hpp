#pragma once

#include <chrono>
#include <iosfwd>
#include <mutex>
#include <string_view>

namespace shodhan {

enum class LogLevel { Silent = 0, Info = 1, Debug = 2 };

/// Minimal leveled logger. Writes whole lines to a configurable stream
/// (std::clog by default). Timestamps are off by default. A message is
/// written when the logger's level is at least the message's level.
/// Individual calls are serialized by an internal mutex.
class Logger {
 public:
  explicit Logger(std::ostream* out = nullptr, LogLevel level = LogLevel::Info);

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  void set_level(LogLevel level) noexcept { level_ = level; }
  LogLevel level() const noexcept { return level_; }
  /// A null stream silences all output.
  void set_stream(std::ostream* out) noexcept { out_ = out; }
  void set_timestamps(bool on) noexcept { timestamps_ = on; }

  bool enabled(LogLevel message_level) const noexcept;

  void info(std::string_view message) const { log(LogLevel::Info, message); }
  void debug(std::string_view message) const { log(LogLevel::Debug, message); }

  /// Maps Params::verbosity (0/1/2, clamped) to a level.
  static LogLevel level_from_verbosity(int verbosity) noexcept;

 private:
  void log(LogLevel message_level, std::string_view message) const;

  std::ostream* out_;
  LogLevel level_;
  bool timestamps_ = false;
  std::chrono::steady_clock::time_point start_;
  mutable std::mutex mutex_;
};

}  // namespace shodhan
