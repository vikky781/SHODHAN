#include "shodhan/logger.hpp"

#include <cstdio>
#include <iostream>
#include <ostream>
#include <string>

namespace shodhan {

Logger::Logger(std::ostream* out, LogLevel level)
    : out_(out != nullptr ? out : &std::clog),
      level_(level),
      start_(std::chrono::steady_clock::now()) {}

bool Logger::enabled(LogLevel message_level) const noexcept {
  return out_ != nullptr && message_level != LogLevel::Silent &&
         static_cast<int>(level_) >= static_cast<int>(message_level);
}

LogLevel Logger::level_from_verbosity(int verbosity) noexcept {
  if (verbosity <= 0) return LogLevel::Silent;
  if (verbosity == 1) return LogLevel::Info;
  return LogLevel::Debug;
}

void Logger::log(LogLevel message_level, std::string_view message) const {
  if (!enabled(message_level)) return;
  std::string line;
  if (timestamps_) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    char buf[32];
    std::snprintf(buf, sizeof(buf), "[%9.3fs] ", seconds);
    line += buf;
  }
  line.append(message);
  line += '\n';
  const std::lock_guard<std::mutex> lock(mutex_);
  out_->write(line.data(), static_cast<std::streamsize>(line.size()));
  out_->flush();
}

}  // namespace shodhan
