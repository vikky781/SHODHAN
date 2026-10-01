#include <sstream>

#include "shodhan/logger.hpp"
#include "shodhan/params.hpp"
#include "test_harness.hpp"

using namespace shodhan;

TEST_CASE(params_documented_defaults) {
  const Params p;
  CHECK_EQ(p.primal_tol, 1e-6);
  CHECK_EQ(p.dual_tol, 1e-6);
  CHECK_EQ(p.int_tol, 1e-5);
  CHECK(is_inf(p.time_limit));
  CHECK_EQ(p.threads, 1);
  CHECK_EQ(p.seed, std::uint64_t{0});
  CHECK_EQ(p.verbosity, 1);
}

TEST_CASE(logger_levels_filter_messages) {
  std::ostringstream os;
  Logger log(&os, LogLevel::Info);
  log.info("hello");
  log.debug("hidden");
  CHECK_EQ(os.str(), std::string("hello\n"));

  log.set_level(LogLevel::Debug);
  log.debug("shown");
  CHECK_EQ(os.str(), std::string("hello\nshown\n"));

  std::ostringstream quiet;
  Logger silent(&quiet, LogLevel::Silent);
  silent.info("x");
  silent.debug("y");
  CHECK(quiet.str().empty());
}

TEST_CASE(logger_timestamps_off_by_default_and_switchable) {
  std::ostringstream os;
  Logger log(&os);
  log.info("plain");
  CHECK_EQ(os.str(), std::string("plain\n"));

  std::ostringstream ts;
  Logger timed(&ts);
  timed.set_timestamps(true);
  timed.info("msg");
  CHECK(ts.str().front() == '[');
  CHECK_CONTAINS(ts.str(), "s] msg\n");
}

TEST_CASE(logger_stream_is_configurable) {
  std::ostringstream a, b;
  Logger log(&a);
  log.info("one");
  log.set_stream(&b);
  log.info("two");
  CHECK_EQ(a.str(), std::string("one\n"));
  CHECK_EQ(b.str(), std::string("two\n"));
  log.set_stream(nullptr);
  log.info("dropped");
  CHECK_EQ(b.str(), std::string("two\n"));
}

TEST_CASE(logger_verbosity_mapping) {
  CHECK(Logger::level_from_verbosity(-3) == LogLevel::Silent);
  CHECK(Logger::level_from_verbosity(0) == LogLevel::Silent);
  CHECK(Logger::level_from_verbosity(1) == LogLevel::Info);
  CHECK(Logger::level_from_verbosity(2) == LogLevel::Debug);
  CHECK(Logger::level_from_verbosity(9) == LogLevel::Debug);
}
