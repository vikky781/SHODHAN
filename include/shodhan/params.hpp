#pragma once

#include <cstdint>

#include "shodhan/constants.hpp"

namespace shodhan {

/// Solver parameters. The values below are defaults, not guarantees: they are
/// starting points and say nothing about the accuracy a solver will achieve.
struct Params {
  /// Absolute/relative primal feasibility tolerance. Default: 1e-6.
  double primal_tol = 1e-6;
  /// Dual feasibility (reduced-cost) tolerance. Default: 1e-6.
  double dual_tol = 1e-6;
  /// A value within int_tol of an integer counts as integral. Default: 1e-5.
  double int_tol = 1e-5;
  /// Wall-clock limit in seconds. Default: kInf (no limit).
  double time_limit = kInf;
  /// Number of worker threads. Default: 1.
  int threads = 1;
  /// Seed for any randomized component. Default: 0.
  std::uint64_t seed = 0;
  /// 0 = silent, 1 = info, 2 = debug. Default: 1.
  int verbosity = 1;
};

}  // namespace shodhan
