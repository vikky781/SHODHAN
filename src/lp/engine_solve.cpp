#include <algorithm>
#include <cmath>

#include "shodhan/simplex_engine.hpp"

namespace shodhan {

EngineStatus SimplexEngine::solve() {
  t0_ = std::chrono::steady_clock::now();
  if (!refactor()) return EngineStatus::NumericalError;
  compute_primal();
  compute_dual();
  const Index remaining = fix_dual_infeasibilities(false);
  if (primal_stale_) compute_primal();
  if (remaining > 0) return EngineStatus::NumericalError;  // dual phase 1 comes with stage D
  const EngineStatus st = run_dual_simplex();
  stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  return st;
}

}  // namespace shodhan
