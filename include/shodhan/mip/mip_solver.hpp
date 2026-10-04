#pragma once

#include "shodhan/lp_model.hpp"
#include "shodhan/mip/options.hpp"
#include "shodhan/mip/result.hpp"

namespace shodhan::mip {

/// Branch-and-bound for MILP (docs/MIP.md): presolve (MIP-safe reductions), scaling (integer columns keep scale
/// 1), dual simplex node LPs with warm starts, best-bound node selection with plunging, reliability branching
/// and primal heuristics. Every solution reported was verified against the ORIGINAL model by the incumbent
/// manager; nothing is returned as optimal that the search did not complete.
class MipSolver {
 public:
  explicit MipSolver(const MipOptions& options = {}) : options_(options) {}
  MipResult solve(const LpModel& model) const;
  const MipOptions& options() const { return options_; }

 private:
  MipOptions options_;
};

}  // namespace shodhan::mip
