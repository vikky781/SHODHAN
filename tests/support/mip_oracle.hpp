#pragma once

// TEST ORACLES ONLY for MILP. Compiled into the test executable (and the corpus/stress programs), never into
// shodhan_core.
//
//  solve_mip_brute_force   enumerates every combination of the integer columns (all must have finite bounds)
//                          and, when continuous columns exist, solves the continuous part with the dense
//                          reference LP. Exact for the sizes the generators produce.
//  solve_mip_dense_bb      a deliberately simple depth-first branch and bound that uses the dense reference LP
//                          at every node; for instances too large to enumerate.

#include <vector>

#include "shodhan/lp_model.hpp"
#include "shodhan/status.hpp"

namespace shodhan::testing {

struct MipRefResult {
  /// Optimal, Infeasible, Unbounded (an improving direction among the continuous columns for some feasible
  /// integer assignment) or NumericalError (the dense LP could not decide some subproblem, or too large).
  Status status = Status::NumericalError;
  double objective = 0.0;  ///< in the model's own sense (valid for Optimal)
  std::vector<double> x;
  long long evaluations = 0;
  bool too_large = false;
};

MipRefResult solve_mip_brute_force(const LpModel& model, long long max_combinations = 400000);
MipRefResult solve_mip_dense_bb(const LpModel& model, long long max_nodes = 200000);

}  // namespace shodhan::testing
