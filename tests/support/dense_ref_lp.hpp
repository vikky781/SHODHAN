#pragma once

// TEST ORACLE ONLY. This is a small dense two-phase simplex with Bland's rule,
// written so that presolve, scaling and postsolve can be checked against an
// independent solver before a real LP engine exists. It is meant for models of
// up to about 40 rows and 60 columns. It is compiled into the test executable
// only; it must never be linked into shodhan_core or the CLI.

#include "shodhan/lp_model.hpp"
#include "shodhan/solution.hpp"
#include "shodhan/status.hpp"

namespace shodhan::testing {

struct RefLpResult {
  /// Optimal, Infeasible, Unbounded, or NumericalError (iteration limit).
  Status status = Status::NumericalError;
  /// Filled when status == Optimal. y and d follow docs/CONVENTIONS.md.
  Solution solution;
  long iterations = 0;
};

/// Solves the LP relaxation of `model` (integrality is ignored).
RefLpResult solve_dense_lp(const LpModel& model);

}  // namespace shodhan::testing
