#pragma once

// Test support for the cutting-plane tests: runs the root cut loop on a model WITHOUT presolve, and checks cuts
// against every feasible integer point of the model (test-only code).

#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"
#include "shodhan/mip/cuts.hpp"

namespace shodhan::testing {

/// The minimization form of a model (costs and offset negated for a maximization model).
LpModel cut_min_form(const LpModel& m);  // named apart from mip_harness.hpp's inline min_form: one symbol, one definition (MSVC link)

struct CutRunResult {
  bool ran = false;               ///< the root LP was optimal
  bool root_fractional = false;   ///< the root LP solution has a fractional integer column
  double root_bound = 0.0;        ///< LP bound before cuts (minimization form)
  double bound_after = 0.0;       ///< LP bound after the loop
  mip::CutLoopResult loop;
  std::vector<double> lo, hi;     ///< root bounds used (integer bounds integral)
  LpModel pm;                     ///< the minimization-form model
};

/// Presolve is NOT applied. Cliques and implications come from find_mip_structure(pm) when `use_structure`.
CutRunResult run_cut_loop_on(const LpModel& model, const mip::MipOptions& options, bool use_structure,
                             bool keep_candidates);

struct ValidityResult {
  bool enumerated = false;        ///< false: the model was too large to enumerate
  long long feasible_points = 0;  ///< feasible integer assignments (with a feasible continuous part)
  long long checks = 0;           ///< (cut, assignment) pairs checked
  long long inconclusive = 0;     ///< continuous LP could not be decided
  int violated_cut = -1;          ///< index of the first violated cut, -1 if none
  double worst_violation = 0.0;   ///< largest (max LHS - rhs) relative to 1 + |rhs|
  std::string detail;
};

/// Enumerates every integer assignment of `pm` within its (finite) bounds and checks every cut on every feasible
/// point; for continuous columns the maximum of the cut over the continuous polytope is computed with the dense
/// reference LP.
ValidityResult check_cuts_valid(const LpModel& pm, const std::vector<mip::Cut>& cuts, long long max_assignments = 40000);

}  // namespace shodhan::testing
