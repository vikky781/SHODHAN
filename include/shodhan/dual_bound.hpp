#pragma once

#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan {

/// Weak-duality bound of a multiplier vector y (minimization form, one entry per row):
///
///   LB(y) = offset + sum_i (y_i > 0 ? y_i row_lower_i : y_i row_upper_i)
///                  + sum_j (d_j > 0 ? d_j col_lower_j : d_j col_upper_j),     d = c - A^T y,
///
/// a lower bound on the objective of every feasible point; with a convex quadratic term the bound is
///   LB(y; x~) = offset + sum_i ... + sum_j (d_j > 0 ? d_j col_lower_j : d_j col_upper_j) - (1/2) x~^T Q x~,   d = c + Q x~ - A^T y,
/// for the supplied point x~ (first-order underestimate of the convex quadratic form, docs/QP.md); for a maximization model it is reported as an
/// upper bound in the model's sense. This is the same bound KASAUTI computes (docs/CERTIFICATES.md), with
/// the same two-pass rule: first strictly (a value that needs an infinite bound makes the bound -infinity),
/// then, only if that fails, treating a value below `zero_tol` of its scale as zero.
///
/// Evaluated in double precision with compensated (Neumaier) summation. Every rounding error of the
/// evaluation is bounded (the standard floating-point error model) and added outward to the bound, so the
/// bound is conservative; `rigorous` additionally requires that nothing was dropped. The exact-arithmetic
/// verifier remains the arbiter.
struct DualBound {
  bool finite = false;     ///< a bound exists (strictly, or after dropping tiny values)
  bool rigorous = false;   ///< finite and nothing was dropped: the strict bound
  int dropped = 0;         ///< multipliers / reduced costs treated as zero (0 when rigorous)
  double drop_effect = 0;  ///< bound on the objective change caused by the dropped values at the supplied x
  double bound = 0.0;      ///< model sense: a lower bound for min, an upper bound for max (valid if finite)
  double allowance = 0.0;  ///< rounding allowance already applied to `bound` (it makes the bound conservative)
  double gap_rel = 0.0;    ///< |primal objective - nominal bound| / (1 + |primal objective|), nominal = without the allowance
  double explained = 0.0;  ///< sum |y_i| viol_i + sum |d_j| viol_j at the primal point: how far it may beat the bound
  bool gap_ok = false;     ///< finite and the gap is within the tolerance passed in: bound - explained - tol <= objective <= bound + tol
  std::string note;        ///< why the bound is missing or not rigorous
};

/// Default `zero_tol` (relative): the same default as KASAUTI's --dual-zero-tol. It was chosen after seeing
/// a failing case (a multiplier of 1.6e-12 of the largest one on a correct answer), not derived in advance.
inline constexpr double kDualZeroTol = 1e-9;

/// `x` is the primal point (original space), `primal_objective` its objective in the model's sense.
DualBound compute_dual_bound(const LpModel& model, const std::vector<double>& x, const std::vector<double>& y,
                             double primal_objective, double gap_tol, double zero_tol = kDualZeroTol);

}  // namespace shodhan
