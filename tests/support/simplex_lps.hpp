#pragma once

// Seeded random LPs for the simplex tests (test-only code). Unlike make_random_lp
// these do not plant an optimal pair: the answer is whatever the dense oracle says.

#include <cstdint>

#include "shodhan/lp_model.hpp"

namespace shodhan::testing {

struct SimplexLpOptions {
  int rows = 10;
  int cols = 14;
  double density = 0.3;
  double boxed_fraction = 0.3;   ///< columns with both bounds
  double fixed_fraction = 0.03;  ///< fixed columns
  double free_fraction = 0.0;    ///< free columns
  double upper_only_fraction = 0.0;  ///< columns with only an upper bound
  double equality_fraction = 0.2;    ///< rows with lo == hi
  double ranged_fraction = 0.2;      ///< rows with two finite sides
  double free_row_fraction = 0.0;
  /// Probability that the generating point sits exactly at a bound / a row
  /// is active at it: many tied ratios and degenerate vertices.
  double degenerate = 0.0;
  bool wide_coefficients = false;  ///< magnitudes 1e-4 .. 1e4
  /// Costs have the sign that makes the slack basis dual feasible (c >= 0 for
  /// lower-bounded, c <= 0 for upper-only, 0 for free columns, any for boxed).
  /// When false the costs are arbitrary, so the LP may be unbounded.
  bool dual_feasible_start = true;
  /// The right-hand sides are derived from a point inside the bounds, so the LP
  /// is feasible. Set to false to move one row out of reach (likely infeasible).
  bool feasible = true;
};

/// Minimization LP on a grid of values with the structure described above.
LpModel make_simplex_lp(std::uint64_t seed, const SimplexLpOptions& options);

}  // namespace shodhan::testing
