#pragma once

// Seeded random LP generators for tests (test-only code).
//
// make_random_lp() builds an LP together with a known primal-dual optimal
// pair: it picks x*, y*, d* first, derives the bounds and row ranges so that
// the complementarity and sign rules of docs/CONVENTIONS.md hold, and sets
// c = A^T y* + d*. The pair therefore satisfies the KKT conditions exactly (up
// to rounding), so it is optimal for both the primal and the dual.

#include <cstdint>
#include <string>

#include "shodhan/lp_model.hpp"
#include "shodhan/solution.hpp"

namespace shodhan::testing {

struct RandomLpOptions {
  int rows = 8;
  int cols = 10;
  double density = 0.4;
  /// Probability that an active bound/row carries a zero multiplier.
  double degeneracy = 0.0;
  /// Probability that a column sits at a bound / a row is active at x*.
  double active_fraction = 0.5;
  double free_col_fraction = 0.0;
  double ranged_row_fraction = 0.0;
  double free_row_fraction = 0.0;
  /// Matrix coefficients with magnitudes from 1e-4 to 1e4 instead of 0.3..3.
  bool wide_coefficients = false;

  // Planted structure (counts are clamped to what fits).
  int fixed_cols = 0;
  int empty_cols = 0;
  int empty_rows = 0;
  int singleton_rows = 0;
  int doubleton_eqs = 0;
  int forcing_rows = 0;
};

struct RandomLp {
  LpModel model;
  /// Optimal primal-dual pair (minimization form), objective includes offset.
  Solution known;
};

RandomLp make_random_lp(std::uint64_t seed, const RandomLpOptions& options);

/// A feasible LP made infeasible by a contradictory construction (chosen by
/// the seed). `kind` receives a short description when non-null.
LpModel make_random_infeasible_lp(std::uint64_t seed, const RandomLpOptions& options,
                                  std::string* kind = nullptr);

/// A feasible LP made unbounded below by an added improving ray (chosen by
/// the seed). `kind` receives a short description when non-null.
LpModel make_random_unbounded_lp(std::uint64_t seed, const RandomLpOptions& options,
                                 std::string* kind = nullptr);

}  // namespace shodhan::testing
