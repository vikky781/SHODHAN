#pragma once

// Seeded convex QPs (and LPs) with a KNOWN optimal primal-dual pair, built by construction (test-only code):
// x*, y*, d* are chosen first with complementarity, bounds and row ranges are derived around x*, Q = L L^T, and
// c = -Q x* + A^T y* + d*. Also planted infeasible and unbounded QPs.

#include <cstdint>
#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan::testing {

enum class QpVariant {
  PositiveDefinite,
  Singular,       // rank-deficient Q
  Degenerate,     // many active bounds / rows with a zero multiplier
  FreeHeavy,      // many free columns
  RangedHeavy,    // many ranged and inactive rows
  EqualityHeavy,  // mostly equality rows
  Wide,           // matrix entries over six orders of magnitude
  Lp,             // Q = 0
  kCount
};

const char* qp_variant_name(QpVariant v);

struct PlantedQp {
  LpModel model;
  std::vector<double> x, y, d;  ///< the planted optimal pair (minimization form)
  double objective = 0.0;       ///< the optimal value in the model's sense
};

PlantedQp make_planted_qp(std::uint64_t seed, QpVariant variant);

/// A QP that is infeasible (two contradictory copies of a row), or unbounded (an extra column with negative cost, no
/// entries and no upper bound, outside Q). `kind` describes it.
LpModel make_infeasible_qp(std::uint64_t seed, std::string* kind = nullptr);
LpModel make_unbounded_qp(std::uint64_t seed, std::string* kind = nullptr);

}  // namespace shodhan::testing
