#pragma once

#include <cstddef>

#include "shodhan/lp_model.hpp"

namespace shodhan {

/// Range of absolute values; `valid` is false when there was nothing to
/// measure.
struct AbsRange {
  double min = 0.0;
  double max = 0.0;
  bool valid = false;
};

/// Descriptive statistics of a model, as printed by `shodhan info`.
struct ModelStats {
  Index rows = 0;
  Index cols = 0;
  std::size_t nnz = 0;
  double density = 0.0;  // nnz / (rows * cols), 0 for an empty matrix

  Index continuous_cols = 0;
  Index integer_cols = 0;
  Index binary_cols = 0;

  Index rows_le = 0;      // [-inf, b]
  Index rows_ge = 0;      // [b, inf]
  Index rows_eq = 0;      // [b, b]
  Index rows_ranged = 0;  // both finite, lower < upper
  Index rows_free = 0;    // (-inf, inf)

  Index cols_fixed = 0;      // lower == upper
  Index cols_free = 0;       // (-inf, inf)
  Index cols_boxed = 0;      // both finite, lower < upper
  Index cols_one_sided = 0;  // exactly one infinite bound

  AbsRange coefficient;  // nonzero matrix entries
  double coefficient_ratio = 0.0;  // max / min of the above, 0 if not valid
  AbsRange cost;         // nonzero costs
  AbsRange bound;        // finite nonzero column bounds
};

ModelStats compute_stats(const LpModel& model);

}  // namespace shodhan
