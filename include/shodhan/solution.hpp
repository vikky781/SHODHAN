#pragma once

#include <vector>

namespace shodhan {

/// A primal-dual point of an LpModel. See docs/CONVENTIONS.md.
///
///  x          primal values, one per column
///  y          row duals, one per row   (minimization-form convention)
///  d          reduced costs d = c - A^T y, one per column (may be empty)
///  objective  c^T x + offset, in the model's own sense (max models report the
///             maximization value)
///
/// y and d always follow the minimization form: a max model is treated as
/// min (-c) and its duals are those of that minimization problem.
struct Solution {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> d;
  double objective = 0.0;
};

}  // namespace shodhan
