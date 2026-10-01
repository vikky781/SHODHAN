#pragma once

#include <vector>

#include "shodhan/lp_model.hpp"
#include "shodhan/solution.hpp"

namespace shodhan {

/// Row, column and objective scale factors, all exact powers of two.
///
/// The scaled model uses  A' = R A C,  c' = s C c,  x = C x'  with
/// R = diag(row_scale), C = diag(col_scale), s = obj_scale:
///   row bounds'    = R * row bounds        column bounds' = C^-1 * column bounds
///   objective'     = s * objective         (offset' = s * offset)
/// and the optimal pair maps back as
///   x = C x',   y = R y' / s,   d = d' / (s C),   objective = objective' / s.
/// Integer and binary columns always have col_scale = 1.
struct Scaling {
  std::vector<double> row_scale;
  std::vector<double> col_scale;
  double obj_scale = 1.0;
};

struct ScalingOptions {
  /// Maximum number of geometric-mean passes (one pass = rows then columns).
  int max_passes = 10;
  /// Stop when a pass improves the max/min coefficient ratio by less than this fraction.
  double min_improvement = 0.10;
};

struct CoefficientRange {
  double min = 0.0;    // smallest nonzero |a_ij|
  double max = 0.0;    // largest |a_ij|
  double ratio = 0.0;  // max / min (0 when there are no nonzeros)
  bool valid = false;
};

struct ScalingReport {
  CoefficientRange before;
  CoefficientRange after;
  int geometric_passes = 0;
  bool objective_scaled = false;
};

/// Coefficient range of the nonzero entries of a matrix.
CoefficientRange coefficient_range(const SparseMatrix& a);

/// Computes scale factors: geometric-mean passes, then one max-abs
/// equilibration of columns and rows, every factor rounded to a power of two.
/// Deterministic. Returns the identity scaling for an empty matrix.
Scaling compute_scaling(const LpModel& model, const ScalingOptions& options = {},
                        ScalingReport* report = nullptr);

/// The scaled copy of `model` (infinite bounds stay infinite).
LpModel apply_scaling(const LpModel& model, const Scaling& scaling);

/// Maps a solution of the scaled model back to the original model.
Solution unscale_solution(const Scaling& scaling, const Solution& scaled);

/// Inverse of unscale_solution: maps an original solution into the scaled space.
Solution scale_solution(const Scaling& scaling, const Solution& original);

}  // namespace shodhan
