#pragma once

#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan {

/// Result of a certificate check. `gap` is how decisively the certificate holds
/// and `scale` the magnitude of the terms it is computed from; the check passes
/// when gap > tol * scale (Farkas) or all violations are at most tol * scale (ray).
struct RayCheck {
  bool ok = false;
  double gap = 0.0;
  double scale = 0.0;
  std::string message;
};

/// Checks that row multipliers y (length n_rows) prove that the LP has no
/// feasible point. For every feasible (x, r) with r = A x within the row ranges
/// and x within the column bounds, sum_i y_i * (a_i x - r_i) = 0. The check
/// computes the largest and smallest value of
///     sum_j (A^T y)_j x_j  -  sum_i y_i r_i
/// over the box of column and row bounds; the LP is infeasible if the largest is
/// below zero or the smallest above it, by more than tol times the magnitude of
/// the terms. An infinite bound that a nonzero coefficient points at makes the
/// check fail; coefficients below 1e-12 times the largest one count as zero
/// (floating-point noise on terms that are zero in theory). Integrality is
/// ignored (LP relaxation).
RayCheck check_farkas(const LpModel& model, const std::vector<double>& y, double tol);

/// Checks that `ray` (length n_cols) is an improving direction of unbounded
/// recession: the column bounds and the row ranges allow moving along it
/// forever (a column with a finite bound in the direction of motion must have
/// ray_j = 0 up to the tolerance, A ray must stay inside the recession cone of
/// the row ranges) and the objective strictly improves along it (c^T ray < 0 for
/// a minimization model, > 0 for maximization). Whether a feasible point exists
/// is not checked here.
RayCheck check_unbounded_ray(const LpModel& model, const std::vector<double>& ray, double tol);

}  // namespace shodhan
