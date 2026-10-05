#pragma once

// Quadratic objective support (docs/QP.md).
//
// Convention: objective = offset + c^T x + (1/2) x^T Q x with Q symmetric. LpModel::quadratic stores the LOWER
// TRIANGLE of Q (row index >= column index, diagonal included) in CSC, n_cols x n_cols; the empty 0x0 matrix means
// "no quadratic term". The full symmetric action of Q is implied: the entry q_ij with i > j stands for both q_ij and
// q_ji. A diagonal entry q_jj adds (1/2) q_jj x_j^2 to the objective, an off-diagonal entry q_ij (i > j) adds
// q_ij x_i x_j once in total. Internally everything is a minimization: a maximization model has c and Q negated, and
// "convex" then means that the NEGATED Q is positive semidefinite.

#include <span>
#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan {

/// True when the model has at least one stored quadratic entry.
inline bool has_quadratic(const LpModel& m) { return m.quadratic.nnz() > 0; }

/// y = Q x for the symmetric Q stored as the lower triangle `lower` (y is overwritten; sizes must equal the order).
void quad_multiply(const SparseMatrix& lower, std::span<const double> x, std::span<double> y);

/// x^T Q x (not halved).
double quad_form(const SparseMatrix& lower, std::span<const double> x);

/// The full symmetric matrix (both triangles) in CSC.
SparseMatrix quad_full(const SparseMatrix& lower);

/// Objective offset + c^T x + (1/2) x^T Q x in the model's own sense.
double model_objective(const LpModel& m, std::span<const double> x);

struct ConvexityReport {
  /// True when Q (in the minimization form of the model) was found positive semidefinite.
  bool convex = false;
  /// True when the test could decide (false only if the sparse factorization failed).
  bool decided = true;
  /// For a non-convex Q: the pivot (negative, or the offending zero pivot's row entry) and the column of the
  /// ORIGINAL model it belongs to.
  double pivot = 0.0;
  Index column = -1;
  /// Largest absolute entry of Q and the absolute tolerance used for the pivots.
  double scale = 0.0;
  double tolerance = 0.0;
  /// Rank found (number of positive pivots).
  Index rank = 0;
  std::string note;
};

/// Pivoted LDL^T (largest remaining diagonal first) of Q in minimization form for up to 1200 columns; larger Q uses the
/// sparse LDL^T with an AMD ordering (docs/QP.md). In exact arithmetic every pivot of a
/// positive semidefinite matrix is >= 0 and a zero pivot forces its whole remaining row to be zero. Here a pivot below
/// -tolerance, or a zero pivot (|pivot| <= tolerance) with a remaining entry above tolerance, proves that Q is not
/// positive semidefinite; tolerance = relative_tolerance * max|q_ij|. Models without a quadratic term are convex.
ConvexityReport check_convexity(const LpModel& model, double relative_tolerance = 1e-9);

}  // namespace shodhan
