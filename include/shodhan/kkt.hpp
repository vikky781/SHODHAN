#pragma once

#include <string>

#include "shodhan/lp_model.hpp"
#include "shodhan/solution.hpp"

namespace shodhan {

/// Result of check_kkt(). All quantities refer to the minimization form of the
/// model (a max model is checked as min of the negated costs). "rel" values
/// are the absolute value divided by (1 + norm of the relevant data), where the
/// relevant data is the magnitude of the terms the quantity is computed from:
///   primal infeasibility  max finite |bound| of the rows / columns,
///   reduced costs         per column, |c_j| + sum_i |a_ij y_i|,
///   duality gap           the larger of |p| + |d| and the absolute dual and
///                         primal objective terms (sum |y b|, sum |d b|, sum |c x|).
/// This is a backward-error scaling: it does not hide error that is large
/// relative to the data, but it does not demand more accuracy than cancellation
/// in the data itself allows.
struct KktReport {
  /// Largest violation of row bounds and column bounds.
  double primal_infeasibility_abs = 0.0;
  double primal_infeasibility_rel = 0.0;  // / (1 + max finite |bound|), rows and columns separately
  /// Sign-rule violations of y and of d recomputed as c - A^T y (a multiplier
  /// that points at an infinite bound counts as its full magnitude).
  double dual_infeasibility_abs = 0.0;
  double dual_infeasibility_rel = 0.0;    // rows / (1 + max |c|), columns per-column (see above)
  /// max |d_supplied - (c - A^T y)|; 0 when no d was supplied.
  double dual_mismatch_abs = 0.0;
  double dual_mismatch_rel = 0.0;         // per-column (see above)
  /// max over rows/columns of |multiplier| * distance to the bound it pushes on.
  double complementarity_abs = 0.0;
  double complementarity_rel = 0.0;       // / objective scale (see above)
  double primal_objective = 0.0;          // offset + c^T x   (minimization form)
  double dual_objective = 0.0;            // see docs/CONVENTIONS.md
  double gap_abs = 0.0;
  double gap_rel = 0.0;                   // |p - d| / objective scale (see above)
  /// |supplied objective - model objective at x| / (1 + |model objective|).
  double objective_mismatch_rel = 0.0;
  bool ok = false;

  std::string summary() const;
};

/// Checks primal feasibility, dual feasibility, complementarity and the
/// duality gap of `sol` for `model`. `ok` is true when every relative measure
/// is at most `tol`. Independent of any presolve code.
KktReport check_kkt(const LpModel& model, const Solution& sol, double tol);

}  // namespace shodhan
