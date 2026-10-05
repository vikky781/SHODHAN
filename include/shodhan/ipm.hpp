#pragma once

// Interior-point method for LPs and convex QPs (docs/IPM.md).
//
//   min  offset + c^T x + (1/2) x^T Q x      s.t.  row_lower <= A x <= row_upper,  col_lower <= x <= col_upper
//
// Mehrotra predictor-corrector with an infeasible start on the formulation with row activities r = A x as extra bounded
// variables, Newton systems solved through the quasi-definite augmented system with the sparse LDL^T.

#include <iosfwd>
#include <string>
#include <vector>

#include "shodhan/constants.hpp"
#include "shodhan/lp_model.hpp"
#include "shodhan/solution.hpp"
#include "shodhan/sparse_ldl.hpp"
#include "shodhan/status.hpp"

namespace shodhan {

struct IpmOptions {
  /// Convergence: the relative primal residual, relative dual residual and relative complementarity gap are each at most
  /// `tol` (default 1e-8, a target).
  double tol = 1e-8;
  int max_iterations = 200;
  double time_limit = kInf;  ///< seconds
  /// Static regularization of the augmented system (primal rho on the (1,1) block, dual delta on the (2,2) block).
  double rho = 1e-11;
  double delta = 1e-11;
  /// Smallest Theta^-1 = z/s used in the Newton matrix for a variable with a bound (target). A smaller value would be
  /// smaller than the static regularization and make refinement against the exact matrix diverge; the perturbation
  /// it causes in the dual equation is theta_floor * |dx|.
  double theta_floor = 0.0;
  /// Largest Theta^-1 used (target): beyond it the pivots of the augmented system span more orders of magnitude than double
  /// precision resolves.
  double theta_cap = 1e300;
  /// Fraction-to-the-boundary: a step may use at most this share of the distance to the boundary (target).
  double step_fraction = 0.95;
  int refinement_steps = 15;
  /// Iterations without progress of the merit function before the run is stopped as stagnating (target).
  int stall_iterations = 30;
  /// Gondzio-style multiple centrality correctors per iteration (0 disables them).
  int centrality_correctors = 2;
  int verbosity = 0;  ///< 1: one line per iteration to `log`
  std::ostream* log = nullptr;
};

struct IpmIteration {
  int iteration = 0;
  double primal_residual = 0.0;  ///< relative
  double dual_residual = 0.0;    ///< relative
  double gap = 0.0;              ///< relative complementarity gap
  double mu = 0.0;
  double step_primal = 0.0;
  double step_dual = 0.0;
};

struct IpmResult {
  /// Optimal (converged within options.tol), InfeasibleOrUnbounded (divergence), NumericalError (stagnation or a
  /// failed factorization), IterationLimit, TimeLimit, NotImplemented (integer columns).
  Status status = Status::NumericalError;
  /// x, y (row duals) and d = c + Q x - A^T y in the minimization form; objective in the model's sense.
  Solution solution;
  int iterations = 0;
  int attempts = 1;  ///< runs made (a numerical failure is retried with a larger regularization)
  double primal_residual = 0.0;
  double dual_residual = 0.0;
  double gap = 0.0;
  long long nnz_l = 0;
  long long dynamic_regularizations = 0;
  long long refinement_steps = 0;
  int factorizations = 0;
  double seconds = 0.0;
  std::vector<IpmIteration> history;
  std::string message;
};

/// Solves the continuous relaxation of `model` (integrality is ignored by the caller; a model with integer columns is
/// rejected with NotImplemented here). Q must be positive semidefinite in the minimization form: this is NOT checked
/// (see check_convexity).
IpmResult solve_ipm(const LpModel& model, const IpmOptions& options = {});

}  // namespace shodhan
