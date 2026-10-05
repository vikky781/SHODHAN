#pragma once

#include <chrono>
#include <iosfwd>
#include <string>
#include <vector>

#include "shodhan/dual_bound.hpp"
#include "shodhan/ipm.hpp"
#include "shodhan/kkt.hpp"
#include "shodhan/lp_model.hpp"
#include "shodhan/params.hpp"
#include "shodhan/presolve.hpp"
#include "shodhan/solution.hpp"
#include "shodhan/status.hpp"

namespace shodhan {

/// Which algorithm solves the continuous problem. Auto: the dual simplex for an LP, the interior-point method for a QP.
/// The simplex cannot solve a QP (NotImplemented with an explanation).
enum class LpMethod { Auto, Simplex, Ipm, IpmCrossover };

const char* to_string(LpMethod method) noexcept;
bool parse_method(const std::string& name, LpMethod* out);

struct LpOptions {
  /// Tolerances (primal_tol, dual_tol), time limit, seed and verbosity.
  Params params;
  bool presolve = true;
  bool scaling = true;
  bool perturb = true;
  LpMethod method = LpMethod::Auto;
  /// Convergence tolerance of the interior-point method on the scaled problem (default 1e-8, a target).
  double ipm_tol = 1e-9;
  int ipm_max_iterations = 200;
  long long iteration_limit = 100000000;
  /// Tolerance of the final KKT check on the ORIGINAL model.
  double kkt_tol = 1e-6;
  /// Progress lines (verbosity >= 2) and notes about fallbacks go here; null is silent.
  std::ostream* log = nullptr;
};

struct LpResult {
  /// Optimal, Infeasible, Unbounded, TimeLimit, IterationLimit or NumericalError.
  /// Optimal only when the KKT check on the original model passed; Infeasible
  /// and Unbounded only with a verified certificate.
  Status status = Status::NumericalError;
  /// x, y, d (minimization-form multipliers) and the objective in the model's own
  /// sense. Filled for Optimal.
  Solution solution;
  KktReport kkt;  ///< check on the original model (Optimal)
  /// Weak-duality bound of the multipliers on the original model (Optimal); the answer was accepted only if
  /// its gap to the objective is within the KKT tolerance. `dual_bound.rigorous` is false when tiny
  /// multipliers had to be treated as zero (tolerance-checked, not a proof).
  DualBound dual_bound;
  bool rigorous = false;  ///< == dual_bound.rigorous

  long long iterations = 0;
  long long phase1_iterations = 0;
  long long primal_iterations = 0;
  int refactors = 0;
  bool perturbation_used = false;
  int attempts = 1;  ///< 1 unless a fallback was needed (see message)
  /// The algorithm that produced the result: "dual simplex", "interior point", "interior point + crossover".
  std::string method_used;
  bool quadratic = false;  ///< the model has a quadratic term
  /// Interior-point statistics of the accepted (or last) run; `iterations` above counts its iterations.
  long long ipm_nnz_l = 0;
  long long ipm_regularizations = 0;
  long long ipm_refinement_steps = 0;
  int ipm_factorizations = 0;
  double ipm_primal_residual = 0.0, ipm_dual_residual = 0.0, ipm_gap = 0.0;
  std::vector<IpmIteration> ipm_history;
  /// Crossover (IpmCrossover): simplex iterations spent after the interior-point solution.
  long long crossover_iterations = 0;

  double presolve_seconds = 0.0;
  double scaling_seconds = 0.0;
  double simplex_seconds = 0.0;
  double total_seconds = 0.0;
  PresolveStats presolve_stats;
  bool presolve_ran = false;
  std::string presolve_status_note;  ///< what presolve concluded on its own, if anything

  /// Infeasible: row multipliers y (length n_rows) that pass check_farkas on the
  /// original model.
  std::vector<double> farkas_ray;
  /// Unbounded: a direction (length n_cols) that passes check_unbounded_ray.
  std::vector<double> unbounded_ray;
  /// Unbounded: a feasible point (length n_cols, original space) from which the ray
  /// can be followed; with unbounded_ray it proves unboundedness.
  std::vector<double> unbounded_point;
  /// The configuration that produced the result, e.g. "presolve+scaling", "scaling",
  /// "none"; "+tight" is appended when the tolerances were tightened.
  std::string configuration;
  /// Human-readable remarks (fallbacks, why a status was downgraded, ...).
  std::string message;
};

/// Full LP pipeline: presolve -> scaling -> SimplexEngine -> unscale -> postsolve
/// -> check_kkt on the original model. Integrality is ignored (LP relaxation).
/// A solution is never returned as Optimal unless the KKT check on the original
/// model passed; if the first attempt fails the check, fallbacks are tried
/// (without presolve, with tighter tolerances, without scaling) before reporting
/// NumericalError.
class LpSolver {
 public:
  explicit LpSolver(const LpOptions& options = {}) : options_(options) {}
  LpResult solve(const LpModel& model) const;
  const LpOptions& options() const { return options_; }

 private:
  LpResult solve_with_ipm(const LpModel& model, std::chrono::steady_clock::time_point t_start) const;

  LpOptions options_;
};

}  // namespace shodhan
