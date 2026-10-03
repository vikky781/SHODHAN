#pragma once

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "shodhan/constants.hpp"
#include "shodhan/lp_model.hpp"
#include "shodhan/solution.hpp"
#include "shodhan/sparse_work.hpp"

namespace shodhan {

/// Status of a variable of the computational form (docs/CONVENTIONS.md): the n
/// structural variables followed by the m logical variables (row activities).
enum class VarStatus : std::uint8_t {
  Basic,
  AtLower,     ///< nonbasic at its (finite) lower bound
  AtUpper,     ///< nonbasic at its (finite) upper bound
  Fixed,       ///< nonbasic with lower == upper
  FreeAtZero,  ///< nonbasic, no finite bound, value 0
};

/// Outcome of SimplexEngine::solve().
enum class EngineStatus {
  Optimal,
  Infeasible,       ///< proven by the dual simplex; farkas_ray() holds the certificate
  Unbounded,        ///< primal ray found; unbounded_ray() holds it
  IterationLimit,
  TimeLimit,
  NumericalError,
};

const char* to_string(EngineStatus status) noexcept;

/// Tunable parameters. Values are defaults/targets, not guarantees.
struct SimplexOptions {
  double primal_tol = 1e-6;  ///< absolute bound violation accepted for a basic variable
  double dual_tol = 1e-6;    ///< wrong-signed reduced cost accepted
  double time_limit = kInf;  ///< seconds
  long long iteration_limit = 100000000;
  int verbosity = 0;         ///< 0 silent, 1 progress lines every log_interval iterations
  int log_interval = 50;
  std::ostream* log = nullptr;  ///< null: nothing is printed

  Index refactor_interval = 100;  ///< updates between refactorizations (target)
  double max_growth = 1e8;        ///< refactor when the LU growth indicator exceeds this (target)
  double min_pivot_ratio = 1e-9;  ///< refactor/ distrust when the smallest relative pivot is below (target)
  double pivot_agreement_tol = 1e-7;  ///< row vs column pivot value, relative (target)

  bool dual_steepest_edge = true;  ///< false: largest scaled infeasibility with unit weights
  bool bound_flipping = true;      ///< long-step ratio test
  bool harris = true;              ///< Harris two-pass ratio test with the two pivot thresholds below
  double min_pivot_abs = 1e-9;     ///< smallest acceptable |alpha| in the ratio test (target)
  double min_pivot_rel = 1e-7;     ///< ... relative to the largest |alpha| in the row (target)

  bool perturb = true;             ///< cost perturbation against dual degeneracy
  double perturb_scale = 5e-7;     ///< relative size of the initial perturbation (target)
  std::uint64_t seed = 0;
  long long stall_iterations = 500;  ///< iterations without dual objective progress before a larger perturbation (target)

  int max_trouble = 6;             ///< consecutive numerical-trouble events before NumericalError (target)
  int max_cleanup_rounds = 5;      ///< perturbation-removal / primal cleanup attempts (target)
  double artificial_bound = 1000.0;  ///< dual phase 1: box for free variables (target)
  bool final_check = true;         ///< check_kkt on the final point (can be disabled by branch and bound)
  double final_tol = 1e-6;         ///< tolerance of the final KKT check

  FactorParams factor;
};

struct SimplexStats {
  long long iterations = 0;        ///< all simplex iterations
  long long dual_iterations = 0;
  long long phase1_iterations = 0; ///< dual phase 1 iterations (counted in dual_iterations too)
  long long primal_iterations = 0; ///< primal cleanup iterations
  long long bound_flips = 0;
  int refactors = 0;
  int basis_repairs = 0;
  int trouble_events = 0;
  int perturbation_rounds = 0;
  int cleanup_rounds = 0;
  int cost_shifts = 0;
  bool perturbed = false;          ///< a perturbation was applied at some point
  double seconds = 0.0;
};

/// Sums and counts of bound violations of basic variables and of wrong-signed
/// reduced costs of nonbasic ones.
struct InfeasibilitySummary {
  double primal_sum = 0.0;
  double primal_max = 0.0;
  Index primal_count = 0;
  double dual_sum = 0.0;
  double dual_max = 0.0;
  Index dual_count = 0;
};

/// Bounded dual simplex (with a primal simplex for cleanup) on the computational
/// form  A x - r = 0, variables 0..n-1 structural and n..n+m-1 logical (column
/// -e_i). The model is minimized internally (a Maximize model has its costs
/// negated; y and d follow the minimization form like everywhere else).
///
/// The engine keeps its factorization and basis between solve() calls, so that
/// branch and bound can change column bounds and resolve from the current basis.
class SimplexEngine {
 public:
  SimplexEngine(const LpModel& model, const SimplexOptions& options = {});

  /// Solves from the current basis (the slack basis at construction). May be
  /// called again after change_col_bounds / change_row_bounds.
  EngineStatus solve();

  // ---- warm start ------------------------------------------------------
  /// Changes the bounds of a structural column. A nonbasic variable moves to
  /// its new bound (the primal values are updated through one ftran at the next
  /// solve); a basic one just gets new bounds.
  void change_col_bounds(Index j, double lo, double hi);
  /// Same for the logical variable of row i (its bounds are the row bounds).
  void change_row_bounds(Index i, double lo, double hi);

  // ---- state and its computation (public for tests and reuse) -----------
  void initialize_slack_basis();
  /// Sets the basis (m distinct variable indices) and statuses of the others
  /// from the current bounds (nonbasic at the bound that is dual feasible when
  /// possible), then refactorizes.
  bool set_basis(const std::vector<Index>& basis);
  /// Refactorizes the basis, repairing rank deficiency by logicals (statuses
  /// are adjusted). Pricing weights are reset to 1 only after a repair or when
  /// reset_weights is true. Returns false if the factorization is unusable.
  bool refactor(bool reset_weights = false);
  /// Dual steepest edge weights ||e_i^T B^-1||^2 computed exactly (one btran per
  /// row). Used for a non-slack start basis; after a basis repair the weights
  /// stay at 1 (an approximation, weights_exact() is false).
  void compute_exact_weights();
  const std::vector<double>& weights() const { return weights_; }
  bool weights_exact() const { return weights_exact_; }
  /// x_B from the nonbasic values by one ftran.
  void compute_primal();
  /// y (btran of the basic costs) and d = c - A^T y for all variables.
  void compute_dual();
  /// Moves nonbasic boxed variables to the bound their d_j favours, and counts
  /// (and, if allow_shift, removes by cost shifting) wrong-signed reduced
  /// costs. Returns the number that remain.
  Index fix_dual_infeasibilities(bool allow_shift);
  InfeasibilitySummary infeasibility() const;
  /// c^T x + offset in minimization form, with the original costs.
  double objective() const;
  /// Same with the working (perturbed or shifted) costs.
  double working_objective() const;

  // ---- results ---------------------------------------------------------
  /// x (structural), y, d (structural reduced costs) and the objective in the
  /// model's own sense, from the current state.
  Solution solution() const;
  const std::vector<double>& primal_all() const { return x_; }
  const std::vector<double>& dual_all() const { return d_; }
  const std::vector<double>& row_duals() const { return y_; }
  const std::vector<Index>& basis() const { return basis_; }
  VarStatus status(Index var) const { return status_[to_size(var)]; }
  const SimplexStats& stats() const { return stats_; }
  const BasisFactor& factor() const { return factor_; }
  const SimplexOptions& options() const { return opt_; }
  SimplexOptions& options() { return opt_; }
  Index n_structural() const { return n_; }
  Index n_rows() const { return m_; }
  /// Row multipliers y (length m) proving infeasibility after Infeasible: the
  /// combination sum_i y_i * (row i) has no feasible value (see check_farkas).
  const std::vector<double>& farkas_ray() const { return farkas_; }
  /// Direction (length n) of an unbounded ray after Unbounded.
  const std::vector<double>& unbounded_ray() const { return ray_; }
  /// Whether the working costs differ from the original ones.
  bool costs_modified() const { return costs_modified_; }

 private:
  // setup / state
  void build_state();
  double nonbasic_value(Index j) const;
  void set_status_from_bounds(Index j, bool prefer_upper);
  void sync_nonbasic_values();
  bool time_exceeded() const;
  void log_line(const char* phase, double objective);
  void accumulate_trouble_reset() { trouble_run_ = 0; }

  // dual simplex (src/lp/dual_simplex.cpp)
  EngineStatus run_dual_simplex();
  Index choose_leaving_row(double* score) const;
  void compute_pivot_row();
  bool select_entering(double sigma, double delta, double* theta_dual, Index* entering,
                       std::vector<Index>* flips);
  void apply_bound_flips(const std::vector<Index>& flips, double* delta, Index r);
  void record_farkas(Index r);
  bool handle_trouble(const char* what);
  bool refactor_and_recompute();
  void update_weights(Index r, double alpha_r);

  SimplexOptions opt_;
  const LpModel& model_;
  Index n_ = 0, m_ = 0, N_ = 0;
  double sgn_ = 1.0;
  CsrMatrix csr_;
  std::vector<double> lo_, hi_;
  std::vector<double> cost_, cost_orig_;
  std::vector<Index> basis_, pos_;
  std::vector<VarStatus> status_;
  std::vector<double> x_, d_, y_;
  std::vector<double> weights_;
  BasisFactor factor_;
  bool weights_exact_ = true;
  bool costs_modified_ = false;
  bool primal_stale_ = false;  ///< nonbasic values changed since compute_primal()

  SimplexStats stats_;
  int trouble_run_ = 0;
  Index updates_since_refactor_ = 0;
  std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

  // work vectors
  SparseWork rho_, col_, tau_, rhs_, row_alpha_;
  std::vector<Index> candidates_, flips_, banned_;
  std::vector<double> cand_ratio_;

  std::vector<double> farkas_, ray_;
  double dual_objective_ = 0.0;
  Index last_leaving_ = -1;
};

}  // namespace shodhan
