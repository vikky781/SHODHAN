#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iosfwd>
#include <memory>
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
  /// Bound violation accepted for a basic variable, relative to the bound: a violation of at
  /// most primal_tol * (1 + |bound|) is tolerated (the same measure as check_kkt).
  double primal_tol = 1e-6;
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

  bool perturb = true;             ///< cost perturbation against dual degeneracy (start of phase 2 and stalls)
  bool perturb_at_start = true;    ///< false: only perturb when a stall is detected
  bool polish = true;              ///< after optimality, remove tolerated-but-visible violations
  double polish_tol = 1e-13;       ///< tolerance (primal relative, dual absolute) the polishing pass aims for (target)
  double perturb_scale = 5e-7;     ///< relative size of the initial perturbation (target)
  std::uint64_t seed = 0;
  long long stall_iterations = 500;  ///< iterations without dual objective progress before a larger perturbation (target)

  int max_trouble = 6;             ///< consecutive numerical-trouble events before NumericalError (target)
  int max_cleanup_rounds = 5;      ///< perturbation-removal / primal cleanup attempts (target)
  double artificial_bound = 1000.0;  ///< dual phase 1: box for free variables (target)
  bool profile = false;            ///< accumulate the time spent in the parts of the dual simplex (SimplexStats::profile)
  bool final_check = true;         ///< check_kkt on the final point (can be disabled by branch and bound)
  double final_tol = 1e-6;         ///< tolerance of the final KKT check

  FactorParams factor;
};

/// Compact record of a basis: the VarStatus of each of the N = n + m variables, one byte each. Restoring it
/// reproduces the basis and the bound each nonbasic variable sits at; the factorization and the primal and
/// dual values are recomputed (so they agree with a cold solve up to rounding, not bit for bit).
struct BasisSnapshot {
  std::vector<std::uint8_t> status;
  bool empty() const { return status.empty(); }
  friend bool operator==(const BasisSnapshot&, const BasisSnapshot&) = default;
};

/// Seconds spent in the parts of a solve, filled when SimplexOptions::profile is set (simple timers, no profiler).
struct SimplexProfile {
  double setup = 0.0;            ///< solve(): refactor, primal and dual values, dual feasibility fix
  double choose_row = 0.0;       ///< pricing: leaving row
  double btran = 0.0;            ///< row of B^-1
  double pivot_row = 0.0;        ///< the pivot row over the nonbasic columns
  double select_entering = 0.0;  ///< ratio test
  double ftran_column = 0.0;     ///< entering column
  double ftran_tau = 0.0;        ///< steepest-edge vector and weight update
  double updates = 0.0;          ///< dual and primal value updates
  double factor_update = 0.0;    ///< Forrest-Tomlin update
  double refactor = 0.0;         ///< refactorizations during iterations
  double finish = 0.0;           ///< perturbation removal, cleanup and polishing after the dual simplex
  double accept = 0.0;           ///< iterative refinement and the final KKT check
};

struct SimplexStats {
  SimplexProfile profile;
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

/// Everything a probe can change: bound changes followed by dual simplex iterations (strong branching, diving).
/// Saving it costs a few vector copies of size N (no matrix, no factorization), so a probe needs no copy of the
/// engine. The factorization is NOT saved: after restore_state() the basis is back but the factorization is
/// marked stale and is rebuilt on first use (solve() always refactorizes first, so a following solve() is
/// identical, bit for bit, to one on an engine that never probed).
struct EngineState {
  std::vector<double> lo, hi, cost, x, d, y, weights;
  std::vector<Index> basis, pos;
  std::vector<VarStatus> status;
  bool weights_exact = true, costs_modified = false, primal_stale = false, bounds_modified = false;
  SimplexStats stats;
  double dual_objective = 0.0, best_dual_objective = 0.0, flip_objective = 0.0;
  long long last_progress_iter = 0;
  int stall_rounds = 0, trouble_run = 0;
  Index last_leaving = -1;
};

/// A row to append to the model the engine works on: sum val[k] x[idx[k]] in [lo, hi].
struct RowSpec {
  std::vector<Index> idx;
  std::vector<double> val;
  double lo = -kInf;
  double hi = kInf;
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

  // ---- probes ---------------------------------------------------------
  /// The state that a probe can change (see EngineState).
  EngineState save_state() const;
  /// Puts the saved state back; the factorization is rebuilt lazily.
  void restore_state(const EngineState& state);
  /// False after restore_state() until the next refactorization.
  bool factor_valid() const { return factor_valid_; }

  // ---- rows (cutting planes) ---------------------------------------------
  /// Appends rows to the model the engine works on (it then keeps its own copy of the model; the model passed
  /// to the constructor is not touched). The logical variable of each new row enters the basis, so the basis
  /// stays dual feasible and the next solve() is a warm dual simplex start. Must not be called while a probe
  /// (EngineProbe, a saved EngineState) is outstanding: saved states have the old dimensions.
  void add_rows(const std::vector<RowSpec>& rows);
  /// Removes rows whose logical variable is basic (a slack row, so the basis stays valid). Returns false and
  /// changes nothing if any listed row is tight (nonbasic logical) or out of range. Row indices above a removed
  /// row shift down; `rows` need not be sorted.
  bool remove_rows(const std::vector<Index>& rows);
  /// The model including appended rows.
  const LpModel& current_model() const { return *mp_; }

  // ---- warm start ------------------------------------------------------
  /// Solves like solve() but stops with IterationLimit after at most `max_iterations` further iterations.
  EngineStatus solve_limited(long long max_iterations);
  /// The basis as a snapshot (one byte per variable).
  BasisSnapshot get_basis_snapshot() const;
  /// Restores a snapshot taken from an engine on the same model (any bounds): the basic variables and the
  /// nonbasic ones are set from the snapshot, a nonbasic status that its variable's CURRENT bounds do not
  /// allow is replaced by the nearest allowed one, and the basis is refactorized. Returns false if the
  /// snapshot does not have exactly m basic variables or the factorization is unusable.
  bool set_basis(const BasisSnapshot& snapshot);
  /// Current bounds of a structural column / of the logical variable of a row (the row bounds).
  double col_lower(Index j) const { return lo_[to_size(j)]; }
  double col_upper(Index j) const { return hi_[to_size(j)]; }
  double row_lower(Index i) const { return lo_[to_size(n_ + i)]; }
  double row_upper(Index i) const { return hi_[to_size(n_ + i)]; }
  /// c_work^T x + offset of the current (possibly primal infeasible) basic solution. When the basis is dual
  /// feasible and the costs are unmodified this is the dual objective, a valid lower bound on the LP optimum
  /// (minimization form), also after an IterationLimit.
  double current_dual_objective() const { return working_objective(); }
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
  /// A primal feasible point (length n, within the tolerances) from which the
  /// ray was found; with the ray it proves unboundedness.
  const std::vector<double>& unbounded_point() const { return point_; }
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
  bool select_entering(double sigma, double delta, double margin, bool relaxed, double* theta_dual, Index* entering,
                       std::vector<Index>* flips);
  bool farkas_valid();
  double ptol(double bound) const { return opt_.primal_tol * (1.0 + std::fabs(bound)); }
  void apply_bound_flips(const std::vector<Index>& flips, double* delta, Index r);
  void record_farkas(Index r);
  bool handle_trouble(const char* what);
  bool refactor_and_recompute();
  void update_weights(Index r, double alpha_r);

  // primal simplex (src/lp/primal_simplex.cpp)
  EngineStatus run_primal_simplex();
  bool refresh_after_refactor_primal();

  // driver (src/lp/engine_solve.cpp)
  void perturb_costs(double multiplier);
  EngineStatus finish_after_dual();
  EngineStatus phase2();
  EngineStatus run_dual_phase1(bool* dual_feasible);
  EngineStatus resolve_dual_infeasible();
  EngineStatus accept(EngineStatus status);
  void refine_solution(int rounds);
  void update_duals_from_y();
  const LpModel& checked_model();

  SimplexOptions opt_;
  const LpModel* mp_ = nullptr;
  std::shared_ptr<LpModel> owned_;  ///< set once rows were added; shared by copies of the engine
  const LpModel& mdl() const { return *mp_; }
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
  bool factor_valid_ = true;   ///< false after restore_state(): the factorization does not match the basis

  SimplexStats stats_;
  int trouble_run_ = 0;
  Index updates_since_refactor_ = 0;
  std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

  // work vectors
  SparseWork rho_, col_, tau_, rhs_, row_alpha_;
  struct Candidate {
    double t;          ///< exact breakpoint
    double r;          ///< Harris-relaxed breakpoint
    double abs_alpha;
    double range;      ///< hi - lo (infinite unless boxed)
    Index var;
  };
  std::vector<Candidate> cand_;
  std::vector<double> suffix_r_;
  std::vector<Index> flips_, banned_;

  std::vector<double> farkas_, ray_, point_;
  bool bounds_modified_ = false;
  bool in_phase1_ = false;
  LpModel check_model_;
  double dual_objective_ = 0.0;
  double flip_objective_ = 0.0;
  double best_dual_objective_ = 0.0;
  long long last_progress_iter_ = 0;
  int stall_rounds_ = 0;
  Index last_leaving_ = -1;
};

/// RAII probe: saves the engine state on construction and restores it on destruction. Whatever the probe does
/// (change_col_bounds, solve_limited, ...) is undone; iterations() reports what it spent before the restore.
class EngineProbe {
 public:
  explicit EngineProbe(SimplexEngine& engine) : engine_(engine), saved_(engine.save_state()), start_iterations_(engine.stats().iterations) {}
  ~EngineProbe() { engine_.restore_state(saved_); }
  EngineProbe(const EngineProbe&) = delete;
  EngineProbe& operator=(const EngineProbe&) = delete;
  /// Simplex iterations spent inside the probe so far.
  long long iterations() const { return engine_.stats().iterations - start_iterations_; }

 private:
  SimplexEngine& engine_;
  EngineState saved_;
  long long start_iterations_;
};

}  // namespace shodhan
