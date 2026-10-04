#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "shodhan/lp_model.hpp"
#include "shodhan/solution.hpp"

namespace shodhan {

/// Outcome of presolve (not a solver status; see docs/PRESOLVE.md).
enum class PresolveStatus {
  Reduced,                // `reduced` is a (possibly smaller) model to solve
  SolvedByPresolve,       // every row and column was eliminated; postsolve gives the solution
  Infeasible,             // proven infeasible
  Unbounded,              // proven unbounded (a feasible point exists and an improving ray)
  InfeasibleOrUnbounded,  // an improving ray exists, feasibility was not established
};

const char* to_string(PresolveStatus status) noexcept;

struct PresolveOptions {
  /// Restrict to reductions that are valid for integer programs. Duals are
  /// not reconstructed for MIP.
  bool is_mip = false;
  /// Reconstruct row duals y and reduced costs d in postsolve (LP only).
  bool need_duals = true;
  int max_passes = 20;
  double feasibility_tol = 1e-9;
  /// Doubleton aggregation: pivot must be at least this fraction of the
  /// largest magnitude in its column.
  double doubleton_pivot_tolerance = 1e-3;
  /// Doubleton aggregation: refuse if a new coefficient would exceed this
  /// factor times the larger of the two coefficients it combines.
  double max_coefficient_growth = 1e3;
  /// Doubleton aggregation: refuse if a new coefficient is nonzero but smaller
  /// than this fraction of the terms that cancelled to form it.
  double min_cancellation_ratio = 1e-3;

  // Per-reduction switches.
  bool empty_rows = true;
  bool empty_columns = true;
  bool fixed_columns = true;
  bool singleton_rows = true;
  bool redundant_rows = true;
  bool forcing_rows = true;
  bool doubleton_equations = true;
  bool dual_fixing = true;
  bool integer_bounds = true;

  // ---- MIP presolve (only with is_mip; all need primal postsolve only: no duals are reconstructed) ----
  /// Activity-based bound propagation to a fixpoint (integer rounding, infeasibility detection).
  bool mip_propagation = true;
  /// Coefficient tightening (big-M reduction) of one-sided rows for binary columns.
  bool coefficient_tightening = true;
  /// Probing on binary columns: fix to 0 and to 1, propagate, derive fixings, bounds and implications.
  bool probing = true;
  /// Duplicate and parallel rows (the tighter range is kept), duplicate and dominated columns.
  bool parallel_rows = true;
  bool duplicate_columns = true;
  /// Clique table from at-most-one structure and probing implications.
  bool clique_table = true;
  /// Rounds of (MIP reductions, standard passes); each round stops when nothing changes.
  int mip_rounds = 5;
  /// Work limits, in row or column entries touched (targets, not tuned).
  long long propagation_work_limit = 20000000;
  long long probing_work_limit = 20000000;
  int probing_column_limit = 2000;  ///< binary columns probed per call
  int max_cliques = 20000;
  int max_clique_literals = 400000;
  int max_implications = 200000;
};

struct PresolveStats {
  Index rows_before = 0;
  Index cols_before = 0;
  std::size_t nnz_before = 0;
  Index rows_after = 0;
  Index cols_after = 0;
  std::size_t nnz_after = 0;
  int passes = 0;
  double seconds = 0.0;

  // One counter per reduction.
  int empty_rows = 0;
  int empty_columns = 0;
  int fixed_columns = 0;
  int singleton_rows = 0;
  int redundant_rows = 0;         // rows removed because both sides are implied
  int redundant_row_sides = 0;    // one implied side dropped, row kept
  int forcing_rows = 0;
  int doubleton_equations = 0;
  int dual_fixed_columns = 0;
  int integer_bounds_rounded = 0;
  int integer_bounds_tightened = 0;
  int implied_bound_checks = 0;   // LP: implied bounds computed for detection only
  int unbounded_columns = 0;      // columns found to be free to improve without limit
  // MIP reductions
  int propagated_bounds = 0;      // bounds tightened by the propagation to a fixpoint (integer columns)
  int coefficients_tightened = 0; // coefficients changed by coefficient tightening
  int probing_fixings = 0;        // columns fixed by probing
  int probing_bounds = 0;         // bounds tightened by probing (both branches imply them)
  int implications = 0;           // implications stored
  int cliques = 0;                // cliques in the table
  int parallel_rows = 0;          // rows merged into a parallel row
  int duplicate_columns = 0;      // identical integer columns merged
  int dominated_columns = 0;      // columns fixed because a parallel column dominates them
  long long mip_work = 0;         // entries touched by the MIP reductions

  /// (name, count) of every reduction counter, in a fixed order.
  std::vector<std::pair<std::string, int>> reduction_counts() const;
};

class PostsolveRecord;  // defined in the implementation

/// Everything postsolve needs. Records are replayed in reverse order.
struct PostsolveStack {
  Index n_rows = 0;  // original dimensions
  Index n_cols = 0;
  std::vector<double> original_cost;  // in the original sense
  double original_offset = 0.0;
  bool compute_duals = false;
  std::vector<Index> row_map;  // reduced row -> original row
  std::vector<Index> col_map;  // reduced column -> original column
  std::vector<std::shared_ptr<const PostsolveRecord>> records;
};

/// An implication found by probing: when column `var` takes the value `var_value` (a binary column), column `other`
/// satisfies x_other <= bound (is_upper) or x_other >= bound. Indices are columns of the REDUCED model.
struct Implication {
  int var = 0;
  int var_value = 0;
  int other = 0;
  bool is_upper = true;
  double bound = 0.0;
};

/// Cliques of binary literals: a literal is 2 * column for x and 2 * column + 1 for 1 - x, columns of the REDUCED
/// model. At most one literal of a clique can be 1 in an integer feasible point.
struct CliqueTable {
  std::vector<std::vector<int>> cliques;
};

/// Structure found by the MIP presolve for later use (cuts), in the index space of the reduced model.
struct MipPresolveInfo {
  std::vector<Implication> implications;
  CliqueTable cliques;
};

struct PresolveResult {
  PresolveStatus status = PresolveStatus::Reduced;
  /// Always a minimization model (a max model is converted by negating costs
  /// and offset). Empty unless status is Reduced or SolvedByPresolve.
  LpModel reduced;
  PostsolveStack stack;
  PresolveStats stats;
  /// For Infeasible: the first conflict found (original row/column indices).
  std::string note;
  /// Implications and cliques (MIP presolve only, status Reduced).
  MipPresolveInfo mip;
};

/// Deterministic LP/MIP presolve. See docs/PRESOLVE.md.
PresolveResult presolve(const LpModel& model, const PresolveOptions& options = {});

/// Cliques and implications of `model` itself (no reduction is applied, indices are the model's own): bound
/// propagation, probing on binary columns and the clique table of the rows. Every statement holds for every
/// integer-feasible point of `model`. Used by the cut separators when presolve is off. Empty when probing finds
/// the model infeasible.
MipPresolveInfo find_mip_structure(const LpModel& model, const PresolveOptions& options = {});

/// Maps a solution of `stack`'s reduced model back to the original model:
/// primal x, row duals y, reduced costs d (when the stack was built with
/// duals and `reduced` carries them) and the objective in the original sense.
/// For SolvedByPresolve pass a default-constructed Solution.
Solution postsolve(const PostsolveStack& stack, const Solution& reduced);

}  // namespace shodhan
