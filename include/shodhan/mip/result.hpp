#pragma once

#include <map>
#include <string>
#include <vector>

#include "shodhan/solution.hpp"
#include "shodhan/status.hpp"

namespace shodhan::mip {

struct HeuristicStats {
  std::string name;
  long long calls = 0;       ///< times it ran
  long long submitted = 0;   ///< candidate points it handed to the incumbent manager
  long long successes = 0;   ///< candidates that became the new incumbent
  double seconds = 0.0;
};

/// Why candidate points were rejected by the incumbent manager (counts).
struct RejectionStats {
  long long not_better = 0;          ///< verified, but not strictly better than the incumbent
  long long not_integral = 0;        ///< an integer column could not be made integral (bounds)
  long long row_violated = 0;        ///< a row bound violated beyond primal_tol on the ORIGINAL model
  long long column_violated = 0;     ///< a column bound violated beyond primal_tol
  long long lp_resolve_failed = 0;   ///< the continuous part had no feasible completion
  long long wrong_size = 0;
  long long total() const { return not_better + not_integral + row_violated + column_violated + lp_resolve_failed + wrong_size; }
};

/// One separator's share of the root cut loop.
struct CutSeparatorSummary {
  std::string name;
  long long generated = 0;   ///< raw candidates
  long long candidates = 0;  ///< passed cleaning and the efficacy filter
  long long added = 0;       ///< rows added to the LP
  double seconds = 0.0;
};

/// What the root cut loop did (docs/CUTS.md). Bounds are in the model's own sense.
struct CutSummary {
  bool ran = false;
  int rounds = 0;
  long long cuts_added = 0, cuts_removed = 0, cuts_kept = 0;
  bool has_bounds = false;
  double root_bound_without_cuts = 0.0;  ///< the LP bound before the first round
  double root_bound_with_cuts = 0.0;     ///< the LP bound after the last round
  bool abandoned = false;                ///< a resolve failed numerically; the cuts were dropped
  bool infeasible = false;               ///< the LP with cuts was infeasible
  std::string stopped_because;
  std::vector<CutSeparatorSummary> separators;
};

/// Result of a MILP solve.
///
/// status semantics (docs/MIP.md):
///   Optimal               the search finished with the gap within tolerance and no node had to be discarded
///                         for numerical reasons; has_solution is true.
///   Infeasible            the search proved there is no integer feasible point (no certificate unless the LP
///                         relaxation itself is infeasible, see lp_infeasible_certified).
///   InfeasibleOrUnbounded the LP relaxation is unbounded (or presolve found an improving ray): not certified
///                         either way.
///   TimeLimit / NodeLimit stopped; has_solution says whether an incumbent exists. best_bound is valid.
///   NumericalError        a node could not be solved reliably and was dropped (its bound is kept in best_bound),
///                         or the root LP failed. has_solution may be true.
struct MipResult {
  Status status = Status::NumericalError;
  bool has_solution = false;
  /// The verified incumbent in the ORIGINAL space (integer columns exactly integral) and its objective in the
  /// model's own sense.
  Solution solution;
  double objective = 0.0;
  /// A valid bound on the optimum in the model's own sense: a lower bound for min, an upper bound for max.
  /// Infinite (-inf for min, +inf for max) when nothing was proven.
  double best_bound = 0.0;
  bool has_bound = false;
  double abs_gap = 0.0;  ///< |objective - best_bound| (valid when has_solution && has_bound)
  double rel_gap = 0.0;  ///< abs_gap / max(1, |objective|)

  long long nodes_processed = 0;
  long long nodes_open = 0;
  long long lp_iterations = 0;            ///< all simplex iterations: root + nodes + diving + cuts (strong branching is separate)
  long long root_lp_iterations = 0;       ///< iterations of the root LP
  long long node_lp_iterations = 0;       ///< iterations of the node LPs after the root
  long long diving_iterations = 0;        ///< iterations spent by the diving heuristics
  long long cut_lp_iterations = 0;        ///< iterations of the re-solves of the root cut loop
  long long strong_branching_iterations = 0;
  long long strong_branching_calls = 0;
  int max_depth = 0;
  long long nodes_created = 0;     ///< nodes ever created (ids are never reused)
  long long peak_live_nodes = 0;   ///< the largest number of nodes in memory at the same time (finished nodes are released)
  long long numerical_trouble_nodes = 0;  ///< nodes whose LP failed twice (the search is then incomplete)
  long long nodes_pruned_by_bound = 0;
  long long nodes_pruned_infeasible = 0;

  std::vector<HeuristicStats> heuristics;  ///< one entry per heuristic, in a fixed order
  long long lp_solutions_integral = 0;     ///< node LPs whose solution was already integral
  RejectionStats rejections;
  long long incumbents_found = 0;
  std::string incumbent_source;  ///< name of the heuristic (or "lp") that found the final incumbent

  double seconds_total = 0.0;
  double seconds_presolve = 0.0;
  double seconds_root_lp = 0.0;
  double seconds_node_lps = 0.0;
  double seconds_strong_branching = 0.0;
  double seconds_heuristics = 0.0;   ///< all primal heuristics (includes diving)
  double seconds_diving = 0.0;       ///< of which the diving heuristics
  double seconds_cuts = 0.0;         ///< separation, cut selection and the LP re-solves of the root cut loop

  /// The LP relaxation was proven infeasible by the simplex (Farkas): `lp_farkas` (original rows) certifies
  /// it. When status is Infeasible and this is false, infeasibility was proved by branching (or presolve) and
  /// has NO certificate.
  bool lp_infeasible_certified = false;
  std::vector<double> lp_farkas;

  CutSummary cuts;

  std::string message;
  /// Presolve statistics, for the report.
  int presolve_rows_before = 0, presolve_rows_after = 0, presolve_cols_before = 0, presolve_cols_after = 0;
};

}  // namespace shodhan::mip
