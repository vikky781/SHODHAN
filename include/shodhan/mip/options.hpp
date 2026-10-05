#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

#include "shodhan/constants.hpp"
#include "shodhan/params.hpp"

namespace shodhan::mip {

enum class BranchingKind { Reliability, Pseudocost, MostFractional, FirstIndex };
enum class NodeSelectKind { BestBound, DepthFirst, BestEstimate };

const char* to_string(BranchingKind kind) noexcept;
const char* to_string(NodeSelectKind kind) noexcept;
/// Parses the command-line names (reliability, pseudocost, mostfrac, first / bestbound, depth, bestestimate).
bool parse_branching(const std::string& name, BranchingKind* out);
bool parse_node_select(const std::string& name, NodeSelectKind* out);

/// When and how often one primal heuristic runs. `frequency` is the number of processed nodes between two
/// runs at nodes (0: never at nodes, only at the root).
struct HeuristicSetting {
  bool enabled = true;
  int frequency = 0;
};

/// Parameters of the MILP solver. Values are defaults/targets, not guarantees; tests override them.
struct MipOptions {
  /// primal_tol (rows/columns, relative form), int_tol, time_limit (seconds, wall clock), seed, verbosity.
  Params params;

  // ---- termination ----
  /// Stop when (incumbent - best bound) <= mip_abs_gap or <= mip_gap * max(1, |incumbent|). Default 1e-4 / 1e-6.
  double mip_gap = 1e-4;
  double mip_abs_gap = 1e-6;
  long long node_limit = 1000000000LL;

  // ---- search ----
  BranchingKind branching = BranchingKind::Reliability;
  NodeSelectKind node_select = NodeSelectKind::BestBound;
  /// Continue into one child immediately (without restoring a basis) after branching.
  bool plunging = true;
  /// While an incumbent exists, plunging stops when the child's bound exceeds
  /// best_bound + plunge_gap_fraction * (incumbent - best_bound) (target 0.25).
  double plunge_gap_fraction = 0.25;
  /// Maximum number of node bases kept in memory; beyond it a node continues from the engine's current basis.
  std::size_t max_stored_bases = 2000;

  // ---- reliability branching (Achterberg, Koch, Martin 2005); all values are targets ----
  int reliability_threshold = 4;
  int strong_candidate_limit = 10;
  long long strong_iteration_limit = 100;  ///< dual simplex iterations per strong-branching child
  int strong_lookahead = 4;                ///< stop after this many candidates without a better score

  // ---- heuristics ----
  bool heuristics = true;  ///< master switch
  HeuristicSetting trivial{true, 0};
  HeuristicSetting rounding{true, 1};
  HeuristicSetting diving_fractional{true, 20};
  HeuristicSetting diving_coefficient{true, 20};
  HeuristicSetting feasibility_jump{true, 0};
  /// Diving may use at most this fraction of the LP iterations of the whole search so far (plus a floor).
  double dive_iteration_fraction = 0.1;
  long long dive_iteration_floor = 500;
  /// Feasibility Jump work limit: matrix entries touched before giving up.
  long long fj_work_limit = 2000000;

  // ---- cutting planes at the root (docs/CUTS.md); all values are targets ----
  bool cuts = true;                  ///< master switch
  int cut_rounds = 20;               ///< maximum rounds of separation
  int cut_stall_rounds = 3;          ///< stop after this many consecutive rounds of small progress
  double cut_min_progress = 1e-6;    ///< relative LP bound gain below which a round counts as small progress
  int cut_max_per_round = 100;       ///< cuts added per round at most
  double cut_min_efficacy = 1e-4;    ///< violation / ||coefficients|| a cut needs at the LP point
  double cut_max_parallelism = 0.95; ///< cosine above which a cut is dropped in favour of a better one of the round
  double cut_max_dynamism = 1e6;     ///< largest allowed ratio of the largest to the smallest |coefficient|
  double cut_max_density = 0.6;      ///< maximum share of the columns a cut may touch (when there are more than 40)
  double cut_rhs_relaxation = 1e-9;  ///< right-hand sides are relaxed by this relative amount against rounding
  int cut_age_limit = 5;             ///< a cut slack for this many rounds is removed from the LP
  int cut_mir_aggregation = 5;       ///< rows combined by the MIR separator at most
  int cut_gomory_rows = 100;         ///< tableau rows examined per round at most
  bool cut_gomory = true, cut_mir = true, cut_cover = true, cut_clique = true, cut_implied_bound = true;
  /// Without presolve, compute cliques and implications by probing the model before the root cut loop.
  bool cut_structure = true;
  /// MIR aggregation uses the detected variable-upper-bound rows as variable bounds and prefers flow-balance rows when it
  /// eliminates a continuous column (docs/STRUCTURE.md).
  bool cut_structure_aware = true;

  // ---- pipeline ----
  bool presolve = true;
  bool probing = true;  ///< probing in the MIP presolve (implications and cliques for the cuts)
  /// Big-M tightening of variable-upper-bound and indicator rows with the implied bounds of the propagation (docs/STRUCTURE.md).
  bool implied_bound_tightening = true;
  bool scaling = true;
  /// Dual simplex iteration limit for one node LP before it counts as numerical trouble.
  long long node_iteration_limit = 1000000;

  // ---- output ----
  int log_interval = 100;  ///< nodes between progress lines (verbosity >= 1)
  std::ostream* log = nullptr;  ///< null: silent
};

}  // namespace shodhan::mip
