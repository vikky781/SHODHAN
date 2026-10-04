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

  // ---- pipeline ----
  bool presolve = true;
  bool scaling = true;
  /// Dual simplex iteration limit for one node LP before it counts as numerical trouble.
  long long node_iteration_limit = 1000000;

  // ---- output ----
  int log_interval = 100;  ///< nodes between progress lines (verbosity >= 1)
  std::ostream* log = nullptr;  ///< null: silent
};

}  // namespace shodhan::mip
