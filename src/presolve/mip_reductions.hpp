#pragma once

// Internal to the presolve module: the reductions that are only used for mixed-integer models. All of them keep
// every integer-feasible point of the model, except the dominated-column fixing, which keeps at least one
// optimal point. They need no dual postsolve; the only one that changes how a solution is recovered is the
// duplicate-column merge (DuplicateColumnRecord). docs/PRESOLVE.md states the validity conditions.

#include <vector>

#include "reductions.hpp"

namespace shodhan::presolve_detail {

struct MipWork {
  long long prop_work = 0;
  long long probe_work = 0;
  /// Implications found by probing, with ORIGINAL column indices.
  std::vector<Implication> implications;
  /// Columns whose meaning changed (merged columns): structure that mentions them is dropped.
  std::vector<char> tainted;
};

/// One round: propagation to a fixpoint, coefficient tightening, parallel rows, duplicate and dominated columns,
/// probing. Returns true when the model changed (or infeasibility was found, see c.infeasible).
bool run_mip_round(Context& c, MipWork& mw);

/// Cliques (from rows) and implications of the final model, restricted to alive, untainted binary columns.
/// Indices are ORIGINAL column indices; the caller maps them to the reduced model.
MipPresolveInfo collect_mip_structure(Context& c, const MipWork& mw);

}  // namespace shodhan::presolve_detail
