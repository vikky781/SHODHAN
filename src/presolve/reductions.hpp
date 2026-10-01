#pragma once

// Internal to the presolve module: the reductions. Each is a class with a
// static apply() that inspects one row or column, performs the reduction if it
// is valid, pushes the matching postsolve record, and returns true when the
// model changed. docs/PRESOLVE.md states the validity conditions.

#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include <sstream>
#include <string>

#include "postsolve_records.hpp"
#include "shodhan/presolve.hpp"
#include "work_model.hpp"

namespace shodhan::presolve_detail {

/// Full-precision text of a double, for diagnostics.
inline std::string num(double v) {
  std::ostringstream os;
  os << std::setprecision(17) << v;
  return os.str();
}

struct Context {
  WorkModel& w;
  const PresolveOptions& opt;
  PostsolveStack& stack;
  PresolveStats& stats;

  Context(WorkModel& work, const PresolveOptions& options, PostsolveStack& post, PresolveStats& st)
      : w(work), opt(options), stack(post), stats(st) {}

  bool infeasible = false;
  bool unbounded = false;  // an improving ray was found and dropped
  std::string reason;      // why infeasibility was concluded

  /// Absolute tolerance for comparing against `v`.
  double tol(double v) const { return opt.feasibility_tol * (1.0 + std::fabs(v)); }
  /// Rounding-error allowance for a quantity computed from data of magnitude `mag`
  /// (about 16 ulps): used where a wrong "yes" would change the problem.
  static double rounding(double mag) { return 16.0 * 2.220446049250313e-16 * mag; }
  /// Allowance for conditioning noise when INFEASIBILITY is concluded: a conflict
  /// is only real if it exceeds the feasibility tolerance relative to the
  /// magnitude of the row data behind it. (Wide coefficient ranges let
  /// substitutions leave errors of ~1e-12 relative to that magnitude.)
  double noise(double mag) const { return opt.feasibility_tol * mag; }
  /// Tolerance for comparisons that involve row i's bounds or activity when
  /// deciding infeasibility.
  double rtol(int i, double v) const {
    return tol(v) + noise(w.row_mag[static_cast<std::size_t>(i)]);
  }
  /// Strict tolerance for decisions that CHANGE the problem (dropping an implied row
  /// side, forcing a row): a few ulps of the compared value only. A looser margin
  /// would perturb the row by that amount, and a large dual turns that into a
  /// visible objective change.
  double stol(int, double v) const { return rounding(std::fabs(v)) / 4.0; }
  /// Same for a column bound derived from row i by dividing by coefficient a.
  double btol(int i, double a, double v) const {
    return tol(v) + noise(w.row_mag[static_cast<std::size_t>(i)] / std::fabs(a));
  }
  void record(std::shared_ptr<const PostsolveRecord> r) { stack.records.push_back(std::move(r)); }
  void mark_infeasible(std::string why) {
    if (!infeasible) reason = std::move(why);
    infeasible = true;
  }
};

// ---- row reductions ----

/// 1. Empty row: 0 must lie in the row range.
struct EmptyRowReduction {
  static bool apply(Context& c, int row);
};

/// 4. Singleton row -> column bound (rounded for integer columns in a MIP).
struct SingletonRowReduction {
  static bool apply(Context& c, int row);
};

/// 5. Redundant row via activity bounds; also detects infeasible rows.
struct RedundantRowReduction {
  /// Sets c.infeasible when the activity range cannot meet the row range.
  static bool detect_infeasible(Context& c, int row, const Activity& act);
  /// Drops row sides that the column bounds already imply; removes the row
  /// when both are gone.
  static bool apply(Context& c, int row, const Activity& act);
};

/// 6. Forcing row: activity extreme equals a row bound, so all its columns
/// are fixed at the corresponding bounds.
struct ForcingRowReduction {
  static bool apply(Context& c, int row, const Activity& act);
};

/// 7. Doubleton equation a_j x_j + a_k x_k = b: substitute x_j out.
struct DoubletonEquationReduction {
  static bool apply(Context& c, int row);
};

/// Implied column bounds from row activities. LP: infeasibility detection
/// only (nothing is written). MIP: integer columns are tightened.
struct ImpliedBoundsReduction {
  static bool apply(Context& c, int row, const Activity& act);
};

// ---- column reductions ----

/// 2. Empty column: fix at the bound its cost favours (or report an improving ray).
struct EmptyColumnReduction {
  static bool apply(Context& c, int col);
};

/// 3. Fixed column: substitute out.
struct FixedColumnReduction {
  static bool apply(Context& c, int col);
  /// Records and performs x_col = value (shared with dual fixing).
  static void fix(Context& c, int col, double value, bool dual_fixing);
};

/// 8. Dual fixing / dominated column.
struct DualFixingReduction {
  static bool apply(Context& c, int col);
};

/// 9. Integer bound rounding (MIP only).
struct IntegerBoundsReduction {
  static bool apply(Context& c, int col);
};

}  // namespace shodhan::presolve_detail
