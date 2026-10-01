#pragma once

// Internal to the presolve module. A mutable model with row-wise and
// column-wise access, lazy deletion and dirty flags. All indices are the
// ORIGINAL row/column indices; compaction into a fresh LpModel happens once, at
// the end of presolve.

#include <cmath>
#include <utility>
#include <vector>

#include "shodhan/constants.hpp"
#include "shodhan/lp_model.hpp"

namespace shodhan::presolve_detail {

struct Entry {
  int idx;  // column index in a row vector, row index in a column vector
  double val;
};

using Entries = std::vector<std::pair<int, double>>;

/// Activity bounds of a row over the current column bounds. The *_fin sums
/// hold the finite contributions; *_inf count the infinite ones.
struct Activity {
  double min_fin = 0.0;
  double max_fin = 0.0;
  int min_inf = 0;
  int max_inf = 0;
  bool min_finite() const { return min_inf == 0; }
  bool max_finite() const { return max_inf == 0; }
};

class WorkModel {
 public:
  /// Converts a max model to min (costs and offset negated) and drops
  /// explicit zero coefficients.
  explicit WorkModel(const LpModel& model);

  int m = 0;
  int n = 0;
  std::vector<std::vector<Entry>> rows;
  std::vector<std::vector<Entry>> cols;
  std::vector<double> rl, ru, cl, cu, cost;
  double offset = 0.0;
  std::vector<char> is_int;
  std::vector<char> row_alive, col_alive;
  std::vector<char> row_dirty, col_dirty;
  std::vector<int> row_cnt, col_cnt;  // alive entries per row / column
  /// Magnitude of the data that fed each row's bounds (bounds, |a| * |column bound|
  /// sums, shifts). Feasibility tolerances scale with it.
  std::vector<double> row_mag;

  int alive_rows() const;
  int alive_cols() const;

  template <typename F>
  void for_row(int i, F&& f) const {
    for (const Entry& e : rows[static_cast<std::size_t>(i)]) {
      if (col_alive[static_cast<std::size_t>(e.idx)]) f(e.idx, e.val);
    }
  }
  template <typename F>
  void for_col(int j, F&& f) const {
    for (const Entry& e : cols[static_cast<std::size_t>(j)]) {
      if (row_alive[static_cast<std::size_t>(e.idx)]) f(e.idx, e.val);
    }
  }

  /// Alive entries of a column / row as (index, value) pairs, in storage order.
  Entries col_snapshot(int j) const;
  Entries row_snapshot(int i) const;

  /// Removes a row/column (lazy: entries stay in the other view until the
  /// next compaction) and marks the affected neighbours dirty.
  void remove_row(int i);
  void remove_col(int j);

  /// Fixes x_j = v: shifts row bounds and the offset, then removes the column.
  void substitute_fixed(int j, double v);

  double coef(int r, int c) const;  // 0 when absent
  /// Sets, inserts or (v == 0) erases entry (r, c) in both views.
  void set_coef(int r, int c, double v);

  /// Bounds of column j changed: its rows' activities changed.
  void touch_col_bounds(int j);
  /// Bounds of row i changed: its columns' dual-fixing conditions changed.
  void touch_row_bounds(int i);

  Activity activity(int i) const;

  /// Drops dead entries from all vectors.
  void compact();

  bool any_dirty() const;

 private:
  void erase_entry(std::vector<Entry>& v, int idx);
};

}  // namespace shodhan::presolve_detail
