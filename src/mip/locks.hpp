#pragma once

// Variable locks (internal to the MIP module): the number of rows that could become violated when a column
// is decreased (down locks) or increased (up locks). A column with no down locks can always be rounded down
// without violating a row; one with no up locks can be rounded up.

#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan::mip {

struct Locks {
  std::vector<int> down, up;
};

inline Locks compute_locks(const LpModel& m) {
  Locks l;
  l.down.assign(to_size(m.n_cols), 0);
  l.up.assign(to_size(m.n_cols), 0);
  for (Index j = 0; j < m.n_cols; ++j) {
    for (Index p = m.A.col_start[to_size(j)]; p < m.A.col_start[to_size(j) + 1]; ++p) {
      const Index i = m.A.row_index[to_size(p)];
      const double a = m.A.value[to_size(p)];
      const bool has_lo = !is_inf(m.row_lower[to_size(i)]), has_hi = !is_inf(m.row_upper[to_size(i)]);
      if (a > 0.0) {
        if (has_lo) ++l.down[to_size(j)];
        if (has_hi) ++l.up[to_size(j)];
      } else if (a < 0.0) {
        if (has_hi) ++l.down[to_size(j)];
        if (has_lo) ++l.up[to_size(j)];
      }
    }
  }
  return l;
}

}  // namespace shodhan::mip
