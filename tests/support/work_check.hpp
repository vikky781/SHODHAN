#pragma once

// Invariant checker for SparseWork (test-only code).

#include <vector>

#include "shodhan/sparse_work.hpp"

namespace shodhan::testing {

/// True when every nonzero entry of w is listed exactly once and the list holds
/// nothing else than valid, distinct positions whose marks are set.
inline bool work_invariant_holds(const SparseWork& w) {
  std::vector<char> seen(to_size(w.size()), 0);
  for (const Index i : w.indices()) {
    if (i < 0 || i >= w.size() || seen[to_size(i)]) return false;
    seen[to_size(i)] = 1;
    if (!w.is_listed(i)) return false;
  }
  for (Index i = 0; i < w.size(); ++i) {
    if (w[i] != 0.0 && !seen[to_size(i)]) return false;
    if (w.is_listed(i) != (seen[to_size(i)] != 0)) return false;
  }
  return true;
}

}  // namespace shodhan::testing
