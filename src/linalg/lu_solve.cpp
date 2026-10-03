// FTRAN / BTRAN for BasisFactor.
//
//   B^-1 = U^-1 R_k ... R_1 L^-1     (R_i = Forrest-Tomlin row etas)
//   B^-T = L^-T R_1^T ... R_k^T U^-T
//
// Every triangular stage has a dense path (a loop over all pivots) and a
// hypersparse path (symbolic reach by depth-first search, then only the
// reachable pivots in topological order). Both compute the same arithmetic in
// an order that respects the dependencies, so they agree to rounding error.

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "shodhan/basis_factor.hpp"

namespace shodhan {

bool BasisFactor::use_hyper(const SparseWork& w) const noexcept {
  return static_cast<double>(w.count()) < params_.hyper_threshold * static_cast<double>(m_);
}

// Depth-first search from every listed index of w over the graph given by
// adj(node) -> pair of pointers (begin, end) of the neighbours. Leaves the
// nodes in postorder in reach_; iterate it backwards for a topological order
// (every node before the nodes it points to).
template <class Adj>
void BasisFactor::dfs_reach(const SparseWork& w, Adj&& adj) {
  if (++stamp_ == 0) {
    std::fill(visit_.begin(), visit_.end(), 0u);
    stamp_ = 1;
  }
  reach_.clear();
  for (const Index s : w.indices()) {
    if (visit_[to_size(s)] == stamp_) continue;
    visit_[to_size(s)] = stamp_;
    std::size_t top = 0;
    dfs_node_[0] = s;
    {
      const auto range = adj(s);
      dfs_cur_[0] = range.first;
      dfs_end_[0] = range.second;
    }
    for (;;) {
      if (dfs_cur_[top] != dfs_end_[top]) {
        const Index nb = *dfs_cur_[top]++;
        if (visit_[to_size(nb)] != stamp_) {
          visit_[to_size(nb)] = stamp_;
          ++top;
          dfs_node_[top] = nb;
          const auto range = adj(nb);
          dfs_cur_[top] = range.first;
          dfs_end_[top] = range.second;
        }
      } else {
        reach_.push_back(dfs_node_[top]);
        if (top == 0) break;
        --top;
      }
    }
  }
}

// ---------------------------------------------------------------- FTRAN ----

void BasisFactor::ftran_l_dense(SparseWork& w) {
  double* v = w.raw();
  const std::size_t ne = l_pivot_.size();
  for (std::size_t e = 0; e < ne; ++e) {
    const double x = v[l_pivot_[e]];
    if (x == 0.0) continue;
    for (std::size_t t = l_start_[e]; t < l_start_[e + 1]; ++t) v[l_idx_[t]] -= l_val_[t] * x;
  }
  w.reindex();
}

void BasisFactor::ftran_l_hyper(SparseWork& w) {
  dfs_reach(w, [this](Index r) {
    const Index e = eta_of_row_[to_size(r)];
    if (e < 0) return std::pair<const Index*, const Index*>(nullptr, nullptr);
    const Index* base = l_idx_.data();
    return std::pair<const Index*, const Index*>(base + l_start_[to_size(e)], base + l_start_[to_size(e) + 1]);
  });
  for (std::size_t k = reach_.size(); k-- > 0;) {
    const Index r = reach_[k];
    const Index e = eta_of_row_[to_size(r)];
    if (e < 0) continue;
    const double x = w[r];
    if (x == 0.0) continue;
    for (std::size_t t = l_start_[to_size(e)]; t < l_start_[to_size(e) + 1]; ++t) {
      w.add(l_idx_[t], -l_val_[t] * x);
    }
  }
}

void BasisFactor::ftran_r(SparseWork& w) {
  const std::size_t ne = r_pivot_.size();
  for (std::size_t e = 0; e < ne; ++e) {
    double s = 0.0;
    for (std::size_t t = r_start_[e]; t < r_start_[e + 1]; ++t) s += r_val_[t] * w[r_idx_[t]];
    if (s != 0.0) w.add(r_pivot_[e], -s);
  }
}

// Row space in, position space out (through scratch_, then swapped into w).
void BasisFactor::ftran_u_dense(SparseWork& w) {
  double* v = w.raw();
  double* out = scratch_.raw();
  for (Index s = n_slots_; s-- > 0;) {
    const Index r = row_of_slot_[to_size(s)];
    if (r < 0) continue;
    const double rhs = v[r];
    if (rhs == 0.0) continue;
    const double x = rhs / diag_[to_size(r)];
    v[r] = 0.0;
    const Index p = pos_of_row_[to_size(r)];
    out[p] = x;
    const auto idx = u_cols_.indices(p);
    const auto val = u_cols_.values(p);
    for (std::size_t t = 0; t < idx.size(); ++t) v[idx[t]] -= val[t] * x;
  }
  scratch_.reindex();
  w.clear();
  w.swap(scratch_);
}

void BasisFactor::ftran_u_hyper(SparseWork& w) {
  dfs_reach(w, [this](Index r) {
    const auto nb = u_cols_.indices(pos_of_row_[to_size(r)]);
    return std::pair<const Index*, const Index*>(nb.data(), nb.data() + nb.size());
  });
  for (std::size_t k = reach_.size(); k-- > 0;) {
    const Index r = reach_[k];
    const double rhs = w[r];
    if (rhs == 0.0) continue;
    const double x = rhs / diag_[to_size(r)];
    const Index p = pos_of_row_[to_size(r)];
    scratch_.set(p, x);
    const auto idx = u_cols_.indices(p);
    const auto val = u_cols_.values(p);
    for (std::size_t t = 0; t < idx.size(); ++t) w.add(idx[t], -val[t] * x);
  }
  w.clear();
  w.swap(scratch_);
}

void BasisFactor::ftran(SparseWork& rhs, bool save_spike) {
  if (!valid_) throw std::logic_error("BasisFactor::ftran: no valid factorization");
  if (rhs.size() != m_) throw std::invalid_argument("BasisFactor::ftran: vector length does not match the basis");
  ++ftran_calls_;
  if (save_spike) spike_valid_ = false;

  if (use_hyper(rhs)) {
    ++hyper_solves_;
    ftran_l_hyper(rhs);
  } else {
    ++dense_solves_;
    ftran_l_dense(rhs);
  }
  ftran_r(rhs);
  if (save_spike) {
    spike_.assign(rhs);
    spike_.drop_small(params_.drop_tol);
  }
  if (use_hyper(rhs)) {
    ++hyper_solves_;
    ftran_u_hyper(rhs);
  } else {
    ++dense_solves_;
    ftran_u_dense(rhs);
  }
  rhs.drop_small(params_.drop_tol);
  if (save_spike) {
    saved_x_.assign(rhs);
    spike_valid_ = true;
  }
}

// ---------------------------------------------------------------- BTRAN ----

// Position space in, row space out.
void BasisFactor::btran_u_dense(SparseWork& w) {
  double* v = w.raw();
  double* out = scratch_.raw();
  for (Index s = 0; s < n_slots_; ++s) {
    const Index r = row_of_slot_[to_size(s)];
    if (r < 0) continue;
    const Index p = pos_of_row_[to_size(r)];
    const double rhs = v[p];
    if (rhs == 0.0) continue;
    const double z = rhs / diag_[to_size(r)];
    v[p] = 0.0;
    out[r] = z;
    const auto idx = u_rows_.indices(r);
    const auto val = u_rows_.values(r);
    for (std::size_t t = 0; t < idx.size(); ++t) v[idx[t]] -= val[t] * z;
  }
  scratch_.reindex();
  w.clear();
  w.swap(scratch_);
}

void BasisFactor::btran_u_hyper(SparseWork& w) {
  // Nodes are basis positions; position p points to the positions in the row of U it pivots on.
  dfs_reach(w, [this](Index p) {
    const auto nb = u_rows_.indices(row_of_pos_[to_size(p)]);
    return std::pair<const Index*, const Index*>(nb.data(), nb.data() + nb.size());
  });
  for (std::size_t k = reach_.size(); k-- > 0;) {
    const Index p = reach_[k];
    const double rhs = w[p];
    if (rhs == 0.0) continue;
    const Index r = row_of_pos_[to_size(p)];
    const double z = rhs / diag_[to_size(r)];
    scratch_.set(r, z);
    const auto idx = u_rows_.indices(r);
    const auto val = u_rows_.values(r);
    for (std::size_t t = 0; t < idx.size(); ++t) w.add(idx[t], -val[t] * z);
  }
  w.clear();
  w.swap(scratch_);
}

void BasisFactor::btran_r(SparseWork& w) {
  for (std::size_t e = r_pivot_.size(); e-- > 0;) {
    const double x = w[r_pivot_[e]];
    if (x == 0.0) continue;
    for (std::size_t t = r_start_[e]; t < r_start_[e + 1]; ++t) w.add(r_idx_[t], -r_val_[t] * x);
  }
}

void BasisFactor::btran_lt_dense(SparseWork& w) {
  double* v = w.raw();
  for (std::size_t e = l_pivot_.size(); e-- > 0;) {
    double s = 0.0;
    for (std::size_t t = l_start_[e]; t < l_start_[e + 1]; ++t) s += l_val_[t] * v[l_idx_[t]];
    if (s != 0.0) v[l_pivot_[e]] -= s;
  }
  w.reindex();
}

void BasisFactor::btran_lt_hyper(SparseWork& w) {
  dfs_reach(w, [this](Index i) {
    const Index* base = lt_idx_.data();
    return std::pair<const Index*, const Index*>(base + lt_start_[to_size(i)], base + lt_start_[to_size(i) + 1]);
  });
  for (std::size_t k = reach_.size(); k-- > 0;) {
    const Index i = reach_[k];
    const double x = w[i];
    if (x == 0.0) continue;
    for (std::size_t t = lt_start_[to_size(i)]; t < lt_start_[to_size(i) + 1]; ++t) {
      w.add(lt_idx_[t], -lt_val_[t] * x);
    }
  }
}

void BasisFactor::btran(SparseWork& rhs) {
  if (!valid_) throw std::logic_error("BasisFactor::btran: no valid factorization");
  if (rhs.size() != m_) throw std::invalid_argument("BasisFactor::btran: vector length does not match the basis");
  ++btran_calls_;

  if (use_hyper(rhs)) {
    ++hyper_solves_;
    btran_u_hyper(rhs);
  } else {
    ++dense_solves_;
    btran_u_dense(rhs);
  }
  btran_r(rhs);
  if (use_hyper(rhs)) {
    ++hyper_solves_;
    btran_lt_hyper(rhs);
  } else {
    ++dense_solves_;
    btran_lt_dense(rhs);
  }
  rhs.drop_small(params_.drop_tol);
}

}  // namespace shodhan
