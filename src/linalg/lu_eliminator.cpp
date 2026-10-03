#include "lu_eliminator.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace shodhan {

namespace {

// An updated entry whose magnitude is at most this fraction of the terms that
// were subtracted is treated as exact cancellation (it is rounding noise).
constexpr double kCancel = 1e-14;

}  // namespace

void LuEliminator::Buckets::init(Index n_items, Index max_count) {
  head.assign(to_size(max_count) + 1, -1);
  next.assign(to_size(n_items), -1);
  prev.assign(to_size(n_items), -1);
  where.assign(to_size(n_items), -1);
}

void LuEliminator::Buckets::insert(Index item, Index count) noexcept {
  const std::size_t it = to_size(item);
  const std::size_t c = to_size(count);
  next[it] = head[c];
  prev[it] = -1;
  if (head[c] >= 0) prev[to_size(head[c])] = item;
  head[c] = item;
  where[it] = count;
}

void LuEliminator::Buckets::remove(Index item) noexcept {
  const std::size_t it = to_size(item);
  const Index c = where[it];
  if (c < 0) return;
  if (prev[it] >= 0) {
    next[to_size(prev[it])] = next[it];
  } else {
    head[to_size(c)] = next[it];
  }
  if (next[it] >= 0) prev[to_size(next[it])] = prev[it];
  where[it] = -1;
}

LuEliminator::LuEliminator(const FactorParams& params, const SparseMatrix& A,
                           const std::vector<Index>& basis)
    : m(static_cast<Index>(basis.size())), params_(params) {
  nsearch_ = std::max<Index>(1, params.markowitz_search);
  const std::size_t sm = to_size(m);
  const Index n = A.n_cols;

  // Count entries per row and per column (skipping explicit zeros).
  std::vector<Index> row_cnt(sm, 0);
  std::vector<Index> col_cnt(sm, 0);
  for (std::size_t p = 0; p < sm; ++p) {
    const Index v = basis[p];
    if (v < n) {
      const Index lo = A.col_start[to_size(v)];
      const Index hi = A.col_start[to_size(v) + 1];
      for (Index t = lo; t < hi; ++t) {
        if (A.value[to_size(t)] != 0.0) {
          ++col_cnt[p];
          ++row_cnt[to_size(A.row_index[to_size(t)])];
        }
      }
    } else {
      ++col_cnt[p];
      ++row_cnt[to_size(v - n)];
    }
  }
  std::vector<Index> col_caps(sm), row_caps(sm);
  for (std::size_t k = 0; k < sm; ++k) {
    col_caps[k] = col_cnt[k] + 2;
    row_caps[k] = row_cnt[k] + 2;
  }
  cols_.reset(m, true, col_caps.data());
  rows_.reset(m, false, row_caps.data());
  col_orig_max_.assign(sm, 0.0);
  for (std::size_t p = 0; p < sm; ++p) {
    const Index v = basis[p];
    const Index pi = static_cast<Index>(p);
    if (v < n) {
      const Index lo = A.col_start[to_size(v)];
      const Index hi = A.col_start[to_size(v) + 1];
      for (Index t = lo; t < hi; ++t) {
        const double a = A.value[to_size(t)];
        if (a == 0.0) continue;
        const Index i = A.row_index[to_size(t)];
        cols_.push(pi, i, a);
        rows_.push(i, pi);
        col_orig_max_[p] = std::max(col_orig_max_[p], std::fabs(a));
        ++nnz_basis;
      }
    } else {
      const Index i = v - n;
      cols_.push(pi, i, -1.0);
      rows_.push(i, pi);
      col_orig_max_[p] = 1.0;
      ++nnz_basis;
    }
  }

  col_b_.init(m, m);
  row_b_.init(m, m);
  col_done_.assign(sm, 0);
  row_done_.assign(sm, 0);
  wpos_.assign(sm, -1);
  row_stamp_.assign(sm, 0);
  col_stamp_.assign(sm, 0);
  piv_row.reserve(sm);
  piv_pos.reserve(sm);
  piv_val.reserve(sm);

  // Insert in descending index order so that bucket heads ascend.
  for (Index p = m - 1; p >= 0; --p) {
    const Index c = cols_.size(p);
    if (c == 0) {
      col_done_[to_size(p)] = 1;
      deficient.push_back(p);
    } else {
      col_b_.insert(p, c);
      ++active_cols_;
    }
  }
  for (Index i = m - 1; i >= 0; --i) {
    const Index c = rows_.size(i);
    if (c > 0) row_b_.insert(i, c);
  }
}

bool LuEliminator::column_is_empty(Index j, double cmax) const noexcept {
  return cmax <= params_.abs_pivot_tol || cmax <= params_.rel_pivot_tol * col_orig_max_[to_size(j)];
}

void LuEliminator::consider(Index i, Index j, double v, long long mk, Candidate& best) const {
  bool take = false;
  if (best.row < 0) {
    take = true;
  } else if (mk != best.mk) {
    take = mk < best.mk;
  } else if (std::fabs(v) != std::fabs(best.val)) {
    take = std::fabs(v) > std::fabs(best.val);
  } else if (i != best.row) {
    take = i < best.row;
  } else {
    take = j < best.col;
  }
  if (take) {
    best.row = i;
    best.col = j;
    best.val = v;
    best.mk = mk;
  }
}

void LuEliminator::examine_column(Index j, Index cnt, Candidate& best, bool& restart) {
  const auto idx = cols_.indices(j);
  const auto val = cols_.values(j);
  double cmax = 0.0;
  for (const double v : val) cmax = std::max(cmax, std::fabs(v));
  if (column_is_empty(j, cmax)) {
    drop_column(j);
    restart = true;
    return;
  }
  const double thr = params_.pivot_threshold * cmax;
  for (std::size_t t = 0; t < idx.size(); ++t) {
    if (std::fabs(val[t]) >= thr) {
      const long long mk =
          static_cast<long long>(rows_.size(idx[t]) - 1) * static_cast<long long>(cnt - 1);
      consider(idx[t], j, val[t], mk, best);
    }
  }
}

void LuEliminator::examine_row(Index i, Index cnt, Candidate& best, bool& restart) {
  for (Index t = 0; t < cnt; ++t) {
    const Index j = rows_.index_at(i, t);
    const auto idx = cols_.indices(j);
    const auto val = cols_.values(j);
    double cmax = 0.0;
    double v = 0.0;
    for (std::size_t q = 0; q < idx.size(); ++q) {
      cmax = std::max(cmax, std::fabs(val[q]));
      if (idx[q] == i) v = val[q];
    }
    if (column_is_empty(j, cmax)) {
      drop_column(j);
      restart = true;
      return;
    }
    if (v != 0.0 && std::fabs(v) >= params_.pivot_threshold * cmax) {
      const long long mk = static_cast<long long>(cnt - 1) * static_cast<long long>(idx.size() - 1);
      consider(i, j, v, mk, best);
    }
  }
}

LuEliminator::Candidate LuEliminator::find_pivot() {
  for (;;) {
    Candidate best;
    Index examined = 0;
    bool restart = false;
    for (Index cnt = 1; cnt <= m && !restart; ++cnt) {
      for (Index j = col_b_.head[to_size(cnt)]; j >= 0; j = col_b_.next[to_size(j)]) {
        examine_column(j, cnt, best, restart);
        if (restart) break;
        ++examined;
        if (best.row >= 0 && (best.mk == 0 || examined >= nsearch_)) return best;
      }
      if (restart) break;
      for (Index i = row_b_.head[to_size(cnt)]; i >= 0; i = row_b_.next[to_size(i)]) {
        examine_row(i, cnt, best, restart);
        if (restart) break;
        ++examined;
        if (best.row >= 0 && (best.mk == 0 || examined >= nsearch_)) return best;
      }
      if (restart) break;
      if (best.row >= 0 && best.mk <= static_cast<long long>(cnt) * cnt) return best;
    }
    if (!restart) return best;
  }
}

void LuEliminator::rebucket_row(Index i) {
  row_b_.remove(i);
  const Index c = rows_.size(i);
  if (c > 0 && !row_done_[to_size(i)]) row_b_.insert(i, c);
}

void LuEliminator::touch_row(Index i) {
  if (row_stamp_[to_size(i)] != stamp_) {
    row_stamp_[to_size(i)] = stamp_;
    touched_rows_.push_back(i);
  }
}

void LuEliminator::touch_col(Index j) {
  if (col_stamp_[to_size(j)] != stamp_) {
    col_stamp_[to_size(j)] = stamp_;
    touched_cols_.push_back(j);
  }
}

void LuEliminator::drop_column(Index j) {
  const auto idx = cols_.indices(j);
  for (const Index i : idx) {
    rows_.erase(i, rows_.find(i, j));
    rebucket_row(i);
  }
  cols_.clear_list(j);
  col_b_.remove(j);
  col_done_[to_size(j)] = 1;
  deficient.push_back(j);
  --active_cols_;
}

// Updates column j for the pivot row's entry arj: a_ij -= l_i * arj for every
// multiplier (i, l_i); creates fill-in and removes exact cancellations.
void LuEliminator::eliminate_column(Index j, double arj) {
  const Index len0 = cols_.size(j);
  for (Index t = 0; t < len0; ++t) wpos_[to_size(cols_.index_at(j, t))] = t;
  zero_list_.clear();
  for (std::size_t q = 0; q < mult_row_.size(); ++q) {
    const Index i = mult_row_[q];
    const double prod = mult_val_[q] * arj;
    const Index p = wpos_[to_size(i)];
    if (p >= 0) {
      double& v = cols_.value_at(j, p);
      const double nv = v - prod;
      if (std::fabs(nv) <= kCancel * (std::fabs(v) + std::fabs(prod))) {
        v = 0.0;
        zero_list_.push_back(p);
      } else {
        v = nv;
      }
    } else {
      cols_.push(j, i, -prod);
      rows_.push(i, j);
    }
  }
  for (Index t = 0; t < len0; ++t) wpos_[to_size(cols_.index_at(j, t))] = -1;
  std::sort(zero_list_.begin(), zero_list_.end(), std::greater<Index>());
  for (const Index p : zero_list_) {
    const Index i = cols_.index_at(j, p);
    cols_.erase(j, p);
    rows_.erase(i, rows_.find(i, j));
  }
}

void LuEliminator::pivot(const Candidate& cand) {
  const Index r = cand.row;
  const Index c = cand.col;
  const double piv = cand.val;
  if (cols_.size(c) == 1) {
    ++n_col_singletons;
  } else if (rows_.size(r) == 1) {
    ++n_row_singletons;
  } else {
    ++n_markowitz;
  }
  ++stamp_;
  touched_rows_.clear();
  touched_cols_.clear();

  // Multipliers of the rows below the pivot in column c.
  mult_row_.clear();
  mult_val_.clear();
  {
    const auto ci = cols_.indices(c);
    const auto cv = cols_.values(c);
    for (std::size_t t = 0; t < ci.size(); ++t) {
      if (ci[t] == r) continue;
      mult_row_.push_back(ci[t]);
      mult_val_.push_back(cv[t] / piv);
    }
  }
  if (!mult_row_.empty()) {
    l_pivot.push_back(r);
    l_idx.insert(l_idx.end(), mult_row_.begin(), mult_row_.end());
    l_val.insert(l_val.end(), mult_val_.begin(), mult_val_.end());
    l_start.push_back(l_idx.size());
  }

  // Column c leaves the active submatrix: drop it from every row it touches.
  for (const Index i : cols_.indices(c)) {
    rows_.erase(i, rows_.find(i, c));
    if (i != r) touch_row(i);
  }
  cols_.clear_list(c);
  col_b_.remove(c);
  col_done_[to_size(c)] = 1;
  --active_cols_;

  piv_row.push_back(r);
  piv_pos.push_back(c);
  piv_val.push_back(piv);

  // Row r becomes a row of U; the columns it touches are updated.
  const auto rp = rows_.indices(r);
  rowpat_.assign(rp.begin(), rp.end());
  for (const Index j : rowpat_) {
    const Index pos = cols_.find(j, r);
    if (pos < 0) throw std::logic_error("LuEliminator: row/column structure out of sync");
    const double arj = cols_.value_at(j, pos);
    cols_.erase(j, pos);
    u_pos.push_back(j);
    u_val.push_back(arj);
    if (!mult_row_.empty()) eliminate_column(j, arj);
    touch_col(j);
  }
  u_start.push_back(u_pos.size());
  rows_.clear_list(r);
  row_b_.remove(r);
  row_done_[to_size(r)] = 1;

  for (const Index i : touched_rows_) rebucket_row(i);
  for (const Index j : touched_cols_) {
    col_b_.remove(j);
    const Index cnt = cols_.size(j);
    if (cnt == 0) {
      // Everything cancelled: the column is numerically dependent.
      col_done_[to_size(j)] = 1;
      deficient.push_back(j);
      --active_cols_;
    } else {
      col_b_.insert(j, cnt);
    }
  }
}

void LuEliminator::run() {
  while (active_cols_ > 0) {
    const Candidate cand = find_pivot();
    if (cand.row < 0) {
      // No active column is left to pivot on (every remaining one was dropped).
      break;
    }
    pivot(cand);
  }
  std::sort(deficient.begin(), deficient.end());
  for (Index i = 0; i < m; ++i) {
    if (!row_done_[to_size(i)]) unpivoted.push_back(i);
  }
}

}  // namespace shodhan
