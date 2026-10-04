#include <algorithm>
#include <stdexcept>
#include <string>

#include "shodhan/simplex_engine.hpp"

namespace shodhan {

namespace {

// Rebuilds a matrix keeping the rows with new_index >= 0 (renumbered) and appending extra rows.
SparseMatrix rebuilt_matrix(const SparseMatrix& a, Index n_cols, Index n_rows_new, const std::vector<Index>& new_index,
                            const std::vector<RowSpec>& extra, Index first_extra_row) {
  std::vector<Triplet> t;
  t.reserve(a.nnz() + 16 * extra.size());
  for (Index j = 0; j < n_cols; ++j) {
    for (Index p = a.col_start[to_size(j)]; p < a.col_start[to_size(j) + 1]; ++p) {
      const Index r = new_index[to_size(a.row_index[to_size(p)])];
      if (r >= 0) t.push_back({r, j, a.value[to_size(p)]});
    }
  }
  for (std::size_t k = 0; k < extra.size(); ++k) {
    for (std::size_t q = 0; q < extra[k].idx.size(); ++q) {
      t.push_back({first_extra_row + static_cast<Index>(k), extra[k].idx[q], extra[k].val[q]});
    }
  }
  SparseMatrix out;
  std::string err;
  if (!SparseMatrix::from_triplets(n_rows_new, n_cols, std::move(t), &out, &err)) {
    throw std::logic_error("SimplexEngine: cannot rebuild the matrix: " + err);
  }
  return out;
}

}  // namespace

void SimplexEngine::add_rows(const std::vector<RowSpec>& rows) {
  if (rows.empty()) return;
  for (const RowSpec& r : rows) {
    if (r.idx.size() != r.val.size()) throw std::invalid_argument("SimplexEngine::add_rows: idx/val size mismatch");
    for (const Index j : r.idx) {
      if (j < 0 || j >= n_) throw std::invalid_argument("SimplexEngine::add_rows: column index out of range");
    }
  }
  const Index k_new = static_cast<Index>(rows.size());
  const Index m_old = m_;
  const Index n_old_total = N_;
  auto nm = std::make_shared<LpModel>(mdl());
  std::vector<Index> identity(to_size(m_old));
  for (Index i = 0; i < m_old; ++i) identity[to_size(i)] = i;
  nm->A = rebuilt_matrix(mdl().A, n_, m_old + k_new, identity, rows, m_old);
  nm->n_rows = m_old + k_new;
  for (const RowSpec& r : rows) {
    nm->row_lower.push_back(r.lo);
    nm->row_upper.push_back(r.hi);
    if (!nm->row_names.empty()) nm->row_names.push_back("cut" + std::to_string(nm->row_names.size()));
  }
  owned_ = nm;
  mp_ = owned_.get();
  csr_ = mp_->A.to_csr();

  m_ = m_old + k_new;
  N_ = n_old_total + k_new;
  const std::size_t sn = to_size(N_);
  lo_.resize(sn);
  hi_.resize(sn);
  cost_.resize(sn, 0.0);
  cost_orig_.resize(sn, 0.0);
  x_.resize(sn, 0.0);
  d_.resize(sn, 0.0);
  status_.resize(sn, VarStatus::Basic);
  pos_.resize(sn, -1);
  basis_.resize(to_size(m_));
  for (Index k = 0; k < k_new; ++k) {
    const Index v = n_old_total + k;
    lo_[to_size(v)] = rows[to_size(k)].lo;
    hi_[to_size(v)] = rows[to_size(k)].hi;
    status_[to_size(v)] = VarStatus::Basic;
    basis_[to_size(m_old + k)] = v;
    pos_[to_size(v)] = m_old + k;
  }
  y_.resize(to_size(m_), 0.0);
  weights_.resize(to_size(m_), 1.0);
  weights_exact_ = false;
  rho_.resize(m_);
  col_.resize(m_);
  tau_.resize(m_);
  rhs_.resize(m_);
  row_alpha_.resize(N_);
  farkas_.clear();
  banned_.clear();
  last_leaving_ = -1;
  factor_valid_ = false;
  if (!refactor()) throw std::logic_error("SimplexEngine::add_rows: the extended basis could not be factorized");
  compute_primal();
  compute_dual();
}

bool SimplexEngine::remove_rows(const std::vector<Index>& rows) {
  if (rows.empty()) return true;
  std::vector<char> del(to_size(m_), 0);
  for (const Index r : rows) {
    if (r < 0 || r >= m_) return false;
    if (status_[to_size(n_ + r)] != VarStatus::Basic) return false;
    del[to_size(r)] = 1;
  }
  std::vector<Index> new_row(to_size(m_), -1);
  Index kept = 0;
  for (Index i = 0; i < m_; ++i) {
    if (!del[to_size(i)]) new_row[to_size(i)] = kept++;
  }
  if (kept == m_) return true;
  const Index m_new = kept;
  auto nm = std::make_shared<LpModel>(mdl());
  nm->A = rebuilt_matrix(mdl().A, n_, m_new, new_row, {}, 0);
  nm->n_rows = m_new;
  nm->row_lower.clear();
  nm->row_upper.clear();
  std::vector<std::string> names;
  for (Index i = 0; i < m_; ++i) {
    if (del[to_size(i)]) continue;
    nm->row_lower.push_back(mdl().row_lower[to_size(i)]);
    nm->row_upper.push_back(mdl().row_upper[to_size(i)]);
    if (!mdl().row_names.empty()) names.push_back(mdl().row_names[to_size(i)]);
  }
  nm->row_names = std::move(names);

  // Variable renumbering: structural unchanged, logical n + i -> n + new_row[i] (or removed).
  const Index n_struct = n_;
  auto map_var = [&](Index v) -> Index {
    if (v < n_struct) return v;
    const Index r = new_row[to_size(v - n_struct)];
    return r < 0 ? Index{-1} : n_struct + r;
  };
  const Index n_total_new = n_ + m_new;
  std::vector<double> lo(to_size(n_total_new)), hi(to_size(n_total_new)), cost(to_size(n_total_new)),
      cost_orig(to_size(n_total_new)), x(to_size(n_total_new)), d(to_size(n_total_new));
  std::vector<VarStatus> status(to_size(n_total_new));
  for (Index v = 0; v < N_; ++v) {
    const Index nv = map_var(v);
    if (nv < 0) continue;
    lo[to_size(nv)] = lo_[to_size(v)];
    hi[to_size(nv)] = hi_[to_size(v)];
    cost[to_size(nv)] = cost_[to_size(v)];
    cost_orig[to_size(nv)] = cost_orig_[to_size(v)];
    x[to_size(nv)] = x_[to_size(v)];
    d[to_size(nv)] = d_[to_size(v)];
    status[to_size(nv)] = status_[to_size(v)];
  }
  std::vector<Index> basis;
  std::vector<double> weights;
  for (Index p = 0; p < m_; ++p) {
    const Index nv = map_var(basis_[to_size(p)]);
    if (nv < 0) continue;
    basis.push_back(nv);
    weights.push_back(weights_[to_size(p)]);
  }
  if (static_cast<Index>(basis.size()) != m_new) return false;  // cannot happen: every removed logical was basic
  std::vector<double> y(to_size(m_new), 0.0);
  for (Index i = 0; i < m_; ++i) {
    if (new_row[to_size(i)] >= 0) y[to_size(new_row[to_size(i)])] = y_[to_size(i)];
  }
  owned_ = nm;
  mp_ = owned_.get();
  csr_ = mp_->A.to_csr();
  m_ = m_new;
  N_ = n_total_new;
  lo_ = std::move(lo);
  hi_ = std::move(hi);
  cost_ = std::move(cost);
  cost_orig_ = std::move(cost_orig);
  x_ = std::move(x);
  d_ = std::move(d);
  status_ = std::move(status);
  basis_ = std::move(basis);
  weights_ = std::move(weights);
  y_ = std::move(y);
  pos_.assign(to_size(N_), -1);
  for (Index p = 0; p < m_; ++p) pos_[to_size(basis_[to_size(p)])] = p;
  weights_exact_ = false;
  rho_.resize(m_);
  col_.resize(m_);
  tau_.resize(m_);
  rhs_.resize(m_);
  row_alpha_.resize(N_);
  farkas_.clear();
  banned_.clear();
  last_leaving_ = -1;
  factor_valid_ = false;
  if (!refactor()) throw std::logic_error("SimplexEngine::remove_rows: the reduced basis could not be factorized");
  compute_primal();
  compute_dual();
  return true;
}

}  // namespace shodhan
