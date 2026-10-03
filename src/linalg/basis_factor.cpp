#include "shodhan/basis_factor.hpp"

#include <limits>
#include <stdexcept>
#include <string>

#include "lu_eliminator.hpp"

namespace shodhan {

const char* to_string(FactorStatus status) noexcept {
  switch (status) {
    case FactorStatus::Ok: return "Ok";
    case FactorStatus::RankDeficient: return "RankDeficient";
    case FactorStatus::NeedRefactor: return "NeedRefactor";
  }
  return "Unknown";
}

FactorStatus BasisFactor::factorize(const SparseMatrix& A, const std::vector<Index>& basis_vars) {
  constexpr std::size_t kMaxIndex = static_cast<std::size_t>(std::numeric_limits<Index>::max());
  const std::size_t m = basis_vars.size();
  if (m >= kMaxIndex / 2) throw std::overflow_error("BasisFactor: dimension does not fit in int32");
  if (A.nnz() >= kMaxIndex / 2) throw std::overflow_error("BasisFactor: nonzero count does not fit in int32");
  if (static_cast<std::size_t>(A.n_rows) != m) {
    throw std::invalid_argument("BasisFactor: basis size " + std::to_string(m) +
                                " does not match the number of rows " + std::to_string(A.n_rows));
  }
  const Index n = A.n_cols;
  const Index mi = static_cast<Index>(m);
  for (const Index v : basis_vars) {
    if (v < 0 || v >= n + mi) {
      throw std::invalid_argument("BasisFactor: basis variable " + std::to_string(v) + " out of range");
    }
  }

  valid_ = false;
  spike_valid_ = false;
  m_ = mi;
  deficient_positions_.clear();
  unpivoted_rows_.clear();

  LuEliminator elim(params_, A, basis_vars);
  elim.run();

  if (!elim.deficient.empty() || elim.piv_row.size() != m) {
    deficient_positions_ = elim.deficient;
    unpivoted_rows_ = elim.unpivoted;
    status_ = FactorStatus::RankDeficient;
    return status_;
  }
  install(elim);
  valid_ = true;
  status_ = FactorStatus::Ok;
  return status_;
}

void BasisFactor::install(LuEliminator& e) {
  const std::size_t sm = to_size(m_);
  const std::size_t slot_cap = sm + to_size(std::max<Index>(params_.max_updates, 0)) + 1;

  n_slots_ = m_;
  row_of_slot_.assign(slot_cap, -1);
  slot_of_row_.assign(sm, -1);
  pos_of_row_.assign(sm, -1);
  row_of_pos_.assign(sm, -1);
  diag_.assign(sm, 0.0);
  for (std::size_t k = 0; k < sm; ++k) {
    const Index r = e.piv_row[k];
    const Index p = e.piv_pos[k];
    row_of_slot_[k] = r;
    slot_of_row_[to_size(r)] = static_cast<Index>(k);
    pos_of_row_[to_size(r)] = p;
    row_of_pos_[to_size(p)] = r;
    diag_[to_size(r)] = e.piv_val[k];
  }

  // U: row-wise and column-wise lists with room for later updates.
  std::vector<Index> row_caps(sm, 0), col_caps(sm, 0);
  for (std::size_t k = 0; k < sm; ++k) {
    const std::size_t len = e.u_start[k + 1] - e.u_start[k];
    row_caps[to_size(e.piv_row[k])] = static_cast<Index>(len) + 2;
  }
  for (const Index p : e.u_pos) ++col_caps[to_size(p)];
  for (std::size_t p = 0; p < sm; ++p) col_caps[p] += 2;
  const std::size_t slack = e.u_pos.size() + 4 * sm + 16;
  u_rows_.reset(m_, true, row_caps.data(), slack);
  u_cols_.reset(m_, true, col_caps.data(), slack);
  for (std::size_t k = 0; k < sm; ++k) {
    const Index r = e.piv_row[k];
    for (std::size_t t = e.u_start[k]; t < e.u_start[k + 1]; ++t) {
      u_rows_.push(r, e.u_pos[t], e.u_val[t]);
      u_cols_.push(e.u_pos[t], r, e.u_val[t]);
    }
  }
  nnz_u_ = e.u_pos.size() + sm;

  // L.
  l_pivot_ = e.l_pivot;
  l_start_ = e.l_start;
  l_idx_ = e.l_idx;
  l_val_ = e.l_val;
  eta_of_row_.assign(sm, -1);
  for (std::size_t k = 0; k < l_pivot_.size(); ++k) eta_of_row_[to_size(l_pivot_[k])] = static_cast<Index>(k);
  lt_start_.assign(sm + 1, 0);
  for (const Index i : l_idx_) ++lt_start_[to_size(i) + 1];
  for (std::size_t i = 0; i < sm; ++i) lt_start_[i + 1] += lt_start_[i];
  lt_idx_.assign(l_idx_.size(), 0);
  lt_val_.assign(l_idx_.size(), 0.0);
  {
    std::vector<std::size_t> fill(lt_start_.begin(), lt_start_.end() - 1);
    for (std::size_t k = 0; k < l_pivot_.size(); ++k) {
      for (std::size_t t = l_start_[k]; t < l_start_[k + 1]; ++t) {
        const std::size_t at = fill[to_size(l_idx_[t])]++;
        lt_idx_[at] = l_pivot_[k];
        lt_val_[at] = l_val_[t];
      }
    }
  }

  // Row etas of Forrest-Tomlin updates.
  r_pivot_.clear();
  r_start_.assign(1, 0);
  r_idx_.clear();
  r_val_.clear();
  update_count_ = 0;

  nnz_basis_ = e.nnz_basis;
  fresh_nnz_ = l_idx_.size() + nnz_u_;
  fill_ratio_ = nnz_basis_ == 0 ? 0.0 : static_cast<double>(fresh_nnz_) / static_cast<double>(nnz_basis_);
  n_col_singletons_ = e.n_col_singletons;
  n_row_singletons_ = e.n_row_singletons;
  n_markowitz_ = e.n_markowitz;
  max_abs_basis_ = e.max_abs_basis;
  max_abs_u_ = e.max_abs_u;
  min_pivot_ratio_ = sm == 0 ? 0.0 : e.min_rel_pivot;

  // Work buffers.
  if (scratch_.size() != m_) {
    scratch_.resize(m_);
    row_work_.resize(m_);
    spike_.resize(m_);
    saved_x_.resize(m_);
  } else {
    scratch_.clear();
    row_work_.clear();
    spike_.clear();
    saved_x_.clear();
  }
  visit_.assign(sm, 0);
  stamp_ = 0;
  reach_.clear();
  reach_.reserve(sm);
  dfs_node_.assign(sm + 1, 0);
  dfs_cur_.assign(sm + 1, nullptr);
  dfs_end_.assign(sm + 1, nullptr);
  heap_.clear();
  heap_.reserve(sm + 1);
  r_pivot_.reserve(to_size(std::max<Index>(params_.max_updates, 0)) + 1);
  r_start_.reserve(to_size(std::max<Index>(params_.max_updates, 0)) + 2);
}

void BasisFactor::reset_counters() noexcept {
  ftran_calls_ = btran_calls_ = hyper_solves_ = dense_solves_ = 0;
}

FactorStats BasisFactor::stats() const {
  FactorStats s;
  s.m = m_;
  s.nnz_basis = nnz_basis_;
  s.nnz_l = l_idx_.size();
  s.nnz_u = nnz_u_;
  s.nnz_r = r_idx_.size();
  s.fill_ratio = fill_ratio_;
  s.column_singleton_pivots = n_col_singletons_;
  s.row_singleton_pivots = n_row_singletons_;
  s.singleton_pivots = n_col_singletons_ + n_row_singletons_;
  s.markowitz_pivots = n_markowitz_;
  s.updates = update_count_;
  s.ftran_calls = ftran_calls_;
  s.btran_calls = btran_calls_;
  s.hyper_solves = hyper_solves_;
  s.dense_solves = dense_solves_;
  s.max_abs_basis = max_abs_basis_;
  s.max_abs_u = max_abs_u_;
  s.growth = max_abs_basis_ > 0.0 ? max_abs_u_ / max_abs_basis_ : 0.0;
  s.min_pivot_ratio = min_pivot_ratio_;
  return s;
}

FactorDump BasisFactor::dump() const {
  FactorDump d;
  auto put_i = [&d](Index v) { d.ints.push_back(v); };
  put_i(m_);
  put_i(n_slots_);
  for (const Index v : row_of_slot_) put_i(v);
  for (const Index v : slot_of_row_) put_i(v);
  for (const Index v : pos_of_row_) put_i(v);
  for (const Index v : row_of_pos_) put_i(v);
  for (const double v : diag_) d.reals.push_back(v);
  for (Index r = 0; r < u_rows_.n_lists(); ++r) {
    put_i(u_rows_.size(r));
    for (const Index v : u_rows_.indices(r)) put_i(v);
    for (const double v : u_rows_.values(r)) d.reals.push_back(v);
  }
  for (Index p = 0; p < u_cols_.n_lists(); ++p) {
    put_i(u_cols_.size(p));
    for (const Index v : u_cols_.indices(p)) put_i(v);
    for (const double v : u_cols_.values(p)) d.reals.push_back(v);
  }
  for (const Index v : l_pivot_) put_i(v);
  for (const std::size_t v : l_start_) put_i(static_cast<Index>(v));
  for (const Index v : l_idx_) put_i(v);
  for (const double v : l_val_) d.reals.push_back(v);
  for (const Index v : r_pivot_) put_i(v);
  for (const std::size_t v : r_start_) put_i(static_cast<Index>(v));
  for (const Index v : r_idx_) put_i(v);
  for (const double v : r_val_) d.reals.push_back(v);
  return d;
}

}  // namespace shodhan
