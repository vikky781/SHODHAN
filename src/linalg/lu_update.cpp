// Forrest-Tomlin update of the basis factorization.
//
// Replacing basis column p by a_q gives, in the transformed space, U with
// column p replaced by the spike s = R_k ... R_1 L^-1 a_q. The pivot row r0 of
// position p moves to the end of the triangular order, which leaves the
// entries of its row in the columns of the later pivots below the diagonal.
// They are eliminated by a row eta R: row r0 -= sum_j mu_j * (row of pivot r_j).
// The new diagonal is s[r0] - sum_j mu_j s[r_j]; it must equal old_diag * alpha,
// where alpha is the pivot element of the saved ftran solution, and the two
// numbers are compared as a stability check.

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "shodhan/basis_factor.hpp"

namespace shodhan {

FactorStatus BasisFactor::update(Index leaving_position) {
  if (leaving_position < 0 || leaving_position >= m_) {
    throw std::invalid_argument("BasisFactor::update: leaving position out of range");
  }
  if (!valid_ || !spike_valid_) return FactorStatus::NeedRefactor;

  // Limits (the factorization is untouched when one of them is hit).
  if (update_count_ >= params_.max_updates) return FactorStatus::NeedRefactor;
  if (to_size(n_slots_) >= row_of_slot_.size()) return FactorStatus::NeedRefactor;
  const std::size_t total = l_idx_.size() + nnz_u_ + r_idx_.size();
  if (static_cast<double>(total) > params_.max_growth * static_cast<double>(fresh_nnz_)) {
    return FactorStatus::NeedRefactor;
  }

  const Index p = leaving_position;
  const Index r0 = row_of_pos_[to_size(p)];
  const double d0 = diag_[to_size(r0)];
  const double alpha = saved_x_[p];
  if (std::fabs(alpha) <= params_.update_pivot_tol) return FactorStatus::NeedRefactor;

  // Eliminate row r0 of U against the rows of the later pivots, in triangular
  // order, through a heap on slots. Nothing in U is modified yet.
  const std::size_t r_idx_mark = r_idx_.size();
  const std::size_t r_val_mark = r_val_.size();
  row_work_.clear();
  heap_.clear();
  const auto greater = std::greater<Index>();
  {
    const auto idx = u_rows_.indices(r0);
    const auto val = u_rows_.values(r0);
    for (std::size_t t = 0; t < idx.size(); ++t) {
      row_work_.set(idx[t], val[t]);
      heap_.push_back(slot_of_row_[to_size(row_of_pos_[to_size(idx[t])])]);
      std::push_heap(heap_.begin(), heap_.end(), greater);
    }
  }
  double new_diag = spike_[r0];
  while (!heap_.empty()) {
    std::pop_heap(heap_.begin(), heap_.end(), greater);
    const Index slot = heap_.back();
    heap_.pop_back();
    const Index rj = row_of_slot_[to_size(slot)];
    const Index pj = pos_of_row_[to_size(rj)];
    const double val = row_work_[pj];
    if (val == 0.0) continue;
    const double mu = val / diag_[to_size(rj)];
    r_idx_.push_back(rj);
    r_val_.push_back(mu);
    new_diag -= mu * spike_[rj];
    const auto idx = u_rows_.indices(rj);
    const auto uv = u_rows_.values(rj);
    for (std::size_t t = 0; t < idx.size(); ++t) {
      const Index pp = idx[t];
      if (pp == p) continue;  // column p is replaced by the spike
      if (!row_work_.is_listed(pp)) {
        heap_.push_back(slot_of_row_[to_size(row_of_pos_[to_size(pp)])]);
        std::push_heap(heap_.begin(), heap_.end(), greater);
      }
      row_work_.add(pp, -mu * uv[t]);
    }
  }
  row_work_.clear();

  // Stability: the new diagonal computed two ways.
  const double other = d0 * alpha;
  const double scale = std::max(std::fabs(new_diag), std::fabs(other));
  const bool unstable = !(std::fabs(new_diag - other) <= params_.update_mismatch_tol * scale) ||
                        std::fabs(new_diag) <= params_.update_pivot_tol || !std::isfinite(new_diag);
  if (unstable) {
    r_idx_.resize(r_idx_mark);
    r_val_.resize(r_val_mark);
    return FactorStatus::NeedRefactor;
  }

  // Commit. Remove the old column p (and its mirror entries in the row lists).
  std::size_t removed = u_cols_.size(p);
  for (Index t = 0; t < u_cols_.size(p); ++t) {
    const Index row = u_cols_.index_at(p, t);
    u_rows_.erase(row, u_rows_.find(row, p));
  }
  u_cols_.clear_list(p);
  // Remove the row of r0 (its entries are eliminated) from the column lists.
  removed += to_size(u_rows_.size(r0));
  for (Index t = 0; t < u_rows_.size(r0); ++t) {
    const Index pos = u_rows_.index_at(r0, t);
    u_cols_.erase(pos, u_cols_.find(pos, r0));
  }
  u_rows_.clear_list(r0);

  // Insert the spike as the new column p.
  std::size_t added = 0;
  for (const Index row : spike_.indices()) {
    if (row == r0) continue;
    const double v = spike_[row];
    if (v == 0.0) continue;
    u_cols_.push(p, row, v);
    u_rows_.push(row, p, v);
    ++added;
  }
  nnz_u_ = nnz_u_ + added - removed;

  // Move the pivot to the end of the order.
  row_of_slot_[to_size(slot_of_row_[to_size(r0)])] = -1;
  slot_of_row_[to_size(r0)] = n_slots_;
  row_of_slot_[to_size(n_slots_)] = r0;
  ++n_slots_;
  diag_[to_size(r0)] = new_diag;

  r_pivot_.push_back(r0);
  r_start_.push_back(r_idx_.size());
  ++update_count_;
  spike_valid_ = false;
  return FactorStatus::Ok;
}

}  // namespace shodhan
