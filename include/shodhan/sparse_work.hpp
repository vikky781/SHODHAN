#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "shodhan/sparse_matrix.hpp"

namespace shodhan {

/// Dense value array of length n plus a list of the positions that may be
/// nonzero. Invariant: every nonzero entry of the value array is in the index
/// list, and no position is listed twice. A listed position may hold zero
/// (after cancellation or a drop).
///
/// clear() costs time proportional to count(), not n, so a sparse right-hand
/// side stays cheap to set up, solve and reset. All BasisFactor solves operate
/// on this type, in place.
class SparseWork {
 public:
  /// Default absolute drop tolerance used by drop_small().
  static constexpr double kDefaultDropTol = 1e-14;

  SparseWork() = default;
  explicit SparseWork(Index n) { resize(n); }

  /// Changes the length to n and clears the vector. Reserves the index list so
  /// that later operations never allocate.
  void resize(Index n) {
    const std::size_t sz = to_size(n < 0 ? 0 : n);
    values_.assign(sz, 0.0);
    mark_.assign(sz, 0);
    index_.clear();
    index_.reserve(sz);
  }

  Index size() const noexcept { return static_cast<Index>(values_.size()); }
  Index count() const noexcept { return static_cast<Index>(index_.size()); }
  /// count() / size(); 0 for an empty vector.
  double density() const noexcept {
    return values_.empty() ? 0.0 : static_cast<double>(index_.size()) / static_cast<double>(values_.size());
  }

  double operator[](Index i) const noexcept { return values_[to_size(i)]; }
  bool is_listed(Index i) const noexcept { return mark_[to_size(i)] != 0; }

  std::span<const Index> indices() const noexcept { return index_; }
  std::span<const double> values() const noexcept { return values_; }
  /// Raw access to the dense array for kernels that scan or fill it directly.
  /// After writing through it, call reindex() before relying on the index list.
  double* raw() noexcept { return values_.data(); }

  /// values[i] += v, listing i if it was not listed.
  void add(Index i, double v) noexcept {
    const std::size_t k = to_size(i);
    if (mark_[k] == 0) {
      mark_[k] = 1;
      index_.push_back(i);
    }
    values_[k] += v;
  }

  /// values[i] = v, listing i if it was not listed.
  void set(Index i, double v) noexcept {
    const std::size_t k = to_size(i);
    if (mark_[k] == 0) {
      mark_[k] = 1;
      index_.push_back(i);
    }
    values_[k] = v;
  }

  /// Sets every listed entry to zero and empties the list: O(count()).
  void clear() noexcept {
    for (const Index i : index_) {
      values_[to_size(i)] = 0.0;
      mark_[to_size(i)] = 0;
    }
    index_.clear();
  }

  /// Zeroes and unlists every entry with |value| <= tol (absolute, not
  /// relative to anything). Returns the number of entries removed. With
  /// tol = 0 only exact zeros are removed.
  Index drop_small(double tol = kDefaultDropTol) noexcept {
    std::size_t keep = 0;
    for (std::size_t t = 0; t < index_.size(); ++t) {
      const std::size_t k = to_size(index_[t]);
      if (std::fabs(values_[k]) <= tol) {
        values_[k] = 0.0;
        mark_[k] = 0;
      } else {
        index_[keep++] = index_[t];
      }
    }
    const Index dropped = static_cast<Index>(index_.size() - keep);
    index_.resize(keep);
    return dropped;
  }

  /// Rebuilds the index list from the value array (O(n)): lists exactly the
  /// nonzero entries. Use after writing through raw().
  void reindex() noexcept {
    for (const Index i : index_) mark_[to_size(i)] = 0;
    index_.clear();
    const std::size_t n = values_.size();
    for (std::size_t k = 0; k < n; ++k) {
      if (values_[k] != 0.0) {
        mark_[k] = 1;
        index_.push_back(static_cast<Index>(k));
      }
    }
  }

  /// Replaces the contents with the nonzeros of a dense vector (entries with
  /// |v| <= tol are skipped). The length becomes dense.size().
  void load_dense(std::span<const double> dense, double tol = 0.0) {
    if (static_cast<std::size_t>(size()) != dense.size()) {
      resize(static_cast<Index>(dense.size()));
    } else {
      clear();
    }
    for (std::size_t k = 0; k < dense.size(); ++k) {
      if (std::fabs(dense[k]) > tol) set(static_cast<Index>(k), dense[k]);
    }
  }

  /// Replaces the contents with (idx[t], val[t]) pairs; repeated indices add.
  void load_sparse(std::span<const Index> idx, std::span<const double> val) {
    clear();
    const std::size_t cnt = idx.size() < val.size() ? idx.size() : val.size();
    for (std::size_t t = 0; t < cnt; ++t) add(idx[t], val[t]);
  }

  /// Writes the vector into a dense array of length size().
  void to_dense(std::span<double> out) const {
    for (double& v : out) v = 0.0;
    for (const Index i : index_) out[to_size(i)] = values_[to_size(i)];
  }

  /// Copies the contents of other (resizing if the lengths differ).
  void assign(const SparseWork& other) {
    if (size() != other.size()) {
      resize(other.size());
    } else {
      clear();
    }
    for (const Index i : other.index_) set(i, other.values_[to_size(i)]);
  }

  void swap(SparseWork& other) noexcept {
    values_.swap(other.values_);
    mark_.swap(other.mark_);
    index_.swap(other.index_);
  }

  /// Largest absolute value over the listed entries.
  double norm_inf() const noexcept {
    double m = 0.0;
    for (const Index i : index_) m = std::fmax(m, std::fabs(values_[to_size(i)]));
    return m;
  }

 private:
  std::vector<double> values_;
  std::vector<unsigned char> mark_;
  std::vector<Index> index_;
};

}  // namespace shodhan
