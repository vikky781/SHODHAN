#pragma once

// Internal storage helper for the sparse LU: many short growable lists packed
// into one flat arena (no vector of vectors). Not part of the public API.

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include "shodhan/sparse_matrix.hpp"

namespace shodhan::detail {

/// A set of n lists of (Index, optional double) entries stored in two flat
/// arrays. A list owns the slots [start, start + cap) of the arena and uses the
/// first len of them. A list that outgrows its capacity moves to the end of the
/// arena (leaving a hole); when holes pile up the arena is compacted into a
/// second buffer that is kept, so the steady state does not allocate.
/// Offsets are std::size_t; indices and lengths are Index (int32).
class ListArena {
 public:
  /// Creates n empty lists, list l with initial capacity caps[l] (all zero when
  /// caps is null) and `slack` spare slots at the end of the arena. Buffers
  /// from an earlier use are reused.
  void reset(Index n, bool with_values, const Index* caps = nullptr, std::size_t slack = 0) {
    n_ = n;
    with_values_ = with_values;
    const std::size_t sn = to_size(n);
    start_.assign(sn, 0);
    len_.assign(sn, 0);
    cap_.assign(sn, 0);
    std::size_t pos = 0;
    for (std::size_t l = 0; l < sn; ++l) {
      start_[l] = pos;
      const Index c = caps != nullptr ? caps[l] : 0;
      cap_[l] = c;
      pos += to_size(c);
    }
    end_ = pos;
    waste_ = 0;
    ensure_size(end_ + slack);
  }

  Index n_lists() const noexcept { return n_; }
  Index size(Index l) const noexcept { return len_[to_size(l)]; }

  std::span<const Index> indices(Index l) const noexcept {
    return {idx_.data() + start_[to_size(l)], to_size(len_[to_size(l)])};
  }
  std::span<const double> values(Index l) const noexcept {
    return {val_.data() + start_[to_size(l)], to_size(len_[to_size(l)])};
  }
  Index index_at(Index l, Index p) const noexcept { return idx_[start_[to_size(l)] + to_size(p)]; }
  double& value_at(Index l, Index p) noexcept { return val_[start_[to_size(l)] + to_size(p)]; }
  double value_at(Index l, Index p) const noexcept { return val_[start_[to_size(l)] + to_size(p)]; }

  /// Position of index i inside list l, or -1.
  Index find(Index l, Index i) const noexcept {
    const std::size_t s = start_[to_size(l)];
    const Index len = len_[to_size(l)];
    for (Index p = 0; p < len; ++p) {
      if (idx_[s + to_size(p)] == i) return p;
    }
    return -1;
  }

  void push(Index l, Index i, double v = 0.0) {
    const std::size_t sl = to_size(l);
    if (len_[sl] == cap_[sl]) grow(l);
    const std::size_t at = start_[sl] + to_size(len_[sl]);
    idx_[at] = i;
    if (with_values_) val_[at] = v;
    ++len_[sl];
  }

  /// Removes the entry at position p by moving the last entry into its place.
  void erase(Index l, Index p) noexcept {
    const std::size_t sl = to_size(l);
    const std::size_t s = start_[sl];
    const std::size_t last = s + to_size(len_[sl]) - 1;
    const std::size_t at = s + to_size(p);
    idx_[at] = idx_[last];
    if (with_values_) val_[at] = val_[last];
    --len_[sl];
  }

  /// Empties list l; its capacity stays reserved until the next compaction.
  void clear_list(Index l) noexcept { len_[to_size(l)] = 0; }

  /// Total number of live entries.
  std::size_t live_entries() const noexcept {
    std::size_t s = 0;
    for (const Index len : len_) s += to_size(len);
    return s;
  }

 private:
  void ensure_size(std::size_t sz) {
    if (idx_.size() < sz) {
      const std::size_t target = std::max(sz, idx_.size() * 2);
      idx_.resize(target);
      if (with_values_) val_.resize(target);
    }
  }

  void grow(Index l) {
    const std::size_t sl = to_size(l);
    std::size_t newcap = std::max<std::size_t>(4, to_size(cap_[sl]) * 2);
    if (end_ + newcap > idx_.size() && waste_ > end_ / 4 + 16) {
      compact();
      newcap = std::max<std::size_t>(4, to_size(cap_[sl]) * 2);
    }
    ensure_size(end_ + newcap);
    const std::size_t old_start = start_[sl];
    const std::size_t len = to_size(len_[sl]);
    for (std::size_t t = 0; t < len; ++t) idx_[end_ + t] = idx_[old_start + t];
    if (with_values_) {
      for (std::size_t t = 0; t < len; ++t) val_[end_ + t] = val_[old_start + t];
    }
    waste_ += to_size(cap_[sl]);
    start_[sl] = end_;
    cap_[sl] = static_cast<Index>(newcap);
    end_ += newcap;
  }

  // Packs every list tightly (capacity := length) into the back buffer, then
  // swaps buffers. The back buffer is kept at the same size as the main one.
  void compact() {
    idx2_.resize(idx_.size());
    if (with_values_) val2_.resize(val_.size());
    std::size_t pos = 0;
    const std::size_t sn = to_size(n_);
    for (std::size_t l = 0; l < sn; ++l) {
      const std::size_t s = start_[l];
      const std::size_t len = to_size(len_[l]);
      for (std::size_t t = 0; t < len; ++t) idx2_[pos + t] = idx_[s + t];
      if (with_values_) {
        for (std::size_t t = 0; t < len; ++t) val2_[pos + t] = val_[s + t];
      }
      start_[l] = pos;
      cap_[l] = len_[l];
      pos += len;
    }
    idx_.swap(idx2_);
    if (with_values_) val_.swap(val2_);
    end_ = pos;
    waste_ = 0;
  }

  Index n_ = 0;
  bool with_values_ = false;
  std::vector<std::size_t> start_;
  std::vector<Index> len_;
  std::vector<Index> cap_;
  std::vector<Index> idx_, idx2_;
  std::vector<double> val_, val2_;
  std::size_t end_ = 0;
  std::size_t waste_ = 0;
};

}  // namespace shodhan::detail
