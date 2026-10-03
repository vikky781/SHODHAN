#pragma once

// Internal: sparse right-looking LU elimination with Markowitz pivot choice and
// threshold partial pivoting, used by BasisFactor::factorize. See docs/LU.md.

#include <cstddef>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "shodhan/detail/list_arena.hpp"
#include "shodhan/sparse_matrix.hpp"

namespace shodhan {

class LuEliminator {
 public:
  LuEliminator(const FactorParams& params, const SparseMatrix& A, const std::vector<Index>& basis);

  /// Runs the elimination to completion (or until every column is either
  /// pivoted or declared deficient).
  void run();

  Index m = 0;
  std::size_t nnz_basis = 0;

  // One entry per pivot step, in elimination order.
  std::vector<Index> piv_row;
  std::vector<Index> piv_pos;
  std::vector<double> piv_val;
  // Row-wise U off-diagonals of each step: entries [u_start[k], u_start[k+1]).
  std::vector<std::size_t> u_start{0};
  std::vector<Index> u_pos;
  std::vector<double> u_val;
  // L etas (only nonempty ones): pivot row and entries [l_start[e], l_start[e+1]).
  std::vector<Index> l_pivot;
  std::vector<std::size_t> l_start{0};
  std::vector<Index> l_idx;
  std::vector<double> l_val;

  std::vector<Index> deficient;  // ascending positions
  std::vector<Index> unpivoted;  // ascending rows
  Index n_col_singletons = 0;
  Index n_row_singletons = 0;
  Index n_markowitz = 0;

 private:
  struct Buckets {
    std::vector<Index> head, next, prev, where;
    void init(Index n_items, Index max_count);
    void insert(Index item, Index count) noexcept;
    void remove(Index item) noexcept;
  };

  struct Candidate {
    Index row = -1;
    Index col = -1;
    double val = 0.0;
    long long mk = 0;
  };

  Candidate find_pivot();
  void examine_column(Index j, Index cnt, Candidate& best, bool& restart);
  void examine_row(Index i, Index cnt, Candidate& best, bool& restart);
  void consider(Index i, Index j, double v, long long mk, Candidate& best) const;
  void drop_column(Index j);
  void pivot(const Candidate& cand);
  void eliminate_column(Index j, double arj);
  void rebucket_row(Index i);
  void touch_row(Index i);
  void touch_col(Index j);
  bool column_is_empty(Index j, double cmax) const noexcept;

  FactorParams params_;
  Index nsearch_ = 4;
  Index active_cols_ = 0;

  detail::ListArena cols_;  // by position: (row, value)
  detail::ListArena rows_;  // by row: (position)
  std::vector<double> col_orig_max_;
  Buckets col_b_, row_b_;
  std::vector<char> col_done_, row_done_;

  // Scratch for pivot().
  std::vector<Index> mult_row_;
  std::vector<double> mult_val_;
  std::vector<Index> rowpat_;
  std::vector<Index> wpos_;
  std::vector<Index> zero_list_;
  std::vector<Index> touched_rows_, touched_cols_;
  std::vector<std::uint32_t> row_stamp_, col_stamp_;
  std::uint32_t stamp_ = 0;
};

}  // namespace shodhan
