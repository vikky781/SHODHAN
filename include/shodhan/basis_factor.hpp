#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "shodhan/detail/list_arena.hpp"
#include "shodhan/sparse_matrix.hpp"
#include "shodhan/sparse_work.hpp"

namespace shodhan {

enum class FactorStatus {
  Ok,
  /// The basis matrix is (numerically) singular. See BasisFactor::deficient_positions().
  RankDeficient,
  /// update() declined: an instability was detected or a refactor limit was reached.
  NeedRefactor,
};

const char* to_string(FactorStatus status) noexcept;

class LuEliminator;

/// Tunable parameters. The numbers are defaults/targets, not guarantees.
struct FactorParams {
  /// Relative pivot threshold u: a pivot must satisfy |a| >= u * (largest
  /// magnitude in its column of the active submatrix). Default (target): 0.1.
  double pivot_threshold = 0.1;
  /// A column whose largest active entry is at most this is numerically empty
  /// (rank deficient). Default (target): 1e-11.
  double abs_pivot_tol = 1e-11;
  /// A column whose largest active entry is at most this fraction of the
  /// largest magnitude it started with is numerically empty. Default: 1e-11.
  double rel_pivot_tol = 1e-11;
  /// Number of candidate columns/rows examined by the Markowitz search once a
  /// pivot has been found. Default (target): 4.
  Index markowitz_search = 4;
  /// Absolute drop tolerance applied to solve results and saved spikes.
  /// Default: 1e-14.
  double drop_tol = 1e-14;
  /// Forrest-Tomlin updates allowed before update() asks for a refactor.
  /// Default (target): 100.
  Index max_updates = 100;
  /// update() asks for a refactor when nnz(L)+nnz(U)+nnz(R) exceeds this
  /// multiple of nnz(L)+nnz(U) right after the last factorization.
  /// Default (target): 3.
  double max_growth = 3.0;
  /// update() asks for a refactor when the two computations of the new
  /// diagonal differ by more than this relative amount. Default (target): 1e-8.
  double update_mismatch_tol = 1e-8;
  /// update() asks for a refactor when the new diagonal (or the pivot element
  /// of the entering column) is at most this in magnitude. Default (target): 1e-11.
  double update_pivot_tol = 1e-11;
  /// A solve takes the hypersparse path for a stage when the number of
  /// listed nonzeros is below this fraction of m. 0 forces the dense path,
  /// values above 1 force the hypersparse path. Default (target): 0.10.
  double hyper_threshold = 0.10;
};

/// One basis column replaced by repair().
struct BasisSubstitution {
  Index position = 0;
  Index old_var = 0;
  Index new_var = 0;
};

/// Counters and sizes of the current factorization. Call counters accumulate
/// until reset_counters().
struct FactorStats {
  Index m = 0;
  std::size_t nnz_basis = 0;
  std::size_t nnz_l = 0;  ///< off-diagonal entries of L (unit diagonal not counted)
  std::size_t nnz_u = 0;  ///< entries of U including the diagonal, current (after updates)
  std::size_t nnz_r = 0;  ///< entries of the row etas added by Forrest-Tomlin updates
  double fill_ratio = 0.0;  ///< (nnz_l + nnz_u) right after factorization / nnz_basis
  Index column_singleton_pivots = 0;
  Index row_singleton_pivots = 0;
  Index singleton_pivots = 0;  ///< column + row singletons
  Index markowitz_pivots = 0;
  Index updates = 0;  ///< Forrest-Tomlin updates since the last factorization
  std::size_t ftran_calls = 0;
  std::size_t btran_calls = 0;
  /// Triangular-solve stages (L, R, U) that took each path. Every ftran or
  /// btran runs two such solves (L and U).
  std::size_t hyper_solves = 0;
  std::size_t dense_solves = 0;
};

/// All integer and floating-point state of a factorization, flattened in a
/// fixed order, so that two factorizations can be compared bit for bit.
struct FactorDump {
  std::vector<Index> ints;
  std::vector<double> reals;
};

/// Sparse LU factorization of a simplex basis with Forrest-Tomlin updates.
///
/// Computational form (docs/CONVENTIONS.md): the model is A x - r = 0. The
/// basis is a list of m variable indices, structural 0..n-1 (column j of A) and
/// logical n..n+m-1 (the column -e_i, i = var - n). The basis matrix B has one
/// column per basis position.
///
/// Spaces: ftran takes a right-hand side indexed by row and returns the
/// solution indexed by basis position; btran takes a vector indexed by basis
/// position and returns the solution indexed by row.
///
/// Not thread-safe. Identical inputs give bit-identical results.
class BasisFactor {
 public:
  BasisFactor() = default;
  explicit BasisFactor(const FactorParams& params) : params_(params) {}

  const FactorParams& params() const noexcept { return params_; }
  void set_params(const FactorParams& params) noexcept { params_ = params; }

  /// Factorizes the basis. A.n_rows must equal basis_vars.size(). Throws
  /// std::invalid_argument for a size mismatch or an out-of-range variable and
  /// std::overflow_error when m or the nonzero count does not fit in int32.
  /// Returns RankDeficient when some column has no acceptable pivot; the
  /// factorization is then not usable for solves until repair() succeeds.
  FactorStatus factorize(const SparseMatrix& A, const std::vector<Index>& basis_vars);

  /// After factorize() returned RankDeficient: replaces the deficient basis
  /// columns, in ascending position, by the logical columns of the rows left
  /// without a pivot, in ascending row, then refactorizes. basis_vars is
  /// updated in place. Returns the substitutions made (empty when the last
  /// factorization was not rank deficient); status() is the status of the
  /// refactorization. Deterministic.
  std::vector<BasisSubstitution> repair(const SparseMatrix& A, std::vector<Index>& basis_vars);

  /// Solves B x = a in place: on entry rhs holds a (indexed by row), on exit x
  /// (indexed by basis position). With save_spike the partially transformed
  /// column and the solution are kept for the next update(). Throws
  /// std::logic_error without a valid factorization.
  void ftran(SparseWork& rhs, bool save_spike = false);

  /// Solves B^T y = c in place: on entry rhs holds c (indexed by basis
  /// position), on exit y (indexed by row).
  void btran(SparseWork& rhs);

  /// Forrest-Tomlin update after replacing the basis column at
  /// leaving_position by the column whose ftran(save_spike = true) was done
  /// last. Returns Ok, or NeedRefactor, in which case the factorization is left
  /// unchanged (still valid for the old basis) and the caller should
  /// refactorize the new basis. NeedRefactor is returned for: no saved spike,
  /// the update limit, growth beyond max_growth, a pivot element at most
  /// update_pivot_tol, or a mismatch of the two diagonal computations.
  FactorStatus update(Index leaving_position);

  Index dimension() const noexcept { return m_; }
  /// True when the last factorize()/repair() produced a usable factorization.
  bool valid() const noexcept { return valid_; }
  FactorStatus status() const noexcept { return status_; }

  /// Basis positions without an acceptable pivot, ascending (after a
  /// RankDeficient factorize).
  const std::vector<Index>& deficient_positions() const noexcept { return deficient_positions_; }
  /// Rows left without a pivot, ascending; same length as deficient_positions().
  const std::vector<Index>& unpivoted_rows() const noexcept { return unpivoted_rows_; }

  FactorStats stats() const;
  std::size_t nnz_l() const noexcept { return l_idx_.size(); }
  std::size_t nnz_u() const noexcept { return nnz_u_; }
  Index update_count() const noexcept { return update_count_; }
  void reset_counters() noexcept;

  /// Flattened state for exact comparison (tests, determinism checks).
  FactorDump dump() const;

 private:
  template <class Adj>
  void dfs_reach(const SparseWork& w, Adj&& adj);

  void install(LuEliminator& elim);
  void ftran_l_dense(SparseWork& w);
  void ftran_l_hyper(SparseWork& w);
  void ftran_r(SparseWork& w);
  void ftran_u_dense(SparseWork& w);
  void ftran_u_hyper(SparseWork& w);
  void btran_u_dense(SparseWork& w);
  void btran_u_hyper(SparseWork& w);
  void btran_r(SparseWork& w);
  void btran_lt_dense(SparseWork& w);
  void btran_lt_hyper(SparseWork& w);
  bool use_hyper(const SparseWork& w) const noexcept;

  FactorParams params_;
  Index m_ = 0;
  bool valid_ = false;
  FactorStatus status_ = FactorStatus::Ok;
  std::vector<Index> deficient_positions_;
  std::vector<Index> unpivoted_rows_;

  // Pivot structure. Pivot id = the pivot row. Slots give the triangular
  // order of the rows of U; updates append a slot and kill the old one (-1).
  Index n_slots_ = 0;
  std::vector<Index> row_of_slot_;
  std::vector<Index> slot_of_row_;
  std::vector<Index> pos_of_row_;
  std::vector<Index> row_of_pos_;
  std::vector<double> diag_;  // by pivot row

  // U off-diagonals: by row (entries: position, value) and by position (entries: row, value).
  detail::ListArena u_rows_;
  detail::ListArena u_cols_;
  std::size_t nnz_u_ = 0;  // including the diagonal

  // L: column etas in elimination order (only nonempty ones are stored).
  std::vector<Index> l_pivot_;
  std::vector<std::size_t> l_start_;
  std::vector<Index> l_idx_;
  std::vector<double> l_val_;
  std::vector<Index> eta_of_row_;  // index of the eta pivoting on a row, or -1
  // Row-wise copy of L for hypersparse btran: row i lists (pivot row, multiplier).
  std::vector<std::size_t> lt_start_;
  std::vector<Index> lt_idx_;
  std::vector<double> lt_val_;

  // Forrest-Tomlin row etas: w[r_pivot] -= sum r_val * w[r_idx].
  std::vector<Index> r_pivot_;
  std::vector<std::size_t> r_start_;
  std::vector<Index> r_idx_;
  std::vector<double> r_val_;

  Index update_count_ = 0;
  std::size_t fresh_nnz_ = 0;
  std::size_t nnz_basis_ = 0;
  double fill_ratio_ = 0.0;
  Index n_col_singletons_ = 0;
  Index n_row_singletons_ = 0;
  Index n_markowitz_ = 0;

  // Saved by ftran(save_spike = true).
  bool spike_valid_ = false;
  SparseWork spike_;
  SparseWork saved_x_;

  // Work buffers, sized in factorize() and reused.
  SparseWork scratch_;
  SparseWork row_work_;
  std::vector<Index> reach_;
  std::vector<Index> dfs_node_;
  std::vector<const Index*> dfs_cur_;
  std::vector<const Index*> dfs_end_;
  std::vector<std::uint32_t> visit_;
  std::uint32_t stamp_ = 0;
  std::vector<Index> heap_;

  mutable std::size_t ftran_calls_ = 0;
  mutable std::size_t btran_calls_ = 0;
  mutable std::size_t hyper_solves_ = 0;
  mutable std::size_t dense_solves_ = 0;
};

}  // namespace shodhan
