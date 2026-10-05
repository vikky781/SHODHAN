#pragma once

// Sparse LDL^T factorization of symmetric (quasi-definite) matrices (docs/LDL.md).
//
// A symmetric matrix is passed as its LOWER triangle in CSC (a SparseMatrix with row >= column, diagonal included, no
// duplicate entries), like the quadratic term of an LpModel.
//
//   1. amd_order()      approximate minimum degree ordering on the quotient graph (Amestoy, Davis, Duff 1996).
//   2. analyze_ldl()    elimination tree (Liu 1990), postorder, column counts and the nonzero count of L.
//   3. SparseLdl        numeric up-looking LDL^T with a static pivot order, static and dynamic regularization and
//                       iterative refinement against the unregularized matrix.

#include <cstdint>
#include <span>
#include <vector>

#include "shodhan/sparse_matrix.hpp"

namespace shodhan {

struct AmdStats {
  Index pivots = 0;            ///< supervariable pivots chosen
  Index supervariable_merges = 0;  ///< variables found indistinguishable and merged into another
  Index mass_eliminations = 0;     ///< variables eliminated together with the pivot (same closed neighbourhood)
  Index absorbed_elements = 0;     ///< elements absorbed into a newer element
};

/// Approximate minimum degree ordering of the symmetric matrix with the given lower-triangle pattern. Returns
/// perm with perm[new] = old; it is always a permutation of 0..n-1.
///
/// What is implemented: the quotient graph with elements and supervariables, element absorption (including the
/// aggressive absorption of elements with an empty external part), the approximate external degree of Amestoy, Davis
/// and Duff, indistinguishable-variable detection by hashing, and mass elimination. What is not: the in-place memory
/// layout, garbage collection, and the special treatment of dense rows.
std::vector<Index> amd_order(const SparseMatrix& lower, AmdStats* stats = nullptr);

std::vector<Index> natural_order(Index n);

/// Result of the symbolic analysis of P K P^T for an ordering P.
struct SymbolicLdl {
  Index n = 0;
  std::vector<Index> perm;       ///< perm[new] = old, the given ordering followed by the postorder of its elimination tree
  std::vector<Index> iperm;      ///< iperm[old] = new
  std::vector<Index> parent;     ///< elimination tree of the permuted matrix (-1 for a root)
  std::vector<Index> col_count;  ///< entries strictly below the diagonal in each column of L
  long long nnz_l = 0;           ///< sum of col_count
  double flops = 0.0;            ///< sum of col_count^2 (about the number of multiplications of the factorization)
};

/// Elimination tree, postorder and column counts. `order` is perm[new] = old.
SymbolicLdl analyze_ldl(const SparseMatrix& lower, const std::vector<Index>& order);

/// The nonzero pattern of L (below the diagonal) of the permuted matrix, one sorted row list per column; used to
/// verify the symbolic analysis and for fill reports.
std::vector<std::vector<Index>> ldl_pattern(const SparseMatrix& lower, const SymbolicLdl& sym);

struct LdlParams {
  /// Added to the diagonal entries of rows with sign +1 (the positive definite block) and subtracted from those of
  /// rows with sign -1 (the negative definite block): the static regularization, applied in the factorization only.
  double rho = 1e-9;
  double delta = 1e-9;
  /// Dynamic regularization: a pivot d_k whose sign-adjusted value sign_k * d_k is below pivot_tol * scale is replaced
  /// by sign_k * dynamic_delta * scale, where scale is the largest absolute diagonal entry; each replacement is counted.
  double pivot_tol = 1e-14;
  double dynamic_delta = 1e-8;
};

struct LdlStats {
  Index n = 0;
  long long nnz_l = 0;
  double flops = 0.0;
  long long dynamic_regularizations = 0;  ///< pivots replaced during the last factorization
  double min_pivot = 0.0;                 ///< smallest sign-adjusted pivot of the last factorization, before any replacement
  Index min_pivot_column = -1;            ///< the column of the ORIGINAL matrix that pivot belongs to
  int refinement_steps = 0;               ///< steps taken by the last solve
  double residual = 0.0;                  ///< relative residual of the last solve against the unregularized matrix
  long long factorizations = 0;
};

class SparseLdl {
 public:
  enum class Ordering { Natural, Amd };

  explicit SparseLdl(const LdlParams& params = {}) : params_(params) {}

  /// Orders and analyzes the pattern. `sign[i]` is +1 or -1 for row i (the block it belongs to). Returns false for a
  /// malformed matrix.
  bool analyze(const SparseMatrix& lower, const std::vector<signed char>& sign, Ordering ordering = Ordering::Amd);

  /// Numeric factorization of the matrix with the analyzed pattern (same dimensions and number of entries; the
  /// pattern itself must be the analyzed one). The values given here are also kept for the residuals of the
  /// refinement. Returns false if a pivot was not finite.
  bool factorize(const SparseMatrix& lower);

  /// Solves K x = b with the regularized factors and up to `max_refinement` steps of iterative refinement against the
  /// unregularized K (a step is kept only if it lowers the residual). Does not allocate after the first call.
  void solve(std::span<const double> b, std::span<double> x, int max_refinement = 3, double tol = 1e-13);

  const LdlStats& stats() const { return stats_; }
  const SymbolicLdl& symbolic() const { return sym_; }
  const LdlParams& params() const { return params_; }
  void set_params(const LdlParams& p) { params_ = p; }
  bool factored() const { return factored_; }

 private:
  void solve_factors(std::span<double> y) const;  // in place, permuted space
  double residual(std::span<const double> b, std::span<const double> x, std::vector<double>& r) const;

  LdlParams params_;
  LdlStats stats_;
  SymbolicLdl sym_;
  bool analyzed_ = false;
  bool factored_ = false;
  Index n_ = 0;
  std::size_t nnz_ = 0;
  std::vector<signed char> psign_;  // sign in permuted order

  // The permuted upper triangle (column k holds rows < k) and the position of every input entry in it.
  std::vector<Index> up_, ui_;
  std::vector<double> ux_;
  std::vector<double> diag_;
  std::vector<Index> map_;  // input entry p -> position in ux_, or -1 - j for the diagonal of permuted column j
  // The original lower triangle values (for residuals) and its pattern.
  SparseMatrix k_;

  // Factors: unit lower L in CSC (column i: rows > i), D.
  std::vector<Index> lp_, li_, lnz_;
  std::vector<double> lx_, d_;

  // Work (sized once).
  mutable std::vector<double> y_, w_, res_, dx_, xs_;
  std::vector<Index> flag_, stack_, pattern_;
};

}  // namespace shodhan
