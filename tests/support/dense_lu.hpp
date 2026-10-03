#pragma once

// Dense LU with partial pivoting: a slow, simple oracle for the sparse basis
// factorization (test-only code, never linked into the library).

#include <vector>

#include "shodhan/sparse_matrix.hpp"

namespace shodhan::testing {

/// Row-major dense n x n matrix.
struct DenseMatrix {
  Index n = 0;
  std::vector<double> a;

  DenseMatrix() = default;
  explicit DenseMatrix(Index size) : n(size), a(to_size(size) * to_size(size), 0.0) {}
  double& at(Index i, Index j) { return a[to_size(i) * to_size(n) + to_size(j)]; }
  double at(Index i, Index j) const { return a[to_size(i) * to_size(n) + to_size(j)]; }

  /// Induced infinity norm (maximum absolute row sum).
  double norm_inf() const;
  /// Induced 1-norm (maximum absolute column sum).
  double norm_one() const;
  std::vector<double> multiply(const std::vector<double>& x) const;
  std::vector<double> multiply_transpose(const std::vector<double>& x) const;
};

/// LU factorization P A = L U with partial (row) pivoting.
class DenseLu {
 public:
  explicit DenseLu(const DenseMatrix& a);

  /// True when a pivot was exactly zero (the matrix is singular).
  bool singular() const noexcept { return singular_; }
  /// Smallest absolute pivot divided by the largest.
  double pivot_ratio() const noexcept { return pivot_ratio_; }

  std::vector<double> solve(const std::vector<double>& b) const;            // A x = b
  std::vector<double> solve_transpose(const std::vector<double>& c) const;  // A^T y = c

  /// Infinity norm of the inverse, from n solves. Infinite when singular.
  double inverse_norm_inf() const;

 private:
  Index n_ = 0;
  std::vector<double> lu_;
  std::vector<Index> perm_;
  bool singular_ = false;
  double pivot_ratio_ = 0.0;
};

/// Numerical rank by Gaussian elimination with complete pivoting on the
/// matrix with rows and columns scaled to unit maximum: a pivot is accepted when
/// its magnitude exceeds rel_tol (the scaled matrix has maximum 1).
Index dense_rank(const DenseMatrix& a, double rel_tol);

}  // namespace shodhan::testing
