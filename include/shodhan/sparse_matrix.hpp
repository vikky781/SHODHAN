#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace shodhan {

/// Row/column index and offset type. Matrices are limited to INT32_MAX
/// nonzeros.
using Index = std::int32_t;

constexpr std::size_t to_size(Index i) noexcept { return static_cast<std::size_t>(i); }

struct Triplet {
  Index row = 0;
  Index col = 0;
  double value = 0.0;
};

class CsrMatrix;

/// Compressed sparse column matrix. Column j owns the entries
/// [col_start[j], col_start[j+1]) of row_index/value; row indices within a
/// column are strictly increasing. Explicit zero values are allowed and kept.
class SparseMatrix {
 public:
  Index n_rows = 0;
  Index n_cols = 0;
  std::vector<Index> col_start{0};
  std::vector<Index> row_index;
  std::vector<double> value;

  SparseMatrix() = default;
  /// All-zero matrix of the given shape (negative sizes are treated as 0 in
  /// the storage; validate() reports them).
  SparseMatrix(Index rows, Index cols);

  /// Builds a matrix from triplets (any order). Fails, setting *error, on an
  /// out-of-range index, a NaN/Inf value or a duplicate coordinate; the
  /// duplicate message names the 0-based (row, col). *out is only modified on
  /// success.
  static bool from_triplets(Index n_rows, Index n_cols, std::vector<Triplet> triplets,
                            SparseMatrix* out, std::string* error);

  std::size_t nnz() const noexcept { return value.size(); }

  /// Structural checks: sizes, monotone col_start, sorted unique row indices
  /// in range, finite values. Empty result means valid.
  std::vector<std::string> validate() const;

  /// CSR copy of the same matrix (row-major view of identical contents).
  CsrMatrix to_csr() const;

  /// y = A x. Throws std::invalid_argument when sizes do not match.
  void multiply(std::span<const double> x, std::span<double> y) const;
  /// y = A^T x. Throws std::invalid_argument when sizes do not match.
  void multiply_transpose(std::span<const double> x, std::span<double> y) const;

  std::vector<Index> row_counts() const;
  std::vector<Index> col_counts() const;

  /// Smallest/largest absolute value among nonzero stored values; 0 if there
  /// are none.
  double min_abs() const;
  double max_abs() const;

  friend bool operator==(const SparseMatrix&, const SparseMatrix&) = default;
};

/// Compressed sparse row matrix: row i owns [row_start[i], row_start[i+1]) of
/// col_index/value with strictly increasing column indices.
class CsrMatrix {
 public:
  Index n_rows = 0;
  Index n_cols = 0;
  std::vector<Index> row_start{0};
  std::vector<Index> col_index;
  std::vector<double> value;

  std::size_t nnz() const noexcept { return value.size(); }

  /// Back to CSC. to_csr().to_csc() reproduces the original matrix.
  SparseMatrix to_csc() const;

  /// y = A x. Throws std::invalid_argument when sizes do not match.
  void multiply(std::span<const double> x, std::span<double> y) const;

  friend bool operator==(const CsrMatrix&, const CsrMatrix&) = default;
};

}  // namespace shodhan
