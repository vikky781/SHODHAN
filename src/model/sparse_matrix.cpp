#include "shodhan/sparse_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace shodhan {

namespace {

constexpr std::size_t kMaxReportedProblems = 20;

// Transposes a compressed matrix: input has `n_major` compressed vectors over
// `n_minor` positions; output has `n_minor` vectors over `n_major` positions.
// Used for both CSC->CSR and CSR->CSC.
void transpose_compressed(Index n_major, Index n_minor, const std::vector<Index>& start,
                          const std::vector<Index>& index, const std::vector<double>& value,
                          std::vector<Index>* out_start, std::vector<Index>* out_index,
                          std::vector<double>* out_value) {
  out_start->assign(to_size(n_minor) + 1, 0);
  for (const Index i : index) ++(*out_start)[to_size(i) + 1];
  for (std::size_t k = 0; k < to_size(n_minor); ++k) (*out_start)[k + 1] += (*out_start)[k];

  out_index->resize(index.size());
  out_value->resize(value.size());
  std::vector<Index> next(out_start->begin(), out_start->end() - 1);
  for (Index major = 0; major < n_major; ++major) {
    for (Index p = start[to_size(major)]; p < start[to_size(major) + 1]; ++p) {
      const Index minor = index[to_size(p)];
      const Index dst = next[to_size(minor)]++;
      (*out_index)[to_size(dst)] = major;
      (*out_value)[to_size(dst)] = value[to_size(p)];
    }
  }
}

}  // namespace

SparseMatrix::SparseMatrix(Index rows, Index cols)
    : n_rows(rows), n_cols(cols), col_start(to_size(cols > 0 ? cols : 0) + 1, 0) {}

bool SparseMatrix::from_triplets(Index n_rows, Index n_cols, std::vector<Triplet> triplets,
                                 SparseMatrix* out, std::string* error) {
  auto fail = [&](std::string message) {
    if (error != nullptr) *error = std::move(message);
    return false;
  };
  if (n_rows < 0 || n_cols < 0) return fail("negative matrix dimensions");
  if (triplets.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
    return fail("too many nonzeros for 32-bit indices");
  }
  for (const Triplet& t : triplets) {
    if (t.row < 0 || t.row >= n_rows || t.col < 0 || t.col >= n_cols) {
      return fail("triplet index out of range: (row=" + std::to_string(t.row) +
                  ", col=" + std::to_string(t.col) + ") in a " + std::to_string(n_rows) + "x" +
                  std::to_string(n_cols) + " matrix");
    }
    if (!std::isfinite(t.value)) {
      return fail("non-finite value at (row=" + std::to_string(t.row) +
                  ", col=" + std::to_string(t.col) + ")");
    }
  }
  std::sort(triplets.begin(), triplets.end(), [](const Triplet& a, const Triplet& b) {
    return a.col != b.col ? a.col < b.col : a.row < b.row;
  });
  for (std::size_t k = 1; k < triplets.size(); ++k) {
    if (triplets[k].col == triplets[k - 1].col && triplets[k].row == triplets[k - 1].row) {
      return fail("duplicate entry at (row=" + std::to_string(triplets[k].row) +
                  ", col=" + std::to_string(triplets[k].col) + ") (0-based)");
    }
  }

  SparseMatrix m(n_rows, n_cols);
  m.row_index.reserve(triplets.size());
  m.value.reserve(triplets.size());
  for (const Triplet& t : triplets) {
    ++m.col_start[to_size(t.col) + 1];
    m.row_index.push_back(t.row);
    m.value.push_back(t.value);
  }
  for (std::size_t j = 0; j < to_size(n_cols); ++j) m.col_start[j + 1] += m.col_start[j];
  if (out != nullptr) *out = std::move(m);
  return true;
}

std::vector<std::string> SparseMatrix::validate() const {
  std::vector<std::string> problems;
  auto add = [&](std::string message) {
    if (problems.size() < kMaxReportedProblems) problems.push_back(std::move(message));
  };
  if (n_rows < 0 || n_cols < 0) {
    add("negative dimensions");
    return problems;
  }
  if (col_start.size() != to_size(n_cols) + 1) {
    add("col_start has size " + std::to_string(col_start.size()) + ", expected " +
        std::to_string(to_size(n_cols) + 1));
    return problems;
  }
  if (row_index.size() != value.size()) {
    add("row_index and value sizes differ");
    return problems;
  }
  if (col_start[0] != 0) add("col_start[0] is not 0");
  for (std::size_t j = 0; j < to_size(n_cols); ++j) {
    if (col_start[j + 1] < col_start[j]) {
      add("col_start is not monotone at column " + std::to_string(j));
      return problems;
    }
  }
  if (to_size(col_start.back()) != row_index.size()) {
    add("col_start.back() does not equal the number of stored entries");
    return problems;
  }
  for (std::size_t j = 0; j < to_size(n_cols); ++j) {
    for (Index p = col_start[j]; p < col_start[j + 1]; ++p) {
      const Index r = row_index[to_size(p)];
      if (r < 0 || r >= n_rows) {
        add("row index " + std::to_string(r) + " out of range in column " + std::to_string(j));
      } else if (p > col_start[j] && r <= row_index[to_size(p) - 1]) {
        add("row indices not strictly increasing in column " + std::to_string(j));
      }
      if (!std::isfinite(value[to_size(p)])) {
        add("non-finite value in column " + std::to_string(j));
      }
    }
  }
  return problems;
}

CsrMatrix SparseMatrix::to_csr() const {
  CsrMatrix r;
  r.n_rows = n_rows;
  r.n_cols = n_cols;
  transpose_compressed(n_cols, n_rows, col_start, row_index, value, &r.row_start, &r.col_index,
                       &r.value);
  return r;
}

SparseMatrix CsrMatrix::to_csc() const {
  SparseMatrix m;
  m.n_rows = n_rows;
  m.n_cols = n_cols;
  transpose_compressed(n_rows, n_cols, row_start, col_index, value, &m.col_start, &m.row_index,
                       &m.value);
  return m;
}

void SparseMatrix::multiply(std::span<const double> x, std::span<double> y) const {
  if (x.size() != to_size(n_cols) || y.size() != to_size(n_rows)) {
    throw std::invalid_argument("SparseMatrix::multiply: size mismatch");
  }
  std::fill(y.begin(), y.end(), 0.0);
  for (std::size_t j = 0; j < to_size(n_cols); ++j) {
    const double xj = x[j];
    for (Index p = col_start[j]; p < col_start[j + 1]; ++p) {
      y[to_size(row_index[to_size(p)])] += value[to_size(p)] * xj;
    }
  }
}

void SparseMatrix::multiply_transpose(std::span<const double> x, std::span<double> y) const {
  if (x.size() != to_size(n_rows) || y.size() != to_size(n_cols)) {
    throw std::invalid_argument("SparseMatrix::multiply_transpose: size mismatch");
  }
  for (std::size_t j = 0; j < to_size(n_cols); ++j) {
    double sum = 0.0;
    for (Index p = col_start[j]; p < col_start[j + 1]; ++p) {
      sum += value[to_size(p)] * x[to_size(row_index[to_size(p)])];
    }
    y[j] = sum;
  }
}

void CsrMatrix::multiply(std::span<const double> x, std::span<double> y) const {
  if (x.size() != to_size(n_cols) || y.size() != to_size(n_rows)) {
    throw std::invalid_argument("CsrMatrix::multiply: size mismatch");
  }
  for (std::size_t i = 0; i < to_size(n_rows); ++i) {
    double sum = 0.0;
    for (Index p = row_start[i]; p < row_start[i + 1]; ++p) {
      sum += value[to_size(p)] * x[to_size(col_index[to_size(p)])];
    }
    y[i] = sum;
  }
}

std::vector<Index> SparseMatrix::row_counts() const {
  std::vector<Index> counts(to_size(n_rows), 0);
  for (const Index r : row_index) ++counts[to_size(r)];
  return counts;
}

std::vector<Index> SparseMatrix::col_counts() const {
  std::vector<Index> counts(to_size(n_cols), 0);
  for (std::size_t j = 0; j < to_size(n_cols); ++j) counts[j] = col_start[j + 1] - col_start[j];
  return counts;
}

double SparseMatrix::min_abs() const {
  double best = 0.0;
  bool found = false;
  for (const double v : value) {
    const double a = std::fabs(v);
    if (a == 0.0) continue;
    if (!found || a < best) best = a;
    found = true;
  }
  return found ? best : 0.0;
}

double SparseMatrix::max_abs() const {
  double best = 0.0;
  for (const double v : value) best = std::max(best, std::fabs(v));
  return best;
}

}  // namespace shodhan
