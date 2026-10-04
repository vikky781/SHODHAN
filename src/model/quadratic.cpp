#include "shodhan/quadratic.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace shodhan {

void quad_multiply(const SparseMatrix& q, std::span<const double> x, std::span<double> y) {
  const std::size_t n = to_size(q.n_cols);
  if (q.nnz() == 0 && q.n_cols == 0) {
    std::fill(y.begin(), y.end(), 0.0);
    return;
  }
  if (x.size() != n || y.size() != n) throw std::invalid_argument("quad_multiply: size mismatch");
  std::fill(y.begin(), y.end(), 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (Index p = q.col_start[j]; p < q.col_start[j + 1]; ++p) {
      const std::size_t i = to_size(q.row_index[to_size(p)]);
      const double v = q.value[to_size(p)];
      y[i] += v * x[j];
      if (i != j) y[j] += v * x[i];
    }
  }
}

double quad_form(const SparseMatrix& q, std::span<const double> x) {
  const std::size_t n = to_size(q.n_cols);
  if (q.nnz() == 0) return 0.0;
  if (x.size() != n) throw std::invalid_argument("quad_form: size mismatch");
  double s = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    for (Index p = q.col_start[j]; p < q.col_start[j + 1]; ++p) {
      const std::size_t i = to_size(q.row_index[to_size(p)]);
      const double v = q.value[to_size(p)];
      s += (i == j ? v * x[j] * x[j] : 2.0 * v * x[i] * x[j]);
    }
  }
  return s;
}

SparseMatrix quad_full(const SparseMatrix& q) {
  std::vector<Triplet> t;
  t.reserve(2 * q.nnz());
  for (Index j = 0; j < q.n_cols; ++j) {
    for (Index p = q.col_start[to_size(j)]; p < q.col_start[to_size(j) + 1]; ++p) {
      const Index i = q.row_index[to_size(p)];
      t.push_back({i, j, q.value[to_size(p)]});
      if (i != j) t.push_back({j, i, q.value[to_size(p)]});
    }
  }
  SparseMatrix out;
  std::string err;
  if (!SparseMatrix::from_triplets(q.n_cols, q.n_cols, std::move(t), &out, &err)) {
    throw std::invalid_argument("quad_full: " + err);
  }
  return out;
}

double model_objective(const LpModel& m, std::span<const double> x) {
  double s = m.objective_offset;
  for (std::size_t j = 0; j < to_size(m.n_cols); ++j) s += m.col_cost[j] * x[j];
  if (has_quadratic(m)) s += 0.5 * quad_form(m.quadratic, x);
  return s;
}

ConvexityReport check_convexity(const LpModel& m, double rel_tol) {
  ConvexityReport rep;
  if (!has_quadratic(m)) {
    rep.convex = true;
    return rep;
  }
  const std::size_t n = to_size(m.n_cols);
  constexpr std::size_t kDenseLimit = 2500;
  if (n > kDenseLimit) {
    rep.decided = false;
    rep.note = "Q has " + std::to_string(n) + " columns, above the dense limit of " + std::to_string(kDenseLimit);
    return rep;
  }
  const double sgn = m.sense == Sense::Maximize ? -1.0 : 1.0;
  // Dense lower triangle, row-major packed: element (i, j), i >= j, at i*(i+1)/2 + j.
  std::vector<double> a(n * (n + 1) / 2, 0.0);
  auto at = [&](std::size_t i, std::size_t j) -> double& { return i >= j ? a[i * (i + 1) / 2 + j] : a[j * (j + 1) / 2 + i]; };
  double scale = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    for (Index p = m.quadratic.col_start[j]; p < m.quadratic.col_start[j + 1]; ++p) {
      const std::size_t i = to_size(m.quadratic.row_index[to_size(p)]);
      at(i, j) = sgn * m.quadratic.value[to_size(p)];
      scale = std::max(scale, std::fabs(m.quadratic.value[to_size(p)]));
    }
  }
  rep.scale = scale;
  rep.tolerance = rel_tol * std::max(scale, 1e-300);
  const double tol = rep.tolerance;
  std::vector<std::size_t> perm(n);
  for (std::size_t i = 0; i < n; ++i) perm[i] = i;
  // Symmetric pivoted elimination on the Schur complement; perm tracks the original column of each position.
  std::vector<double> col(n);
  for (std::size_t k = 0; k < n; ++k) {
    // Largest remaining diagonal.
    std::size_t best = k;
    double bd = at(k, k);
    for (std::size_t i = k + 1; i < n; ++i) {
      if (at(i, i) > bd) {
        bd = at(i, i);
        best = i;
      }
    }
    // The most negative diagonal among the rest decides non-convexity.
    std::size_t neg = k;
    double nd = at(k, k);
    for (std::size_t i = k + 1; i < n; ++i) {
      if (at(i, i) < nd) {
        nd = at(i, i);
        neg = i;
      }
    }
    if (nd < -tol) {
      rep.pivot = nd;
      rep.column = static_cast<Index>(perm[neg]);
      rep.note = "negative pivot " + std::to_string(nd) + " at column " + std::to_string(rep.column);
      return rep;
    }
    if (bd <= tol) {
      // All remaining diagonals are zero (within the tolerance): the rest must be zero too.
      for (std::size_t i = k; i < n; ++i) {
        for (std::size_t j = k; j < i; ++j) {
          if (std::fabs(at(i, j)) > tol) {
            rep.pivot = at(i, j);
            rep.column = static_cast<Index>(perm[i]);
            rep.note = "zero pivots but a nonzero entry " + std::to_string(at(i, j)) + " in the remaining block (columns " +
                       std::to_string(perm[i]) + " and " + std::to_string(perm[j]) + "): indefinite";
            return rep;
          }
        }
      }
      rep.convex = true;
      return rep;
    }
    // Swap position k and best (rows and columns of the packed symmetric matrix).
    if (best != k) {
      for (std::size_t i = 0; i < n; ++i) {
        if (i == k || i == best) continue;
        std::swap(at(i, k), at(i, best));
      }
      std::swap(at(k, k), at(best, best));
      std::swap(perm[k], perm[best]);
    }
    const double piv = at(k, k);
    ++rep.rank;
    for (std::size_t i = k + 1; i < n; ++i) col[i] = at(i, k);
    for (std::size_t i = k + 1; i < n; ++i) {
      if (col[i] == 0.0) continue;
      const double f = col[i] / piv;
      for (std::size_t j = k + 1; j <= i; ++j) at(i, j) -= f * col[j];
    }
  }
  rep.convex = true;
  return rep;
}

}  // namespace shodhan
