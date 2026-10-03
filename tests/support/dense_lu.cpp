#include "support/dense_lu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace shodhan::testing {

double DenseMatrix::norm_inf() const {
  double best = 0.0;
  for (Index i = 0; i < n; ++i) {
    double s = 0.0;
    for (Index j = 0; j < n; ++j) s += std::fabs(at(i, j));
    best = std::max(best, s);
  }
  return best;
}

double DenseMatrix::norm_one() const {
  double best = 0.0;
  for (Index j = 0; j < n; ++j) {
    double s = 0.0;
    for (Index i = 0; i < n; ++i) s += std::fabs(at(i, j));
    best = std::max(best, s);
  }
  return best;
}

std::vector<double> DenseMatrix::multiply(const std::vector<double>& x) const {
  std::vector<double> y(to_size(n), 0.0);
  for (Index i = 0; i < n; ++i) {
    double s = 0.0;
    for (Index j = 0; j < n; ++j) s += at(i, j) * x[to_size(j)];
    y[to_size(i)] = s;
  }
  return y;
}

std::vector<double> DenseMatrix::multiply_transpose(const std::vector<double>& x) const {
  std::vector<double> y(to_size(n), 0.0);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) y[to_size(j)] += at(i, j) * x[to_size(i)];
  }
  return y;
}

DenseLu::DenseLu(const DenseMatrix& a) : n_(a.n), lu_(a.a), perm_(to_size(a.n)) {
  const std::size_t n = to_size(n_);
  for (std::size_t i = 0; i < n; ++i) perm_[i] = static_cast<Index>(i);
  double pmin = std::numeric_limits<double>::infinity();
  double pmax = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t best = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::fabs(lu_[i * n + k]) > std::fabs(lu_[best * n + k])) best = i;
    }
    if (best != k) {
      for (std::size_t j = 0; j < n; ++j) std::swap(lu_[k * n + j], lu_[best * n + j]);
      std::swap(perm_[k], perm_[best]);
    }
    const double piv = lu_[k * n + k];
    pmin = std::min(pmin, std::fabs(piv));
    pmax = std::max(pmax, std::fabs(piv));
    if (piv == 0.0) {
      singular_ = true;
      continue;
    }
    for (std::size_t i = k + 1; i < n; ++i) {
      const double l = lu_[i * n + k] / piv;
      lu_[i * n + k] = l;
      if (l == 0.0) continue;
      for (std::size_t j = k + 1; j < n; ++j) lu_[i * n + j] -= l * lu_[k * n + j];
    }
  }
  pivot_ratio_ = (n == 0 || pmax == 0.0) ? 0.0 : pmin / pmax;
}

std::vector<double> DenseLu::solve(const std::vector<double>& b) const {
  const std::size_t n = to_size(n_);
  std::vector<double> x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = b[to_size(perm_[i])];
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) x[i] -= lu_[i * n + j] * x[j];
  }
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t j = i + 1; j < n; ++j) x[i] -= lu_[i * n + j] * x[j];
    x[i] /= lu_[i * n + i];
  }
  return x;
}

std::vector<double> DenseLu::solve_transpose(const std::vector<double>& c) const {
  // A = P^T L U, so A^T y = c  <=>  U^T L^T (P y) = c.
  const std::size_t n = to_size(n_);
  std::vector<double> z(c);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) z[i] -= lu_[j * n + i] * z[j];
    z[i] /= lu_[i * n + i];
  }
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t j = i + 1; j < n; ++j) z[i] -= lu_[j * n + i] * z[j];
  }
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) y[to_size(perm_[i])] = z[i];
  return y;
}

double DenseLu::inverse_norm_inf() const {
  if (singular_) return std::numeric_limits<double>::infinity();
  const std::size_t n = to_size(n_);
  std::vector<double> row_sum(n, 0.0);
  std::vector<double> e(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    e[j] = 1.0;
    const std::vector<double> col = solve(e);
    e[j] = 0.0;
    for (std::size_t i = 0; i < n; ++i) row_sum[i] += std::fabs(col[i]);
  }
  double best = 0.0;
  for (const double s : row_sum) best = std::max(best, s);
  return best;
}

Index dense_rank(const DenseMatrix& a, double rel_tol) {
  const std::size_t n = to_size(a.n);
  std::vector<double> w(a.a);
  // Equilibrate first (columns, then rows, to unit maximum) so that the
  // tolerance does not depend on the units of the columns and rows.
  for (std::size_t j = 0; j < n; ++j) {
    double cm = 0.0;
    for (std::size_t i = 0; i < n; ++i) cm = std::max(cm, std::fabs(w[i * n + j]));
    if (cm > 0.0) for (std::size_t i = 0; i < n; ++i) w[i * n + j] /= cm;
  }
  for (std::size_t i = 0; i < n; ++i) {
    double rm = 0.0;
    for (std::size_t j = 0; j < n; ++j) rm = std::max(rm, std::fabs(w[i * n + j]));
    if (rm > 0.0) for (std::size_t j = 0; j < n; ++j) w[i * n + j] /= rm;
  }
  double amax = 0.0;
  for (const double v : w) amax = std::max(amax, std::fabs(v));
  if (amax == 0.0) return 0;
  Index rank = 0;
  std::vector<char> row_used(n, 0), col_used(n, 0);
  for (std::size_t step = 0; step < n; ++step) {
    double best = 0.0;
    std::size_t bi = 0, bj = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (row_used[i]) continue;
      for (std::size_t j = 0; j < n; ++j) {
        if (col_used[j]) continue;
        if (std::fabs(w[i * n + j]) > best) {
          best = std::fabs(w[i * n + j]);
          bi = i;
          bj = j;
        }
      }
    }
    if (best <= rel_tol * amax) break;
    ++rank;
    row_used[bi] = col_used[bj] = 1;
    for (std::size_t i = 0; i < n; ++i) {
      if (row_used[i]) continue;
      const double l = w[i * n + bj] / w[bi * n + bj];
      if (l == 0.0) continue;
      for (std::size_t j = 0; j < n; ++j) {
        if (!col_used[j]) w[i * n + j] -= l * w[bi * n + j];
      }
    }
  }
  return rank;
}

}  // namespace shodhan::testing
