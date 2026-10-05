#include "support/dense_ldl.hpp"

#include <cmath>

namespace shodhan::testing {

bool dense_ldl_solve(int n, const std::vector<double>& K, const std::vector<double>& b, std::vector<double>* x) {
  std::vector<double> a = K;
  const auto at = [&](int i, int j) -> double& { return a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]; };
  std::vector<double> d(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k) {
    // d_k = a_kk - sum_{j<k} l_kj^2 d_j with l stored in the strict lower triangle.
    double dk = at(k, k);
    for (int j = 0; j < k; ++j) dk -= at(k, j) * at(k, j) * d[static_cast<std::size_t>(j)];
    if (!(std::fabs(dk) > 0.0) || !std::isfinite(dk)) return false;
    d[static_cast<std::size_t>(k)] = dk;
    for (int i = k + 1; i < n; ++i) {
      double s = at(i, k);
      for (int j = 0; j < k; ++j) s -= at(i, j) * at(k, j) * d[static_cast<std::size_t>(j)];
      at(i, k) = s / dk;
    }
  }
  std::vector<double> y = b;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) y[static_cast<std::size_t>(i)] -= at(i, j) * y[static_cast<std::size_t>(j)];
  }
  for (int i = 0; i < n; ++i) y[static_cast<std::size_t>(i)] /= d[static_cast<std::size_t>(i)];
  for (int i = n - 1; i >= 0; --i) {
    for (int j = i + 1; j < n; ++j) y[static_cast<std::size_t>(i)] -= at(j, i) * y[static_cast<std::size_t>(j)];
  }
  *x = y;
  return true;
}

}  // namespace shodhan::testing
