#include <algorithm>

#include "shodhan/mip/search_state.hpp"

namespace shodhan::mip {

void Pseudocosts::resize(Index n_cols) {
  for (int d = 0; d < 2; ++d) {
    sum_[d].assign(static_cast<std::size_t>(n_cols), 0.0);
    n_[d].assign(static_cast<std::size_t>(n_cols), 0);
  }
}

void Pseudocosts::update(Index col, int dir, double gain, double delta) {
  if (!(delta > 1e-9) || !(gain >= 0.0)) return;
  const int d = dir > 0 ? 1 : 0;
  const double unit = gain / delta;
  sum_[d][static_cast<std::size_t>(col)] += unit;
  ++n_[d][static_cast<std::size_t>(col)];
  total_sum_[d] += unit;
  ++total_n_[d];
}

double Pseudocosts::value(Index col, int dir) const {
  const int d = dir > 0 ? 1 : 0;
  const std::size_t c = static_cast<std::size_t>(col);
  if (n_[d][c] > 0) return sum_[d][c] / n_[d][c];
  return total_n_[d] > 0 ? total_sum_[d] / static_cast<double>(total_n_[d]) : 1.0;
}

double Pseudocosts::score(Index col, double frac) const {
  constexpr double kEps = 1e-6;
  const double down = std::max(value(col, -1) * frac, kEps);
  const double up = std::max(value(col, +1) * (1.0 - frac), kEps);
  return down * up;
}

}  // namespace shodhan::mip
