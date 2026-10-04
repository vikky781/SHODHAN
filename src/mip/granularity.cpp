#include "shodhan/mip/granularity.hpp"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace shodhan::mip {

double objective_granularity(const LpModel& model) {
  std::vector<double> costs;
  for (Index j = 0; j < model.n_cols; ++j) {
    const double c = model.col_cost[to_size(j)];
    if (c == 0.0) continue;
    if (!model.is_integer(j)) return 0.0;  // a continuous column with a cost: values are not on a grid
    costs.push_back(std::fabs(c));
  }
  if (costs.empty()) return 0.0;
  static const double kScales[] = {1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 25.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0};
  for (const double scale : kScales) {
    bool all_integer = true;
    std::int64_t g = 0;
    for (const double c : costs) {
      const double v = c * scale;
      const double r = std::round(v);
      if (r < 1.0 || r > 1e12 || std::fabs(v - r) > 1e-9 * std::max(1.0, v)) {
        all_integer = false;
        break;
      }
      g = std::gcd(g, static_cast<std::int64_t>(r));
    }
    if (all_integer && g > 0) return static_cast<double>(g) / scale;
  }
  return 0.0;
}

}  // namespace shodhan::mip
