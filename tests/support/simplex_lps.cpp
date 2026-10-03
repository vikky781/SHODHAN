#include "support/simplex_lps.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "support/rng.hpp"

namespace shodhan::testing {

namespace {

double grid(Rng& rng, double lo, double hi, int denom) {
  const int a = static_cast<int>(std::ceil(lo * denom - 1e-12));
  int b = static_cast<int>(std::floor(hi * denom + 1e-12));
  if (b < a) b = a;
  return static_cast<double>(rng.range(a, b)) / static_cast<double>(denom);
}

double coefficient(Rng& rng, bool wide) {
  const double sign = rng.chance(0.5) ? 1.0 : -1.0;
  if (wide) {
    const double mantissa = static_cast<double>(1024 + rng.range(0, 1023)) / 1024.0;
    return sign * std::ldexp(mantissa, rng.range(-13, 12));
  }
  return sign * grid(rng, 0.25, 3.0, 8);
}

}  // namespace

LpModel make_simplex_lp(std::uint64_t seed, const SimplexLpOptions& o) {
  Rng rng(seed * 6364136223846793005ULL + 1442695040888963407ULL);
  const Index m = std::max(o.rows, 1);
  const Index n = std::max(o.cols, 1);
  LpModel model;
  model.name = "simplex_lp";
  model.n_rows = m;
  model.n_cols = n;
  model.col_cost.assign(to_size(n), 0.0);
  model.col_lower.assign(to_size(n), 0.0);
  model.col_upper.assign(to_size(n), kInf);
  model.col_type.assign(to_size(n), ColType::Continuous);
  model.row_lower.assign(to_size(m), -kInf);
  model.row_upper.assign(to_size(m), kInf);

  std::vector<double> x0(to_size(n), 0.0);
  for (Index j = 0; j < n; ++j) {
    const double u = rng.unit();
    double lo = 0.0, hi = kInf;
    double acc = o.fixed_fraction;
    if (u < acc) {
      lo = hi = static_cast<double>(rng.range(-4, 4)) / 2.0;
    } else if (u < (acc += o.boxed_fraction)) {
      lo = static_cast<double>(rng.range(-6, 2)) / 2.0;
      hi = lo + static_cast<double>(rng.range(1, 10)) / 2.0;
    } else if (u < (acc += o.free_fraction)) {
      lo = -kInf;
      hi = kInf;
    } else if (u < (acc += o.upper_only_fraction)) {
      lo = -kInf;
      hi = static_cast<double>(rng.range(-2, 6)) / 2.0;
    } else {
      lo = static_cast<double>(rng.range(-2, 2)) / 2.0;
    }
    model.col_lower[to_size(j)] = lo;
    model.col_upper[to_size(j)] = hi;
    // Generating point inside the bounds.
    double v;
    if (!is_inf(lo) && !is_inf(hi)) {
      v = rng.chance(o.degenerate) ? (rng.chance(0.5) ? lo : hi) : lo + (hi - lo) * grid(rng, 0.0, 1.0, 8);
    } else if (!is_inf(lo)) {
      v = rng.chance(std::max(o.degenerate, 0.25)) ? lo : lo + grid(rng, 0.0, 4.0, 8);
    } else if (!is_inf(hi)) {
      v = rng.chance(std::max(o.degenerate, 0.25)) ? hi : hi - grid(rng, 0.0, 4.0, 8);
    } else {
      v = grid(rng, -3.0, 3.0, 8);
    }
    x0[to_size(j)] = v;

    // Cost.
    double c = grid(rng, 0.0, 4.0, 8);
    if (!o.dual_feasible_start) {
      c = grid(rng, -4.0, 4.0, 8);
    } else if (is_inf(lo) && is_inf(hi)) {
      c = 0.0;
    } else if (is_inf(lo) && !is_inf(hi)) {
      c = -c;
    } else if (!is_inf(lo) && !is_inf(hi)) {
      if (rng.chance(0.5)) c = -c;
    }
    model.col_cost[to_size(j)] = c;
  }

  std::vector<Triplet> t;
  std::vector<double> act(to_size(m), 0.0);
  for (Index i = 0; i < m; ++i) {
    bool any = false;
    for (Index j = 0; j < n; ++j) {
      if (rng.chance(o.density)) {
        const double a = coefficient(rng, o.wide_coefficients);
        t.push_back({i, j, a});
        act[to_size(i)] += a * x0[to_size(j)];
        any = true;
      }
    }
    if (!any) {
      const Index j = rng.range(0, n - 1);
      const double a = coefficient(rng, o.wide_coefficients);
      t.push_back({i, j, a});
      act[to_size(i)] += a * x0[to_size(j)];
    }
  }
  std::string err;
  SparseMatrix::from_triplets(m, n, t, &model.A, &err);

  for (Index i = 0; i < m; ++i) {
    const double a = act[to_size(i)];
    const double u = rng.unit();
    const double slack1 = static_cast<double>(rng.range(0, 6)) / 2.0;
    const double slack2 = static_cast<double>(rng.range(0, 6)) / 2.0;
    const bool active = rng.chance(o.degenerate);
    double lo, hi;
    if (u < o.equality_fraction) {
      lo = hi = a;
    } else if (u < o.equality_fraction + o.ranged_fraction) {
      lo = a - (active ? 0.0 : slack1);
      hi = a + (active ? 0.0 : slack2);
    } else if (u < o.equality_fraction + o.ranged_fraction + o.free_row_fraction) {
      lo = -kInf;
      hi = kInf;
    } else if (rng.chance(0.5)) {
      lo = a - (active ? 0.0 : slack1);
      hi = kInf;
    } else {
      lo = -kInf;
      hi = a + (active ? 0.0 : slack2);
    }
    model.row_lower[to_size(i)] = lo;
    model.row_upper[to_size(i)] = hi;
  }
  if (!o.feasible) {
    // Push one row past what its columns can reach.
    const Index i = rng.range(0, m - 1);
    model.row_lower[to_size(i)] = act[to_size(i)] + 1000.0;
    model.row_upper[to_size(i)] = kInf;
  }
  return model;
}

}  // namespace shodhan::testing
