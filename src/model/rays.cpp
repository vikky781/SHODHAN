#include "shodhan/rays.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace shodhan {

namespace {

// Accumulates the largest and smallest value of sum coef * v over lo <= v <= hi.
struct BoxSum {
  double max_val = 0.0, min_val = 0.0;
  bool max_inf = false, min_inf = false;
  double scale = 0.0;

  void add(double coef, double lo, double hi) {
    if (coef == 0.0) return;
    const double a = std::fabs(coef);
    double bound_mag = 1.0;
    if (!is_inf(lo)) bound_mag = std::max(bound_mag, std::fabs(lo));
    if (!is_inf(hi)) bound_mag = std::max(bound_mag, std::fabs(hi));
    scale += a * bound_mag;
    if (coef > 0.0) {
      if (is_inf(hi)) max_inf = true; else max_val += coef * hi;
      if (is_inf(lo)) min_inf = true; else min_val += coef * lo;
    } else {
      if (is_inf(lo)) max_inf = true; else max_val += coef * lo;
      if (is_inf(hi)) min_inf = true; else min_val += coef * hi;
    }
  }
};

}  // namespace

RayCheck check_farkas(const LpModel& model, const std::vector<double>& y, double tol) {
  RayCheck r;
  if (y.size() != to_size(model.n_rows)) {
    r.message = "multiplier vector has the wrong length";
    return r;
  }
  // Coefficients of the combination: (A^T y)_j on x_j and -y_i on r_i. A coefficient that is exactly
  // zero in theory (the basic variables of the simplex row) can be 1e-17 in floating point; against an
  // infinite bound it would make every proof impossible, so coefficients below coef_tol times the
  // largest one count as zero.
  std::vector<double> g(to_size(model.n_cols), 0.0);
  double cmax = 0.0;
  for (Index j = 0; j < model.n_cols; ++j) {
    double v = 0.0;
    for (Index t = model.A.col_start[to_size(j)]; t < model.A.col_start[to_size(j) + 1]; ++t) {
      v += model.A.value[to_size(t)] * y[to_size(model.A.row_index[to_size(t)])];
    }
    g[to_size(j)] = v;
    cmax = std::max(cmax, std::fabs(v));
  }
  for (Index i = 0; i < model.n_rows; ++i) cmax = std::max(cmax, std::fabs(y[to_size(i)]));
  const double zero = 1e-12 * cmax;
  BoxSum s;
  for (Index j = 0; j < model.n_cols; ++j) {
    if (std::fabs(g[to_size(j)]) > zero) s.add(g[to_size(j)], model.col_lower[to_size(j)], model.col_upper[to_size(j)]);
  }
  for (Index i = 0; i < model.n_rows; ++i) {
    if (std::fabs(y[to_size(i)]) > zero) s.add(-y[to_size(i)], model.row_lower[to_size(i)], model.row_upper[to_size(i)]);
  }
  r.scale = s.scale;
  if (s.scale == 0.0) {
    r.message = "all multipliers are zero";
    return r;
  }
  // The total must be 0 for a feasible point.
  const double gap_below = s.max_inf ? -kInf : -s.max_val;  // > 0 when even the largest value is negative
  const double gap_above = s.min_inf ? -kInf : s.min_val;   // > 0 when even the smallest value is positive
  r.gap = std::max(gap_below, gap_above);
  r.ok = r.gap > tol * r.scale;
  r.message = r.ok ? "the combination of rows excludes zero by " + std::to_string(r.gap)
                   : "the combination of rows can still be zero (gap " + std::to_string(r.gap) + ", scale " + std::to_string(r.scale) + ")";
  return r;
}

RayCheck check_unbounded_ray(const LpModel& model, const std::vector<double>& ray, double tol) {
  RayCheck r;
  if (ray.size() != to_size(model.n_cols)) {
    r.message = "ray has the wrong length";
    return r;
  }
  double rmax = 0.0;
  for (const double v : ray) rmax = std::max(rmax, std::fabs(v));
  if (rmax == 0.0) {
    r.message = "the ray is zero";
    return r;
  }
  r.scale = rmax;
  double worst = 0.0;
  for (Index j = 0; j < model.n_cols; ++j) {
    const double v = ray[to_size(j)];
    if (v > 0.0 && !is_inf(model.col_upper[to_size(j)])) worst = std::max(worst, v);
    if (v < 0.0 && !is_inf(model.col_lower[to_size(j)])) worst = std::max(worst, -v);
  }
  std::vector<double> ar(to_size(model.n_rows), 0.0), mag(to_size(model.n_rows), 0.0);
  for (Index j = 0; j < model.n_cols; ++j) {
    const double v = ray[to_size(j)];
    if (v == 0.0) continue;
    for (Index t = model.A.col_start[to_size(j)]; t < model.A.col_start[to_size(j) + 1]; ++t) {
      const std::size_t i = to_size(model.A.row_index[to_size(t)]);
      ar[i] += model.A.value[to_size(t)] * v;
      mag[i] += std::fabs(model.A.value[to_size(t)] * v);
    }
  }
  // Row violations are measured against the magnitude of the terms of the row, but at least 1e-6 of
  // the largest contribution the row could have (max|ray| * max|a_ij|): where a row's terms cancel to
  // noise, noise divided by noise would otherwise read as a total violation.
  std::vector<double> row_max(to_size(model.n_rows), 0.0);
  for (Index j = 0; j < model.n_cols; ++j) {
    for (Index t = model.A.col_start[to_size(j)]; t < model.A.col_start[to_size(j) + 1]; ++t) {
      const std::size_t i = to_size(model.A.row_index[to_size(t)]);
      row_max[i] = std::max(row_max[i], std::fabs(model.A.value[to_size(t)]));
    }
  }
  double row_worst = 0.0;
  for (Index i = 0; i < model.n_rows; ++i) {
    const std::size_t k = to_size(i);
    const double denom = std::max(mag[k], 1e-6 * rmax * row_max[k]);
    const double rel = ar[k] / std::max(denom, 1e-300);
    if (denom == 0.0) continue;
    if (ar[k] > 0.0 && !is_inf(model.row_upper[k])) row_worst = std::max(row_worst, rel);
    if (ar[k] < 0.0 && !is_inf(model.row_lower[k])) row_worst = std::max(row_worst, -rel);
  }
  const double sgn = model.sense == Sense::Maximize ? -1.0 : 1.0;
  double rate = 0.0, rate_mag = 0.0;
  for (Index j = 0; j < model.n_cols; ++j) {
    rate += sgn * model.col_cost[to_size(j)] * ray[to_size(j)];
    rate_mag += std::fabs(model.col_cost[to_size(j)] * ray[to_size(j)]);
  }
  r.gap = -rate;
  if (worst > tol * rmax) {
    r.message = "the ray leaves a finite column bound (violation " + std::to_string(worst) + ")";
  } else if (row_worst > tol) {
    r.message = "A * ray leaves a finite row bound (relative violation " + std::to_string(row_worst) + " (" + [&]{ char b[32]; std::snprintf(b, sizeof b, "%.3e", row_worst); return std::string(b); }() + ")" + ")";
  } else if (!(rate < -tol * std::max(rate_mag, 1e-300))) {
    r.message = "the objective does not improve along the ray (rate " + std::to_string(rate) + ")";
  } else {
    r.ok = true;
    r.message = "recession direction with objective rate " + std::to_string(rate);
  }
  return r;
}

}  // namespace shodhan

namespace shodhan {

double max_relative_violation(const LpModel& model, const std::vector<double>& x) {
  if (x.size() != to_size(model.n_cols)) return kInf;
  double worst = 0.0;
  for (Index j = 0; j < model.n_cols; ++j) {
    const double v = x[to_size(j)];
    const double lo = model.col_lower[to_size(j)], hi = model.col_upper[to_size(j)];
    if (!is_inf(lo) && v < lo) worst = std::max(worst, (lo - v) / (1.0 + std::fabs(lo)));
    if (!is_inf(hi) && v > hi) worst = std::max(worst, (v - hi) / (1.0 + std::fabs(hi)));
  }
  std::vector<double> ax(to_size(model.n_rows), 0.0);
  for (Index j = 0; j < model.n_cols; ++j) {
    for (Index t = model.A.col_start[to_size(j)]; t < model.A.col_start[to_size(j) + 1]; ++t) {
      ax[to_size(model.A.row_index[to_size(t)])] += model.A.value[to_size(t)] * x[to_size(j)];
    }
  }
  for (Index i = 0; i < model.n_rows; ++i) {
    const double lo = model.row_lower[to_size(i)], hi = model.row_upper[to_size(i)];
    const double v = ax[to_size(i)];
    if (!is_inf(lo) && v < lo) worst = std::max(worst, (lo - v) / (1.0 + std::fabs(lo)));
    if (!is_inf(hi) && v > hi) worst = std::max(worst, (v - hi) / (1.0 + std::fabs(hi)));
  }
  return worst;
}

}  // namespace shodhan
