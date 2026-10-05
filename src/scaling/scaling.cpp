#include "shodhan/scaling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace shodhan {

namespace {

constexpr int kMaxExponent = 60;

// Nearest power of two to a positive finite value (exponent clamped).
double round_pow2(double v) {
  if (!(v > 0.0) || !std::isfinite(v)) return 1.0;
  int e = static_cast<int>(std::lround(std::log2(v)));
  e = std::max(-kMaxExponent, std::min(kMaxExponent, e));
  return std::ldexp(1.0, e);
}

double ratio_of(const std::vector<double>& abs_values) {
  double mn = 0.0;
  double mx = 0.0;
  bool any = false;
  for (const double a : abs_values) {
    if (a == 0.0) continue;
    mn = any ? std::min(mn, a) : a;
    mx = std::max(mx, a);
    any = true;
  }
  return any ? mx / mn : 0.0;
}

}  // namespace

CoefficientRange coefficient_range(const SparseMatrix& a) {
  CoefficientRange r;
  for (const double v : a.value) {
    const double x = std::fabs(v);
    if (x == 0.0) continue;
    r.min = r.valid ? std::min(r.min, x) : x;
    r.max = std::max(r.max, x);
    r.valid = true;
  }
  if (r.valid) r.ratio = r.max / r.min;
  return r;
}

Scaling compute_scaling(const LpModel& model, const ScalingOptions& options, ScalingReport* report) {
  const std::size_t m = to_size(model.n_rows);
  const std::size_t n = to_size(model.n_cols);
  Scaling sc;
  sc.row_scale.assign(m, 1.0);
  sc.col_scale.assign(n, 1.0);
  sc.obj_scale = 1.0;
  ScalingReport rep;
  rep.before = coefficient_range(model.A);

  const SparseMatrix& A = model.A;
  std::vector<double> v(A.value.size());  // current scaled |a_ij|
  for (std::size_t p = 0; p < v.size(); ++p) v[p] = std::fabs(A.value[p]);
  std::vector<double> r(m, 1.0);  // cumulative, unrounded
  std::vector<double> c(n, 1.0);
  std::vector<char> col_fixed(n, 0);  // integer columns keep scale 1
  for (std::size_t j = 0; j < n; ++j) col_fixed[j] = model.col_type[j] != ColType::Continuous ? 1 : 0;

  std::vector<double> rmin(m);
  std::vector<double> rmax(m);

  auto row_stats = [&]() {
    std::fill(rmin.begin(), rmin.end(), 0.0);
    std::fill(rmax.begin(), rmax.end(), 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      for (Index p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
        const double a = v[to_size(p)];
        if (a == 0.0) continue;
        const std::size_t i = to_size(A.row_index[to_size(p)]);
        rmin[i] = rmin[i] == 0.0 ? a : std::min(rmin[i], a);
        rmax[i] = std::max(rmax[i], a);
      }
    }
  };
  auto scale_rows_geometric = [&]() {
    row_stats();
    for (std::size_t i = 0; i < m; ++i) {
      if (rmax[i] == 0.0) continue;
      const double f = 1.0 / std::sqrt(rmin[i] * rmax[i]);
      r[i] *= f;
      rmin[i] = f;  // reuse as the factor
    }
    for (std::size_t j = 0; j < n; ++j) {
      for (Index p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
        const std::size_t i = to_size(A.row_index[to_size(p)]);
        if (rmax[i] != 0.0) v[to_size(p)] *= rmin[i];
      }
    }
  };
  auto scale_cols_geometric = [&]() {
    for (std::size_t j = 0; j < n; ++j) {
      if (col_fixed[j]) continue;
      double mn = 0.0;
      double mx = 0.0;
      for (Index p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
        const double a = v[to_size(p)];
        if (a == 0.0) continue;
        mn = mn == 0.0 ? a : std::min(mn, a);
        mx = std::max(mx, a);
      }
      if (mx == 0.0) continue;
      const double f = 1.0 / std::sqrt(mn * mx);
      c[j] *= f;
      for (Index p = A.col_start[j]; p < A.col_start[j + 1]; ++p) v[to_size(p)] *= f;
    }
  };

  // Geometric-mean passes.
  double ratio = ratio_of(v);
  for (int pass = 0; pass < options.max_passes && ratio > 0.0; ++pass) {
    const std::vector<double> v_backup = v;
    const std::vector<double> r_backup = r;
    const std::vector<double> c_backup = c;
    scale_rows_geometric();
    scale_cols_geometric();
    const double next = ratio_of(v);
    if (next > ratio) {  // got worse: undo and stop
      v = v_backup;
      r = r_backup;
      c = c_backup;
      break;
    }
    ++rep.geometric_passes;
    const double improvement = (ratio - next) / ratio;
    ratio = next;
    if (improvement < options.min_improvement) break;
  }

  // One max-abs equilibration pass: columns, then rows.
  for (std::size_t j = 0; j < n; ++j) {
    if (col_fixed[j]) continue;
    double mx = 0.0;
    for (Index p = A.col_start[j]; p < A.col_start[j + 1]; ++p) mx = std::max(mx, v[to_size(p)]);
    if (mx == 0.0) continue;
    c[j] /= mx;
    for (Index p = A.col_start[j]; p < A.col_start[j + 1]; ++p) v[to_size(p)] /= mx;
  }
  std::vector<double> rowmax(m, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (Index p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
      const std::size_t i = to_size(A.row_index[to_size(p)]);
      rowmax[i] = std::max(rowmax[i], v[to_size(p)]);
    }
  }
  for (std::size_t i = 0; i < m; ++i) {
    if (rowmax[i] != 0.0) r[i] /= rowmax[i];
  }

  // Round to powers of two so scaling is exact in floating point.
  for (std::size_t i = 0; i < m; ++i) sc.row_scale[i] = round_pow2(r[i]);
  for (std::size_t j = 0; j < n; ++j) sc.col_scale[j] = col_fixed[j] ? 1.0 : round_pow2(c[j]);

  // Objective: one factor from the magnitudes of the scaled costs.
  double cmin = 0.0;
  double cmax = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    const double a = std::fabs(model.col_cost[j]) * sc.col_scale[j];
    if (a == 0.0) continue;
    cmin = cmin == 0.0 ? a : std::min(cmin, a);
    cmax = std::max(cmax, a);
  }
  // The quadratic term s C Q C enters the objective on the same footing as the costs.
  for (std::size_t j = 0; j < n; ++j) {
    for (Index p = model.quadratic.col_start.size() > j + 1 ? model.quadratic.col_start[j] : 0;
         model.quadratic.col_start.size() > j + 1 && p < model.quadratic.col_start[j + 1]; ++p) {
      const double a = std::fabs(model.quadratic.value[to_size(p)]) * sc.col_scale[j] * sc.col_scale[to_size(model.quadratic.row_index[to_size(p)])];
      if (a == 0.0) continue;
      cmin = cmin == 0.0 ? a : std::min(cmin, a);
      cmax = std::max(cmax, a);
    }
  }
  if (cmax > 0.0) {
    sc.obj_scale = round_pow2(1.0 / std::sqrt(cmin * cmax));
    rep.objective_scaled = sc.obj_scale != 1.0;
  }

  if (report != nullptr) {
    SparseMatrix scaled = A;
    for (std::size_t j = 0; j < n; ++j) {
      for (Index p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
        scaled.value[to_size(p)] =
            sc.row_scale[to_size(A.row_index[to_size(p)])] * A.value[to_size(p)] * sc.col_scale[j];
      }
    }
    rep.after = coefficient_range(scaled);
    *report = rep;
  }
  return sc;
}

LpModel apply_scaling(const LpModel& model, const Scaling& sc) {
  const std::size_t m = to_size(model.n_rows);
  const std::size_t n = to_size(model.n_cols);
  if (sc.row_scale.size() != m || sc.col_scale.size() != n) {
    throw std::invalid_argument("apply_scaling: scale vector sizes do not match the model");
  }
  LpModel out = model;
  for (std::size_t j = 0; j < n; ++j) {
    for (Index p = model.A.col_start[j]; p < model.A.col_start[j + 1]; ++p) {
      out.A.value[to_size(p)] =
          sc.row_scale[to_size(model.A.row_index[to_size(p)])] * model.A.value[to_size(p)] *
          sc.col_scale[j];
    }
    out.col_cost[j] = sc.obj_scale * sc.col_scale[j] * model.col_cost[j];
    if (!is_inf(model.col_lower[j])) out.col_lower[j] = model.col_lower[j] / sc.col_scale[j];
    if (!is_inf(model.col_upper[j])) out.col_upper[j] = model.col_upper[j] / sc.col_scale[j];
  }
  for (std::size_t i = 0; i < m; ++i) {
    if (!is_inf(model.row_lower[i])) out.row_lower[i] = sc.row_scale[i] * model.row_lower[i];
    if (!is_inf(model.row_upper[i])) out.row_upper[i] = sc.row_scale[i] * model.row_upper[i];
  }
  out.objective_offset = sc.obj_scale * model.objective_offset;
  // Q' = s C Q C (lower triangle, entry (i, j) scaled by s C_i C_j).
  for (std::size_t j = 0; j < n && model.quadratic.col_start.size() > j + 1; ++j) {
    for (Index p = model.quadratic.col_start[j]; p < model.quadratic.col_start[j + 1]; ++p) {
      out.quadratic.value[to_size(p)] =
          sc.obj_scale * sc.col_scale[to_size(model.quadratic.row_index[to_size(p)])] * model.quadratic.value[to_size(p)] * sc.col_scale[j];
    }
  }
  return out;
}

Solution unscale_solution(const Scaling& sc, const Solution& s) {
  Solution out;
  out.x.resize(s.x.size());
  for (std::size_t j = 0; j < s.x.size(); ++j) out.x[j] = sc.col_scale[j] * s.x[j];
  out.y.resize(s.y.size());
  for (std::size_t i = 0; i < s.y.size(); ++i) out.y[i] = sc.row_scale[i] * s.y[i] / sc.obj_scale;
  out.d.resize(s.d.size());
  for (std::size_t j = 0; j < s.d.size(); ++j) out.d[j] = s.d[j] / (sc.obj_scale * sc.col_scale[j]);
  out.objective = s.objective / sc.obj_scale;
  return out;
}

Solution scale_solution(const Scaling& sc, const Solution& s) {
  Solution out;
  out.x.resize(s.x.size());
  for (std::size_t j = 0; j < s.x.size(); ++j) out.x[j] = s.x[j] / sc.col_scale[j];
  out.y.resize(s.y.size());
  for (std::size_t i = 0; i < s.y.size(); ++i) out.y[i] = sc.obj_scale * s.y[i] / sc.row_scale[i];
  out.d.resize(s.d.size());
  for (std::size_t j = 0; j < s.d.size(); ++j) out.d[j] = sc.obj_scale * sc.col_scale[j] * s.d[j];
  out.objective = sc.obj_scale * s.objective;
  return out;
}

}  // namespace shodhan
