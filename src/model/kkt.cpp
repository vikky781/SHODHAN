#include "shodhan/kkt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace shodhan {

namespace {

constexpr double kBad = std::numeric_limits<double>::infinity();

}  // namespace

std::string KktReport::summary() const {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "primal_inf=%.3g(rel %.3g) dual_inf=%.3g(rel %.3g) d_mismatch=%.3g(rel %.3g) "
                "compl=%.3g(rel %.3g) obj_p=%.12g obj_d=%.12g gap_rel=%.3g obj_mismatch=%.3g ok=%d",
                primal_infeasibility_abs, primal_infeasibility_rel, dual_infeasibility_abs,
                dual_infeasibility_rel, dual_mismatch_abs, dual_mismatch_rel, complementarity_abs,
                complementarity_rel, primal_objective, dual_objective, gap_rel,
                objective_mismatch_rel, ok ? 1 : 0);
  return buf;
}

KktReport check_kkt(const LpModel& m, const Solution& s, double tol) {
  KktReport r;
  const std::size_t rows = to_size(m.n_rows);
  const std::size_t cols = to_size(m.n_cols);
  const bool sizes_ok = s.x.size() == cols && s.y.size() == rows &&
                        (s.d.empty() || s.d.size() == cols) && m.A.n_rows == m.n_rows &&
                        m.A.n_cols == m.n_cols;
  if (!sizes_ok) {
    r.primal_infeasibility_abs = r.primal_infeasibility_rel = kBad;
    r.dual_infeasibility_abs = r.dual_infeasibility_rel = kBad;
    r.gap_abs = r.gap_rel = kBad;
    r.ok = false;
    return r;
  }

  const double sgn = m.sense == Sense::Maximize ? -1.0 : 1.0;

  // ---- primal feasibility ----
  std::vector<double> activity(rows, 0.0);
  m.A.multiply(s.x, activity);
  double row_viol = 0.0;
  double row_norm = 0.0;
  for (std::size_t i = 0; i < rows; ++i) {
    const double lo = m.row_lower[i];
    const double up = m.row_upper[i];
    if (!is_inf(lo)) {
      row_viol = std::max(row_viol, lo - activity[i]);
      row_norm = std::max(row_norm, std::fabs(lo));
    }
    if (!is_inf(up)) {
      row_viol = std::max(row_viol, activity[i] - up);
      row_norm = std::max(row_norm, std::fabs(up));
    }
  }
  double col_viol = 0.0;
  double col_norm = 0.0;
  for (std::size_t j = 0; j < cols; ++j) {
    const double lo = m.col_lower[j];
    const double up = m.col_upper[j];
    if (!is_inf(lo)) {
      col_viol = std::max(col_viol, lo - s.x[j]);
      col_norm = std::max(col_norm, std::fabs(lo));
    }
    if (!is_inf(up)) {
      col_viol = std::max(col_viol, s.x[j] - up);
      col_norm = std::max(col_norm, std::fabs(up));
    }
  }
  r.primal_infeasibility_abs = std::max(row_viol, col_viol);
  r.primal_infeasibility_rel = std::max(row_viol / (1.0 + row_norm), col_viol / (1.0 + col_norm));

  // ---- reduced costs recomputed as c - A^T y (minimization form) ----
  std::vector<double> aty(cols, 0.0);
  m.A.multiply_transpose(s.y, aty);
  std::vector<double> d(cols, 0.0);
  double cnorm = 0.0;
  double primal_obj = sgn * m.objective_offset;
  for (std::size_t j = 0; j < cols; ++j) {
    const double c = sgn * m.col_cost[j];
    d[j] = c - aty[j];
    cnorm = std::max(cnorm, std::fabs(c));
    primal_obj += c * s.x[j];
  }
  r.primal_objective = primal_obj;

  if (!s.d.empty()) {
    for (std::size_t j = 0; j < cols; ++j) {
      r.dual_mismatch_abs = std::max(r.dual_mismatch_abs, std::fabs(s.d[j] - d[j]));
    }
  }
  r.dual_mismatch_rel = r.dual_mismatch_abs / (1.0 + cnorm);

  // ---- dual feasibility, complementarity, dual objective ----
  double dual_obj = sgn * m.objective_offset;
  double dual_viol = 0.0;
  double comp = 0.0;
  for (std::size_t i = 0; i < rows; ++i) {
    const double yi = s.y[i];
    const double lo = m.row_lower[i];
    const double up = m.row_upper[i];
    if (yi > 0.0) {
      if (is_inf(lo)) {
        dual_viol = std::max(dual_viol, yi);
      } else {
        dual_obj += yi * lo;
        comp = std::max(comp, yi * std::fabs(activity[i] - lo));
      }
    } else if (yi < 0.0) {
      if (is_inf(up)) {
        dual_viol = std::max(dual_viol, -yi);
      } else {
        dual_obj += yi * up;
        comp = std::max(comp, -yi * std::fabs(up - activity[i]));
      }
    }
  }
  for (std::size_t j = 0; j < cols; ++j) {
    const double dj = d[j];
    const double lo = m.col_lower[j];
    const double up = m.col_upper[j];
    if (dj > 0.0) {
      if (is_inf(lo)) {
        dual_viol = std::max(dual_viol, dj);
      } else {
        dual_obj += dj * lo;
        comp = std::max(comp, dj * std::fabs(s.x[j] - lo));
      }
    } else if (dj < 0.0) {
      if (is_inf(up)) {
        dual_viol = std::max(dual_viol, -dj);
      } else {
        dual_obj += dj * up;
        comp = std::max(comp, -dj * std::fabs(up - s.x[j]));
      }
    }
  }
  r.dual_objective = dual_obj;
  r.dual_infeasibility_abs = dual_viol;
  r.dual_infeasibility_rel = dual_viol / (1.0 + cnorm);
  r.complementarity_abs = comp;
  const double obj_scale = 1.0 + std::fabs(primal_obj) + std::fabs(dual_obj);
  r.complementarity_rel = comp / obj_scale;
  r.gap_abs = std::fabs(primal_obj - dual_obj);
  r.gap_rel = r.gap_abs / obj_scale;

  // ---- supplied objective (model sense) ----
  const double model_obj = sgn * primal_obj;  // offset + c^T x in the model's own sense
  r.objective_mismatch_rel = std::fabs(s.objective - model_obj) / (1.0 + std::fabs(model_obj));

  r.ok = r.primal_infeasibility_rel <= tol && r.dual_infeasibility_rel <= tol &&
         r.dual_mismatch_rel <= tol && r.complementarity_rel <= tol && r.gap_rel <= tol &&
         r.objective_mismatch_rel <= tol && std::isfinite(r.gap_rel);
  return r;
}

}  // namespace shodhan
