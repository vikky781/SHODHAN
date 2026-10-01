#include <algorithm>
#include <cmath>

#include "reductions.hpp"

namespace shodhan::presolve_detail {

namespace {

std::size_t u(int i) { return static_cast<std::size_t>(i); }

constexpr double kIntTol = 1e-6;

bool is_integral(double v) { return std::fabs(v - std::round(v)) <= 1e-9; }

}  // namespace

// ---------------------------------------------------------------------------
// 1. Empty row
// ---------------------------------------------------------------------------
bool EmptyRowReduction::apply(Context& c, int i) {
  WorkModel& w = c.w;
  if (w.row_cnt[u(i)] != 0) return false;
  const double lo = w.rl[u(i)];
  const double up = w.ru[u(i)];
  if ((!is_neg_inf(lo) && lo > c.rtol(i, lo)) || (!is_pos_inf(up) && up < -c.rtol(i, up))) {
    c.mark_infeasible("empty row " + std::to_string(i) + " has range [" + num(lo) + ", " +
                      num(up) + "] which excludes 0");
    return true;
  }
  w.remove_row(i);  // y_i = 0 needs no record
  ++c.stats.empty_rows;
  return true;
}

// ---------------------------------------------------------------------------
// 4. Singleton row
// ---------------------------------------------------------------------------
bool SingletonRowReduction::apply(Context& c, int i) {
  WorkModel& w = c.w;
  if (w.row_cnt[u(i)] != 1) return false;
  int j = -1;
  double a = 0.0;
  w.for_row(i, [&](int jj, double aa) {
    j = jj;
    a = aa;
  });
  const double lo = w.rl[u(i)];
  const double up = w.ru[u(i)];
  double nl = -kInf;
  double nu = kInf;
  if (a > 0.0) {
    if (!is_inf(lo)) nl = lo / a;
    if (!is_inf(up)) nu = up / a;
  } else {
    if (!is_inf(up)) nl = up / a;
    if (!is_inf(lo)) nu = lo / a;
  }
  if (c.opt.is_mip && w.is_int[u(j)]) {
    if (!is_inf(nl)) nl = std::ceil(nl - kIntTol * (1.0 + std::fabs(nl)));
    if (!is_inf(nu)) nu = std::floor(nu + kIntTol * (1.0 + std::fabs(nu)));
  }
  const bool lb_from_row = !is_inf(nl) && nl >= w.cl[u(j)];
  const bool ub_from_row = !is_inf(nu) && nu <= w.cu[u(j)];
  double new_l = lb_from_row ? nl : w.cl[u(j)];
  double new_u = ub_from_row ? nu : w.cu[u(j)];
  if (new_l > new_u + c.btol(i, a, std::max(std::fabs(new_l), std::fabs(new_u)))) {
    c.mark_infeasible("singleton row " + std::to_string(i) + " implies [" + num(new_l) + ", " +
                      num(new_u) + "] which conflicts with the bounds of column " +
                      std::to_string(j) + " [" + num(w.cl[u(j)]) + ", " + num(w.cu[u(j)]) + "] (a=" + num(a) +
                      ", rl=" + num(lo) + ", ru=" + num(up) + ", mag=" + num(w.row_mag[u(i)]) + ")");
    return true;
  }
  if (new_l > new_u) new_u = new_l;
  w.cl[u(j)] = new_l;
  w.cu[u(j)] = new_u;
  c.record(std::make_shared<SingletonRowRecord>(i, j, a, lb_from_row, ub_from_row));
  w.remove_row(i);
  w.touch_col_bounds(j);
  ++c.stats.singleton_rows;
  return true;
}

// ---------------------------------------------------------------------------
// 5. Redundant row
// ---------------------------------------------------------------------------
bool RedundantRowReduction::detect_infeasible(Context& c, int i, const Activity& act) {
  const WorkModel& w = c.w;
  const double lo = w.rl[u(i)];
  const double up = w.ru[u(i)];
  if (!is_neg_inf(lo) && act.max_finite() && act.max_fin < lo - c.rtol(i, lo)) {
    c.mark_infeasible("row " + std::to_string(i) + ": largest possible activity is below its lower bound");
    return true;
  }
  if (!is_pos_inf(up) && act.min_finite() && act.min_fin > up + c.rtol(i, up)) {
    c.mark_infeasible("row " + std::to_string(i) + ": smallest possible activity is above its upper bound");
    return true;
  }
  return false;
}

bool RedundantRowReduction::apply(Context& c, int i, const Activity& act) {
  WorkModel& w = c.w;
  bool changed = false;
  if (!is_neg_inf(w.rl[u(i)]) && act.min_finite() && act.min_fin >= w.rl[u(i)] - c.stol(i, w.rl[u(i)])) {
    w.rl[u(i)] = -kInf;
    changed = true;
    ++c.stats.redundant_row_sides;
  }
  if (!is_pos_inf(w.ru[u(i)]) && act.max_finite() && act.max_fin <= w.ru[u(i)] + c.stol(i, w.ru[u(i)])) {
    w.ru[u(i)] = kInf;
    changed = true;
    ++c.stats.redundant_row_sides;
  }
  if (is_neg_inf(w.rl[u(i)]) && is_pos_inf(w.ru[u(i)])) {
    // Both sides implied (or the row was free): the row can go. y_i = 0.
    w.remove_row(i);
    ++c.stats.redundant_rows;
    return true;
  }
  if (changed) w.touch_row_bounds(i);
  return changed;
}

// ---------------------------------------------------------------------------
// 6. Forcing row
// ---------------------------------------------------------------------------
bool ForcingRowReduction::apply(Context& c, int i, const Activity& act) {
  WorkModel& w = c.w;
  const double lo = w.rl[u(i)];
  const double up = w.ru[u(i)];
  const bool min_forcing = act.min_finite() && !is_pos_inf(up) && std::fabs(act.min_fin - up) <= c.stol(i, up);
  const bool max_forcing = act.max_finite() && !is_neg_inf(lo) && std::fabs(act.max_fin - lo) <= c.stol(i, lo);
  if (!min_forcing && !max_forcing) return false;
  const bool use_min = min_forcing;

  // Fixing integer columns at a fractional bound would break integrality.
  bool fixable = true;
  w.for_row(i, [&](int j, double a) {
    const double bound = use_min ? (a > 0.0 ? w.cl[u(j)] : w.cu[u(j)])
                                 : (a > 0.0 ? w.cu[u(j)] : w.cl[u(j)]);
    if (c.opt.is_mip && w.is_int[u(j)] && !is_integral(bound)) fixable = false;
  });
  if (!fixable) return false;

  std::vector<ForcingRowRecord::Forced> forced;
  w.for_row(i, [&](int j, double a) {
    const double bound = use_min ? (a > 0.0 ? w.cl[u(j)] : w.cu[u(j)])
                                 : (a > 0.0 ? w.cu[u(j)] : w.cl[u(j)]);
    forced.push_back({j, bound, w.cost[u(j)], a, w.col_snapshot(j)});
  });
  const bool equality = std::fabs(up - lo) <= c.stol(i, lo);
  c.record(std::make_shared<ForcingRowRecord>(i, use_min, equality, forced));
  for (const auto& f : forced) w.substitute_fixed(f.col, f.value);
  w.remove_row(i);
  ++c.stats.forcing_rows;
  return true;
}

// ---------------------------------------------------------------------------
// 7. Doubleton equation aggregation
// ---------------------------------------------------------------------------
namespace {

// Tries to eliminate column `e` (coefficient ae) using row i, keeping column k.
bool try_eliminate(Context& c, int i, int e, double ae, int k, double ak, double b) {
  WorkModel& w = c.w;
  const PresolveOptions& opt = c.opt;

  // Numerical safety: the pivot must not be tiny relative to its column.
  double col_max = 0.0;
  w.for_col(e, [&](int, double a) { col_max = std::max(col_max, std::fabs(a)); });
  if (std::fabs(ae) < opt.doubleton_pivot_tolerance * col_max) return false;

  // Integrality: x_e = beta - alpha x_k must stay integral when x_e is integer.
  if (opt.is_mip && w.is_int[u(e)]) {
    if (!w.is_int[u(k)]) return false;
    if (std::fabs(std::fabs(ae) - 1.0) > 1e-12 || !is_integral(ak) || !is_integral(b)) return false;
  }

  const double alpha = ak / ae;
  const double beta = b / ae;
  // A tiny or huge ratio makes the implied bounds (beta - bound) / alpha carry
  // rounding error of the order eps / |alpha|, which later conflicts would
  // mistake for infeasibility; keep the substitution balanced.
  if (std::fabs(alpha) > opt.max_coefficient_growth || std::fabs(alpha) * opt.max_coefficient_growth < 1.0) {
    return false;
  }

  // Coefficient growth: new a_rk = a_rk - a_re * alpha for every other row of e.
  bool ok = true;
  w.for_col(e, [&](int r, double are) {
    if (r == i) return;
    const double ark = w.coef(r, k);
    const double nw = ark - are * alpha;
    if (std::fabs(nw) > opt.max_coefficient_growth * std::max(std::fabs(are), std::fabs(ark))) ok = false;
    // Near-cancellation leaves a coefficient with a large relative error, which
    // later divisions (singleton rows) would amplify. Exact cancellation is fine.
    const double mag = std::max(std::fabs(ark), std::fabs(are * alpha));
    const double fabs_nw = std::fabs(nw);
    if (fabs_nw > 1e-12 * mag && fabs_nw < opt.min_cancellation_ratio * mag) ok = false;
  });
  if (!ok) return false;

  // Implied bounds on x_k from l_e <= beta - alpha x_k <= u_e.
  const double le = w.cl[u(e)];
  const double ue = w.cu[u(e)];
  double cand_lo = -kInf;
  double cand_up = kInf;
  if (alpha > 0.0) {
    if (!is_inf(ue)) cand_lo = (beta - ue) / alpha;
    if (!is_inf(le)) cand_up = (beta - le) / alpha;
  } else {
    if (!is_inf(le)) cand_lo = (beta - le) / alpha;
    if (!is_inf(ue)) cand_up = (beta - ue) / alpha;
  }
  if (opt.is_mip && w.is_int[u(k)]) {
    if (!is_inf(cand_lo)) cand_lo = std::ceil(cand_lo - kIntTol * (1.0 + std::fabs(cand_lo)));
    if (!is_inf(cand_up)) cand_up = std::floor(cand_up + kIntTol * (1.0 + std::fabs(cand_up)));
  }
  const bool lb_from_elim = !is_inf(cand_lo) && cand_lo > w.cl[u(k)];
  const bool ub_from_elim = !is_inf(cand_up) && cand_up < w.cu[u(k)];
  double new_l = lb_from_elim ? cand_lo : w.cl[u(k)];
  double new_u = ub_from_elim ? cand_up : w.cu[u(k)];
  if (new_l > new_u + c.btol(i, ae, std::max(std::fabs(new_l), std::fabs(new_u)))) {
    c.mark_infeasible("aggregating doubleton row " + std::to_string(i) + " gives column " +
                      std::to_string(k) + " conflicting bounds");
    return true;
  }
  if (new_l > new_u) new_u = new_l;

  // ---- commit ----
  DoubletonRecord::Data d;
  d.row = i;
  d.elim = e;
  d.kept = k;
  d.a_elim = ae;
  d.a_kept = ak;
  d.alpha = alpha;
  d.beta = beta;
  d.cost_elim = w.cost[u(e)];
  d.cost_kept = w.cost[u(k)];
  d.entries_elim = w.col_snapshot(e);
  d.entries_kept = w.col_snapshot(k);
  d.lb_from_elim = lb_from_elim;
  d.ub_from_elim = ub_from_elim;
  c.record(std::make_shared<DoubletonRecord>(std::move(d)));

  w.offset += w.cost[u(e)] * beta;
  w.cost[u(k)] -= w.cost[u(e)] * alpha;
  w.cl[u(k)] = new_l;
  w.cu[u(k)] = new_u;

  const Entries rows_of_e = w.col_snapshot(e);
  auto bound_mag = [&](int col) {
    double mag = 0.0;
    if (!is_inf(w.cl[u(col)])) mag = std::max(mag, std::fabs(w.cl[u(col)]));
    if (!is_inf(w.cu[u(col)])) mag = std::max(mag, std::fabs(w.cu[u(col)]));
    return mag;
  };
  const double bmag_e = bound_mag(e);
  const double bmag_k = bound_mag(k);
  for (const auto& entry : rows_of_e) {
    const int r = entry.first;
    if (r == i) continue;
    const double are = entry.second;
    if (!is_inf(w.rl[u(r)])) w.rl[u(r)] -= are * beta;
    if (!is_inf(w.ru[u(r)])) w.ru[u(r)] -= are * beta;
    const double ark = w.coef(r, k);
    double nw = ark - are * alpha;
    if (std::fabs(nw) <= 1e-12 * std::max(std::fabs(ark), std::fabs(are * alpha))) nw = 0.0;
    w.set_coef(r, k, nw);
    w.row_mag[u(r)] = std::max({w.row_mag[u(r)], std::fabs(are * beta), std::fabs(are) * bmag_e,
                                std::fabs(are * alpha) * bmag_k, std::fabs(ark) * bmag_k});
  }
  w.remove_row(i);
  w.remove_col(e);
  w.touch_col_bounds(k);
  ++c.stats.doubleton_equations;
  return true;
}

}  // namespace

bool DoubletonEquationReduction::apply(Context& c, int i) {
  WorkModel& w = c.w;
  if (w.row_cnt[u(i)] != 2) return false;
  const double lo = w.rl[u(i)];
  const double up = w.ru[u(i)];
  if (is_inf(lo) || is_inf(up) || std::fabs(up - lo) > c.stol(i, lo)) return false;
  int cols[2] = {-1, -1};
  double vals[2] = {0.0, 0.0};
  int n = 0;
  w.for_row(i, [&](int j, double a) {
    if (n < 2) {
      cols[n] = j;
      vals[n] = a;
    }
    ++n;
  });
  if (n != 2) return false;
  const double b = lo;

  // Prefer eliminating the column with fewer entries, then the larger pivot,
  // then the lower index.
  int first = 0;
  const int cnt0 = w.col_cnt[u(cols[0])];
  const int cnt1 = w.col_cnt[u(cols[1])];
  if (cnt1 < cnt0 || (cnt1 == cnt0 && std::fabs(vals[1]) > std::fabs(vals[0])) ||
      (cnt1 == cnt0 && std::fabs(vals[1]) == std::fabs(vals[0]) && cols[1] < cols[0])) {
    first = 1;
  }
  const int second = 1 - first;
  if (try_eliminate(c, i, cols[first], vals[first], cols[second], vals[second], b)) return true;
  return try_eliminate(c, i, cols[second], vals[second], cols[first], vals[first], b);
}

// ---------------------------------------------------------------------------
// Implied bounds
// ---------------------------------------------------------------------------
bool ImpliedBoundsReduction::apply(Context& c, int i, const Activity& act) {
  WorkModel& w = c.w;
  const double lo = w.rl[u(i)];
  const double up = w.ru[u(i)];
  if (is_neg_inf(lo) && is_pos_inf(up)) return false;
  ++c.stats.implied_bound_checks;
  bool changed = false;
  const Entries entries = w.row_snapshot(i);
  for (const auto& entry : entries) {
    const int j = entry.first;
    const double a = entry.second;
    const double cmin = a > 0.0 ? (is_neg_inf(w.cl[u(j)]) ? 0.0 : a * w.cl[u(j)])
                                : (is_pos_inf(w.cu[u(j)]) ? 0.0 : a * w.cu[u(j)]);
    const bool cmin_inf = a > 0.0 ? is_neg_inf(w.cl[u(j)]) : is_pos_inf(w.cu[u(j)]);
    const double cmax = a > 0.0 ? (is_pos_inf(w.cu[u(j)]) ? 0.0 : a * w.cu[u(j)])
                                : (is_neg_inf(w.cl[u(j)]) ? 0.0 : a * w.cl[u(j)]);
    const bool cmax_inf = a > 0.0 ? is_pos_inf(w.cu[u(j)]) : is_neg_inf(w.cl[u(j)]);
    const bool min_excl_ok = act.min_inf - (cmin_inf ? 1 : 0) == 0;
    const bool max_excl_ok = act.max_inf - (cmax_inf ? 1 : 0) == 0;
    const double min_excl = act.min_fin - (cmin_inf ? 0.0 : cmin);
    const double max_excl = act.max_fin - (cmax_inf ? 0.0 : cmax);

    double implied_lo = -kInf;
    double implied_up = kInf;
    if (a > 0.0) {
      if (!is_pos_inf(up) && min_excl_ok) implied_up = (up - min_excl) / a;
      if (!is_neg_inf(lo) && max_excl_ok) implied_lo = (lo - max_excl) / a;
    } else {
      if (!is_pos_inf(up) && min_excl_ok) implied_lo = (up - min_excl) / a;
      if (!is_neg_inf(lo) && max_excl_ok) implied_up = (lo - max_excl) / a;
    }

    const double cl = w.cl[u(j)];
    const double cu = w.cu[u(j)];
    const double slack_up = c.btol(i, a, std::max(std::fabs(is_inf(implied_lo) ? 0.0 : implied_lo), std::fabs(is_inf(cu) ? 0.0 : cu)));
    const double slack_lo = c.btol(i, a, std::max(std::fabs(is_inf(implied_up) ? 0.0 : implied_up), std::fabs(is_inf(cl) ? 0.0 : cl)));
    if ((!is_inf(implied_lo) && !is_inf(cu) && implied_lo > cu + slack_up) ||
        (!is_inf(implied_up) && !is_inf(cl) && implied_up < cl - slack_lo)) {
      c.mark_infeasible("row " + std::to_string(i) + " implies [" + num(implied_lo) + ", " +
                        num(implied_up) + "] on column " + std::to_string(j) +
                        " which conflicts with its bounds [" + num(cl) + ", " +
                        num(cu) + "]");
      return true;
    }
    // LP: detection only. MIP: tighten integer columns.
    if (c.opt.is_mip && w.is_int[u(j)]) {
      bool col_changed = false;
      if (!is_inf(implied_lo)) {
        const double nl = std::ceil(implied_lo - kIntTol * (1.0 + std::fabs(implied_lo)));
        if (nl > w.cl[u(j)] + 0.5) {
          w.cl[u(j)] = nl;
          col_changed = true;
        }
      }
      if (!is_inf(implied_up)) {
        const double nu = std::floor(implied_up + kIntTol * (1.0 + std::fabs(implied_up)));
        if (nu < w.cu[u(j)] - 0.5) {
          w.cu[u(j)] = nu;
          col_changed = true;
        }
      }
      if (col_changed) {
        if (w.cl[u(j)] > w.cu[u(j)]) {
          c.mark_infeasible("tightening integer column " + std::to_string(j) + " from row " +
                            std::to_string(i) + " leaves no integer value");
          return true;
        }
        w.touch_col_bounds(j);
        ++c.stats.integer_bounds_tightened;
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace shodhan::presolve_detail
