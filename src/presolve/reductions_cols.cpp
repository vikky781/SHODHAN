#include <algorithm>
#include <cmath>

#include "reductions.hpp"

namespace shodhan::presolve_detail {

namespace {

std::size_t u(int i) { return static_cast<std::size_t>(i); }

constexpr double kZeroCost = 1e-12;
constexpr double kIntTol = 1e-6;

bool is_integral(double v) { return std::fabs(v - std::round(v)) <= 1e-9; }

}  // namespace

// ---------------------------------------------------------------------------
// 2. Empty column
// ---------------------------------------------------------------------------
bool EmptyColumnReduction::apply(Context& c, int j) {
  WorkModel& w = c.w;
  if (w.col_cnt[u(j)] != 0) return false;
  const double cost = w.cost[u(j)];
  const double lo = w.cl[u(j)];
  const double up = w.cu[u(j)];
  double x = 0.0;
  bool unbounded = false;
  if (cost > 0.0) {
    if (!is_inf(lo)) {
      x = lo;
    } else if (cost <= kZeroCost) {
      x = is_inf(up) ? 0.0 : up;
    } else {
      unbounded = true;
    }
  } else if (cost < 0.0) {
    if (!is_inf(up)) {
      x = up;
    } else if (-cost <= kZeroCost) {
      x = is_inf(lo) ? 0.0 : lo;
    } else {
      unbounded = true;
    }
  } else {
    // Zero cost: any feasible value; take 0 clamped into the bounds.
    if (!is_inf(lo) && x < lo) x = lo;
    if (!is_inf(up) && x > up) x = up;
  }
  if (unbounded) {
    c.unbounded = true;
    ++c.stats.unbounded_columns;
    w.remove_col(j);
    return true;
  }
  c.record(std::make_shared<EmptyColumnRecord>(j, x, cost));
  w.offset += cost * x;
  w.remove_col(j);
  ++c.stats.empty_columns;
  return true;
}

// ---------------------------------------------------------------------------
// 3. Fixed column
// ---------------------------------------------------------------------------
void FixedColumnReduction::fix(Context& c, int j, double value, bool dual_fixing) {
  WorkModel& w = c.w;
  Entries entries = w.col_snapshot(j);
  if (dual_fixing) {
    c.record(std::make_shared<DualFixingRecord>(j, value, w.cost[u(j)], std::move(entries)));
  } else {
    c.record(std::make_shared<FixedColumnRecord>(j, value, w.cost[u(j)], std::move(entries)));
  }
  w.substitute_fixed(j, value);
}

bool FixedColumnReduction::apply(Context& c, int j) {
  WorkModel& w = c.w;
  const double lo = w.cl[u(j)];
  const double up = w.cu[u(j)];
  if (is_inf(lo) || is_inf(up)) return false;
  if (lo > up + c.tol(up)) {
    c.mark_infeasible("column " + std::to_string(j) + " has lower bound above upper bound");
    return true;
  }
  // Fixing a column of width w perturbs every row it is in by up to |a| * w and
  // the objective by |c| * w, so the allowed width shrinks with the column's
  // largest coefficient or cost.
  double amp = std::fabs(w.cost[u(j)]);
  w.for_col(j, [&](int, double a) { amp = std::max(amp, std::fabs(a)); });
  if ((up - lo) * (1.0 + amp) > c.tol(lo)) return false;
  fix(c, j, lo, false);
  ++c.stats.fixed_columns;
  return true;
}

// ---------------------------------------------------------------------------
// 8. Dual fixing
// ---------------------------------------------------------------------------
// Moving x_j towards a bound cannot hurt feasibility when every row it appears
// in has an infinite side in that direction, and cannot hurt the objective when
// the cost has the right sign. Then x_j can be fixed at that bound; its reduced
// cost c_j - A_j^T y is automatically of the right sign (see docs/PRESOLVE.md).
bool DualFixingReduction::apply(Context& c, int j) {
  WorkModel& w = c.w;
  if (w.col_cnt[u(j)] == 0) return false;
  bool dec_safe = true;  // decreasing x_j never violates a row
  bool inc_safe = true;
  w.for_col(j, [&](int r, double a) {
    const double lo = w.rl[u(r)];
    const double up = w.ru[u(r)];
    if (a > 0.0) {
      if (!is_neg_inf(lo)) dec_safe = false;
      if (!is_pos_inf(up)) inc_safe = false;
    } else {
      if (!is_pos_inf(up)) dec_safe = false;
      if (!is_neg_inf(lo)) inc_safe = false;
    }
  });
  const double cost = w.cost[u(j)];
  const bool integer_col = c.opt.is_mip && w.is_int[u(j)];

  auto drop_with_rows = [&]() {
    // An improving ray exists and every row of the column stays satisfiable
    // along it: the column and its rows can be dropped; the final status is
    // decided by the driver.
    const Entries rows = w.col_snapshot(j);
    for (const auto& e : rows) w.remove_row(e.first);
    w.remove_col(j);
    c.unbounded = true;
    ++c.stats.unbounded_columns;
  };

  if (dec_safe && cost >= 0.0) {
    const double lo = w.cl[u(j)];
    if (!is_inf(lo)) {
      if (!integer_col || is_integral(lo)) {
        FixedColumnReduction::fix(c, j, lo, true);
        ++c.stats.dual_fixed_columns;
        return true;
      }
    } else if (cost > kZeroCost) {
      drop_with_rows();
      return true;
    }
  }
  if (inc_safe && cost <= 0.0) {
    const double up = w.cu[u(j)];
    if (!is_inf(up)) {
      if (!integer_col || is_integral(up)) {
        FixedColumnReduction::fix(c, j, up, true);
        ++c.stats.dual_fixed_columns;
        return true;
      }
    } else if (-cost > kZeroCost) {
      drop_with_rows();
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// 9. Integer bound rounding (MIP)
// ---------------------------------------------------------------------------
bool IntegerBoundsReduction::apply(Context& c, int j) {
  WorkModel& w = c.w;
  if (!c.opt.is_mip || !w.is_int[u(j)]) return false;
  const double lo = w.cl[u(j)];
  const double up = w.cu[u(j)];
  const double nl = is_inf(lo) ? lo : std::ceil(lo - kIntTol * (1.0 + std::fabs(lo)));
  const double nu = is_inf(up) ? up : std::floor(up + kIntTol * (1.0 + std::fabs(up)));
  if (nl == lo && nu == up) return false;
  if (nl > nu) {
    c.mark_infeasible("integer column " + std::to_string(j) + " has no integer value within its bounds");
    return true;
  }
  w.cl[u(j)] = nl;
  w.cu[u(j)] = nu;
  w.touch_col_bounds(j);
  ++c.stats.integer_bounds_rounded;
  return true;
}

}  // namespace shodhan::presolve_detail
