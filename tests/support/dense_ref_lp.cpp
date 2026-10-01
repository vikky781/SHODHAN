// TEST ORACLE ONLY - see dense_ref_lp.hpp.

#include "support/dense_ref_lp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace shodhan::testing {

namespace {

using Real = long double;

constexpr Real kPivotTol = 1e-10L;
constexpr Real kCostTol = 1e-10L;
constexpr long kRefactorEvery = 40;
constexpr long kMaxIterations = 200000;

// How model column j maps onto non-negative variables z:
//   x_j = shift + sign * z[a]              (kind 0: lower finite, kind 1: only upper finite)
//   x_j = z[a] - z[b]                      (kind 2: free)
struct ColMap {
  int kind = 0;
  Real shift = 0;
  Real sign = 1;
  int a = -1;
  int b = -1;
};

struct StdRow {
  std::vector<std::pair<int, Real>> terms;  // over z (structural and slack variables)
  Real rhs = 0;
  int origin = -1;  // model row, or -1 for a column upper-bound row
  Real flip = 1;    // tableau row = flip * std row
};

struct Tableau {
  int m = 0;
  int n = 0;  // columns excluding rhs
  std::vector<std::vector<Real>> t;  // m x (n + 1), last column is rhs
  std::vector<int> basis;
};

void pivot(Tableau& tb, int row, int col) {
  std::vector<Real>& pr = tb.t[static_cast<std::size_t>(row)];
  const Real p = pr[static_cast<std::size_t>(col)];
  for (Real& v : pr) v /= p;
  for (int k = 0; k < tb.m; ++k) {
    if (k == row) continue;
    std::vector<Real>& r = tb.t[static_cast<std::size_t>(k)];
    const Real f = r[static_cast<std::size_t>(col)];
    if (f == 0) continue;
    for (std::size_t j = 0; j < r.size(); ++j) r[j] -= f * pr[j];
    r[static_cast<std::size_t>(col)] = 0;
  }
  tb.basis[static_cast<std::size_t>(row)] = col;
}

// Rebuilds the tableau for the current basis from the original data, removing
// the rounding error accumulated by many pivots (Gauss-Jordan, partial pivoting).
void refactor(Tableau& tb, const std::vector<std::vector<Real>>& orig) {
  const std::vector<int> wanted = tb.basis;
  tb.t = orig;
  std::vector<char> row_used(static_cast<std::size_t>(tb.m), 0);
  for (const int col : wanted) {
    int best = -1;
    Real best_abs = 0;
    for (int k = 0; k < tb.m; ++k) {
      if (row_used[static_cast<std::size_t>(k)]) continue;
      const Real a = std::fabs(tb.t[static_cast<std::size_t>(k)][static_cast<std::size_t>(col)]);
      if (a > best_abs) {
        best_abs = a;
        best = k;
      }
    }
    if (best < 0 || best_abs < 1e-13L) continue;  // numerically singular: leave as is
    pivot(tb, best, col);
    row_used[static_cast<std::size_t>(best)] = 1;
  }
}

enum class SimplexOutcome { Optimal, Unbounded, IterationLimit };

// Bland's rule. `allowed(j)` says whether column j may enter the basis.
SimplexOutcome simplex(Tableau& tb, const std::vector<Real>& cost, const std::vector<char>& allowed,
                       const std::vector<std::vector<Real>>& orig, long* iterations) {
  const std::size_t n = static_cast<std::size_t>(tb.n);
  std::vector<Real> red(n);
  std::vector<char> is_basic(n);
  for (;;) {
    if (++*iterations > kMaxIterations) return SimplexOutcome::IterationLimit;
    if (*iterations % kRefactorEvery == 0) refactor(tb, orig);
    // Reduced costs from scratch: red_j = c_j - sum_k c_B[k] T[k][j].
    std::fill(is_basic.begin(), is_basic.end(), 0);
    for (int k = 0; k < tb.m; ++k) is_basic[static_cast<std::size_t>(tb.basis[static_cast<std::size_t>(k)])] = 1;
    for (std::size_t j = 0; j < n; ++j) red[j] = cost[j];
    for (int k = 0; k < tb.m; ++k) {
      const Real cb = cost[static_cast<std::size_t>(tb.basis[static_cast<std::size_t>(k)])];
      if (cb == 0) continue;
      const std::vector<Real>& r = tb.t[static_cast<std::size_t>(k)];
      for (std::size_t j = 0; j < n; ++j) red[j] -= cb * r[j];
    }
    int enter = -1;
    for (std::size_t j = 0; j < n; ++j) {
      if (allowed[j] && !is_basic[j] && red[j] < -kCostTol) {
        enter = static_cast<int>(j);
        break;
      }
    }
    if (enter < 0) {
      refactor(tb, orig);
      return SimplexOutcome::Optimal;
    }

    int leave = -1;
    Real best = 0;
    for (int k = 0; k < tb.m; ++k) {
      const std::vector<Real>& r = tb.t[static_cast<std::size_t>(k)];
      const Real a = r[static_cast<std::size_t>(enter)];
      if (a <= kPivotTol) continue;
      const Real ratio = r[n] / a;
      if (leave < 0 || ratio < best - 1e-12L * (1 + std::fabs(best)) ||
          (ratio <= best + 1e-12L * (1 + std::fabs(best)) &&
           tb.basis[static_cast<std::size_t>(k)] < tb.basis[static_cast<std::size_t>(leave)])) {
        leave = k;
        best = ratio;
      }
    }
    if (leave < 0) return SimplexOutcome::Unbounded;
    pivot(tb, leave, enter);
  }
}

}  // namespace

RefLpResult solve_dense_lp(const LpModel& model) {
  RefLpResult result;
  const int mrows = model.n_rows;
  const int ncols = model.n_cols;
  const Real sgn = model.sense == Sense::Maximize ? -1 : 1;

  // ---- variables ----
  std::vector<ColMap> cmap(static_cast<std::size_t>(ncols));
  int nz = 0;
  std::vector<std::pair<int, Real>> upper_rows;  // (z index, upper bound on z)
  for (int j = 0; j < ncols; ++j) {
    const double lo = model.col_lower[static_cast<std::size_t>(j)];
    const double up = model.col_upper[static_cast<std::size_t>(j)];
    ColMap& c = cmap[static_cast<std::size_t>(j)];
    if (!is_inf(lo)) {
      c.kind = 0;
      c.shift = lo;
      c.sign = 1;
      c.a = nz++;
      if (!is_inf(up)) upper_rows.emplace_back(c.a, static_cast<Real>(up) - static_cast<Real>(lo));
    } else if (!is_inf(up)) {
      c.kind = 1;
      c.shift = up;
      c.sign = -1;
      c.a = nz++;
    } else {
      c.kind = 2;
      c.a = nz++;
      c.b = nz++;
    }
  }
  const int nstruct = nz;

  // Row expression a_i . x = const_i + sum_t coef_t z_t.
  std::vector<std::vector<std::pair<int, Real>>> expr(static_cast<std::size_t>(mrows));
  std::vector<Real> constant(static_cast<std::size_t>(mrows), 0);
  for (int j = 0; j < ncols; ++j) {
    const ColMap& c = cmap[static_cast<std::size_t>(j)];
    for (Index p = model.A.col_start[static_cast<std::size_t>(j)];
         p < model.A.col_start[static_cast<std::size_t>(j) + 1]; ++p) {
      const std::size_t i = static_cast<std::size_t>(model.A.row_index[static_cast<std::size_t>(p)]);
      const Real a = model.A.value[static_cast<std::size_t>(p)];
      if (c.kind == 2) {
        expr[i].emplace_back(c.a, a);
        expr[i].emplace_back(c.b, -a);
      } else {
        constant[i] += a * c.shift;
        expr[i].emplace_back(c.a, a * c.sign);
      }
    }
  }

  // ---- standard-form rows ----
  std::vector<StdRow> rows;
  int nslack = 0;
  auto add_row = [&](int origin, const std::vector<std::pair<int, Real>>& terms, int slack_sign,
                     Real rhs) {
    StdRow r;
    r.origin = origin;
    r.terms = terms;
    if (slack_sign != 0) r.terms.emplace_back(nstruct + nslack++, static_cast<Real>(slack_sign));
    r.rhs = rhs;
    rows.push_back(std::move(r));
  };
  for (int i = 0; i < mrows; ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    const double lo = model.row_lower[si];
    const double up = model.row_upper[si];
    const bool lo_f = !is_inf(lo);
    const bool up_f = !is_inf(up);
    if (lo_f && up_f && lo == up) {
      add_row(i, expr[si], 0, static_cast<Real>(lo) - constant[si]);
    } else {
      if (lo_f) add_row(i, expr[si], -1, static_cast<Real>(lo) - constant[si]);
      if (up_f) add_row(i, expr[si], +1, static_cast<Real>(up) - constant[si]);
    }
  }
  for (const auto& ur : upper_rows) add_row(-1, {{ur.first, 1}}, +1, ur.second);

  const int m = static_cast<int>(rows.size());
  const int ntot = nstruct + nslack;  // z variables plus slacks
  const int n = ntot + m;             // plus one artificial per row

  // ---- scale the standard-form system (powers of two, exact) ----
  // Wide coefficient ranges make absolute pivot tolerances meaningless, so the
  // oracle equilibrates its own matrix. Scaled system: (R M C) z' = R r with
  // z = C z' and duals y = R y'.
  std::vector<std::vector<Real>> dense(static_cast<std::size_t>(m),
                                       std::vector<Real>(static_cast<std::size_t>(ntot), 0));
  for (int k = 0; k < m; ++k) {
    for (const auto& term : rows[static_cast<std::size_t>(k)].terms) {
      dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(term.first)] += term.second;
    }
  }
  std::vector<Real> rs(static_cast<std::size_t>(m), 1);
  std::vector<Real> cs(static_cast<std::size_t>(ntot), 1);
  auto pow2 = [](Real v) { return std::ldexp(static_cast<Real>(1), static_cast<int>(std::lround(std::log2(v)))); };
  for (int pass = 0; pass < 6; ++pass) {
    for (int k = 0; k < m; ++k) {
      Real mn = 0;
      Real mx = 0;
      for (int t = 0; t < ntot; ++t) {
        const Real a = std::fabs(dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)]);
        if (a == 0) continue;
        mn = mn == 0 ? a : std::min(mn, a);
        mx = std::max(mx, a);
      }
      if (mx == 0) continue;
      const Real f = 1 / std::sqrt(mn * mx);
      for (int t = 0; t < ntot; ++t) dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)] *= f;
      rs[static_cast<std::size_t>(k)] *= f;
    }
    for (int t = 0; t < ntot; ++t) {
      Real mn = 0;
      Real mx = 0;
      for (int k = 0; k < m; ++k) {
        const Real a = std::fabs(dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)]);
        if (a == 0) continue;
        mn = mn == 0 ? a : std::min(mn, a);
        mx = std::max(mx, a);
      }
      if (mx == 0) continue;
      const Real f = 1 / std::sqrt(mn * mx);
      for (int k = 0; k < m; ++k) dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)] *= f;
      cs[static_cast<std::size_t>(t)] *= f;
    }
  }
  for (Real& v : rs) v = pow2(v);
  for (Real& v : cs) v = pow2(v);

  // ---- tableau with artificials ----
  Tableau tb;
  tb.m = m;
  tb.n = n;
  tb.t.assign(static_cast<std::size_t>(m), std::vector<Real>(static_cast<std::size_t>(n) + 1, 0));
  tb.basis.resize(static_cast<std::size_t>(m));
  for (int k = 0; k < m; ++k) {
    StdRow& r = rows[static_cast<std::size_t>(k)];
    const Real scaled_rhs = rs[static_cast<std::size_t>(k)] * r.rhs;
    r.flip = scaled_rhs < 0 ? -1 : 1;
    std::vector<Real>& tr = tb.t[static_cast<std::size_t>(k)];
    for (const auto& term : r.terms) {
      tr[static_cast<std::size_t>(term.first)] += r.flip * rs[static_cast<std::size_t>(k)] * term.second *
                                                 cs[static_cast<std::size_t>(term.first)];
    }
    tr[static_cast<std::size_t>(ntot + k)] = 1;
    tr[static_cast<std::size_t>(n)] = r.flip * scaled_rhs;
    tb.basis[static_cast<std::size_t>(k)] = ntot + k;
  }
  const std::vector<std::vector<Real>> original = tb.t;

  // ---- phase 1 ----
  std::vector<Real> cost1(static_cast<std::size_t>(n), 0);
  for (int k = 0; k < m; ++k) cost1[static_cast<std::size_t>(ntot + k)] = 1;
  std::vector<char> allowed(static_cast<std::size_t>(n), 1);
  SimplexOutcome out = simplex(tb, cost1, allowed, original, &result.iterations);
  if (out == SimplexOutcome::IterationLimit) {
    result.status = Status::NumericalError;
    return result;
  }
  Real infeas = 0;
  Real rhs_scale = 1;
  for (int k = 0; k < m; ++k) {
    const std::size_t sk = static_cast<std::size_t>(k);
    if (tb.basis[sk] >= ntot) infeas += tb.t[sk][static_cast<std::size_t>(n)];
    rhs_scale = std::max(rhs_scale, std::fabs(rs[sk] * rows[sk].rhs));
  }
  if (infeas > 1e-8L * rhs_scale) {
    result.status = Status::Infeasible;
    return result;
  }
  // Drive artificials out of the basis where possible.
  for (int k = 0; k < m; ++k) {
    const std::size_t sk = static_cast<std::size_t>(k);
    if (tb.basis[sk] < ntot) continue;
    int best = -1;
    Real best_abs = 1e-9L;
    for (int j = 0; j < ntot; ++j) {
      const Real a = std::fabs(tb.t[sk][static_cast<std::size_t>(j)]);
      if (a > best_abs) {
        best_abs = a;
        best = j;
      }
    }
    if (best >= 0) pivot(tb, k, best);  // rhs is (numerically) zero: no change in the point
  }

  // ---- phase 2 ----
  Real cost_scale = 1;
  std::vector<Real> cost2(static_cast<std::size_t>(n), 0);
  for (int j = 0; j < ncols; ++j) {
    const ColMap& c = cmap[static_cast<std::size_t>(j)];
    const Real cj = sgn * static_cast<Real>(model.col_cost[static_cast<std::size_t>(j)]);
    if (c.kind == 2) {
      cost2[static_cast<std::size_t>(c.a)] = cj * cs[static_cast<std::size_t>(c.a)];
      cost2[static_cast<std::size_t>(c.b)] = -cj * cs[static_cast<std::size_t>(c.b)];
    } else {
      cost2[static_cast<std::size_t>(c.a)] = cj * c.sign * cs[static_cast<std::size_t>(c.a)];
    }
  }
  Real cmax = 0;
  for (const Real v : cost2) cmax = std::max(cmax, std::fabs(v));
  if (cmax > 0) {
    cost_scale = pow2(1 / cmax);
    for (Real& v : cost2) v *= cost_scale;
  }
  for (int k = 0; k < m; ++k) allowed[static_cast<std::size_t>(ntot + k)] = 0;
  out = simplex(tb, cost2, allowed, original, &result.iterations);
  if (out == SimplexOutcome::IterationLimit) {
    result.status = Status::NumericalError;
    return result;
  }
  if (out == SimplexOutcome::Unbounded) {
    result.status = Status::Unbounded;
    return result;
  }

  // ---- primal ----
  std::vector<Real> z(static_cast<std::size_t>(n), 0);
  for (int k = 0; k < m; ++k) {
    const std::size_t bi = static_cast<std::size_t>(tb.basis[static_cast<std::size_t>(k)]);
    z[bi] = tb.t[static_cast<std::size_t>(k)][static_cast<std::size_t>(n)];
    if (bi < static_cast<std::size_t>(ntot)) z[bi] *= cs[bi];  // z = C z'
  }
  Solution& sol = result.solution;
  sol.x.assign(static_cast<std::size_t>(ncols), 0.0);
  for (int j = 0; j < ncols; ++j) {
    const ColMap& c = cmap[static_cast<std::size_t>(j)];
    Real v = 0;
    if (c.kind == 2) {
      v = z[static_cast<std::size_t>(c.a)] - z[static_cast<std::size_t>(c.b)];
    } else {
      v = c.shift + c.sign * z[static_cast<std::size_t>(c.a)];
    }
    sol.x[static_cast<std::size_t>(j)] = static_cast<double>(v);
  }

  // ---- duals: y' = c_B^T B^{-1}, read from the artificial columns ----
  std::vector<Real> yrow(static_cast<std::size_t>(m), 0);
  for (int i = 0; i < m; ++i) {
    Real s = 0;
    for (int k = 0; k < m; ++k) {
      const Real cb = cost2[static_cast<std::size_t>(tb.basis[static_cast<std::size_t>(k)])];
      s += cb * tb.t[static_cast<std::size_t>(k)][static_cast<std::size_t>(ntot + i)];
    }
    yrow[static_cast<std::size_t>(i)] = s;
  }
  std::vector<Real> y(static_cast<std::size_t>(mrows), 0);
  for (int k = 0; k < m; ++k) {
    const StdRow& r = rows[static_cast<std::size_t>(k)];
    if (r.origin >= 0) {
      y[static_cast<std::size_t>(r.origin)] +=
          rs[static_cast<std::size_t>(k)] * r.flip * yrow[static_cast<std::size_t>(k)] / cost_scale;
    }
  }
  sol.y.assign(static_cast<std::size_t>(mrows), 0.0);
  for (int i = 0; i < mrows; ++i) sol.y[static_cast<std::size_t>(i)] = static_cast<double>(y[static_cast<std::size_t>(i)]);

  // d = c - A^T y in the minimization form.
  std::vector<double> aty(static_cast<std::size_t>(ncols), 0.0);
  model.A.multiply_transpose(sol.y, aty);
  sol.d.assign(static_cast<std::size_t>(ncols), 0.0);
  double obj = model.objective_offset;
  for (int j = 0; j < ncols; ++j) {
    const std::size_t sj = static_cast<std::size_t>(j);
    sol.d[sj] = static_cast<double>(sgn) * model.col_cost[sj] - aty[sj];
    obj += model.col_cost[sj] * sol.x[sj];
  }
  sol.objective = obj;
  result.status = Status::Optimal;
  return result;
}

}  // namespace shodhan::testing
