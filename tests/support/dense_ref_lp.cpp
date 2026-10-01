// TEST ORACLE ONLY - see dense_ref_lp.hpp.

#include "support/dense_ref_lp.hpp"

#include "shodhan/kkt.hpp"
#include "shodhan/scaling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace shodhan::testing {

namespace {

using Real = long double;

constexpr Real kCostTol = 1e-12L;
constexpr long kMaxIterations = 20000;

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

enum class SimplexOutcome { Optimal, Unbounded, IterationLimit, Unsafe };

// Numerical settings; solve_dense_lp retries with more conservative ones when an
// attempt ends in a numerical failure.
struct Settings {
  Real pivot_tol;
  long refactor_every;
  bool harris;  // Harris two-pass ratio test; otherwise textbook Bland ratio test
};

// Entering variable: smallest index with a negative reduced cost (Bland); a
// candidate that no row can block with a safe pivot is skipped in favour of the
// next one. Leaving variable: Harris two-pass ratio test with a preference for
// large pivots among near-ties (smallest basis index breaks remaining ties),
// which keeps the tableau well conditioned. `ent_tol[j]` is the reduced-cost
// tolerance of column j. `art_start` is the index of the first artificial
// column when artificials that stayed in the basis at level zero must be kept
// at zero (phase 2), or -1.
SimplexOutcome simplex(Tableau& tb, const std::vector<Real>& cost, const std::vector<char>& allowed,
                       const std::vector<Real>& ent_tol, const std::vector<Real>& colscale,
                       const std::vector<std::vector<Real>>& orig, int art_start, const Settings& set,
                       long* iterations) {
  const std::size_t n = static_cast<std::size_t>(tb.n);
  const std::size_t m = static_cast<std::size_t>(tb.m);
  constexpr Real kHarrisDelta = 1e-9L;
  constexpr Real kClearlyNegative = 1e4L;  // x the column's tolerance: unbounded only beyond this
  std::vector<Real> red(n);
  std::vector<char> is_basic(n);
  std::vector<Real> step(m);
  std::vector<char> eligible(m);
  for (;;) {
    if (++*iterations > kMaxIterations) return SimplexOutcome::IterationLimit;
    if (*iterations % set.refactor_every == 0) refactor(tb, orig);
    // Reduced costs from scratch: red_j = c_j - sum_k c_B[k] T[k][j].
    std::fill(is_basic.begin(), is_basic.end(), 0);
    for (std::size_t k = 0; k < m; ++k) is_basic[static_cast<std::size_t>(tb.basis[k])] = 1;
    for (std::size_t j = 0; j < n; ++j) red[j] = cost[j];
    for (std::size_t k = 0; k < m; ++k) {
      const Real cb = cost[static_cast<std::size_t>(tb.basis[k])];
      if (cb == 0) continue;
      const std::vector<Real>& r = tb.t[k];
      for (std::size_t j = 0; j < n; ++j) red[j] -= cb * r[j];
    }

    int enter = -1;
    int leave = -1;
    bool clearly_unbounded = false;
    bool unsafe_negative = false;  // attractive column whose only blockers are too small to pivot on
    for (std::size_t e = 0; e < n && leave < 0; ++e) {
      if (!allowed[e] || is_basic[e] || !(red[e] < -ent_tol[e])) continue;

      // A pivot must not be noise next to the column's large entries (relative
      // tolerance), but a column whose entries are all small is still usable.
      Real col_max = 0;
      for (std::size_t k = 0; k < m; ++k) col_max = std::max(col_max, std::fabs(tb.t[k][e]));
      const Real ptol = std::max<Real>(set.pivot_tol * col_max, 1e-14L);

      // Pass 1: the largest step that keeps every row within a small tolerance.
      bool any = false;
      Real theta = 0;
      for (std::size_t k = 0; k < m; ++k) {
        const Real a = tb.t[k][e];
        eligible[k] = 0;
        if (art_start >= 0 && tb.basis[k] >= art_start) {
          // A zero-level artificial must not increase: any nonzero entry
          // blocks at step 0 (the pivot may be negative).
          if (std::fabs(a) <= ptol) continue;
          eligible[k] = 1;
          step[k] = 0;
          const Real bound = (kHarrisDelta / colscale[static_cast<std::size_t>(tb.basis[k])]) / std::fabs(a);
          if (!any || bound < theta) theta = bound;
          any = true;
          continue;
        }
        if (a <= ptol) continue;
        eligible[k] = 1;
        const Real rhs = std::max<Real>(tb.t[k][n], 0);
        step[k] = rhs / a;
        const Real delta_k = kHarrisDelta / colscale[static_cast<std::size_t>(tb.basis[k])];
        const Real bound = (rhs + delta_k) / a;
        if (!any || bound < theta) theta = bound;
        any = true;
      }
      if (!any) {
        // Nothing can be pivoted on safely. If there are genuine (if small) blocking
        // entries this is a numerical hazard, not a proof of unboundedness.
        Real max_blocking = 0;
        for (std::size_t k = 0; k < m; ++k) {
          const Real a = tb.t[k][e];
          const bool zero_level_artificial = art_start >= 0 && tb.basis[k] >= art_start;
          max_blocking = std::max(max_blocking, zero_level_artificial ? std::fabs(a) : a);
        }
        if (red[e] < -kClearlyNegative * ent_tol[e]) {
          if (max_blocking <= 1e-12L) {
            clearly_unbounded = true;
          } else {
            unsafe_negative = true;
          }
        }
        continue;  // try the next candidate
      }

      if (set.harris) {
        // Pass 2: among rows that block within theta, take the largest pivot.
        Real best_pivot = 0;
        for (std::size_t k = 0; k < m; ++k) {
          if (!eligible[k] || step[k] > theta) continue;
          const Real a = std::fabs(tb.t[k][e]);
          if (leave < 0 || a > best_pivot * (1 + 1e-9L) ||
              (a >= best_pivot * (1 - 1e-9L) && tb.basis[k] < tb.basis[static_cast<std::size_t>(leave)])) {
            leave = static_cast<int>(k);
            best_pivot = a;
          }
        }
      } else {
        // Textbook Bland: exact minimum ratio, ties to the smallest basis index.
        Real best = 0;
        for (std::size_t k = 0; k < m; ++k) {
          if (!eligible[k]) continue;
          const Real ratio = step[k];
          if (leave < 0 || ratio < best - 1e-12L * (1 + std::fabs(best)) ||
              (ratio <= best + 1e-12L * (1 + std::fabs(best)) &&
               tb.basis[k] < tb.basis[static_cast<std::size_t>(leave)])) {
            leave = static_cast<int>(k);
            best = ratio;
          }
        }
      }
      enter = static_cast<int>(e);
    }

    if (leave < 0) {
      if (clearly_unbounded) return SimplexOutcome::Unbounded;
      if (unsafe_negative) return SimplexOutcome::Unsafe;
      refactor(tb, orig);
      return SimplexOutcome::Optimal;
    }
    pivot(tb, leave, enter);
    // Clean tiny negative right-hand sides that Harris's tolerance allowed.
    if (set.harris) for (std::size_t k = 0; k < m; ++k) {
      if (tb.t[k][n] < 0 && tb.t[k][n] > -(kHarrisDelta * 10) / colscale[static_cast<std::size_t>(tb.basis[k])]) tb.t[k][n] = 0;
    }
  }
}

}  // namespace

static RefLpResult solve_once(const LpModel& model, const Settings& set) {
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
  for (int pass = 0; pass < 12; ++pass) {
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
  // One max-abs equilibration pass: columns, then rows.
  for (int t = 0; t < ntot; ++t) {
    Real mx = 0;
    for (int k = 0; k < m; ++k) mx = std::max(mx, std::fabs(dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)]));
    if (mx == 0) continue;
    for (int k = 0; k < m; ++k) dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)] /= mx;
    cs[static_cast<std::size_t>(t)] /= mx;
  }
  for (int k = 0; k < m; ++k) {
    Real mx = 0;
    for (int t = 0; t < ntot; ++t) mx = std::max(mx, std::fabs(dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)]));
    if (mx == 0) continue;
    for (int t = 0; t < ntot; ++t) dense[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)] /= mx;
    rs[static_cast<std::size_t>(k)] /= mx;
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
  std::vector<Real> cs_full(static_cast<std::size_t>(n), 1);
  for (int t = 0; t < ntot; ++t) cs_full[static_cast<std::size_t>(t)] = cs[static_cast<std::size_t>(t)];

  // ---- phase 1 ----
  std::vector<Real> cost1(static_cast<std::size_t>(n), 0);
  for (int k = 0; k < m; ++k) cost1[static_cast<std::size_t>(ntot + k)] = 1;
  std::vector<char> allowed(static_cast<std::size_t>(n), 1);
  const std::vector<Real> ent_tol1(static_cast<std::size_t>(n), kCostTol);
  SimplexOutcome out = simplex(tb, cost1, allowed, ent_tol1, cs_full, original, -1, set, &result.iterations);
  if (out != SimplexOutcome::Optimal) {
    result.status = Status::NumericalError;
    return result;
  }
  // Each artificial still in the basis is judged in the ORIGINAL units of its own
  // row: a conflict in a small row must not hide under the scale of a large one,
  // and scaling noise must not look like infeasibility.
  Real infeas = 0;
  bool infeasible = false;
  for (int k = 0; k < m; ++k) {
    const std::size_t sk = static_cast<std::size_t>(k);
    if (tb.basis[sk] < ntot) continue;
    const Real value = tb.t[sk][static_cast<std::size_t>(n)];
    infeas += value;
    if (value / rs[sk] > 1e-8L * (1 + std::fabs(rows[sk].rhs))) infeasible = true;
  }
  result.phase1_residual = static_cast<double>(infeas);
  if (infeasible) {
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
  // Reduced-cost tolerance of each column in UNSCALED units: scaled and
  // normalised reduced costs are cs * cost_scale times the original ones.
  std::vector<Real> cmag(static_cast<std::size_t>(n), 0);
  for (int j = 0; j < ncols; ++j) {
    const ColMap& c = cmap[static_cast<std::size_t>(j)];
    const Real cj = std::fabs(static_cast<Real>(model.col_cost[static_cast<std::size_t>(j)]));
    cmag[static_cast<std::size_t>(c.a)] = cj;
    if (c.kind == 2) cmag[static_cast<std::size_t>(c.b)] = cj;
  }
  std::vector<Real> ent_tol2(static_cast<std::size_t>(n), kCostTol);
  for (int t = 0; t < ntot; ++t) {
    const std::size_t st = static_cast<std::size_t>(t);
    ent_tol2[st] = kCostTol * cs[st] * cost_scale * (1 + cmag[st]);
  }
  out = simplex(tb, cost2, allowed, ent_tol2, cs_full, original, ntot, set, &result.iterations);
  if (out == SimplexOutcome::IterationLimit || out == SimplexOutcome::Unsafe) {
    result.status = Status::NumericalError;
    return result;
  }
  if (out == SimplexOutcome::Unbounded) {
    result.status = Status::Unbounded;
    return result;
  }

  // ---- primal ----
  for (int k = 0; k < m; ++k) {
    const std::size_t sk = static_cast<std::size_t>(k);
    if (tb.basis[sk] >= ntot) {
      result.artificial_residual = std::max(result.artificial_residual, static_cast<double>(std::fabs(tb.t[sk][static_cast<std::size_t>(n)])));
    }
  }
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

  // Accept the point only if it satisfies the ORIGINAL constraints: a basis that
  // is infeasible after reinversion means pivoting went wrong numerically.
  {
    std::vector<double> act(static_cast<std::size_t>(mrows), 0.0);
    std::vector<double> mag(static_cast<std::size_t>(mrows), 0.0);
    for (int j = 0; j < ncols; ++j) {
      const double xj = sol.x[static_cast<std::size_t>(j)];
      for (Index p = model.A.col_start[static_cast<std::size_t>(j)]; p < model.A.col_start[static_cast<std::size_t>(j) + 1]; ++p) {
        const std::size_t i = static_cast<std::size_t>(model.A.row_index[static_cast<std::size_t>(p)]);
        const double t = model.A.value[static_cast<std::size_t>(p)] * xj;
        act[i] += t;
        mag[i] += std::fabs(t);
      }
    }
    bool bad = false;
    for (int i = 0; i < mrows; ++i) {
      const std::size_t si = static_cast<std::size_t>(i);
      const double lo = model.row_lower[si];
      const double up = model.row_upper[si];
      if (!is_inf(lo) && lo - act[si] > 1e-9 * (1.0 + std::fabs(lo) + mag[si])) bad = true;
      if (!is_inf(up) && act[si] - up > 1e-9 * (1.0 + std::fabs(up) + mag[si])) bad = true;
    }
    for (int j = 0; j < ncols; ++j) {
      const std::size_t sj = static_cast<std::size_t>(j);
      const double lo = model.col_lower[sj];
      const double up = model.col_upper[sj];
      if (!is_inf(lo) && lo - sol.x[sj] > 1e-9 * (1.0 + std::fabs(lo))) bad = true;
      if (!is_inf(up) && sol.x[sj] - up > 1e-9 * (1.0 + std::fabs(up))) bad = true;
    }
    if (bad) {
      result.status = Status::NumericalError;
      return result;
    }
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

namespace {

// Un-hidden accuracy of an optimal answer: worst of the relative primal
// violation, relative dual violation and the plain relative duality gap.
double quality(const LpModel& model, const Solution& sol) {
  const KktReport k = check_kkt(model, sol, 1.0);
  const double gap = std::fabs(k.primal_objective - k.dual_objective) /
                     (1.0 + std::fabs(k.primal_objective) + std::fabs(k.dual_objective));
  return std::max({k.primal_infeasibility_rel, k.dual_infeasibility_rel, k.dual_mismatch_rel, gap});
}

}  // namespace

RefLpResult solve_dense_lp(const LpModel& model) {
  const Settings attempts[] = {{1e-9L, 40, false}, {1e-8L, 40, true}, {1e-7L, 10, false}, {1e-7L, 10, true},
                               {1e-6L, 4, true},   {1e-9L, 5, false}, {1e-8L, 5, true},   {1e-5L, 2, true}};
  constexpr double kGoodEnough = 1e-9;
  constexpr int kVerdictVotes = 2;  // definite verdicts need two configurations to agree

  RefLpResult best_optimal;  // lowest-residual Optimal answer so far
  best_optimal.status = Status::NumericalError;
  double best_quality = 1e300;
  RefLpResult verdict;  // last definite Infeasible / Unbounded answer
  verdict.status = Status::NumericalError;
  int votes_infeasible = 0;
  int votes_unbounded = 0;
  long total_iterations = 0;

  // Returns true when the search can stop.
  auto consider = [&](RefLpResult r) -> bool {
    total_iterations += r.iterations;
    switch (r.status) {
      case Status::Optimal: {
        const double q = quality(model, r.solution);
        if (q < best_quality) {
          best_quality = q;
          best_optimal = std::move(r);
        }
        return best_quality <= kGoodEnough;
      }
      case Status::Infeasible:
        ++votes_infeasible;
        verdict = std::move(r);
        return votes_infeasible >= kVerdictVotes;
      case Status::Unbounded:
        ++votes_unbounded;
        verdict = std::move(r);
        return votes_unbounded >= kVerdictVotes;
      default:
        return false;
    }
  };

  bool done = false;
  for (const Settings& set : attempts) {
    if (consider(solve_once(model, set))) {
      done = true;
      break;
    }
  }
  if (!done) {
    // Last resort: solve a pre-scaled copy and map the answer back.
    const Scaling sc = compute_scaling(model);
    const LpModel scaled = apply_scaling(model, sc);
    for (const Settings& set : attempts) {
      RefLpResult r2 = solve_once(scaled, set);
      if (r2.status == Status::Optimal) r2.solution = unscale_solution(sc, r2.solution);
      if (consider(std::move(r2))) break;
    }
  }

  RefLpResult result;
  result.status = Status::NumericalError;
  if (best_optimal.status == Status::Optimal && best_quality <= kGoodEnough) {
    result = std::move(best_optimal);
  } else if (votes_infeasible >= kVerdictVotes) {
    result = std::move(verdict);
    result.status = Status::Infeasible;
  } else if (votes_unbounded >= kVerdictVotes) {
    result = std::move(verdict);
    result.status = Status::Unbounded;
  } else if (best_optimal.status == Status::Optimal) {
    result = std::move(best_optimal);  // best available, not fully accurate
  } else if (votes_infeasible + votes_unbounded > 0) {
    result = std::move(verdict);  // a single (unconfirmed) verdict is all there is
  }
  result.iterations = total_iterations;
  return result;
}

}  // namespace shodhan::testing
