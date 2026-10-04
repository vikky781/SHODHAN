#include "support/mip_oracle.hpp"

#include <algorithm>
#include <cmath>

#include "support/dense_ref_lp.hpp"

namespace shodhan::testing {

namespace {

double objective_of(const LpModel& m, const std::vector<double>& x) {
  double o = m.objective_offset;
  for (Index j = 0; j < m.n_cols; ++j) o += m.col_cost[to_size(j)] * x[to_size(j)];
  return o;
}

bool better(const LpModel& m, double a, double b) { return m.sense == Sense::Maximize ? a > b + 1e-9 * (1 + std::fabs(b)) : a < b - 1e-9 * (1 + std::fabs(b)); }

}  // namespace

MipRefResult solve_mip_brute_force(const LpModel& m, long long max_combinations) {
  MipRefResult out;
  std::vector<Index> ints;
  std::vector<long long> lo, hi;
  long long combos = 1;
  bool has_cont = false;
  for (Index j = 0; j < m.n_cols; ++j) {
    if (!m.is_integer(j)) {
      has_cont = true;
      continue;
    }
    const double l = m.col_lower[to_size(j)], h = m.col_upper[to_size(j)];
    if (is_inf(l) || is_inf(h)) {
      out.too_large = true;
      return out;
    }
    const long long a = static_cast<long long>(std::ceil(l - 1e-9)), b = static_cast<long long>(std::floor(h + 1e-9));
    if (b < a) {
      out.status = Status::Infeasible;
      return out;
    }
    ints.push_back(j);
    lo.push_back(a);
    hi.push_back(b);
    combos *= (b - a + 1);
    if (combos > max_combinations) {
      out.too_large = true;
      return out;
    }
  }
  std::vector<long long> v = lo;
  bool have = false;
  double best = 0.0;
  std::vector<double> best_x;
  std::vector<double> x(to_size(m.n_cols), 0.0);
  std::vector<double> act;
  bool inconclusive = false;
  while (true) {
    ++out.evaluations;
    for (std::size_t k = 0; k < ints.size(); ++k) x[to_size(ints[k])] = static_cast<double>(v[k]);
    if (!has_cont) {
      act.assign(to_size(m.n_rows), 0.0);
      for (Index j = 0; j < m.n_cols; ++j) {
        const double xj = x[to_size(j)];
        if (xj == 0.0) continue;
        for (Index p = m.A.col_start[to_size(j)]; p < m.A.col_start[to_size(j) + 1]; ++p) act[to_size(m.A.row_index[to_size(p)])] += m.A.value[to_size(p)] * xj;
      }
      bool ok = true;
      for (Index i = 0; i < m.n_rows && ok; ++i) {
        const double a = act[to_size(i)], l = m.row_lower[to_size(i)], h = m.row_upper[to_size(i)];
        if ((!is_inf(l) && a < l - 1e-9 * (1 + std::fabs(l))) || (!is_inf(h) && a > h + 1e-9 * (1 + std::fabs(h)))) ok = false;
      }
      if (ok) {
        const double o = objective_of(m, x);
        if (!have || better(m, o, best)) {
          have = true;
          best = o;
          best_x = x;
        }
      }
    } else {
      LpModel fixed = m;
      for (std::size_t k = 0; k < ints.size(); ++k) fixed.col_lower[to_size(ints[k])] = fixed.col_upper[to_size(ints[k])] = static_cast<double>(v[k]);
      const RefLpResult r = solve_dense_lp(fixed);
      if (r.status == Status::Unbounded) {
        out.status = Status::Unbounded;
        return out;
      }
      if (r.status == Status::NumericalError) {
        inconclusive = true;
      } else if (r.status == Status::Optimal) {
        const double o = r.solution.objective;
        if (!have || better(m, o, best)) {
          have = true;
          best = o;
          best_x = r.solution.x;
        }
      }
    }
    // odometer
    std::size_t k = 0;
    while (k < ints.size()) {
      if (++v[k] <= hi[k]) break;
      v[k] = lo[k];
      ++k;
    }
    if (k == ints.size()) break;
  }
  if (inconclusive) {
    out.status = Status::NumericalError;
    return out;
  }
  out.status = have ? Status::Optimal : Status::Infeasible;
  if (have) {
    out.objective = best;
    out.x = best_x;
  }
  return out;
}

namespace {

struct DenseBb {
  const LpModel& orig;
  long long max_nodes;
  long long nodes = 0;
  bool have = false, unbounded = false, failed = false;
  double best = 0.0;
  std::vector<double> best_x;

  void dive(LpModel m) {
    if (++nodes > max_nodes) {
      failed = true;
      return;
    }
    const RefLpResult r = solve_dense_lp(m);
    if (r.status == Status::Infeasible) return;
    if (r.status == Status::Unbounded) {
      unbounded = true;
      return;
    }
    if (r.status != Status::Optimal) {
      failed = true;
      return;
    }
    if (have && !better(orig, r.solution.objective, best) ) {
      // The relaxation is no better than the incumbent: prune (minimization: objective >= best).
      if (orig.sense == Sense::Maximize ? r.solution.objective <= best + 1e-9 * (1 + std::fabs(best)) : r.solution.objective >= best - 1e-9 * (1 + std::fabs(best))) return;
    }
    Index frac = -1;
    double worst = 0.0;
    for (Index j = 0; j < m.n_cols; ++j) {
      if (!m.is_integer(j)) continue;
      const double v = r.solution.x[to_size(j)];
      const double f = std::fabs(v - std::round(v));
      if (f > 1e-6 && f > worst) {
        worst = f;
        frac = j;
      }
    }
    if (frac < 0) {
      if (!have || better(orig, r.solution.objective, best)) {
        have = true;
        best = r.solution.objective;
        best_x = r.solution.x;
        for (Index j = 0; j < m.n_cols; ++j) {
          if (m.is_integer(j)) best_x[to_size(j)] = std::round(best_x[to_size(j)]);
        }
      }
      return;
    }
    const double v = r.solution.x[to_size(frac)];
    LpModel down = m, up = m;
    down.col_upper[to_size(frac)] = std::floor(v);
    up.col_lower[to_size(frac)] = std::ceil(v);
    dive(std::move(down));
    dive(std::move(up));
  }
};

}  // namespace

MipRefResult solve_mip_dense_bb(const LpModel& model, long long max_nodes) {
  MipRefResult out;
  DenseBb bb{model, max_nodes, 0, false, false, false, 0.0, {}};
  bb.dive(model);
  out.evaluations = bb.nodes;
  if (bb.unbounded) {
    out.status = Status::Unbounded;
  } else if (bb.failed) {
    out.status = Status::NumericalError;
  } else if (bb.have) {
    out.status = Status::Optimal;
    out.objective = bb.best;
    out.x = bb.best_x;
  } else {
    out.status = Status::Infeasible;
  }
  return out;
}

}  // namespace shodhan::testing
