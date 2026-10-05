#include "support/pooling_families.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "shodhan/lp_solver.hpp"
#include "support/rng.hpp"

namespace shodhan::testing {

using namespace pooling;

namespace {

double eighth(Rng& r, double lo, double hi) { return std::round(r.uniform(lo, hi) * 8.0) / 8.0; }

struct Lp {
  LpModel m;
  std::vector<Triplet> t;
  Index col(double cost, double lo, double hi) {
    m.col_cost.push_back(cost);
    m.col_lower.push_back(lo);
    m.col_upper.push_back(hi);
    m.col_type.push_back(ColType::Continuous);
    return m.n_cols++;
  }
  Index row(double lo, double hi) {
    m.row_lower.push_back(lo);
    m.row_upper.push_back(hi);
    return m.n_rows++;
  }
  void coef(Index r, Index c, double v) {
    if (v != 0.0) t.push_back({r, c, v});
  }
};

}  // namespace

PoolProblem make_single_pool(std::uint64_t seed) {
  Rng r(seed * 2654435761ULL + 17ULL);
  PoolProblem p;
  p.name = "single_pool_" + std::to_string(seed);
  p.synthetic = true;
  p.n_qualities = 1;
  p.quality_names = {"S"};
  const int nS = r.range(2, 4), nT = r.range(1, 2);
  double qmin = 1e9, qmax = -1e9;
  for (int s = 0; s < nS; ++s) {
    Source src;
    src.name = "S" + std::to_string(s + 1);
    double q = eighth(r, 0.5, 6.0);
    if (s == 1 && std::fabs(q - p.sources[0].quality[0]) < 1.0) q = p.sources[0].quality[0] + (q >= p.sources[0].quality[0] ? 1.5 : -1.5);  // at least two distinct qualities
    src.quality = {q};
    src.cost = std::max(2.0, std::round(20.0 - 2.2 * q + r.uniform(-2.0, 2.0)));
    src.supply = r.chance(0.5) ? kInf : eighth(r, 20.0, 80.0);
    qmin = std::min(qmin, q);
    qmax = std::max(qmax, q);
    p.sources.push_back(src);
  }
  p.pools.push_back({"P", eighth(r, 40.0, 120.0)});
  for (int t = 0; t < nT; ++t) {
    Terminal term;
    term.name = "T" + std::to_string(t + 1);
    const double q = eighth(r, qmin + 0.1 * (qmax - qmin), qmax - 0.1 * (qmax - qmin));
    term.spec = {q};
    term.price = std::round(30.0 - 2.0 * q + r.uniform(-1.0, 3.0));
    term.demand = eighth(r, 20.0, 80.0);
    p.terminals.push_back(term);
  }
  for (int s = 0; s < nS; ++s) p.arcs_sp.push_back({s, 0});
  for (int t = 0; t < nT; ++t) p.arcs_pt.push_back({0, t});
  for (int s = 0; s < nS; ++s) {
    for (int t = 0; t < nT; ++t) {
      if (r.chance(0.35)) p.arcs_st.push_back({s, t});
    }
  }
  return p;
}

PoolProblem make_two_pool(std::uint64_t seed) {
  Rng r(seed * 11400714819323198485ULL + 3ULL);
  PoolProblem p;
  p.name = "two_pool_" + std::to_string(seed);
  p.synthetic = true;
  p.n_qualities = 2;
  p.quality_names = {"S", "N"};
  const int nS = r.range(3, 4);
  double qmin[2] = {1e9, 1e9}, qmax[2] = {-1e9, -1e9};
  for (int s = 0; s < nS; ++s) {
    Source src;
    src.name = "S" + std::to_string(s + 1);
    src.quality = {eighth(r, 0.5, 6.0), eighth(r, 0.5, 6.0)};
    for (int k = 0; k < 2; ++k) {
      qmin[k] = std::min(qmin[k], src.quality[static_cast<std::size_t>(k)]);
      qmax[k] = std::max(qmax[k], src.quality[static_cast<std::size_t>(k)]);
    }
    src.cost = std::max(2.0, std::round(22.0 - 1.4 * (src.quality[0] + src.quality[1]) + r.uniform(-2.0, 2.0)));
    src.supply = r.chance(0.4) ? kInf : eighth(r, 25.0, 90.0);
    p.sources.push_back(src);
  }
  p.pools.push_back({"P1", eighth(r, 40.0, 100.0)});
  p.pools.push_back({"P2", eighth(r, 40.0, 100.0)});
  for (int t = 0; t < 2; ++t) {
    Terminal term;
    term.name = "T" + std::to_string(t + 1);
    term.spec.resize(2);
    double mean = 0.0;
    for (int k = 0; k < 2; ++k) {
      term.spec[static_cast<std::size_t>(k)] = eighth(r, qmin[k] + 0.15 * (qmax[k] - qmin[k]), qmax[k] - 0.15 * (qmax[k] - qmin[k]));
      mean += term.spec[static_cast<std::size_t>(k)];
    }
    term.price = std::round(32.0 - 1.2 * mean + r.uniform(-1.0, 3.0));
    term.demand = eighth(r, 30.0, 90.0);
    p.terminals.push_back(term);
  }
  for (int s = 0; s < nS; ++s) {
    for (int pool = 0; pool < 2; ++pool) {
      if (pool == 0 || r.chance(0.7) || s == 0) p.arcs_sp.push_back({s, pool});
    }
  }
  for (int pool = 0; pool < 2; ++pool) {
    for (int t = 0; t < 2; ++t) p.arcs_pt.push_back({pool, t});
  }
  for (int s = 0; s < nS; ++s) {
    for (int t = 0; t < 2; ++t) {
      if (r.chance(0.25)) p.arcs_st.push_back({s, t});
    }
  }
  return p;
}

double fixed_q_enforced_lp(const PoolProblem& p, const std::vector<double>& q) {
  Lp b;
  std::vector<Index> sp, pt, st;
  for (const auto& a : p.arcs_sp) sp.push_back(b.col(-p.sources[to_size(a.first)].cost, 0.0, kInf));
  for (const auto& a : p.arcs_pt) pt.push_back(b.col(p.terminals[to_size(a.second)].price, 0.0, kInf));
  for (const auto& a : p.arcs_st) st.push_back(b.col(p.terminals[to_size(a.second)].price - p.sources[to_size(a.first)].cost, 0.0, kInf));
  for (std::size_t s = 0; s < p.sources.size(); ++s) {
    if (is_inf(p.sources[s].supply)) continue;
    const Index r = b.row(-kInf, p.sources[s].supply);
    for (std::size_t a = 0; a < sp.size(); ++a) if (to_size(p.arcs_sp[a].first) == s) b.coef(r, sp[a], 1.0);
    for (std::size_t a = 0; a < st.size(); ++a) if (to_size(p.arcs_st[a].first) == s) b.coef(r, st[a], 1.0);
  }
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    if (!is_inf(p.pools[pool].capacity)) {
      const Index r = b.row(-kInf, p.pools[pool].capacity);
      for (std::size_t a = 0; a < sp.size(); ++a) if (to_size(p.arcs_sp[a].second) == pool) b.coef(r, sp[a], 1.0);
    }
    const Index m = b.row(0.0, 0.0);  // material balance
    for (std::size_t a = 0; a < sp.size(); ++a) if (to_size(p.arcs_sp[a].second) == pool) b.coef(m, sp[a], 1.0);
    for (std::size_t a = 0; a < pt.size(); ++a) if (to_size(p.arcs_pt[a].first) == pool) b.coef(m, pt[a], -1.0);
    for (int k = 0; k < p.n_qualities; ++k) {  // quality balance with q fixed: sum_s q_sk f_sp - q_pk sum_t f_pt = 0
      const Index e = b.row(0.0, 0.0);
      for (std::size_t a = 0; a < sp.size(); ++a) {
        if (to_size(p.arcs_sp[a].second) == pool) b.coef(e, sp[a], p.sources[to_size(p.arcs_sp[a].first)].quality[to_size(k)]);
      }
      for (std::size_t a = 0; a < pt.size(); ++a) {
        if (to_size(p.arcs_pt[a].first) == pool) b.coef(e, pt[a], -q[pool * to_size(p.n_qualities) + to_size(k)]);
      }
    }
  }
  for (std::size_t t = 0; t < p.terminals.size(); ++t) {
    if (!is_inf(p.terminals[t].demand)) {
      const Index r = b.row(-kInf, p.terminals[t].demand);
      for (std::size_t a = 0; a < pt.size(); ++a) if (to_size(p.arcs_pt[a].second) == t) b.coef(r, pt[a], 1.0);
      for (std::size_t a = 0; a < st.size(); ++a) if (to_size(p.arcs_st[a].second) == t) b.coef(r, st[a], 1.0);
    }
    for (int k = 0; k < p.n_qualities; ++k) {
      const double spec = p.terminals[t].spec[to_size(k)];
      if (is_inf(spec)) continue;
      const Index r = b.row(-kInf, 0.0);
      for (std::size_t a = 0; a < pt.size(); ++a) {
        if (to_size(p.arcs_pt[a].second) == t) b.coef(r, pt[a], q[to_size(p.arcs_pt[a].first) * to_size(p.n_qualities) + to_size(k)] - spec);
      }
      for (std::size_t a = 0; a < st.size(); ++a) {
        if (to_size(p.arcs_st[a].second) == t) b.coef(r, st[a], p.sources[to_size(p.arcs_st[a].first)].quality[to_size(k)] - spec);
      }
    }
  }
  std::string err;
  SparseMatrix::from_triplets(b.m.n_rows, b.m.n_cols, b.t, &b.m.A, &err);
  b.m.sense = Sense::Maximize;
  const LpResult res = LpSolver().solve(b.m);
  if (res.status != Status::Optimal) return -kInf;
  return res.solution.objective;
}

double single_pool_reference(const PoolProblem& p, int grid_points, int refine_steps) {
  const auto range = pool_quality_range(p, 0, 0);
  const double lo = range.first, hi = range.second;
  double best = -kInf, best_q = lo;
  std::vector<double> vals(static_cast<std::size_t>(grid_points));
  for (int i = 0; i < grid_points; ++i) {
    const double q = lo + (hi - lo) * i / (grid_points - 1);
    vals[static_cast<std::size_t>(i)] = fixed_q_enforced_lp(p, {q});
    if (vals[static_cast<std::size_t>(i)] > best) {
      best = vals[static_cast<std::size_t>(i)];
      best_q = q;
    }
  }
  // Ternary search on the neighbouring cells of the best grid point (a fine cell of a piecewise smooth function).
  const double h = (hi - lo) / (grid_points - 1);
  double a = std::max(lo, best_q - h), b = std::min(hi, best_q + h);
  for (int it = 0; it < refine_steps; ++it) {
    const double m1 = a + (b - a) / 3.0, m2 = b - (b - a) / 3.0;
    const double v1 = fixed_q_enforced_lp(p, {m1}), v2 = fixed_q_enforced_lp(p, {m2});
    best = std::max({best, v1, v2});
    if (v1 < v2) a = m1;
    else b = m2;
  }
  return best;
}

namespace {

void grid_rec(const PoolProblem& p, int points, std::size_t idx, std::vector<double>& q, double* best) {
  if (idx == p.n_q()) {
    *best = std::max(*best, fixed_q_enforced_lp(p, q));
    return;
  }
  const std::size_t pool = idx / to_size(p.n_qualities);
  const int k = static_cast<int>(idx % to_size(p.n_qualities));
  const auto range = pool_quality_range(p, pool, k);
  for (int i = 0; i < points; ++i) {
    q[idx] = range.first + (range.second - range.first) * i / (points - 1);
    grid_rec(p, points, idx + 1, q, best);
  }
}

}  // namespace

double grid_reference(const PoolProblem& p, int points_per_axis) {
  std::vector<double> q(p.n_q(), 0.0);
  double best = -kInf;
  grid_rec(p, points_per_axis, 0, q, &best);
  return best;
}

}  // namespace shodhan::testing
