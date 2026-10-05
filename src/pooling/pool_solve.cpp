#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>

#include "shodhan/lp_solver.hpp"
#include "shodhan/pooling.hpp"

namespace shodhan::pooling {

namespace {

using Clock = std::chrono::steady_clock;

struct Builder {
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
  LpModel finish() {
    std::string err;
    SparseMatrix::from_triplets(m.n_rows, m.n_cols, t, &m.A, &err);
    m.sense = Sense::Maximize;
    m.name = "pooling";
    return std::move(m);
  }
};

double flow_cost(const PoolProblem& p, int kind, std::size_t a) {
  if (kind == 0) return -p.sources[to_size(p.arcs_sp[a].first)].cost;
  if (kind == 1) return p.terminals[to_size(p.arcs_pt[a].second)].price;
  return p.terminals[to_size(p.arcs_st[a].second)].price - p.sources[to_size(p.arcs_st[a].first)].cost;
}

// Flow columns in the order of PoolProblem::idx_*, then the rows that do not involve any pool quality:
// supply, pool capacity, terminal demand, pool material balance.
void add_flow_part(const PoolProblem& p, Builder& b) {
  for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) b.col(flow_cost(p, 0, a), 0.0, kInf);
  for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) b.col(flow_cost(p, 1, a), 0.0, kInf);
  for (std::size_t a = 0; a < p.arcs_st.size(); ++a) b.col(flow_cost(p, 2, a), 0.0, kInf);
  for (std::size_t s = 0; s < p.sources.size(); ++s) {
    if (is_inf(p.sources[s].supply)) continue;
    const Index r = b.row(-kInf, p.sources[s].supply);
    for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) if (to_size(p.arcs_sp[a].first) == s) b.coef(r, static_cast<Index>(p.idx_sp(a)), 1.0);
    for (std::size_t a = 0; a < p.arcs_st.size(); ++a) if (to_size(p.arcs_st[a].first) == s) b.coef(r, static_cast<Index>(p.idx_st(a)), 1.0);
  }
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    if (!is_inf(p.pools[pool].capacity)) {
      const Index r = b.row(-kInf, p.pools[pool].capacity);
      for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) if (to_size(p.arcs_sp[a].second) == pool) b.coef(r, static_cast<Index>(p.idx_sp(a)), 1.0);
    }
    const Index r = b.row(0.0, 0.0);
    for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) if (to_size(p.arcs_sp[a].second) == pool) b.coef(r, static_cast<Index>(p.idx_sp(a)), 1.0);
    for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) if (to_size(p.arcs_pt[a].first) == pool) b.coef(r, static_cast<Index>(p.idx_pt(a)), -1.0);
  }
  for (std::size_t t = 0; t < p.terminals.size(); ++t) {
    if (is_inf(p.terminals[t].demand)) continue;
    const Index r = b.row(-kInf, p.terminals[t].demand);
    for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) if (to_size(p.arcs_pt[a].second) == t) b.coef(r, static_cast<Index>(p.idx_pt(a)), 1.0);
    for (std::size_t a = 0; a < p.arcs_st.size(); ++a) if (to_size(p.arcs_st[a].second) == t) b.coef(r, static_cast<Index>(p.idx_st(a)), 1.0);
  }
}

bool solve_lp(const LpModel& m, std::vector<double>* x, double* objective, std::string* why) {
  const LpResult r = LpSolver().solve(m);
  if (r.status != Status::Optimal) {
    if (why != nullptr) *why = std::string("LP status ") + to_string(r.status);
    return false;
  }
  *x = r.solution.x;
  *objective = r.solution.objective;
  return true;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

// Largest range of a pool quality, used to scale quality tolerances.
double quality_scale(const PoolProblem& p) {
  double s = 1.0;
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const auto r = pool_quality_range(p, pool, k);
      s = std::max({s, std::fabs(r.first), std::fabs(r.second)});
    }
  }
  return s;
}

// Our own generator (splitmix64), so that starts do not depend on a library's random numbers.
struct SplitMix {
  unsigned long long s;
  unsigned long long next() {
    unsigned long long z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double unit() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }
};

}  // namespace

bool solve_fixed_quality_lp(const PoolProblem& p, const std::vector<double>& q, std::vector<double>* flow, double* objective) {
  Builder b;
  add_flow_part(p, b);
  for (std::size_t t = 0; t < p.terminals.size(); ++t) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const double spec = p.terminals[t].spec[to_size(k)];
      if (is_inf(spec)) continue;
      const Index r = b.row(-kInf, 0.0);
      for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
        if (to_size(p.arcs_pt[a].second) == t) b.coef(r, static_cast<Index>(p.idx_pt(a)), q[p.qidx(to_size(p.arcs_pt[a].first), k)] - spec);
      }
      for (std::size_t a = 0; a < p.arcs_st.size(); ++a) {
        if (to_size(p.arcs_st[a].second) == t) b.coef(r, static_cast<Index>(p.idx_st(a)), p.sources[to_size(p.arcs_st[a].first)].quality[to_size(k)] - spec);
      }
    }
  }
  const LpModel m = b.finish();
  std::vector<double> x;
  if (!solve_lp(m, &x, objective, nullptr)) return false;
  flow->assign(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(p.n_flows()));
  for (double& v : *flow) v = std::max(0.0, v);
  *objective = pool_objective(p, *flow);
  return true;
}

PoolRun solve_recursion(const PoolProblem& p, const std::vector<double>& q0, const PoolOptions& opt) {
  PoolRun run;
  std::vector<double> q = q0;
  const double scale = quality_scale(p);
  const double conv = 0.01 * opt.tol * scale;
  std::vector<std::vector<double>> history{q};
  PoolRun incumbent;
  incumbent.status = PoolStatus::NoSolution;
  bool have_incumbent = false;
  for (int it = 1; it <= opt.max_iterations; ++it) {
    std::vector<double> f;
    double obj = 0.0;
    if (!solve_fixed_quality_lp(p, q, &f, &obj)) {
      run.status = PoolStatus::NumericalError;
      run.message = "the fixed-quality LP could not be solved";
      return run;
    }
    run.iterations = it;
    const std::vector<double> qi = implied_pool_qualities(p, f, q);
    const double d = max_abs_diff(qi, q);
    // The point with the quality implied by the flows satisfies the pool balances exactly; it is feasible when the
    // terminal specifications hold with it.
    PoolPoint pt{f, qi};
    const PoolCheck chk = check_pool_point(p, pt);
    if (chk.ok(opt.tol) && (!have_incumbent || pool_objective(p, f) > incumbent.objective)) {
      incumbent.point = pt;
      incumbent.objective = pool_objective(p, f);
      incumbent.violation = chk.worst;
      have_incumbent = true;
    }
    if (opt.log != nullptr) *opt.log << "  recursion " << it << ": objective " << obj << ", max change of pool quality " << d << ", residual " << chk.worst << "\n";
    if (d <= conv) {
      if (chk.ok(opt.tol)) {
        run.status = PoolStatus::Converged;
        run.point = pt;
        run.objective = pool_objective(p, f);
        run.violation = chk.worst;
        return run;
      }
      run.status = PoolStatus::NoSolution;
      run.message = "the recursion reached a fixed point that violates the nonlinear model by " + std::to_string(chk.worst);
      return run;
    }
    std::vector<double> next(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) next[i] = q[i] + opt.damping * (qi[i] - q[i]);
    // Revisiting an earlier iterate (other than the last one) is a cycle: the recursion will not converge.
    for (std::size_t h = 0; h + 1 < history.size(); ++h) {
      if (max_abs_diff(next, history[h]) <= 1e-9 * scale) {
        run.status = PoolStatus::Cycling;
        run.message = "the recursion revisits the pool qualities of iteration " + std::to_string(h) + " (period " + std::to_string(history.size() - h) + ")";
        run.iterations = it;
        if (have_incumbent) {
          run.point = incumbent.point;
          run.objective = incumbent.objective;
          run.violation = incumbent.violation;
        }
        return run;
      }
    }
    history.push_back(next);
    q = next;
  }
  run.status = PoolStatus::IterationLimit;
  run.message = "no convergence in " + std::to_string(opt.max_iterations) + " iterations";
  if (have_incumbent) {
    run.point = incumbent.point;
    run.objective = incumbent.objective;
    run.violation = incumbent.violation;
  }
  return run;
}

namespace {

// Residuals of the nonlinear model, unscaled, for the merit function of the SLP.
void nonlinear_residuals(const PoolProblem& p, const PoolPoint& pt, double* balance_abs, double* spec_abs) {
  *balance_abs = 0.0;
  *spec_abs = 0.0;
  const std::vector<double>& f = pt.flow;
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    double out = 0.0;
    for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) if (to_size(p.arcs_pt[a].first) == pool) out += f[p.idx_pt(a)];
    for (int k = 0; k < p.n_qualities; ++k) {
      double lhs = 0.0;
      for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) {
        if (to_size(p.arcs_sp[a].second) == pool) lhs += p.sources[to_size(p.arcs_sp[a].first)].quality[to_size(k)] * f[p.idx_sp(a)];
      }
      *balance_abs += std::fabs(lhs - pt.q[p.qidx(pool, k)] * out);
    }
  }
  for (std::size_t t = 0; t < p.terminals.size(); ++t) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const double spec = p.terminals[t].spec[to_size(k)];
      if (is_inf(spec)) continue;
      double lhs = 0.0, in = 0.0;
      for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
        if (to_size(p.arcs_pt[a].second) != t) continue;
        lhs += pt.q[p.qidx(to_size(p.arcs_pt[a].first), k)] * f[p.idx_pt(a)];
        in += f[p.idx_pt(a)];
      }
      for (std::size_t a = 0; a < p.arcs_st.size(); ++a) {
        if (to_size(p.arcs_st[a].second) != t) continue;
        lhs += p.sources[to_size(p.arcs_st[a].first)].quality[to_size(k)] * f[p.idx_st(a)];
        in += f[p.idx_st(a)];
      }
      *spec_abs += std::max(0.0, lhs - spec * in);
    }
  }
}

}  // namespace

PoolRun solve_slp(const PoolProblem& p, const std::vector<double>& q0, const PoolOptions& opt) {
  PoolRun run;
  const std::size_t nF = p.n_flows(), nQ = p.n_q();
  const double scale = quality_scale(p);
  double max_price = 1.0;
  for (const Terminal& t : p.terminals) max_price = std::max(max_price, std::fabs(t.price));
  for (const Source& s : p.sources) max_price = std::max(max_price, std::fabs(s.cost));
  double mu = opt.penalty > 0.0 ? opt.penalty : 10.0 * (max_price + 1.0) / scale;
  int mu_updates = 0;
  // trust region: absolute radius per pool quality
  std::vector<double> range(nQ), qlo(nQ), qhi(nQ), radius(nQ);
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const auto r = pool_quality_range(p, pool, k);
      const std::size_t i = p.qidx(pool, k);
      qlo[i] = r.first;
      qhi[i] = r.second;
      range[i] = std::max(r.second - r.first, 1e-9);
      radius[i] = opt.trust_radius * range[i];
    }
  }
  PoolPoint cur;
  cur.q = q0;
  double obj0 = 0.0;
  if (!solve_fixed_quality_lp(p, q0, &cur.flow, &obj0)) {
    run.status = PoolStatus::NumericalError;
    run.message = "the fixed-quality LP at the start could not be solved";
    return run;
  }
  auto merit = [&](const PoolPoint& pt) {
    double bal = 0.0, spec = 0.0;
    nonlinear_residuals(p, pt, &bal, &spec);
    return pool_objective(p, pt.flow) - mu * (bal + spec);
  };
  double phi = merit(cur);
  // Best verified point. A point that passes the tolerance only barely can owe part of its objective to the slack of the
  // tolerance, so points that pass a tolerance 1000 times tighter are preferred over those that pass only `tol`.
  PoolPoint best, best_loose;
  double best_obj = -kInf, loose_obj = -kInf;
  bool have_best = false, have_loose = false;
  auto consider = [&](const PoolPoint& pt) {
    // The point with the pool qualities implied by the flows satisfies the pool balances exactly.
    PoolPoint c{pt.flow, implied_pool_qualities(p, pt.flow, pt.q)};
    const PoolCheck chk = check_pool_point(p, c);
    const double o = pool_objective(p, c.flow);
    if (chk.ok(1e-3 * opt.tol) && o > best_obj) {
      best = c;
      best_obj = o;
      have_best = true;
    }
    if (chk.ok(opt.tol) && o > loose_obj) {
      best_loose = c;
      loose_obj = o;
      have_loose = true;
    }
  };
  // Whether the iteration may stop: if the linearization residuals do not vanish at the stationary point, the penalty was
  // too small (an exact penalty needs a weight above the multiplier): raise it and go on.
  auto may_stop = [&]() {
    double bal = 0.0, spec = 0.0;
    nonlinear_residuals(p, cur, &bal, &spec);
    if (bal + spec > 1e-8 * (1.0 + scale) && mu_updates < 6) {
      mu *= 10.0;
      ++mu_updates;
      phi = merit(cur);
      for (std::size_t i = 0; i < nQ; ++i) radius[i] = std::max(radius[i], 0.05 * range[i]);
      return false;
    }
    return true;
  };
  consider(cur);
  bool converged = false;
  int it = 0;
  for (it = 1; it <= opt.max_iterations; ++it) {
    // The linearized problem around (cur.flow, cur.q) with elastic variables for the linearization residuals.
    Builder b;
    add_flow_part(p, b);
    std::vector<Index> qcol(nQ);
    for (std::size_t i = 0; i < nQ; ++i) qcol[i] = b.col(0.0, std::max(qlo[i], cur.q[i] - radius[i]), std::min(qhi[i], cur.q[i] + radius[i]));
    std::vector<double> T0(p.pools.size(), 0.0);
    for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) T0[to_size(p.arcs_pt[a].first)] += cur.flow[p.idx_pt(a)];
    for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
      for (int k = 0; k < p.n_qualities; ++k) {
        const std::size_t i = p.qidx(pool, k);
        // sum_s q_sk f_sp - [q0 T + T0 q - q0 T0] + e+ - e- = 0
        const double q0v = cur.q[i];
        const Index r = b.row(-q0v * T0[pool], -q0v * T0[pool]);
        for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) {
          if (to_size(p.arcs_sp[a].second) == pool) b.coef(r, static_cast<Index>(p.idx_sp(a)), p.sources[to_size(p.arcs_sp[a].first)].quality[to_size(k)]);
        }
        for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
          if (to_size(p.arcs_pt[a].first) == pool) b.coef(r, static_cast<Index>(p.idx_pt(a)), -q0v);
        }
        b.coef(r, qcol[i], -T0[pool]);
        const Index ep = b.col(-mu, 0.0, kInf), em = b.col(-mu, 0.0, kInf);
        b.coef(r, ep, 1.0);
        b.coef(r, em, -1.0);
      }
    }
    for (std::size_t t = 0; t < p.terminals.size(); ++t) {
      for (int k = 0; k < p.n_qualities; ++k) {
        const double spec = p.terminals[t].spec[to_size(k)];
        if (is_inf(spec)) continue;
        // sum_p [q0 f + f0 q - q0 f0] + sum_s q_sk f_st - spec * inflow - slack <= 0
        double const_part = 0.0;
        const Index r = b.row(-kInf, 0.0);
        for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
          if (to_size(p.arcs_pt[a].second) != t) continue;
          const std::size_t pool = to_size(p.arcs_pt[a].first);
          const double q0v = cur.q[p.qidx(pool, k)], f0 = cur.flow[p.idx_pt(a)];
          b.coef(r, static_cast<Index>(p.idx_pt(a)), q0v - spec);
          b.coef(r, qcol[p.qidx(pool, k)], f0);
          const_part -= q0v * f0;
        }
        for (std::size_t a = 0; a < p.arcs_st.size(); ++a) {
          if (to_size(p.arcs_st[a].second) == t) b.coef(r, static_cast<Index>(p.idx_st(a)), p.sources[to_size(p.arcs_st[a].first)].quality[to_size(k)] - spec);
        }
        b.m.row_upper[to_size(r)] = -const_part;
        const Index sl = b.col(-mu, 0.0, kInf);
        b.coef(r, sl, -1.0);
      }
    }
    const LpModel m = b.finish();
    std::vector<double> x;
    double lp_obj = 0.0;
    std::string why;
    if (!solve_lp(m, &x, &lp_obj, &why)) {
      run.status = PoolStatus::NumericalError;
      run.message = "the linearized LP failed: " + why;
      run.iterations = it;
      if (!have_best && have_loose) {
        best = best_loose;
        best_obj = loose_obj;
        have_best = true;
      }
      if (have_best) {
        run.point = best;
        run.objective = best_obj;
        run.violation = check_pool_point(p, best).worst;
      }
      return run;
    }
    PoolPoint cand;
    cand.flow.assign(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(nF));
    for (double& v : cand.flow) v = std::max(0.0, v);
    cand.q.resize(nQ);
    bool at_boundary = false;
    for (std::size_t i = 0; i < nQ; ++i) {
      cand.q[i] = x[to_size(qcol[i])];
      const double lo = std::max(qlo[i], cur.q[i] - radius[i]), hi = std::min(qhi[i], cur.q[i] + radius[i]);
      if (cand.q[i] <= lo + 1e-9 * range[i] && lo > qlo[i] + 1e-12) at_boundary = true;
      if (cand.q[i] >= hi - 1e-9 * range[i] && hi < qhi[i] - 1e-12) at_boundary = true;
    }
    const double phi_new = merit(cand);
    const double predicted = lp_obj - phi;
    const double actual = phi_new - phi;
    const double step = std::max(max_abs_diff(cand.q, cur.q), max_abs_diff(cand.flow, cur.flow));
    if (opt.log != nullptr) {
      *opt.log << "  slp " << it << ": merit " << phi << " -> " << phi_new << " (predicted gain " << predicted << "), step " << step << ", radius " << radius[0] << "\n";
    }
    if (actual > 1e-12 * (1.0 + std::fabs(phi))) {
      cur = cand;
      phi = phi_new;
      consider(cur);
      const double rho = predicted > 0.0 ? actual / predicted : 1.0;
      if (rho > 0.75 && at_boundary) {
        for (std::size_t i = 0; i < nQ; ++i) radius[i] = std::min(2.0 * radius[i], range[i]);
      } else if (rho < 0.25) {
        for (std::size_t i = 0; i < nQ; ++i) radius[i] *= 0.5;
      }
      if ((predicted <= 1e-10 * (1.0 + std::fabs(phi)) || step <= 1e-10 * scale) && may_stop()) {
        converged = true;
        break;
      }
    } else {
      for (std::size_t i = 0; i < nQ; ++i) radius[i] *= 0.5;
      double rmax = 0.0;
      for (std::size_t i = 0; i < nQ; ++i) rmax = std::max(rmax, radius[i] / range[i]);
      if ((rmax < 1e-10 || predicted <= 1e-10 * (1.0 + std::fabs(phi))) && may_stop()) {
        converged = true;
        break;
      }
    }
  }
  run.iterations = it > opt.max_iterations ? opt.max_iterations : it;
  // Final step: the pool qualities implied by the flows, then a verification; if the terminal specifications are slightly
  // violated, one fixed-quality LP at those qualities restores them.
  consider(cur);
  if (!have_best && !have_loose) {
    std::vector<double> qi = implied_pool_qualities(p, cur.flow, cur.q);
    std::vector<double> f2;
    double o2 = 0.0;
    if (solve_fixed_quality_lp(p, qi, &f2, &o2)) consider(PoolPoint{f2, qi});
  }
  if (!have_best && !have_loose) {
    // Last resort: polish with the recursion from the qualities that the flows imply.
    PoolOptions ro = opt;
    ro.max_iterations = 60;
    ro.log = nullptr;
    const PoolRun pr = solve_recursion(p, implied_pool_qualities(p, cur.flow, cur.q), ro);
    if (!pr.point.flow.empty()) consider(pr.point);
  }
  if (!have_best && have_loose) {
    best = best_loose;
    best_obj = loose_obj;
    have_best = true;
  }
  if (have_best) {
    run.point = best;
    run.objective = best_obj;
    run.violation = check_pool_point(p, best).worst;
    run.status = converged ? PoolStatus::Converged : PoolStatus::IterationLimit;
    if (!converged) run.message = "no convergence in " + std::to_string(opt.max_iterations) + " iterations; the best verified point is returned";
  } else {
    run.status = converged ? PoolStatus::NoSolution : PoolStatus::IterationLimit;
    run.message = "no point that satisfies the nonlinear model within the tolerance was found";
  }
  return run;
}

bool mccormick_upper_bound(const PoolProblem& p, double* bound, std::string* message) {
  Builder b;
  add_flow_part(p, b);
  const std::size_t nQ = p.n_q();
  std::vector<Index> qcol(nQ);
  std::vector<double> qlo(nQ), qhi(nQ);
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const auto r = pool_quality_range(p, pool, k);
      const std::size_t i = p.qidx(pool, k);
      qlo[i] = r.first;
      qhi[i] = r.second;
      qcol[i] = b.col(0.0, r.first, r.second);
    }
  }
  // Upper bound of the flow on a pool -> terminal arc: demand, pool capacity, supply that can reach the pool.
  std::vector<double> fU(p.arcs_pt.size());
  for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
    const std::size_t pool = to_size(p.arcs_pt[a].first), t = to_size(p.arcs_pt[a].second);
    double sup = 0.0;
    for (const auto& e : p.arcs_sp) if (to_size(e.second) == pool) sup += p.sources[to_size(e.first)].supply;
    fU[a] = std::min({p.terminals[t].demand, p.pools[pool].capacity, sup});
    if (is_inf(fU[a])) {
      if (message != nullptr) *message = "the flow on arc " + p.arc_name_pt(a) + " has no finite upper bound (demand, pool capacity and supplies are all infinite): no McCormick bound";
      return false;
    }
  }
  std::vector<std::vector<Index>> v(p.arcs_pt.size(), std::vector<Index>(to_size(p.n_qualities), 0));
  for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
    const std::size_t pool = to_size(p.arcs_pt[a].first);
    for (int k = 0; k < p.n_qualities; ++k) {
      const std::size_t i = p.qidx(pool, k);
      const Index vc = b.col(0.0, -kInf, kInf);
      v[a][to_size(k)] = vc;
      const Index f = static_cast<Index>(p.idx_pt(a));
      const double ql = qlo[i], qu = qhi[i], fu = fU[a];
      // v >= ql f ; v >= qu f + fu q - qu fu ; v <= qu f ; v <= ql f + fu q - ql fu   (flow lower bound 0)
      Index r = b.row(0.0, kInf);
      b.coef(r, vc, 1.0);
      b.coef(r, f, -ql);
      r = b.row(-qu * fu, kInf);
      b.coef(r, vc, 1.0);
      b.coef(r, f, -qu);
      b.coef(r, qcol[i], -fu);
      r = b.row(-kInf, 0.0);
      b.coef(r, vc, 1.0);
      b.coef(r, f, -qu);
      r = b.row(-kInf, -ql * fu);
      b.coef(r, vc, 1.0);
      b.coef(r, f, -ql);
      b.coef(r, qcol[i], -fu);
    }
  }
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const Index r = b.row(0.0, 0.0);  // sum_s q_sk f_sp = sum_t v_pkt
      for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) {
        if (to_size(p.arcs_sp[a].second) == pool) b.coef(r, static_cast<Index>(p.idx_sp(a)), p.sources[to_size(p.arcs_sp[a].first)].quality[to_size(k)]);
      }
      for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
        if (to_size(p.arcs_pt[a].first) == pool) b.coef(r, v[a][to_size(k)], -1.0);
      }
    }
  }
  for (std::size_t t = 0; t < p.terminals.size(); ++t) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const double spec = p.terminals[t].spec[to_size(k)];
      if (is_inf(spec)) continue;
      const Index r = b.row(-kInf, 0.0);
      for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
        if (to_size(p.arcs_pt[a].second) != t) continue;
        b.coef(r, v[a][to_size(k)], 1.0);
        b.coef(r, static_cast<Index>(p.idx_pt(a)), -spec);
      }
      for (std::size_t a = 0; a < p.arcs_st.size(); ++a) {
        if (to_size(p.arcs_st[a].second) == t) b.coef(r, static_cast<Index>(p.idx_st(a)), p.sources[to_size(p.arcs_st[a].first)].quality[to_size(k)] - spec);
      }
    }
  }
  const LpModel m = b.finish();
  std::vector<double> x;
  std::string why;
  if (!solve_lp(m, &x, bound, &why)) {
    if (message != nullptr) *message = "the McCormick LP was not solved to an accepted optimum (" + why + ")";
    return false;
  }
  return true;
}

PoolResult solve_pool(const PoolProblem& p, const PoolOptions& opt) {
  PoolResult res;
  const Clock::time_point t0 = Clock::now();
  const std::size_t nQ = p.n_q();
  std::vector<double> qlo(nQ), qhi(nQ);
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const auto r = pool_quality_range(p, pool, k);
      qlo[p.qidx(pool, k)] = r.first;
      qhi[p.qidx(pool, k)] = r.second;
    }
  }
  // Start 0: each pool quality at the strictest specification of the terminals the pool feeds (clipped to the range of its
  // sources), so that every terminal can take material from the pool; start 1 is the midpoint of the ranges; the others are
  // seeded random. (A pool quality above a specification makes the fixed-quality LP leave the pool idle, and an idle pool
  // never changes its quality.)
  std::vector<double> strict_start(nQ);
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const std::size_t i = p.qidx(pool, k);
      double r = kInf;
      for (const auto& arc : p.arcs_pt) {
        if (to_size(arc.first) == pool) r = std::min(r, p.terminals[to_size(arc.second)].spec[to_size(k)]);
      }
      strict_start[i] = is_inf(r) ? 0.5 * (qlo[i] + qhi[i]) : std::min(std::max(r, qlo[i]), qhi[i]);
    }
  }
  SplitMix rng{opt.seed * 0x9e3779b97f4a7c15ULL + 12345ULL};
  std::map<long long, std::pair<double, int>> outcomes;  // rounded objective -> (objective, count)
  bool have = false;
  double best_obj = -kInf;
  for (int s = 0; s < std::max(1, opt.starts); ++s) {
    if (std::chrono::duration<double>(Clock::now() - t0).count() > opt.time_limit) break;
    std::vector<double> q0(nQ);
    for (std::size_t i = 0; i < nQ; ++i) q0[i] = s == 1 ? 0.5 * (qlo[i] + qhi[i]) : qlo[i] + (qhi[i] - qlo[i]) * rng.unit();
    if (s == 0) q0 = strict_start;
    const PoolRun run = opt.method == PoolMethod::Slp ? solve_slp(p, q0, opt) : solve_recursion(p, q0, opt);
    ++res.starts_run;
    res.iterations += run.iterations;
    switch (run.status) {
      case PoolStatus::Converged: ++res.starts_converged; break;
      case PoolStatus::Cycling: ++res.starts_cycled; break;
      default: ++res.starts_failed; break;
    }
    if (opt.log != nullptr) *opt.log << "start " << s << ": " << to_string(run.status) << ", objective " << run.objective << ", " << run.iterations << " iterations\n";
    // A point is accepted only if it passes the check against the original nonlinear model.
    if (run.point.flow.empty()) continue;
    const PoolCheck chk = check_pool_point(p, run.point);
    if (!chk.ok(opt.tol)) continue;
    const double obj = pool_objective(p, run.point.flow);
    if (run.status == PoolStatus::Converged) {
      const long long key = static_cast<long long>(std::llround(obj / (1e-6 * (1.0 + std::fabs(obj)))));
      auto& e = outcomes[key];
      e.first = obj;
      ++e.second;
    }
    if (!have || obj > best_obj) {
      have = true;
      best_obj = obj;
      res.point = run.point;
      res.check = chk;
      res.objective = obj;
    }
  }
  for (const auto& kv : outcomes) res.outcomes.push_back(kv.second);
  std::sort(res.outcomes.begin(), res.outcomes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  const bool all_cycled = res.starts_run > 0 && res.starts_cycled == res.starts_run;
  if (have) {
    res.status = res.starts_converged > 0 ? PoolStatus::Converged : all_cycled ? PoolStatus::Cycling : PoolStatus::IterationLimit;
    if (res.starts_converged == 0) {
      res.message = std::string(all_cycled ? "every start cycled" : "no start converged") +
                    "; the best point met along the way, which satisfies the nonlinear model within the tolerance, is returned (a feasible point, not a converged solution)";
    }
  } else {
    res.status = all_cycled ? PoolStatus::Cycling : PoolStatus::NoSolution;
    res.message = all_cycled ? "every start cycled; no point that satisfies the nonlinear model was met, none is returned" : "no start produced a point that satisfies the nonlinear model";
  }
  if (opt.mccormick) {
    std::string why;
    double bound = 0.0;
    if (mccormick_upper_bound(p, &bound, &why)) {
      res.has_bound = true;
      res.mccormick_bound = bound;
      if (have) res.gap = (bound - res.objective) / std::max(1.0, std::fabs(bound));
    } else if (!res.message.empty()) {
      res.message += "; " + why;
    } else {
      res.message = why;
    }
  }
  return res;
}

}  // namespace shodhan::pooling
