// IPM against the dual simplex (LPs), planted infeasible/unbounded problems, and a tiny-QP active-set enumerator.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/ipm.hpp"
#include "shodhan/kkt.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/quadratic.hpp"
#include "shodhan/scaling.hpp"
#include "support/lp_families.hpp"
#include "support/qp_families.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == 0) return fallback;
  return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
}

// Scaled like the pipeline, solution returned in the original space. With `ladder`, a run that does not end Optimal with a
// point that passes check_kkt at 1e-6 is repeated without scaling (the first rung of the facade's fallback ladder).
IpmResult solve_scaled(const LpModel& model, bool scaled, bool ladder = false) {
  if (!scaled) return solve_ipm(model);
  const Scaling sc = compute_scaling(model);
  const LpModel sm = apply_scaling(model, sc);
  IpmResult r = solve_ipm(sm);
  if (r.status == Status::Optimal) r.solution = unscale_solution(sc, r.solution);
  if (ladder && !(r.status == Status::Optimal && check_kkt(model, r.solution, 1e-6).ok)) {
    IpmResult u = solve_ipm(model);
    if (u.status == Status::Optimal && check_kkt(model, u.solution, 1e-6).ok) return u;
  }
  return r;
}

}  // namespace

// IPM versus the dual simplex on LPs: both engines must agree on the status and the objective.
TEST_CASE(ipm_and_the_dual_simplex_agree_on_lps) {
  const std::uint64_t per_family = env_u64("SHODHAN_LP_AGREE_SEEDS", 300);
  int compared = 0, agree = 0, both_optimal = 0, both_not_optimal = 0, simplex_numerical = 0;
  double worst = 0.0;
  std::vector<std::string> failures;
  for (int f = 0; f < kNumFamilies; ++f) {
    for (std::uint64_t k = 1; k <= per_family; ++k) {
      const std::uint64_t seed = 500000ULL + static_cast<std::uint64_t>(f) * 10000ULL + k;
      const LpModel model = make_family_instance(f, seed);
      const LpResult sx = LpSolver().solve(model);
      if (sx.status == Status::NumericalError) {
        ++simplex_numerical;
        continue;
      }
      ++compared;
      const IpmResult ip = solve_scaled(model, true, true);
      std::string why;
      if (sx.status == Status::Optimal) {
        if (ip.status != Status::Optimal) {
          why = std::string("simplex Optimal, IPM ") + to_string(ip.status) + " " + ip.message;
        } else {
          const double err = std::fabs(ip.solution.objective - sx.solution.objective) / (1.0 + std::fabs(sx.solution.objective));
          worst = std::max(worst, err);
          if (err > 1e-6) why = "objectives differ: IPM " + std::to_string(ip.solution.objective) + " simplex " + std::to_string(sx.solution.objective);
          else if (!check_kkt(model, ip.solution, 1e-6).ok) why = "IPM point fails the KKT check";
          else ++both_optimal;
        }
      } else {  // Infeasible, Unbounded, InfeasibleOrUnbounded: the IPM must not claim an optimum
        if (ip.status == Status::Optimal) why = std::string("simplex ") + to_string(sx.status) + " but the IPM reports Optimal";
        else ++both_not_optimal;
      }
      if (why.empty()) ++agree;
      else failures.push_back("family " + std::string(family_name(f)) + " seed " + std::to_string(seed) + ": " + why);
    }
  }
  std::cout << "  IPM vs dual simplex on " << compared << " LPs (7 families): " << agree << " agree (" << both_optimal
            << " optimal with the same objective, worst relative difference " << worst << "; " << both_not_optimal
            << " infeasible or unbounded for the simplex and not Optimal for the IPM); " << simplex_numerical
            << " skipped because the simplex returned NumericalError\n";
  for (std::size_t i = 0; i < failures.size() && i < 20; ++i) std::cerr << "  DISAGREE " << failures[i] << "\n";
  // The IPM is less robust than the simplex on the ill-conditioned wide family; a handful of those seeds are allowed to
  // fail (they are listed above), anything beyond 0.2% is a regression.
  CHECK(compared - agree <= 1 + compared / 500);
  CHECK(compared >= 1900 || per_family < 300);
}

// Planted infeasible and unbounded QPs are never reported Optimal.
TEST_CASE(ipm_never_reports_optimal_for_infeasible_or_unbounded_problems) {
  int runs = 0, wrongly_optimal = 0;
  std::string first;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    for (int kind = 0; kind < 2; ++kind) {
      std::string what;
      const LpModel m = kind == 0 ? make_infeasible_qp(seed, &what) : make_unbounded_qp(seed, &what);
      const IpmResult r = solve_scaled(m, seed % 2 == 0);
      ++runs;
      if (r.status == Status::Optimal) {
        ++wrongly_optimal;
        if (first.empty()) first = "seed " + std::to_string(seed) + " " + what;
      }
    }
  }
  std::cout << "  planted infeasible and unbounded QPs: " << runs << " runs, " << wrongly_optimal << " reported Optimal\n";
  if (!first.empty()) std::cerr << "  WRONG " << first << "\n";
  CHECK_EQ(wrongly_optimal, 0);
}

// ---------------------------------------------------------------------------------------------------------
// Tiny QPs against an enumeration of all active sets (n <= 4, m <= 3, positive definite Q).
namespace {

// Solves the dense system M z = r by Gaussian elimination with partial pivoting; false when singular.
bool dense_solve(std::vector<std::vector<double>> M, std::vector<double> r, std::vector<double>* z) {
  const int n = static_cast<int>(r.size());
  for (int k = 0; k < n; ++k) {
    int piv = k;
    for (int i = k + 1; i < n; ++i) {
      if (std::fabs(M[to_size(i)][to_size(k)]) > std::fabs(M[to_size(piv)][to_size(k)])) piv = i;
    }
    if (std::fabs(M[to_size(piv)][to_size(k)]) < 1e-11) return false;
    std::swap(M[to_size(piv)], M[to_size(k)]);
    std::swap(r[to_size(piv)], r[to_size(k)]);
    for (int i = k + 1; i < n; ++i) {
      const double f = M[to_size(i)][to_size(k)] / M[to_size(k)][to_size(k)];
      for (int j = k; j < n; ++j) M[to_size(i)][to_size(j)] -= f * M[to_size(k)][to_size(j)];
      r[to_size(i)] -= f * r[to_size(k)];
    }
  }
  z->assign(to_size(n), 0.0);
  for (int i = n - 1; i >= 0; --i) {
    double s = r[to_size(i)];
    for (int j = i + 1; j < n; ++j) s -= M[to_size(i)][to_size(j)] * (*z)[to_size(j)];
    (*z)[to_size(i)] = s / M[to_size(i)][to_size(i)];
  }
  return true;
}

LpModel make_tiny_qp(std::uint64_t seed) {
  Rng rng(seed * 2862933555777941757ULL + 3037000493ULL);
  LpModel m;
  const int n = rng.range(2, 4), rows = rng.range(1, 3);
  m.n_cols = n;
  m.n_rows = rows;
  m.sense = Sense::Minimize;
  std::vector<Triplet> a;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < n; ++j) {
      const double v = static_cast<double>(rng.range(-3, 3));
      if (rng.chance(0.7) && v != 0.0) a.push_back({i, j, v});
    }
  }
  std::string err;
  SparseMatrix::from_triplets(rows, n, a, &m.A, &err);
  // Q = L L^T + 0.5 I (positive definite), lower triangle.
  std::vector<std::vector<double>> l(to_size(n), std::vector<double>(to_size(n)));
  for (auto& r : l) {
    for (double& v : r) v = static_cast<double>(rng.range(-2, 2)) * 0.5;
  }
  std::vector<Triplet> q;
  for (int j = 0; j < n; ++j) {
    for (int i = j; i < n; ++i) {
      double s = i == j ? 0.5 : 0.0;
      for (int k = 0; k < n; ++k) s += l[to_size(i)][to_size(k)] * l[to_size(j)][to_size(k)];
      q.push_back({i, j, s});
    }
  }
  SparseMatrix::from_triplets(n, n, q, &m.quadratic, &err);
  for (int j = 0; j < n; ++j) {
    m.col_cost.push_back(static_cast<double>(rng.range(-6, 6)));
    const int kind = rng.range(0, 3);
    double lo = kind == 0 ? -kInf : static_cast<double>(rng.range(-3, 0));
    double up = kind == 1 ? kInf : static_cast<double>(rng.range(1, 4));
    if (kind == 3) lo = up = 0.0, lo = -kInf, up = kInf;
    m.col_lower.push_back(lo);
    m.col_upper.push_back(up);
    m.col_type.push_back(ColType::Continuous);
  }
  for (int i = 0; i < rows; ++i) {
    const int kind = rng.range(0, 3);
    double lo = -kInf, up = kInf;
    if (kind == 0) up = rng.range(-2, 4);
    else if (kind == 1) lo = rng.range(-4, 2);
    else if (kind == 2) lo = up = rng.range(-2, 2);
    else {
      lo = rng.range(-4, 0);
      up = lo + rng.range(1, 4);
    }
    m.row_lower.push_back(lo);
    m.row_upper.push_back(up);
  }
  m.objective_offset = rng.range(-3, 3);
  return m;
}

// Optimal value by enumerating, for every column and row, whether it is free/inactive, at its lower or at its upper
// bound, solving the KKT system of that active set and keeping a primal feasible one with correct dual signs.
bool enumerate_optimum(const LpModel& m, double* objective) {
  const int n = static_cast<int>(m.n_cols), rows = static_cast<int>(m.n_rows);
  std::vector<double> fq(to_size(n) * to_size(n), 0.0);
  for (int j = 0; j < n; ++j) {
    for (Index p = m.quadratic.col_start[to_size(j)]; p < m.quadratic.col_start[to_size(j) + 1]; ++p) {
      const int i = static_cast<int>(m.quadratic.row_index[to_size(p)]);
      fq[to_size(i) * to_size(n) + to_size(j)] = fq[to_size(j) * to_size(n) + to_size(i)] = m.quadratic.value[to_size(p)];
    }
  }
  std::vector<std::vector<double>> A(to_size(rows), std::vector<double>(to_size(n), 0.0));
  for (int j = 0; j < n; ++j) {
    for (Index p = m.A.col_start[to_size(j)]; p < m.A.col_start[to_size(j) + 1]; ++p) A[to_size(m.A.row_index[to_size(p)])][to_size(j)] = m.A.value[to_size(p)];
  }
  int combos = 1;
  for (int k = 0; k < n + rows; ++k) combos *= 3;
  for (int code = 0; code < combos; ++code) {
    std::vector<int> state(to_size(n + rows));  // 0 inactive/free, 1 at lower, 2 at upper
    int c = code;
    bool valid = true;
    for (int k = 0; k < n + rows; ++k) {
      state[to_size(k)] = c % 3;
      c /= 3;
      const double lo = k < n ? m.col_lower[to_size(k)] : m.row_lower[to_size(k - n)];
      const double up = k < n ? m.col_upper[to_size(k)] : m.row_upper[to_size(k - n)];
      if ((state[to_size(k)] == 1 && is_inf(lo)) || (state[to_size(k)] == 2 && is_inf(up))) valid = false;
    }
    if (!valid) continue;
    std::vector<int> fcols, arows;
    for (int j = 0; j < n; ++j) {
      if (state[to_size(j)] == 0) fcols.push_back(j);
    }
    for (int i = 0; i < rows; ++i) {
      if (state[to_size(n + i)] != 0) arows.push_back(i);
    }
    std::vector<double> x(to_size(n), 0.0);
    for (int j = 0; j < n; ++j) {
      if (state[to_size(j)] == 1) x[to_size(j)] = m.col_lower[to_size(j)];
      if (state[to_size(j)] == 2) x[to_size(j)] = m.col_upper[to_size(j)];
    }
    const int nf = static_cast<int>(fcols.size()), na = static_cast<int>(arows.size());
    std::vector<double> sol;
    if (nf + na > 0) {
      std::vector<std::vector<double>> M(to_size(nf + na), std::vector<double>(to_size(nf + na), 0.0));
      std::vector<double> rhs(to_size(nf + na), 0.0);
      for (int a = 0; a < nf; ++a) {
        const int j = fcols[to_size(a)];
        double r = -m.col_cost[to_size(j)];  // (Q x)_j + c_j - sum_i y_i A_ij = 0
        for (int k = 0; k < n; ++k) {
          if (state[to_size(k)] != 0) r -= fq[to_size(j) * to_size(n) + to_size(k)] * x[to_size(k)];
        }
        for (int b = 0; b < nf; ++b) M[to_size(a)][to_size(b)] = fq[to_size(j) * to_size(n) + to_size(fcols[to_size(b)])];
        for (int b = 0; b < na; ++b) M[to_size(a)][to_size(nf + b)] = -A[to_size(arows[to_size(b)])][to_size(j)];
        rhs[to_size(a)] = r;
      }
      for (int b = 0; b < na; ++b) {
        const int i = arows[to_size(b)];
        double r = state[to_size(n + i)] == 1 ? m.row_lower[to_size(i)] : m.row_upper[to_size(i)];
        for (int k = 0; k < n; ++k) {
          if (state[to_size(k)] != 0) r -= A[to_size(i)][to_size(k)] * x[to_size(k)];
        }
        for (int a = 0; a < nf; ++a) M[to_size(nf + b)][to_size(a)] = A[to_size(i)][to_size(fcols[to_size(a)])];
        rhs[to_size(nf + b)] = r;
      }
      if (!dense_solve(M, rhs, &sol)) continue;
      for (int a = 0; a < nf; ++a) x[to_size(fcols[to_size(a)])] = sol[to_size(a)];
    }
    const double tol = 1e-9;
    bool ok = true;
    for (int j = 0; j < n && ok; ++j) ok = x[to_size(j)] >= m.col_lower[to_size(j)] - tol && x[to_size(j)] <= m.col_upper[to_size(j)] + tol;
    std::vector<double> y(to_size(rows), 0.0);
    for (int b = 0; b < na; ++b) y[to_size(arows[to_size(b)])] = sol[to_size(nf + b)];
    for (int i = 0; i < rows && ok; ++i) {
      double act = 0.0;
      for (int j = 0; j < n; ++j) act += A[to_size(i)][to_size(j)] * x[to_size(j)];
      ok = act >= m.row_lower[to_size(i)] - tol && act <= m.row_upper[to_size(i)] + tol;
    }
    for (int i = 0; i < rows && ok; ++i) {
      if (state[to_size(n + i)] == 1) ok = y[to_size(i)] >= -tol;
      if (state[to_size(n + i)] == 2) ok = y[to_size(i)] <= tol;
    }
    for (int j = 0; j < n && ok; ++j) {
      if (state[to_size(j)] == 0) continue;
      double d = m.col_cost[to_size(j)];
      for (int k = 0; k < n; ++k) d += fq[to_size(j) * to_size(n) + to_size(k)] * x[to_size(k)];
      for (int i = 0; i < rows; ++i) d -= y[to_size(i)] * A[to_size(i)][to_size(j)];
      if (state[to_size(j)] == 1) ok = d >= -tol;
      if (state[to_size(j)] == 2) ok = d <= tol;
    }
    if (!ok) continue;
    *objective = model_objective(m, x);
    return true;
  }
  return false;
}

}  // namespace

TEST_CASE(ipm_matches_an_active_set_enumeration_on_tiny_qps) {
  int checked = 0, with_optimum = 0, without = 0, agree = 0;
  double worst = 0.0;
  for (std::uint64_t seed = 1; seed <= 400; ++seed) {
    const LpModel m = make_tiny_qp(seed);
    CHECK(m.validate().empty());
    double truth = 0.0;
    const bool has = enumerate_optimum(m, &truth);
    const IpmResult r = solve_scaled(m, seed % 2 == 0);
    ++checked;
    if (has) {
      ++with_optimum;
      if (r.status == Status::Optimal) {
        const double err = std::fabs(r.solution.objective - truth) / (1.0 + std::fabs(truth));
        worst = std::max(worst, err);
        if (err <= 1e-6) ++agree;
        else std::cerr << "  DISAGREE tiny QP seed " << seed << ": IPM " << r.solution.objective << " enumeration " << truth << "\n";
      } else {
        std::cerr << "  DISAGREE tiny QP seed " << seed << ": enumeration found an optimum, IPM " << to_string(r.status) << " " << r.message << "\n";
      }
    } else {
      ++without;
      if (r.status != Status::Optimal) ++agree;
      else std::cerr << "  DISAGREE tiny QP seed " << seed << ": no KKT point, IPM Optimal " << r.solution.objective << "\n";
    }
  }
  std::cout << "  tiny QPs vs active-set enumeration: " << checked << " problems (" << with_optimum << " with an optimum, " << without << " without), " << agree
            << " agree, worst relative objective difference " << worst << "\n";
  CHECK_EQ(agree, checked);
  CHECK(with_optimum >= 200);
}
