// Structured, highly degenerate LP families with independently known optima:
// assignment problems (Hungarian algorithm), transportation problems with equal
// supply and demand (successive shortest paths), Klee-Minty cubes (analytic optimum)
// and staircase LPs (dense oracle).

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/lp_solver.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

LpModel empty_model(Index m, Index n) {
  LpModel model;
  model.n_rows = m;
  model.n_cols = n;
  model.col_cost.assign(to_size(n), 0.0);
  model.col_lower.assign(to_size(n), 0.0);
  model.col_upper.assign(to_size(n), kInf);
  model.col_type.assign(to_size(n), ColType::Continuous);
  model.row_lower.assign(to_size(m), 0.0);
  model.row_upper.assign(to_size(m), 0.0);
  return model;
}

void set_matrix(LpModel* m, std::vector<Triplet> t) {
  std::string err;
  if (!SparseMatrix::from_triplets(m->n_rows, m->n_cols, std::move(t), &m->A, &err)) throw std::runtime_error(err);
}

// Hungarian algorithm (potentials, O(n^3)) for the minimum-cost perfect matching.
long long hungarian(const std::vector<std::vector<long long>>& cost) {
  const int n = static_cast<int>(cost.size());
  const long long inf = std::numeric_limits<long long>::max() / 4;
  std::vector<long long> u(to_size(n + 1), 0), v(to_size(n + 1), 0);
  std::vector<int> p(to_size(n + 1), 0), way(to_size(n + 1), 0);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<long long> minv(to_size(n + 1), inf);
    std::vector<char> used(to_size(n + 1), 0);
    do {
      used[to_size(j0)] = 1;
      const int i0 = p[to_size(j0)];
      long long delta = inf;
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[to_size(j)]) continue;
        const long long cur = cost[to_size(i0 - 1)][to_size(j - 1)] - u[to_size(i0)] - v[to_size(j)];
        if (cur < minv[to_size(j)]) {
          minv[to_size(j)] = cur;
          way[to_size(j)] = j0;
        }
        if (minv[to_size(j)] < delta) {
          delta = minv[to_size(j)];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[to_size(j)]) {
          u[to_size(p[to_size(j)])] += delta;
          v[to_size(j)] -= delta;
        } else {
          minv[to_size(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[to_size(j0)] != 0);
    do {
      const int j1 = way[to_size(j0)];
      p[to_size(j0)] = p[to_size(j1)];
      j0 = j1;
    } while (j0 != 0);
  }
  long long total = 0;
  for (int j = 1; j <= n; ++j) total += cost[to_size(p[to_size(j)] - 1)][to_size(j - 1)];
  return total;
}

// Minimum-cost flow by successive shortest paths (Bellman-Ford), integer data.
struct MinCostFlow {
  struct Edge { int to; long long cap, cost; };
  std::vector<Edge> edges;
  std::vector<std::vector<int>> adj;
  explicit MinCostFlow(int n) : adj(to_size(n)) {}
  void add(int a, int b, long long cap, long long cost) {
    adj[to_size(a)].push_back(static_cast<int>(edges.size()));
    edges.push_back({b, cap, cost});
    adj[to_size(b)].push_back(static_cast<int>(edges.size()));
    edges.push_back({a, 0, -cost});
  }
  long long run(int s, int t, long long* flow_out) {
    long long flow = 0, cost = 0;
    const long long inf = std::numeric_limits<long long>::max() / 4;
    for (;;) {
      std::vector<long long> dist(adj.size(), inf);
      std::vector<int> prev_edge(adj.size(), -1);
      dist[to_size(s)] = 0;
      for (std::size_t it = 0; it < adj.size(); ++it) {
        bool changed = false;
        for (std::size_t a = 0; a < adj.size(); ++a) {
          if (dist[a] == inf) continue;
          for (const int e : adj[a]) {
            if (edges[to_size(e)].cap > 0 && dist[a] + edges[to_size(e)].cost < dist[to_size(edges[to_size(e)].to)]) {
              dist[to_size(edges[to_size(e)].to)] = dist[a] + edges[to_size(e)].cost;
              prev_edge[to_size(edges[to_size(e)].to)] = e;
              changed = true;
            }
          }
        }
        if (!changed) break;
      }
      if (dist[to_size(t)] == inf) break;
      long long push = inf;
      for (int v = t; v != s;) {
        const int e = prev_edge[to_size(v)];
        push = std::min(push, edges[to_size(e)].cap);
        v = edges[to_size(e ^ 1)].to;
      }
      for (int v = t; v != s;) {
        const int e = prev_edge[to_size(v)];
        edges[to_size(e)].cap -= push;
        edges[to_size(e ^ 1)].cap += push;
        v = edges[to_size(e ^ 1)].to;
      }
      flow += push;
      cost += push * dist[to_size(t)];
    }
    *flow_out = flow;
    return cost;
  }
};

LpOptions config(int variant) {
  LpOptions o;
  o.presolve = (variant & 1) != 0;
  o.scaling = (variant & 2) != 0;
  return o;
}

}  // namespace

TEST_CASE(lp_structured_assignment_problems) {
  int solved = 0;
  long long iterations = 0;
  double worst_int = 0.0;
  for (int n = 2; n <= 30; ++n) {
    for (int rep = 0; rep < 2; ++rep) {
      Rng rng(static_cast<std::uint64_t>(n) * 977ULL + static_cast<std::uint64_t>(rep));
      std::vector<std::vector<long long>> cost(to_size(n), std::vector<long long>(to_size(n)));
      const int range = rep == 0 ? 100 : 3;  // rep 1: many ties, very degenerate
      for (auto& row : cost)
        for (auto& c : row) c = rng.range(1, range);
      LpModel m = empty_model(2 * n, n * n);
      std::vector<Triplet> t;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          const Index col = i * n + j;
          m.col_cost[to_size(col)] = static_cast<double>(cost[to_size(i)][to_size(j)]);
          t.push_back({i, col, 1.0});
          t.push_back({n + j, col, 1.0});
        }
      set_matrix(&m, t);
      for (Index i = 0; i < 2 * n; ++i) m.row_lower[to_size(i)] = m.row_upper[to_size(i)] = 1.0;
      const long long expected = hungarian(cost);
      for (int variant = 0; variant < 4; ++variant) {
        const LpResult r = LpSolver(config(variant)).solve(m);
        if (r.status != Status::Optimal || std::fabs(r.solution.objective - static_cast<double>(expected)) > 1e-6 * (1.0 + static_cast<double>(expected))) {
          std::cerr << "FAILING assignment n = " << n << " rep " << rep << " variant " << variant << ": " << to_string(r.status) << " objective "
                    << r.solution.objective << ", Hungarian " << expected << "\n";
          CHECK(false);
          continue;
        }
        for (const double x : r.solution.x) worst_int = std::max(worst_int, std::min(std::fabs(x), std::fabs(x - 1.0)));
        CHECK(r.kkt.ok);
        iterations += r.iterations;
        ++solved;
      }
    }
  }
  std::cout << "    assignment problems n = 2..30: " << solved << " solves match the Hungarian optimum; largest distance of a variable from 0/1 " << std::scientific
            << std::setprecision(1) << worst_int << std::defaultfloat << ", " << iterations << " iterations in total\n";
  CHECK_EQ(solved, 29 * 2 * 4);
}

TEST_CASE(lp_structured_transportation_problems_with_equal_supply_and_demand) {
  int solved = 0;
  for (int size = 2; size <= 12; ++size) {
    for (int rep = 0; rep < 3; ++rep) {
      Rng rng(static_cast<std::uint64_t>(size) * 131ULL + static_cast<std::uint64_t>(rep) + 7ULL);
      const int S = size, D = rep == 0 ? size : std::max(2, size - 1 + rep);
      std::vector<long long> supply(to_size(S)), demand(to_size(D));
      long long total = 0;
      for (auto& s : supply) {
        s = rng.range(1, 9);
        total += s;
      }
      // Demands with the same total.
      long long left = total;
      for (int j = 0; j < D; ++j) {
        const long long d = j + 1 == D ? left : std::min<long long>(left - (D - 1 - j), rng.range(1, 9));
        demand[to_size(j)] = std::max<long long>(d, 0);
        left -= demand[to_size(j)];
      }
      std::vector<std::vector<long long>> cost(to_size(S), std::vector<long long>(to_size(D)));
      for (auto& row : cost)
        for (auto& c : row) c = rng.range(1, rep == 2 ? 3 : 20);  // rep 2: heavy ties
      LpModel m = empty_model(S + D, S * D);
      std::vector<Triplet> t;
      for (int i = 0; i < S; ++i)
        for (int j = 0; j < D; ++j) {
          const Index col = i * D + j;
          m.col_cost[to_size(col)] = static_cast<double>(cost[to_size(i)][to_size(j)]);
          t.push_back({i, col, 1.0});
          t.push_back({S + j, col, 1.0});
        }
      set_matrix(&m, t);
      for (int i = 0; i < S; ++i) m.row_lower[to_size(i)] = m.row_upper[to_size(i)] = static_cast<double>(supply[to_size(i)]);
      for (int j = 0; j < D; ++j) m.row_lower[to_size(S + j)] = m.row_upper[to_size(S + j)] = static_cast<double>(demand[to_size(j)]);
      MinCostFlow mcf(S + D + 2);
      for (int i = 0; i < S; ++i) mcf.add(S + D, i, supply[to_size(i)], 0);
      for (int j = 0; j < D; ++j) mcf.add(S + j, S + D + 1, demand[to_size(j)], 0);
      for (int i = 0; i < S; ++i)
        for (int j = 0; j < D; ++j) mcf.add(i, S + j, total, cost[to_size(i)][to_size(j)]);
      long long flow = 0;
      const long long expected = mcf.run(S + D, S + D + 1, &flow);
      REQUIRE(flow == total);
      for (int variant = 0; variant < 4; ++variant) {
        const LpResult r = LpSolver(config(variant)).solve(m);
        if (r.status != Status::Optimal || std::fabs(r.solution.objective - static_cast<double>(expected)) > 1e-6 * (1.0 + static_cast<double>(expected))) {
          std::cerr << "FAILING transportation " << S << "x" << D << " rep " << rep << " variant " << variant << ": " << to_string(r.status) << " objective "
                    << r.solution.objective << ", min-cost flow " << expected << "\n";
          CHECK(false);
          continue;
        }
        CHECK(r.kkt.ok);
        ++solved;
      }
    }
  }
  std::cout << "    transportation problems (equal supply and demand, up to 12 x 12): " << solved << " solves match the min-cost-flow optimum\n";
  CHECK_EQ(solved, 11 * 3 * 4);
}

TEST_CASE(lp_structured_klee_minty_cubes) {
  // max sum_j 2^(D-j) x_j  s.t.  sum_{i<j} 2^(j-i+1) x_i + x_j <= 5^j  (j = 1..D), x >= 0.
  // The optimum is x_D = 5^D, all others 0, objective 5^D.
  int solved = 0;
  long long iterations = 0;
  for (int D = 2; D <= 12; ++D) {
    LpModel m = empty_model(D, D);
    m.sense = Sense::Maximize;
    std::vector<Triplet> t;
    for (int j = 0; j < D; ++j) {
      m.col_cost[to_size(j)] = std::ldexp(1.0, D - 1 - j);
      m.row_lower[to_size(j)] = -kInf;
      m.row_upper[to_size(j)] = std::pow(5.0, j + 1);
      for (int i = 0; i < j; ++i) t.push_back({j, i, std::ldexp(1.0, j - i + 1)});
      t.push_back({j, j, 1.0});
    }
    set_matrix(&m, t);
    const double expected = std::pow(5.0, D);
    for (int variant = 0; variant < 4; ++variant) {
      const LpResult r = LpSolver(config(variant)).solve(m);
      if (r.status != Status::Optimal || std::fabs(r.solution.objective - expected) > 1e-6 * expected) {
        std::cerr << "FAILING Klee-Minty D = " << D << " variant " << variant << ": " << to_string(r.status) << " objective " << r.solution.objective
                  << ", expected " << expected << " [" << r.message << "]\n";
        CHECK(false);
        continue;
      }
      CHECK_NEAR(r.solution.x[to_size(D - 1)], expected, 1e-6 * expected);
      for (int j = 0; j + 1 < D; ++j) CHECK_NEAR(r.solution.x[to_size(j)], 0.0, 1e-6 * expected);
      CHECK(r.kkt.ok);
      iterations += r.iterations;
      ++solved;
    }
  }
  std::cout << "    Klee-Minty cubes D = 2..12: " << solved << " solves reach the analytic optimum 5^D, " << iterations << " iterations in total\n";
  CHECK_EQ(solved, 11 * 4);
}

TEST_CASE(lp_structured_staircase_lps_against_the_oracle) {
  int compared = 0, optimal = 0, infeasible = 0;
  for (int T = 3; T <= 6; ++T) {
    for (int P = 2; P <= 3; ++P) {
      for (int rep = 0; rep < 8; ++rep) {
        Rng rng(static_cast<std::uint64_t>(T) * 1000ULL + static_cast<std::uint64_t>(P) * 100ULL + static_cast<std::uint64_t>(rep));
        // Columns: production p(t,k) then inventory s(t,k); rows: balance (t,k) then capacity t.
        const int n = 2 * T * P, m_rows = T * P + T;
        LpModel m = empty_model(m_rows, n);
        std::vector<Triplet> tr;
        auto prod = [&](int t, int k) { return t * P + k; };
        auto inv = [&](int t, int k) { return T * P + t * P + k; };
        std::vector<double> weight(to_size(P));
        for (auto& w : weight) w = static_cast<double>(rng.range(1, 3));
        for (int t = 0; t < T; ++t) {
          for (int k = 0; k < P; ++k) {
            m.col_cost[to_size(prod(t, k))] = static_cast<double>(rng.range(1, 6));
            m.col_cost[to_size(inv(t, k))] = static_cast<double>(rng.range(0, 2)) / 2.0;
            const int row = t * P + k;
            tr.push_back({row, prod(t, k), 1.0});
            tr.push_back({row, inv(t, k), -1.0});
            if (t > 0) tr.push_back({row, inv(t - 1, k), 1.0});
            m.row_lower[to_size(row)] = m.row_upper[to_size(row)] = static_cast<double>(rng.range(0, 8));
            tr.push_back({T * P + t, prod(t, k), weight[to_size(k)]});
          }
          m.row_lower[to_size(T * P + t)] = -kInf;
          m.row_upper[to_size(T * P + t)] = static_cast<double>(rng.range(rep < 4 ? 6 : 20, rep < 4 ? 16 : 40));  // tight capacities may be infeasible
        }
        set_matrix(&m, tr);
        const RefLpResult ref = solve_dense_lp(m);
        if (ref.status == Status::NumericalError) continue;
        const LpResult r = LpSolver().solve(m);
        ++compared;
        if (r.status != ref.status ||
            (r.status == Status::Optimal && std::fabs(r.solution.objective - ref.solution.objective) > 1e-6 * (1.0 + std::fabs(ref.solution.objective)))) {
          std::cerr << "FAILING staircase T = " << T << " P = " << P << " rep " << rep << ": " << to_string(r.status) << ", oracle " << to_string(ref.status) << "\n";
          CHECK(false);
          continue;
        }
        if (r.status == Status::Optimal) {
          ++optimal;
          CHECK(r.kkt.ok);
        } else {
          ++infeasible;
        }
      }
    }
  }
  std::cout << "    staircase LPs: " << compared << " compared with the oracle (" << optimal << " optimal, " << infeasible << " infeasible)\n";
  CHECK(compared > 40);
}
