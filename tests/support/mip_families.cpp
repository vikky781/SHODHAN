#include "support/mip_families.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "support/rng.hpp"

namespace shodhan::testing {

const char* mip_family_name(int family) {
  switch (family) {
    case kMipKnapsack: return "knapsack";
    case kMipSetCover: return "set cover";
    case kMipAssignmentSide: return "assignment+side";
    case kMipFixedCharge: return "fixed-charge flow";
    case kMipFacility: return "facility location";
    case kMipLotSizing: return "lot sizing";
    case kMipGeneralInt: return "general integer";
    case kMipMixed: return "mixed";
    case kMipParity: return "parity infeasible";
    case kMipUnbounded: return "unbounded";
    default: return "?";
  }
}

namespace {

struct Builder {
  LpModel m;
  std::vector<Triplet> t;
  Index add_col(double cost, double lo, double hi, ColType type) {
    m.col_cost.push_back(cost);
    m.col_lower.push_back(lo);
    m.col_upper.push_back(hi);
    m.col_type.push_back(type);
    return m.n_cols++;
  }
  Index add_row(double lo, double hi) {
    m.row_lower.push_back(lo);
    m.row_upper.push_back(hi);
    return m.n_rows++;
  }
  void coef(Index row, Index col, double v) { t.push_back({row, col, v}); }
  LpModel finish(std::uint64_t seed) {
    std::string err;
    SparseMatrix::from_triplets(m.n_rows, m.n_cols, t, &m.A, &err);
    if (seed % 3 == 0) m.sense = Sense::Maximize;
    m.name = "mip";
    return std::move(m);
  }
};

LpModel knapsack(Rng& r, std::uint64_t seed) {
  Builder b;
  const int n = r.range(6, 10), rows = r.range(1, 3);
  for (int j = 0; j < n; ++j) b.add_col(r.range(1, 20), 0, 1, ColType::Binary);
  for (int i = 0; i < rows; ++i) {
    const Index row = b.add_row(-kInf, 0);
    double sum = 0;
    for (int j = 0; j < n; ++j) {
      const int w = r.range(1, 12);
      b.coef(row, j, w);
      sum += w;
    }
    b.m.row_upper[to_size(row)] = std::floor(sum * r.uniform(0.3, 0.6));
  }
  return b.finish(seed);
}

LpModel set_cover(Rng& r, std::uint64_t seed) {
  Builder b;
  const int sets = r.range(6, 10), elems = r.range(4, 7);
  for (int j = 0; j < sets; ++j) b.add_col(r.range(1, 9), 0, 1, ColType::Binary);
  for (int e = 0; e < elems; ++e) {
    const Index row = b.add_row(1, kInf);
    int cnt = 0;
    for (int j = 0; j < sets; ++j) {
      if (r.chance(0.4)) {
        b.coef(row, j, 1);
        ++cnt;
      }
    }
    if (cnt == 0) b.coef(row, r.range(0, sets - 1), 1);
  }
  LpModel m = b.finish(seed);
  m.sense = Sense::Minimize;  // set cover minimizes cost
  return m;
}

LpModel assignment_side(Rng& r, std::uint64_t seed) {
  Builder b;
  const int k = 3;
  std::vector<Index> x(static_cast<std::size_t>(k * k));
  for (int i = 0; i < k; ++i)
    for (int j = 0; j < k; ++j) x[static_cast<std::size_t>(i * k + j)] = b.add_col(r.range(1, 15), 0, 1, ColType::Binary);
  for (int i = 0; i < k; ++i) {
    const Index row = b.add_row(1, 1);
    for (int j = 0; j < k; ++j) b.coef(row, x[static_cast<std::size_t>(i * k + j)], 1);
  }
  for (int j = 0; j < k; ++j) {
    const Index row = b.add_row(1, 1);
    for (int i = 0; i < k; ++i) b.coef(row, x[static_cast<std::size_t>(i * k + j)], 1);
  }
  const Index side = b.add_row(-kInf, 0);
  double total = 0;
  for (int c = 0; c < k * k; ++c) {
    const int w = r.range(1, 9);
    b.coef(side, x[static_cast<std::size_t>(c)], w);
    total += w;
  }
  // The budget sometimes makes the model infeasible.
  b.m.row_upper[to_size(side)] = std::floor(total * r.uniform(0.15, 0.5));
  LpModel m = b.finish(seed);
  m.sense = Sense::Minimize;
  return m;
}

LpModel fixed_charge(Rng& r, std::uint64_t seed) {
  Builder b;
  // Nodes 0..3: node 0 is the source (supply), node 3 the sink; arcs go forward; weak big-M.
  const int nodes = 4;
  struct Arc {
    int from, to;
  };
  std::vector<Arc> arcs;
  for (int i = 0; i < nodes; ++i)
    for (int j = i + 1; j < nodes; ++j)
      if (r.chance(0.75) || j == i + 1) arcs.push_back({i, j});
  const double demand = r.range(3, 8);
  std::vector<Index> f, y;
  for (std::size_t a = 0; a < arcs.size(); ++a) {
    f.push_back(b.add_col(r.range(1, 4), 0, kInf, ColType::Continuous));
    y.push_back(b.add_col(r.range(5, 25), 0, 1, ColType::Binary));
  }
  for (int v = 0; v < nodes; ++v) {
    const double rhs = v == 0 ? demand : (v == nodes - 1 ? -demand : 0.0);
    const Index row = b.add_row(rhs, rhs);
    for (std::size_t a = 0; a < arcs.size(); ++a) {
      if (arcs[a].from == v) b.coef(row, f[a], 1);
      if (arcs[a].to == v) b.coef(row, f[a], -1);
    }
  }
  for (std::size_t a = 0; a < arcs.size(); ++a) {
    const Index row = b.add_row(-kInf, 0);  // f - M y <= 0 with a weak M
    b.coef(row, f[a], 1);
    b.coef(row, y[a], -(demand * r.uniform(2.0, 6.0)));
  }
  LpModel m = b.finish(seed);
  m.sense = Sense::Minimize;
  return m;
}

LpModel facility(Rng& r, std::uint64_t seed) {
  Builder b;
  const int F = 3, C = r.range(3, 4);
  std::vector<Index> open(static_cast<std::size_t>(F));
  std::vector<std::vector<Index>> x(static_cast<std::size_t>(F), std::vector<Index>(static_cast<std::size_t>(C)));
  std::vector<double> cap(static_cast<std::size_t>(F)), dem(static_cast<std::size_t>(C));
  double total = 0;
  for (int c = 0; c < C; ++c) {
    dem[static_cast<std::size_t>(c)] = r.range(2, 6);
    total += dem[static_cast<std::size_t>(c)];
  }
  for (int f = 0; f < F; ++f) {
    cap[static_cast<std::size_t>(f)] = std::ceil(total * r.uniform(0.5, 0.9));
    open[static_cast<std::size_t>(f)] = b.add_col(r.range(8, 30), 0, 1, ColType::Binary);
  }
  for (int f = 0; f < F; ++f)
    for (int c = 0; c < C; ++c) x[static_cast<std::size_t>(f)][static_cast<std::size_t>(c)] = b.add_col(r.range(1, 9), 0, 1, ColType::Continuous);
  for (int c = 0; c < C; ++c) {
    const Index row = b.add_row(1, 1);
    for (int f = 0; f < F; ++f) b.coef(row, x[static_cast<std::size_t>(f)][static_cast<std::size_t>(c)], 1);
  }
  for (int f = 0; f < F; ++f) {
    const Index row = b.add_row(-kInf, 0);
    for (int c = 0; c < C; ++c) b.coef(row, x[static_cast<std::size_t>(f)][static_cast<std::size_t>(c)], dem[static_cast<std::size_t>(c)]);
    b.coef(row, open[static_cast<std::size_t>(f)], -cap[static_cast<std::size_t>(f)]);
  }
  LpModel m = b.finish(seed);
  m.sense = Sense::Minimize;
  return m;
}

LpModel lot_sizing(Rng& r, std::uint64_t seed) {
  Builder b;
  const int T = r.range(4, 5);
  std::vector<Index> p(static_cast<std::size_t>(T)), s(static_cast<std::size_t>(T)), y(static_cast<std::size_t>(T));
  std::vector<double> d(static_cast<std::size_t>(T));
  double total = 0;
  for (int t = 0; t < T; ++t) {
    d[static_cast<std::size_t>(t)] = r.range(2, 8);
    total += d[static_cast<std::size_t>(t)];
  }
  for (int t = 0; t < T; ++t) {
    const std::size_t u = static_cast<std::size_t>(t);
    p[u] = b.add_col(r.uniform(0.5, 2.0), 0, kInf, ColType::Continuous);
    s[u] = b.add_col(r.uniform(0.2, 1.0), 0, kInf, ColType::Continuous);
    y[u] = b.add_col(r.range(6, 20), 0, 1, ColType::Binary);
  }
  for (int t = 0; t < T; ++t) {
    const std::size_t u = static_cast<std::size_t>(t);
    const Index row = b.add_row(d[u], d[u]);
    if (t > 0) b.coef(row, s[u - 1], 1);
    b.coef(row, p[u], 1);
    b.coef(row, s[u], -1);
  }
  for (int t = 0; t < T; ++t) {
    const std::size_t u = static_cast<std::size_t>(t);
    const Index row = b.add_row(-kInf, 0);
    b.coef(row, p[u], 1);
    b.coef(row, y[u], -total * r.uniform(1.0, 1.5));  // weak big-M
  }
  LpModel m = b.finish(seed);
  m.sense = Sense::Minimize;
  return m;
}

LpModel general_int(Rng& r, std::uint64_t seed) {
  Builder b;
  const int n = r.range(3, 5), rows = r.range(2, 4);
  for (int j = 0; j < n; ++j) b.add_col(r.range(-6, 9), 0, 5, ColType::Integer);
  for (int i = 0; i < rows; ++i) {
    const Index row = b.add_row(-kInf, 0);
    double sum = 0;
    for (int j = 0; j < n; ++j) {
      const int a = r.range(-3, 7);
      if (a != 0) b.coef(row, j, a);
      sum += std::max(0, a) * 5;
    }
    b.m.row_upper[to_size(row)] = std::floor(sum * r.uniform(0.2, 0.5));
    if (r.chance(0.25)) b.m.row_lower[to_size(row)] = std::floor(b.m.row_upper[to_size(row)] * 0.5) - 2;
  }
  return b.finish(seed);
}

LpModel mixed(Rng& r, std::uint64_t seed) {
  Builder b;
  const int ni = r.range(3, 4), nc = r.range(2, 3), rows = r.range(2, 4);
  for (int j = 0; j < ni; ++j) b.add_col(r.range(-4, 8), 0, 3, ColType::Integer);
  for (int j = 0; j < nc; ++j) b.add_col(r.uniform(-2.0, 4.0), 0, r.range(2, 6), ColType::Continuous);
  for (int i = 0; i < rows; ++i) {
    const Index row = b.add_row(-kInf, 0);
    double sum = 0;
    for (int j = 0; j < ni + nc; ++j) {
      const double a = r.range(-2, 6) * (j < ni ? 1.0 : 0.5);
      if (a != 0.0) b.coef(row, j, a);
      sum += std::max(0.0, a) * (j < ni ? 3 : 4);
    }
    b.m.row_upper[to_size(row)] = std::floor(sum * r.uniform(0.25, 0.55)) + 0.5;
  }
  return b.finish(seed);
}

LpModel parity(Rng& r, std::uint64_t seed) {
  Builder b;
  const int n = r.range(3, 5);
  for (int j = 0; j < n; ++j) b.add_col(r.range(-3, 5), 0, 4, ColType::Integer);
  // 2 * (integer combination) = odd: the LP relaxation is feasible, there is no integer point.
  const double rhs = 2.0 * r.range(2, 6) + 1.0;
  const Index row = b.add_row(rhs, rhs);
  for (int j = 0; j < n; ++j) b.coef(row, j, 2.0 * r.range(1, 3));
  if (r.chance(0.5)) {  // an extra constraint that keeps the LP relaxation feasible
    const Index row2 = b.add_row(-kInf, 12);
    for (int j = 0; j < n; ++j) b.coef(row2, j, 1);
  }
  return b.finish(seed);
}

LpModel unbounded_mip(Rng& r, std::uint64_t seed) {
  Builder b;
  const int n = r.range(2, 3);
  for (int j = 0; j < n; ++j) b.add_col(r.range(1, 5), 0, 4, ColType::Integer);
  const Index free_col = b.add_col(-1.0, 0, kInf, ColType::Continuous);  // improves without limit (minimization)
  const Index row = b.add_row(-kInf, 10);
  for (int j = 0; j < n; ++j) b.coef(row, j, r.range(1, 3));
  const Index row2 = b.add_row(-kInf, kInf);  // a free row: nothing limits free_col
  b.coef(row2, free_col, 1);
  b.coef(row2, 0, -1);
  LpModel m = b.finish(seed);
  m.sense = Sense::Minimize;
  return m;
}

}  // namespace

LpModel make_mip_instance(int family, std::uint64_t seed) {
  Rng r(seed * 7919ULL + static_cast<std::uint64_t>(family) * 104729ULL + 17ULL);
  switch (family) {
    case kMipKnapsack: return knapsack(r, seed);
    case kMipSetCover: return set_cover(r, seed);
    case kMipAssignmentSide: return assignment_side(r, seed);
    case kMipFixedCharge: return fixed_charge(r, seed);
    case kMipFacility: return facility(r, seed);
    case kMipLotSizing: return lot_sizing(r, seed);
    case kMipGeneralInt: return general_int(r, seed);
    case kMipMixed: return mixed(r, seed);
    case kMipParity: return parity(r, seed);
    case kMipUnbounded: return unbounded_mip(r, seed);
    default: return LpModel{};
  }
}

}  // namespace shodhan::testing
