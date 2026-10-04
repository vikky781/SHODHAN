#include "support/cut_families.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "support/rng.hpp"

namespace shodhan::testing {

const char* cut_family_name(int family) {
  switch (family) {
    case kCutKnapsackHard: return "knapsack (hard)";
    case kCutSetCoverHard: return "set cover (hard)";
    case kCutIndependentSet: return "independent set";
    case kCutSetPartition: return "set packing";
    case kCutFixedChargeVub: return "fixed charge (VUB)";
    case kCutGeneralIntEq: return "general integer";
    case kCutMixedBinary: return "mixed binary";
    case kCutLotSizingBigM: return "lot sizing (big-M)";
    default: return "?";
  }
}

namespace {

struct Builder {
  LpModel m;
  std::vector<Triplet> t;
  Index col(double cost, double lo, double hi, ColType type) {
    m.col_cost.push_back(cost);
    m.col_lower.push_back(lo);
    m.col_upper.push_back(hi);
    m.col_type.push_back(type);
    return m.n_cols++;
  }
  Index row(double lo, double hi) {
    m.row_lower.push_back(lo);
    m.row_upper.push_back(hi);
    return m.n_rows++;
  }
  void coef(Index r, Index c, double v) { t.push_back({r, c, v}); }
  LpModel finish(const char* name, Sense sense) {
    std::string err;
    SparseMatrix::from_triplets(m.n_rows, m.n_cols, t, &m.A, &err);
    m.sense = sense;
    m.name = name;
    return std::move(m);
  }
};

LpModel knapsack_hard(Rng& r) {
  Builder b;
  const int n = r.range(8, 12), rows = r.range(2, 3);
  std::vector<std::vector<int>> w(static_cast<std::size_t>(rows), std::vector<int>(static_cast<std::size_t>(n)));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < n; ++j) w[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = r.range(3, 25);
  }
  for (int j = 0; j < n; ++j) {
    double v = 0;
    for (int i = 0; i < rows; ++i) v += w[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    b.col(std::floor(v / rows) + r.range(0, 5), 0, 1, ColType::Binary);
  }
  for (int i = 0; i < rows; ++i) {
    double sum = 0;
    const Index row = b.row(-kInf, 0);
    for (int j = 0; j < n; ++j) {
      b.coef(row, j, w[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
      sum += w[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
    b.m.row_upper[static_cast<std::size_t>(row)] = std::floor(sum * r.uniform(0.35, 0.55)) + 0.0;
  }
  return b.finish("knapsack_hard", Sense::Maximize);
}

LpModel set_cover_hard(Rng& r) {
  // Every element is covered by exactly two sets (an edge of a random graph): the LP relaxation of such a vertex
  // cover is half-integral and fractional whenever the graph has an odd cycle.
  Builder b;
  const int sets = r.range(8, 11);
  for (int j = 0; j < sets; ++j) b.col(r.chance(0.7) ? 1 : 2, 0, 1, ColType::Binary);
  const int elems = r.range(sets + 1, sets + 5);
  for (int e = 0; e < elems; ++e) {
    const Index row = b.row(1, kInf);
    const int p = r.range(0, sets - 1);
    int q = r.range(0, sets - 1);
    if (q == p) q = (p + 1) % sets;
    b.coef(row, p, 1.0);
    b.coef(row, q, 1.0);
  }
  return b.finish("set_cover_hard", Sense::Minimize);
}

LpModel independent_set(Rng& r) {
  Builder b;
  const int n = r.range(8, 12);
  for (int j = 0; j < n; ++j) b.col(r.range(4, 6), 0, 1, ColType::Binary);
  int edges = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (!r.chance(0.4)) continue;
      const Index row = b.row(-kInf, 1);
      b.coef(row, i, 1.0);
      b.coef(row, j, 1.0);
      ++edges;
    }
  }
  if (edges == 0) {
    const Index row = b.row(-kInf, 1);
    b.coef(row, 0, 1.0);
    b.coef(row, 1, 1.0);
  }
  return b.finish("independent_set", Sense::Maximize);
}

LpModel set_partition(Rng& r) {
  // Set packing with overlapping rows (the name of the family is kept for the enum): maximize the weight of the
  // chosen columns such that every row contains at most one of them.
  Builder b;
  const int n = r.range(8, 12), rows = r.range(5, 8);
  for (int j = 0; j < n; ++j) b.col(r.range(3, 9), 0, 1, ColType::Binary);
  for (int i = 0; i < rows; ++i) {
    const Index row = b.row(-kInf, 1);
    std::vector<char> in(static_cast<std::size_t>(n), 0);
    const int cnt = r.range(3, 4);
    int have = 0;
    while (have < cnt) {
      const int j = r.range(0, n - 1);
      if (!in[static_cast<std::size_t>(j)]) {
        in[static_cast<std::size_t>(j)] = 1;
        ++have;
      }
    }
    for (int j = 0; j < n; ++j) {
      if (in[static_cast<std::size_t>(j)]) b.coef(row, j, 1.0);
    }
  }
  return b.finish("set_packing", Sense::Maximize);
}

LpModel fixed_charge_vub(Rng& r) {
  Builder b;
  const int n = r.range(4, 6);
  double total = 0;
  std::vector<Index> xs, ys;
  for (int i = 0; i < n; ++i) {
    const double cap = r.range(4, 12);
    total += cap;
    xs.push_back(b.col(r.range(8, 30), 0, 1, ColType::Binary));
    ys.push_back(b.col(r.range(1, 4), 0, cap, ColType::Continuous));
  }
  for (int i = 0; i < n; ++i) {
    const Index row = b.row(-kInf, 0);
    const double bigm = b.m.col_upper[static_cast<std::size_t>(ys[static_cast<std::size_t>(i)])] * r.uniform(1.4, 3.0);
    b.coef(row, ys[static_cast<std::size_t>(i)], 1.0);
    b.coef(row, xs[static_cast<std::size_t>(i)], -bigm);
  }
  const Index demand = b.row(std::floor(total * r.uniform(0.4, 0.7)), kInf);
  for (const Index y : ys) b.coef(demand, y, 1.0);
  return b.finish("fixed_charge_vub", Sense::Minimize);
}

LpModel general_int_eq(Rng& r) {
  Builder b;
  const int n = r.range(5, 6);
  std::vector<int> x0;
  for (int j = 0; j < n; ++j) {
    b.col(r.range(-6, 6), 0, 4, ColType::Integer);
    x0.push_back(r.range(0, 4));
  }
  const int rows = r.range(2, 3);
  for (int i = 0; i < rows; ++i) {
    double act = 0;
    std::vector<std::pair<int, int>> e;
    for (int j = 0; j < n; ++j) {
      if (r.chance(0.6)) {
        int a = r.range(-3, 5);
        if (a == 0) a = 1;
        e.emplace_back(j, a);
        act += a * x0[static_cast<std::size_t>(j)];
      }
    }
    if (e.empty()) e.emplace_back(0, 2), act = 2 * x0[0];
    const double kind = r.unit();
    Index row;
    if (kind < 0.35) row = b.row(act, act);
    else if (kind < 0.7) row = b.row(-kInf, act + r.range(0, 2));
    else row = b.row(act - r.range(0, 2), kInf);
    for (const auto& p : e) b.coef(row, p.first, p.second);
  }
  return b.finish("general_int_eq", Sense::Minimize);
}

LpModel mixed_binary(Rng& r) {
  Builder b;
  const int nb = r.range(4, 6), nc = r.range(2, 3);
  for (int j = 0; j < nb; ++j) b.col(r.range(2, 9), 0, 1, ColType::Binary);
  for (int k = 0; k < nc; ++k) b.col(r.range(1, 4), 0, r.range(3, 8), ColType::Continuous);
  const int rows = r.range(3, 5);
  for (int i = 0; i < rows; ++i) {
    const Index row = b.row(-kInf, 0);
    double mx = 0;
    int cnt = 0;
    for (int j = 0; j < nb + nc; ++j) {
      if (!r.chance(0.65)) continue;
      const double a = r.range(1, 7) * (r.chance(0.15) ? -1.0 : 1.0);
      b.coef(row, j, a);
      ++cnt;
      mx += a > 0 ? a * b.m.col_upper[static_cast<std::size_t>(j)] : 0.0;
    }
    if (cnt == 0) b.coef(row, 0, 3.0), mx = 3.0;
    b.m.row_upper[static_cast<std::size_t>(row)] = std::floor(mx * r.uniform(0.3, 0.6));
  }
  return b.finish("mixed_binary", Sense::Maximize);
}

LpModel lot_sizing_big_m(Rng& r) {
  Builder b;
  const int T = r.range(4, 5);
  std::vector<Index> prod, setup, stock;
  std::vector<double> demand;
  double total = 0;
  for (int t = 0; t < T; ++t) {
    demand.push_back(r.range(2, 8));
    total += demand.back();
  }
  for (int t = 0; t < T; ++t) {
    setup.push_back(b.col(r.range(10, 25), 0, 1, ColType::Binary));
    prod.push_back(b.col(r.range(1, 3), 0, total, ColType::Continuous));
    stock.push_back(b.col(r.range(1, 2), 0, total, ColType::Continuous));
  }
  for (int t = 0; t < T; ++t) {  // stock[t-1] + prod[t] - stock[t] = demand[t]
    const Index row = b.row(demand[static_cast<std::size_t>(t)], demand[static_cast<std::size_t>(t)]);
    if (t > 0) b.coef(row, stock[static_cast<std::size_t>(t - 1)], 1.0);
    b.coef(row, prod[static_cast<std::size_t>(t)], 1.0);
    b.coef(row, stock[static_cast<std::size_t>(t)], -1.0);
  }
  for (int t = 0; t < T; ++t) {  // prod[t] <= M setup[t]
    const Index row = b.row(-kInf, 0);
    b.coef(row, prod[static_cast<std::size_t>(t)], 1.0);
    b.coef(row, setup[static_cast<std::size_t>(t)], -total * r.uniform(1.0, 1.5));
  }
  return b.finish("lot_sizing_bigm", Sense::Minimize);
}

}  // namespace

LpModel make_cut_instance(int family, std::uint64_t seed) {
  Rng r(seed * 2654435761ULL + 17);
  switch (family) {
    case kCutKnapsackHard: return knapsack_hard(r);
    case kCutSetCoverHard: return set_cover_hard(r);
    case kCutIndependentSet: return independent_set(r);
    case kCutSetPartition: return set_partition(r);
    case kCutFixedChargeVub: return fixed_charge_vub(r);
    case kCutGeneralIntEq: return general_int_eq(r);
    case kCutMixedBinary: return mixed_binary(r);
    default: return lot_sizing_big_m(r);
  }
}

}  // namespace shodhan::testing
