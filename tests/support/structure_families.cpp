#include "support/structure_families.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "support/rng.hpp"

namespace shodhan::testing {

const char* bigm_kind_name(int kind) {
  switch (kind) {
    case kBigMFixedCharge: return "fixed-charge, capacity rows";
    case kBigMScheduling: return "single-machine sequencing";
    case kBigMLotSizing: return "lot sizing with setups";
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
  LpModel finish(const char* name) {
    std::string err;
    SparseMatrix::from_triplets(m.n_rows, m.n_cols, t, &m.A, &err);
    m.name = name;
    return std::move(m);
  }
};

struct RowSpec {
  std::vector<std::pair<Index, double>> entries;
  double lo = -kInf, hi = kInf;
  int kind = 0;  // 0 decoy, 1 vub, 2 balance, 3 set
  VubRow vub;
  SetRowKind set_kind = SetRowKind::Packing;
};

}  // namespace

PlantedStructure make_planted_structure(std::uint64_t seed) {
  Rng r(seed * 7919ULL + 11ULL);
  Builder b;
  const int nb = r.range(8, 14), nc = r.range(8, 14), ng = r.range(0, 2);
  std::vector<Index> bins, conts, gens;
  for (int j = 0; j < nb; ++j) bins.push_back(b.add_col(r.range(1, 9), 0.0, 1.0, ColType::Binary));
  for (int j = 0; j < nc; ++j) conts.push_back(b.add_col(r.range(0, 5), 0.0, r.chance(0.5) ? kInf : r.range(10, 60), ColType::Continuous));
  for (int j = 0; j < ng; ++j) gens.push_back(b.add_col(r.range(0, 5), 0.0, 4.0, ColType::Integer));

  std::vector<RowSpec> specs;
  const double pow2[4] = {0.5, 1.0, 2.0, 4.0};
  auto pick = [&](const std::vector<Index>& v) { return v[static_cast<std::size_t>(r.range(0, static_cast<int>(v.size()) - 1))]; };
  auto distinct = [&](const std::vector<Index>& v, int k) {
    std::vector<Index> c = v;
    r.shuffle(c);
    c.resize(static_cast<std::size_t>(std::min<int>(k, static_cast<int>(c.size()))));
    return c;
  };
  PlantedStructure out;

  // Variable upper bounds in all four spellings.
  const int nv = r.range(3, 8);
  for (int k = 0; k < nv; ++k) {
    RowSpec s;
    s.kind = 1;
    const Index x = (!gens.empty() && r.chance(0.25)) ? pick(gens) : pick(conts);
    const Index y = pick(bins);
    const double a = pow2[r.range(0, 3)], u = r.range(3, 40);
    const int form = r.range(0, 3);
    s.vub = {-1, x, y, u, form >= 2};
    switch (form) {
      case 0: s.entries = {{x, a}, {y, -a * u}}; s.lo = -kInf; s.hi = 0.0; break;
      case 1: s.entries = {{x, -a}, {y, a * u}}; s.lo = 0.0; s.hi = kInf; break;
      case 2: s.entries = {{x, a}, {y, a * u}}; s.lo = -kInf; s.hi = a * u; break;
      default: s.entries = {{x, -a}, {y, -a * u}}; s.lo = -a * u; s.hi = kInf; break;
    }
    specs.push_back(s);
  }
  // Flow balances.
  const int nbal = r.range(1, 4);
  for (int k = 0; k < nbal; ++k) {
    RowSpec s;
    s.kind = 2;
    const std::vector<Index> cols = distinct(conts, r.range(3, 5));
    for (std::size_t q = 0; q < cols.size(); ++q) {
      double v = r.chance(0.5) ? 1.0 : -1.0;
      if (q == 0) v = 1.0;
      if (q == 1) v = -1.0;
      s.entries.push_back({cols[q], v});
    }
    s.lo = s.hi = r.range(-3, 3);
    specs.push_back(s);
  }
  // Set rows.
  const int nset = r.range(2, 6);
  for (int k = 0; k < nset; ++k) {
    RowSpec s;
    s.kind = 3;
    for (const Index j : distinct(bins, r.range(2, 5))) s.entries.push_back({j, 1.0});
    const int t = r.range(0, 2);
    s.set_kind = t == 0 ? SetRowKind::Partition : t == 1 ? SetRowKind::Packing : SetRowKind::Covering;
    if (t == 0) s.lo = s.hi = 1.0;
    else if (t == 1) s.hi = 1.0;
    else s.lo = 1.0;
    specs.push_back(s);
  }
  // Decoys: similar looking rows that must not be recognised.
  auto decoy = [&](std::vector<std::pair<Index, double>> e, double lo, double hi) {
    RowSpec s;
    s.entries = std::move(e);
    s.lo = lo;
    s.hi = hi;
    specs.push_back(s);
    ++out.decoys;
  };
  for (int k = 0; k < 2; ++k) {
    const Index x = pick(conts), y = pick(bins), x2 = pick(conts);
    const Index a = pick(bins), c = pick(bins), c3 = pick(bins);
    const double u = r.range(3, 40);
    decoy({{x, 1.0}, {y, -u}}, -kInf, 1.0);              // x <= 1 + u y: a general variable bound, not x <= u y
    if (x2 != x) decoy({{x, 1.0}, {x2, 1.0}, {y, -u}}, -kInf, 0.0);  // aggregated: three columns
    decoy({{x, 1.0}, {y, -u}}, 0.0, 0.0);                // equality
    decoy({{x, 1.0}, {y, -u}}, -3.0, 0.0);               // range
    decoy({{x, -1.0}, {y, u}}, -kInf, 0.0);              // x >= u y: a lower bound
    if (x2 != x) {
      decoy({{x, 2.0}, {x2, -1.0}}, 0.0, 0.0);          // equality with a coefficient 2
      decoy({{x, 1.0}, {x2, 1.0}}, 1.0, 1.0);           // equality without a minus sign
      decoy({{x, 1.0}, {x2, -1.0}}, -kInf, 0.0);        // inequality
    }
    if (a != c) {
      decoy({{a, 1.0}, {c, -1.0}}, 0.0, 0.0);           // equality of two binaries
      decoy({{a, 2.0}, {c, 1.0}}, -kInf, 1.0);          // coefficient 2
      if (c3 != a && c3 != c) decoy({{a, 1.0}, {c, 1.0}, {c3, 1.0}}, -kInf, 2.0);  // right-hand side 2
      decoy({{a, 1.0}, {c, 1.0}, {x, 1.0}}, -kInf, 1.0);  // a continuous column in a set row
    }
  }
  r.shuffle(specs);
  for (RowSpec& s : specs) {
    const Index row = b.add_row(s.lo, s.hi);
    for (const auto& e : s.entries) b.coef(row, e.first, e.second);
    if (s.kind == 1) {
      s.vub.row = row;
      out.vubs.push_back(s.vub);
    } else if (s.kind == 2) {
      out.balance_rows.push_back(row);
    } else if (s.kind == 3) {
      out.set_rows.push_back({row, s.set_kind});
    }
  }
  out.model = b.finish("planted_structure");
  return out;
}

LpModel make_bigm_instance(int kind, std::uint64_t seed) {
  Rng r(seed * 104729ULL + static_cast<std::uint64_t>(kind) * 31ULL + 5ULL);
  Builder b;
  if (kind == kBigMFixedCharge) {
    const int k = r.range(3, 6);
    double cap_sum = 0.0;
    std::vector<Index> xs;
    const double M = 1000.0 + r.range(0, 500);
    for (int i = 0; i < k; ++i) {
      const Index y = b.add_col(r.range(10, 50), 0.0, 1.0, ColType::Binary);
      const Index x = b.add_col(r.range(1, 5), 0.0, kInf, ColType::Continuous);
      const Index s = b.add_col(0.0, 0.0, kInf, ColType::Continuous);
      const double cap = r.range(15, 60);
      cap_sum += cap;
      const Index vub = b.add_row(-kInf, 0.0);  // x <= M y
      b.coef(vub, x, 1.0);
      b.coef(vub, y, -M);
      const Index capr = b.add_row(-kInf, cap);  // x + s <= cap: the implied bound of x
      b.coef(capr, x, 1.0);
      b.coef(capr, s, 1.0);
      xs.push_back(x);
    }
    const Index dem = b.add_row(std::floor(cap_sum * r.uniform(0.3, 0.7)), kInf);
    for (const Index x : xs) b.coef(dem, x, 1.0);
  } else if (kind == kBigMScheduling) {
    const int n = r.range(3, 4);
    std::vector<double> p(static_cast<std::size_t>(n));
    double H = 0.0;
    for (double& v : p) {
      v = r.range(1, 6);
      H += v;
    }
    H += r.range(0, 4);
    const double M = 10.0 * H + 100.0;
    std::vector<Index> s;
    for (int j = 0; j < n; ++j) {
      s.push_back(b.add_col(r.range(1, 5), 0.0, kInf, ColType::Continuous));
      const Index z = b.add_col(0.0, 0.0, kInf, ColType::Continuous);
      const Index hr = b.add_row(-kInf, H);  // s_j + z_j <= H: s_j <= H is implied
      b.coef(hr, s.back(), 1.0);
      b.coef(hr, z, 1.0);
    }
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const Index y = b.add_col(0.0, 0.0, 1.0, ColType::Binary);
        const Index r1 = b.add_row(p[static_cast<std::size_t>(i)] - M, kInf);  // y = 1: i before j
        b.coef(r1, s[static_cast<std::size_t>(j)], 1.0);
        b.coef(r1, s[static_cast<std::size_t>(i)], -1.0);
        b.coef(r1, y, -M);
        const Index r2 = b.add_row(p[static_cast<std::size_t>(j)], kInf);  // y = 0: j before i
        b.coef(r2, s[static_cast<std::size_t>(i)], 1.0);
        b.coef(r2, s[static_cast<std::size_t>(j)], -1.0);
        b.coef(r2, y, M);
      }
    }
  } else {
    const int T = r.range(4, 7);
    std::vector<double> d(static_cast<std::size_t>(T));
    for (double& v : d) v = r.chance(0.2) ? 0.0 : r.range(5, 25);
    const double M = 5000.0;
    std::vector<Index> inv(static_cast<std::size_t>(T));
    for (int t = 0; t < T; ++t) inv[static_cast<std::size_t>(t)] = b.add_col(r.range(0, 2), 0.0, t == T - 1 ? 0.0 : kInf, ColType::Continuous);
    for (int t = 0; t < T; ++t) {
      const Index y = b.add_col(r.range(20, 60), 0.0, 1.0, ColType::Binary);
      const Index x = b.add_col(r.range(1, 3), 0.0, kInf, ColType::Continuous);
      const Index bal = b.add_row(d[static_cast<std::size_t>(t)], d[static_cast<std::size_t>(t)]);  // I_{t-1} + x_t - I_t = d_t
      if (t > 0) b.coef(bal, inv[static_cast<std::size_t>(t - 1)], 1.0);
      b.coef(bal, x, 1.0);
      b.coef(bal, inv[static_cast<std::size_t>(t)], -1.0);
      const Index vub = b.add_row(-kInf, 0.0);
      b.coef(vub, x, 1.0);
      b.coef(vub, y, -M);
    }
  }
  return b.finish("bigm");
}

}  // namespace shodhan::testing
