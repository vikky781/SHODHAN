#include "support/random_lp.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "support/rng.hpp"

namespace shodhan::testing {

namespace {

enum RowRole { kRowRandom, kRowEmpty, kRowSingleton, kRowDoubleton, kRowForcing };
enum ColRole { kColNormal, kColEmpty, kColFixed, kColForcing };

struct ColInfo {
  double lo = 0.0;
  double up = kInf;
  double x = 0.0;
  double d = 0.0;
};

// All generated data lies on coarse dyadic grids so that every product and sum
// below is EXACT in double precision. The planted pair is then exactly
// feasible/optimal, and any inaccuracy seen by presolve tests comes from the
// code under test, not from the generator.

/// Uniform multiple of 1/denom in [lo, hi].
double grid(Rng& rng, double lo, double hi, int denom) {
  const int a = static_cast<int>(std::ceil(lo * denom - 1e-12));
  int b = static_cast<int>(std::floor(hi * denom + 1e-12));
  if (b < a) b = a;
  return static_cast<double>(rng.range(a, b)) / static_cast<double>(denom);
}

double coefficient(Rng& rng, bool wide) {
  const double sign = rng.chance(0.5) ? 1.0 : -1.0;
  if (wide) {
    // 11-bit mantissa times 2^e, e in [-13, 12]: magnitudes from 1.2e-4 to 8.2e3.
    const double mantissa = static_cast<double>(1024 + rng.range(0, 1023)) / 1024.0;
    return sign * std::ldexp(mantissa, rng.range(-13, 12));
  }
  return sign * grid(rng, 0.3, 3.0, 1024);
}

double multiplier(Rng& rng, double degeneracy) {
  if (rng.chance(degeneracy)) return 0.0;
  return grid(rng, 0.2, 3.0, 16);
}

}  // namespace

RandomLp make_random_lp(std::uint64_t seed, const RandomLpOptions& o) {
  Rng rng(seed);
  const int m = std::max(o.rows, 0);
  const int n = std::max(o.cols, 1);

  // ---- roles ----
  std::vector<int> row_role(static_cast<std::size_t>(m), kRowRandom);
  std::vector<int> col_role(static_cast<std::size_t>(n), kColNormal);
  std::vector<int> row_perm(static_cast<std::size_t>(m));
  std::vector<int> col_perm(static_cast<std::size_t>(n));
  for (int i = 0; i < m; ++i) row_perm[static_cast<std::size_t>(i)] = i;
  for (int j = 0; j < n; ++j) col_perm[static_cast<std::size_t>(j)] = j;
  rng.shuffle(row_perm);
  rng.shuffle(col_perm);

  std::size_t rp = 0;
  std::size_t cp = 0;
  struct Forcing {
    int row;
    std::vector<int> cols;
    bool max_forcing;
    bool equality;
  };
  std::vector<Forcing> forcing;
  for (int f = 0; f < o.forcing_rows && rp < row_perm.size(); ++f) {
    const int k = rng.range(2, 4);
    if (cp + static_cast<std::size_t>(k) + 1 > col_perm.size()) break;
    Forcing fr;
    fr.row = row_perm[rp++];
    for (int t = 0; t < k; ++t) {
      fr.cols.push_back(col_perm[cp]);
      col_role[static_cast<std::size_t>(col_perm[cp])] = kColForcing;
      ++cp;
    }
    fr.max_forcing = rng.chance(0.5);
    fr.equality = rng.chance(0.2);
    row_role[static_cast<std::size_t>(fr.row)] = kRowForcing;
    forcing.push_back(std::move(fr));
  }
  auto assign_rows = [&](int count, int role) {
    for (int t = 0; t < count && rp < row_perm.size(); ++t) {
      row_role[static_cast<std::size_t>(row_perm[rp++])] = role;
    }
  };
  assign_rows(o.doubleton_eqs, kRowDoubleton);
  assign_rows(o.singleton_rows, kRowSingleton);
  assign_rows(o.empty_rows, kRowEmpty);
  // Keep at least one ordinary column so that the structure below is non-trivial.
  for (int t = 0; t < o.empty_cols && cp + 1 < col_perm.size(); ++t) {
    col_role[static_cast<std::size_t>(col_perm[cp++])] = kColEmpty;
  }
  for (int t = 0; t < o.fixed_cols && cp + 1 < col_perm.size(); ++t) {
    col_role[static_cast<std::size_t>(col_perm[cp++])] = kColFixed;
  }

  std::vector<int> usable;  // columns that may carry entries in ordinary rows
  for (int j = 0; j < n; ++j) {
    if (col_role[static_cast<std::size_t>(j)] != kColEmpty) usable.push_back(j);
  }

  // ---- matrix entries, row by row ----
  std::vector<std::vector<std::pair<int, double>>> entries(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    auto& row = entries[static_cast<std::size_t>(i)];
    switch (row_role[static_cast<std::size_t>(i)]) {
      case kRowEmpty:
        break;
      case kRowSingleton:
        row.emplace_back(usable[static_cast<std::size_t>(rng.range(0, static_cast<int>(usable.size()) - 1))],
                         coefficient(rng, o.wide_coefficients));
        break;
      case kRowDoubleton: {
        std::vector<int> pool = usable;
        rng.shuffle(pool);
        for (int t = 0; t < 2 && t < static_cast<int>(pool.size()); ++t) {
          row.emplace_back(pool[static_cast<std::size_t>(t)], coefficient(rng, o.wide_coefficients));
        }
        break;
      }
      case kRowForcing:
        for (const Forcing& f : forcing) {
          if (f.row != i) continue;
          for (const int j : f.cols) row.emplace_back(j, coefficient(rng, o.wide_coefficients));
        }
        break;
      default:
        for (const int j : usable) {
          if (rng.chance(o.density)) row.emplace_back(j, coefficient(rng, o.wide_coefficients));
        }
        break;
    }
    std::sort(row.begin(), row.end());
  }

  // ---- columns: bounds, x*, d* ----
  std::vector<ColInfo> col(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    ColInfo& c = col[static_cast<std::size_t>(j)];
    const int role = col_role[static_cast<std::size_t>(j)];
    if (role == kColForcing) continue;  // set below
    if (role == kColFixed) {
      c.lo = c.up = c.x = grid(rng, -5.0, 5.0, 8);
      c.d = rng.chance(0.3) ? 0.0 : grid(rng, -3.0, 3.0, 16);
      continue;
    }
    const bool is_free = rng.chance(o.free_col_fraction);
    if (is_free) {
      c.lo = -kInf;
      c.up = kInf;
      c.x = grid(rng, -5.0, 5.0, 8);
      c.d = 0.0;
      continue;
    }
    const double kind = rng.unit();
    const double base = grid(rng, -5.0, 5.0, 8);
    const double width = grid(rng, 1.0, 6.0, 8);
    if (kind < 0.35) {  // lower only
      c.lo = base;
      c.up = kInf;
    } else if (kind < 0.5) {  // upper only
      c.lo = -kInf;
      c.up = base;
    } else {  // boxed
      c.lo = base;
      c.up = base + width;
    }
    const bool at_bound = rng.chance(o.active_fraction);
    if (at_bound) {
      const bool lower = !is_inf(c.lo) && (is_inf(c.up) || rng.chance(0.5));
      if (lower) {
        c.x = c.lo;
        c.d = multiplier(rng, o.degeneracy);
      } else {
        c.x = c.up;
        c.d = -multiplier(rng, o.degeneracy);
      }
    } else if (is_inf(c.lo)) {
      c.x = c.up - grid(rng, 0.5, 3.0, 8);
    } else if (is_inf(c.up)) {
      c.x = c.lo + grid(rng, 0.5, 3.0, 8);
    } else {
      c.x = c.lo + grid(rng, 0.15 * (c.up - c.lo), 0.85 * (c.up - c.lo), 8);
    }
  }
  for (const Forcing& f : forcing) {
    const auto& row = entries[static_cast<std::size_t>(f.row)];
    for (const int j : f.cols) {
      double a = 0.0;
      for (const auto& e : row) {
        if (e.first == j) a = e.second;
      }
      ColInfo& c = col[static_cast<std::size_t>(j)];
      c.lo = grid(rng, -4.0, 4.0, 8);
      c.up = c.lo + grid(rng, 1.0, 5.0, 8);
      const bool at_upper = f.max_forcing ? (a > 0) : (a < 0);
      c.x = at_upper ? c.up : c.lo;
      c.d = at_upper ? -multiplier(rng, o.degeneracy) : multiplier(rng, o.degeneracy);
    }
  }

  // ---- rows: activity, bounds, y* ----
  LpModel model;
  model.n_rows = m;
  model.n_cols = n;
  std::vector<double> y(static_cast<std::size_t>(m), 0.0);
  model.row_lower.assign(static_cast<std::size_t>(m), 0.0);
  model.row_upper.assign(static_cast<std::size_t>(m), 0.0);
  for (int i = 0; i < m; ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    double r = 0.0;
    for (const auto& e : entries[si]) r += e.second * col[static_cast<std::size_t>(e.first)].x;
    double lo = r;
    double up = r;
    double yi = 0.0;
    const int role = row_role[si];
    if (role == kRowForcing) {
      const Forcing* fr = nullptr;
      for (const Forcing& f : forcing) {
        if (f.row == i) fr = &f;
      }
      if (fr->equality) {
        yi = rng.chance(o.degeneracy) ? 0.0 : grid(rng, -3.0, 3.0, 16);
      } else if (fr->max_forcing) {
        up = kInf;
        yi = multiplier(rng, o.degeneracy);
      } else {
        lo = -kInf;
        yi = -multiplier(rng, o.degeneracy);
      }
    } else if (role == kRowDoubleton) {
      yi = rng.chance(o.degeneracy) ? 0.0 : grid(rng, -3.0, 3.0, 16);
    } else {
      const double kind = rng.unit();
      const bool active = rng.chance(o.active_fraction);
      const double slack1 = grid(rng, 0.3, 3.0, 8);
      const double slack2 = grid(rng, 0.3, 3.0, 8);
      if (rng.chance(o.free_row_fraction)) {
        lo = -kInf;
        up = kInf;
      } else if (rng.chance(o.ranged_row_fraction)) {
        if (active) {
          if (rng.chance(0.5)) {
            lo = r;
            up = r + slack2;
            yi = multiplier(rng, o.degeneracy);
          } else {
            lo = r - slack1;
            up = r;
            yi = -multiplier(rng, o.degeneracy);
          }
        } else {
          lo = r - slack1;
          up = r + slack2;
        }
      } else if (kind < 0.15) {  // equality
        yi = rng.chance(o.degeneracy) ? 0.0 : grid(rng, -3.0, 3.0, 16);
      } else if (kind < 0.575) {  // <=
        lo = -kInf;
        if (active) {
          yi = -multiplier(rng, o.degeneracy);
        } else {
          up = r + slack1;
        }
      } else {  // >=
        up = kInf;
        if (active) {
          yi = multiplier(rng, o.degeneracy);
        } else {
          lo = r - slack1;
        }
      }
    }
    model.row_lower[si] = lo;
    model.row_upper[si] = up;
    y[si] = yi;
  }

  // ---- assemble ----
  std::vector<Triplet> triplets;
  for (int i = 0; i < m; ++i) {
    for (const auto& e : entries[static_cast<std::size_t>(i)]) triplets.push_back({i, e.first, e.second});
  }
  std::string err;
  SparseMatrix::from_triplets(m, n, std::move(triplets), &model.A, &err);

  std::vector<double> d(static_cast<std::size_t>(n));
  std::vector<double> x(static_cast<std::size_t>(n));
  model.col_lower.resize(static_cast<std::size_t>(n));
  model.col_upper.resize(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    const std::size_t sj = static_cast<std::size_t>(j);
    model.col_lower[sj] = col[sj].lo;
    model.col_upper[sj] = col[sj].up;
    x[sj] = col[sj].x;
    d[sj] = col[sj].d;
  }
  std::vector<double> aty(static_cast<std::size_t>(n), 0.0);
  model.A.multiply_transpose(y, aty);
  model.col_cost.resize(static_cast<std::size_t>(n));
  double obj = 0.0;
  model.objective_offset = rng.chance(0.5) ? grid(rng, -5.0, 5.0, 8) : 0.0;
  for (int j = 0; j < n; ++j) {
    const std::size_t sj = static_cast<std::size_t>(j);
    model.col_cost[sj] = aty[sj] + d[sj];
    obj += model.col_cost[sj] * x[sj];
  }
  model.col_type.assign(static_cast<std::size_t>(n), ColType::Continuous);

  RandomLp out;
  out.model = std::move(model);
  out.known.x = std::move(x);
  out.known.y = std::move(y);
  out.known.d = std::move(d);
  out.known.objective = obj + out.model.objective_offset;
  return out;
}

namespace {

// Appends rows given as (coefficients over columns, lower, upper).
void append_row(LpModel* m, const std::vector<std::pair<int, double>>& coefs, double lo, double up) {
  std::vector<Triplet> t;
  for (Index j = 0; j < m->n_cols; ++j) {
    for (Index p = m->A.col_start[to_size(j)]; p < m->A.col_start[to_size(j) + 1]; ++p) {
      t.push_back({m->A.row_index[to_size(p)], j, m->A.value[to_size(p)]});
    }
  }
  for (const auto& c : coefs) t.push_back({m->n_rows, c.first, c.second});
  ++m->n_rows;
  std::string err;
  SparseMatrix::from_triplets(m->n_rows, m->n_cols, std::move(t), &m->A, &err);
  m->row_lower.push_back(lo);
  m->row_upper.push_back(up);
}

void append_col(LpModel* m, const std::vector<std::pair<int, double>>& coefs, double cost, double lo,
                double up) {
  std::vector<Triplet> t;
  for (Index j = 0; j < m->n_cols; ++j) {
    for (Index p = m->A.col_start[to_size(j)]; p < m->A.col_start[to_size(j) + 1]; ++p) {
      t.push_back({m->A.row_index[to_size(p)], j, m->A.value[to_size(p)]});
    }
  }
  for (const auto& c : coefs) t.push_back({c.first, m->n_cols, c.second});
  ++m->n_cols;
  std::string err;
  SparseMatrix::from_triplets(m->n_rows, m->n_cols, std::move(t), &m->A, &err);
  m->col_cost.push_back(cost);
  m->col_lower.push_back(lo);
  m->col_upper.push_back(up);
  m->col_type.push_back(ColType::Continuous);
}

}  // namespace

LpModel make_random_infeasible_lp(std::uint64_t seed, const RandomLpOptions& options,
                                  std::string* kind) {
  RandomLp lp = make_random_lp(seed, options);
  Rng rng(seed ^ 0xabcdef12345ULL);
  LpModel m = std::move(lp.model);
  const int variant = rng.range(0, 2);
  if (variant == 0 && m.n_rows > 0) {
    // Two rows with identical coefficients and contradictory ranges.
    std::vector<std::pair<int, double>> coefs;
    for (Index i0 = 0; i0 < m.n_rows && coefs.empty(); ++i0) {
      const Index i = static_cast<Index>((i0 + rng.range(0, std::max(m.n_rows - 1, 0))) % m.n_rows);
      for (Index j = 0; j < m.n_cols; ++j) {
        for (Index p = m.A.col_start[to_size(j)]; p < m.A.col_start[to_size(j) + 1]; ++p) {
          if (m.A.row_index[to_size(p)] == i) coefs.emplace_back(j, m.A.value[to_size(p)]);
        }
      }
    }
    if (!coefs.empty()) {
      double r = 0.0;
      for (const auto& c : coefs) r += c.second * lp.known.x[to_size(c.first)];
      append_row(&m, coefs, -kInf, r - 1.0);
      append_row(&m, coefs, r + 1.0, kInf);
      if (kind != nullptr) *kind = "contradictory duplicate rows";
      return m;
    }
  }
  if (variant == 1) {
    append_row(&m, {}, 1.0, 2.0);
    if (kind != nullptr) *kind = "empty row with 0 outside its range";
    return m;
  }
  const int j = rng.range(0, m.n_cols - 1);
  append_row(&m, {{j, 1.0}}, 5.0 + std::max(0.0, is_inf(m.col_upper[to_size(j)]) ? 0.0 : m.col_upper[to_size(j)]), kInf);
  append_row(&m, {{j, 1.0}}, -kInf, 3.0);
  if (kind != nullptr) *kind = "conflicting singleton rows";
  return m;
}

LpModel make_random_unbounded_lp(std::uint64_t seed, const RandomLpOptions& options,
                                 std::string* kind) {
  RandomLp lp = make_random_lp(seed, options);
  Rng rng(seed ^ 0x1234567abcULL);
  LpModel m = std::move(lp.model);
  if (rng.chance(0.5)) {
    // Ray along a column whose coefficients only ever relax one-sided rows.
    std::vector<std::pair<int, double>> coefs;
    for (Index i = 0; i < m.n_rows; ++i) {
      const bool lo_f = !is_inf(m.row_lower[to_size(i)]);
      const bool up_f = !is_inf(m.row_upper[to_size(i)]);
      if (lo_f == up_f) {
        if (!lo_f && rng.chance(0.5)) coefs.emplace_back(i, grid(rng, -2.0, 2.0, 8));  // free row
        continue;
      }
      if (rng.chance(0.7)) coefs.emplace_back(i, (lo_f ? 1.0 : -1.0) * grid(rng, 0.375, 2.0, 8));
    }
    if (!coefs.empty()) {
      append_col(&m, coefs, -1.0, 0.0, kInf);
      if (kind != nullptr) *kind = "column with an improving ray inside one-sided rows";
      return m;
    }
  }
  append_col(&m, {}, -1.0, rng.chance(0.5) ? 0.0 : -kInf, kInf);
  if (kind != nullptr) *kind = "empty column with an improving direction";
  return m;
}

}  // namespace shodhan::testing
