#include "support/qp_families.hpp"

#include <algorithm>
#include <cmath>

#include "shodhan/quadratic.hpp"
#include "support/rng.hpp"

namespace shodhan::testing {

const char* qp_variant_name(QpVariant v) {
  switch (v) {
    case QpVariant::PositiveDefinite: return "positive definite";
    case QpVariant::Singular: return "singular Q";
    case QpVariant::Degenerate: return "degenerate";
    case QpVariant::FreeHeavy: return "free columns";
    case QpVariant::RangedHeavy: return "ranged rows";
    case QpVariant::EqualityHeavy: return "equality rows";
    case QpVariant::Wide: return "wide coefficients";
    case QpVariant::Lp: return "LP (Q = 0)";
    default: return "?";
  }
}

PlantedQp make_planted_qp(std::uint64_t seed, QpVariant variant) {
  Rng rng(seed * 6364136223846793005ULL + 1442695040888963407ULL);
  PlantedQp out;
  LpModel& m = out.model;
  const bool lp = variant == QpVariant::Lp;
  const int n = rng.range(2, 14);
  const int rows = rng.range(1, 10);
  m.n_cols = n;
  m.n_rows = rows;
  m.name = "qp" + std::to_string(seed);
  m.sense = rng.chance(0.25) ? Sense::Maximize : Sense::Minimize;
  const double sgn = m.sense == Sense::Maximize ? -1.0 : 1.0;  // the model is built in minimization form, then negated

  // ---- A ----
  std::vector<Triplet> a;
  const double density = rng.uniform(0.25, 0.8);
  for (int i = 0; i < rows; ++i) {
    int cnt = 0;
    for (int j = 0; j < n; ++j) {
      if (!rng.chance(density)) continue;
      double v = rng.uniform(0.3, 2.0) * (rng.chance(0.5) ? 1.0 : -1.0);
      if (variant == QpVariant::Wide) v *= std::pow(10.0, rng.uniform(-3.0, 3.0));
      a.push_back({i, j, v});
      ++cnt;
    }
    if (cnt == 0) a.push_back({i, rng.range(0, n - 1), 1.0});
  }
  std::string err;
  SparseMatrix::from_triplets(rows, n, a, &m.A, &err);

  // ---- Q = L L^T (lower triangle) ----
  std::vector<double> qdense(to_size(n) * to_size(n), 0.0);
  if (!lp) {
    int rank = n;
    if (variant == QpVariant::Singular) rank = rng.range(1, std::max(1, n - 1));
    std::vector<std::vector<double>> l(to_size(n), std::vector<double>(to_size(rank), 0.0));
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < rank; ++k) {
        if (rng.chance(0.5)) l[to_size(i)][to_size(k)] = rng.uniform(-1.5, 1.5);
      }
      if (variant != QpVariant::Singular && i < rank) l[to_size(i)][to_size(i)] += rng.uniform(0.3, 1.0);
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int k = 0; k < rank; ++k) s += l[to_size(i)][to_size(k)] * l[to_size(j)][to_size(k)];
        qdense[to_size(i) * to_size(n) + to_size(j)] = qdense[to_size(j) * to_size(n) + to_size(i)] = s;
      }
    }
    std::vector<Triplet> q;
    for (int j = 0; j < n; ++j) {
      for (int i = j; i < n; ++i) {
        const double v = qdense[to_size(i) * to_size(n) + to_size(j)];
        if (v != 0.0) q.push_back({i, j, v});
      }
    }
    if (!q.empty()) SparseMatrix::from_triplets(n, n, q, &m.quadratic, &err);
  }

  // ---- planted x*, d* with complementarity ----
  const double p_free = variant == QpVariant::FreeHeavy ? 0.5 : 0.1;
  const double p_zero_mult = variant == QpVariant::Degenerate ? 0.7 : 0.1;
  out.x.assign(to_size(n), 0.0);
  out.d.assign(to_size(n), 0.0);
  m.col_lower.assign(to_size(n), 0.0);
  m.col_upper.assign(to_size(n), kInf);
  m.col_type.assign(to_size(n), ColType::Continuous);
  for (int j = 0; j < n; ++j) {
    const double u = rng.unit();
    if (u < p_free) {  // free column
      out.x[to_size(j)] = rng.uniform(-3.0, 3.0);
      m.col_lower[to_size(j)] = -kInf;
      m.col_upper[to_size(j)] = kInf;
    } else if (u < p_free + 0.3) {  // at its lower bound
      const double lb = std::round(rng.uniform(-3.0, 3.0) * 4.0) / 4.0;
      out.x[to_size(j)] = lb;
      m.col_lower[to_size(j)] = lb;
      m.col_upper[to_size(j)] = rng.chance(0.4) ? kInf : lb + rng.uniform(1.0, 4.0);
      out.d[to_size(j)] = rng.chance(p_zero_mult) ? 0.0 : rng.uniform(0.1, 2.0);
    } else if (u < p_free + 0.45) {  // at its upper bound
      const double ub = std::round(rng.uniform(-3.0, 3.0) * 4.0) / 4.0;
      out.x[to_size(j)] = ub;
      m.col_upper[to_size(j)] = ub;
      m.col_lower[to_size(j)] = rng.chance(0.4) ? -kInf : ub - rng.uniform(1.0, 4.0);
      out.d[to_size(j)] = rng.chance(p_zero_mult) ? 0.0 : -rng.uniform(0.1, 2.0);
    } else if (u < p_free + 0.5) {  // fixed
      const double v = rng.uniform(-2.0, 2.0);
      out.x[to_size(j)] = v;
      m.col_lower[to_size(j)] = m.col_upper[to_size(j)] = v;
      out.d[to_size(j)] = rng.uniform(-1.0, 1.0);
    } else {  // interior
      const double v = rng.uniform(-3.0, 3.0);
      out.x[to_size(j)] = v;
      m.col_lower[to_size(j)] = rng.chance(0.3) ? -kInf : v - rng.uniform(0.5, 3.0);
      m.col_upper[to_size(j)] = rng.chance(0.3) ? kInf : v + rng.uniform(0.5, 3.0);
    }
  }
  // ---- rows around the activity of x* ----
  std::vector<double> act(to_size(rows), 0.0);
  m.A.multiply(out.x, act);
  out.y.assign(to_size(rows), 0.0);
  m.row_lower.assign(to_size(rows), -kInf);
  m.row_upper.assign(to_size(rows), kInf);
  const double p_eq = variant == QpVariant::EqualityHeavy ? 0.7 : 0.15;
  const double p_inactive = variant == QpVariant::RangedHeavy ? 0.5 : 0.25;
  for (int i = 0; i < rows; ++i) {
    const double u = rng.unit();
    const double ai = act[to_size(i)];
    if (u < p_eq) {  // equality
      m.row_lower[to_size(i)] = m.row_upper[to_size(i)] = ai;
      out.y[to_size(i)] = rng.chance(p_zero_mult * 0.5) ? 0.0 : rng.uniform(-2.0, 2.0);
    } else if (u < p_eq + 0.3) {  // active at the lower side
      m.row_lower[to_size(i)] = ai;
      m.row_upper[to_size(i)] = rng.chance(0.5) ? kInf : ai + rng.uniform(0.5, 3.0);
      out.y[to_size(i)] = rng.chance(p_zero_mult) ? 0.0 : rng.uniform(0.1, 2.0);
    } else if (u < p_eq + 0.5) {  // active at the upper side
      m.row_upper[to_size(i)] = ai;
      m.row_lower[to_size(i)] = rng.chance(0.5) ? -kInf : ai - rng.uniform(0.5, 3.0);
      out.y[to_size(i)] = rng.chance(p_zero_mult) ? 0.0 : -rng.uniform(0.1, 2.0);
    } else if (u < p_eq + 0.5 + p_inactive) {  // inactive, ranged or one-sided
      m.row_lower[to_size(i)] = rng.chance(0.3) ? -kInf : ai - rng.uniform(0.5, 3.0);
      m.row_upper[to_size(i)] = rng.chance(0.3) ? kInf : ai + rng.uniform(0.5, 3.0);
    }  // else a free row
  }
  // ---- c = -Q x* + A^T y* + d* ----
  std::vector<double> qx(to_size(n), 0.0), aty(to_size(n), 0.0);
  if (!lp) quad_multiply(m.quadratic, out.x, qx);
  m.A.multiply_transpose(out.y, aty);
  m.col_cost.assign(to_size(n), 0.0);
  for (int j = 0; j < n; ++j) m.col_cost[to_size(j)] = -qx[to_size(j)] + aty[to_size(j)] + out.d[to_size(j)];
  m.objective_offset = rng.uniform(-3.0, 3.0);
  // The planted point is optimal for this minimization problem; for a maximization model negate c, Q and the offset.
  double obj = m.objective_offset;
  for (int j = 0; j < n; ++j) obj += m.col_cost[to_size(j)] * out.x[to_size(j)];
  if (!lp) obj += 0.5 * quad_form(m.quadratic, out.x);
  out.objective = sgn * obj;
  if (m.sense == Sense::Maximize) {
    for (double& v : m.col_cost) v = -v;
    for (double& v : m.quadratic.value) v = -v;
    m.objective_offset = -m.objective_offset;
  }
  return out;
}

LpModel make_infeasible_qp(std::uint64_t seed, std::string* kind) {
  PlantedQp p = make_planted_qp(seed, QpVariant::PositiveDefinite);
  LpModel m = p.model;
  // Append two contradictory copies of row 0: a x >= big and a x <= big - 1.
  std::vector<Triplet> t;
  for (Index j = 0; j < m.n_cols; ++j) {
    for (Index q = m.A.col_start[to_size(j)]; q < m.A.col_start[to_size(j) + 1]; ++q) t.push_back({m.A.row_index[to_size(q)], j, m.A.value[to_size(q)]});
  }
  const Index r0 = m.n_rows, r1 = m.n_rows + 1;
  for (Index j = 0; j < m.n_cols; ++j) {
    const double v = m.A.col_start[to_size(j) + 1] > m.A.col_start[to_size(j)] ? m.A.value[to_size(m.A.col_start[to_size(j)])] : 0.0;
    if (v != 0.0 && m.A.row_index[to_size(m.A.col_start[to_size(j)])] == 0) {
      t.push_back({r0, j, v});
      t.push_back({r1, j, v});
    }
  }
  // Guarantee at least one entry: use column 0 coefficient 1 if row 0 has none in the scan above.
  bool any = false;
  for (const Triplet& tr : t) any = any || tr.row == r0;
  if (!any) {
    t.push_back({r0, 0, 1.0});
    t.push_back({r1, 0, 1.0});
  }
  m.n_rows += 2;
  m.row_lower.push_back(1e3);
  m.row_upper.push_back(kInf);
  m.row_lower.push_back(-kInf);
  m.row_upper.push_back(1e3 - 1.0);
  // The columns are bounded far below 1e3 only if their bounds say so; make the contradiction independent of the data.
  std::string err;
  SparseMatrix::from_triplets(m.n_rows, m.n_cols, t, &m.A, &err);
  if (kind != nullptr) *kind = "contradictory row copies";
  return m;
}

LpModel make_unbounded_qp(std::uint64_t seed, std::string* kind) {
  PlantedQp p = make_planted_qp(seed, QpVariant::Singular);
  LpModel m = p.model;
  const Index j = m.n_cols;
  m.n_cols += 1;
  m.col_cost.push_back(m.sense == Sense::Maximize ? 1.0 : -1.0);  // improving in the model's own sense
  m.col_lower.push_back(0.0);
  m.col_upper.push_back(kInf);
  m.col_type.push_back(ColType::Continuous);
  std::vector<Triplet> t;
  for (Index c = 0; c < m.A.n_cols; ++c) {
    for (Index q = m.A.col_start[to_size(c)]; q < m.A.col_start[to_size(c) + 1]; ++q) t.push_back({m.A.row_index[to_size(q)], c, m.A.value[to_size(q)]});
  }
  std::string err;
  SparseMatrix::from_triplets(m.n_rows, m.n_cols, t, &m.A, &err);
  if (m.quadratic.nnz() > 0) {
    std::vector<Triplet> q;
    for (Index c = 0; c < m.quadratic.n_cols; ++c) {
      for (Index k = m.quadratic.col_start[to_size(c)]; k < m.quadratic.col_start[to_size(c) + 1]; ++k) {
        q.push_back({m.quadratic.row_index[to_size(k)], c, m.quadratic.value[to_size(k)]});
      }
    }
    SparseMatrix::from_triplets(m.n_cols, m.n_cols, q, &m.quadratic, &err);
  }
  (void)j;
  if (kind != nullptr) *kind = "extra free-direction column";
  return m;
}

}  // namespace shodhan::testing
