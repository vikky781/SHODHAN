// Sparse LDL^T: ordering, symbolic analysis and numeric factorization against dense oracles (docs/LDL.md).

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "shodhan/sparse_ldl.hpp"
#include "support/dense_ldl.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

struct Sym {
  Index n = 0;
  std::vector<std::vector<double>> dense;  // full symmetric
  SparseMatrix lower;
  std::vector<signed char> sign;
};

Sym from_dense(const std::vector<std::vector<double>>& m, const std::vector<signed char>& sign) {
  Sym s;
  s.n = static_cast<Index>(m.size());
  s.dense = m;
  s.sign = sign;
  std::vector<Triplet> t;
  for (Index j = 0; j < s.n; ++j) {
    for (Index i = j; i < s.n; ++i) {
      if (m[to_size(i)][to_size(j)] != 0.0) t.push_back({i, j, m[to_size(i)][to_size(j)]});
    }
  }
  std::string err;
  SparseMatrix::from_triplets(s.n, s.n, t, &s.lower, &err);
  return s;
}

// Random quasi-definite matrix [[H + shift, A^T], [A, -D]] with its rows randomly interleaved.
Sym random_quasi_definite(Rng& rng) {
  const int n1 = rng.range(1, 30), n2 = rng.range(0, 25), n = n1 + n2;
  const double density = rng.uniform(0.03, 0.4);
  std::vector<std::vector<double>> m(to_size(n), std::vector<double>(to_size(n), 0.0));
  // H = diagonal dominant symmetric positive definite.
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < i; ++j) {
      if (rng.chance(density)) m[to_size(i)][to_size(j)] = m[to_size(j)][to_size(i)] = rng.uniform(-1.0, 1.0);
    }
  }
  for (int i = 0; i < n1; ++i) {
    double s = 0.0;
    for (int j = 0; j < n1; ++j) s += std::fabs(m[to_size(i)][to_size(j)]);
    m[to_size(i)][to_size(i)] = s + rng.uniform(0.1, 2.0);
  }
  for (int i = 0; i < n2; ++i) {
    m[to_size(n1 + i)][to_size(n1 + i)] = -rng.uniform(0.05, 3.0);
    for (int j = 0; j < n1; ++j) {
      if (rng.chance(density)) m[to_size(n1 + i)][to_size(j)] = m[to_size(j)][to_size(n1 + i)] = rng.uniform(-2.0, 2.0);
    }
  }
  std::vector<signed char> sign(to_size(n));
  for (int i = 0; i < n; ++i) sign[to_size(i)] = i < n1 ? 1 : -1;
  // Random symmetric permutation.
  std::vector<int> p(to_size(n));
  std::iota(p.begin(), p.end(), 0);
  rng.shuffle(p);
  std::vector<std::vector<double>> pm(to_size(n), std::vector<double>(to_size(n)));
  std::vector<signed char> ps(to_size(n));
  for (int i = 0; i < n; ++i) {
    ps[to_size(i)] = sign[to_size(p[to_size(i)])];
    for (int j = 0; j < n; ++j) pm[to_size(i)][to_size(j)] = m[to_size(p[to_size(i)])][to_size(p[to_size(j)])];
  }
  return from_dense(pm, ps);
}

bool is_permutation(const std::vector<Index>& p, Index n) {
  if (p.size() != to_size(n)) return false;
  std::vector<char> seen(to_size(n), 0);
  for (const Index v : p) {
    if (v < 0 || v >= n || seen[to_size(v)]) return false;
    seen[to_size(v)] = 1;
  }
  return true;
}

// Pattern matrices for the fill table.
SparseMatrix lower_from_edges(Index n, const std::vector<std::pair<Index, Index>>& edges) {
  std::set<std::pair<Index, Index>> s;
  for (Index i = 0; i < n; ++i) s.insert({i, i});
  for (const auto& e : edges) s.insert({std::max(e.first, e.second), std::min(e.first, e.second)});
  std::vector<Triplet> t;
  for (const auto& e : s) t.push_back({e.first, e.second, e.first == e.second ? 4.0 : -1.0});
  SparseMatrix m;
  std::string err;
  SparseMatrix::from_triplets(n, n, t, &m, &err);
  return m;
}

long long fill_with(const SparseMatrix& lower, bool amd) {
  const std::vector<Index> order = amd ? amd_order(lower) : natural_order(lower.n_cols);
  return analyze_ldl(lower, order).nnz_l;
}

}  // namespace

TEST_CASE(amd_returns_a_permutation_and_reports_its_work) {
  Rng rng(3);
  for (int t = 0; t < 200; ++t) {
    const Sym s = random_quasi_definite(rng);
    AmdStats st;
    const std::vector<Index> p = amd_order(s.lower, &st);
    CHECK(is_permutation(p, s.n));
    CHECK(st.pivots >= 1 || s.n == 0);
  }
  // Degenerate inputs.
  SparseMatrix empty;
  CHECK(amd_order(empty).empty());
  const SparseMatrix diag = lower_from_edges(5, {});
  CHECK(is_permutation(amd_order(diag), 5));
}

TEST_CASE(symbolic_analysis_matches_a_dense_symbolic_elimination) {
  Rng rng(5);
  int checked = 0;
  for (int t = 0; t < 300; ++t) {
    const Sym s = random_quasi_definite(rng);
    std::vector<Index> order = natural_order(s.n);
    if (t % 3 == 1) order = amd_order(s.lower);
    if (t % 3 == 2) {
      std::vector<int> r(to_size(s.n));
      std::iota(r.begin(), r.end(), 0);
      rng.shuffle(r);
      for (Index i = 0; i < s.n; ++i) order[to_size(i)] = r[to_size(i)];
    }
    const SymbolicLdl sym = analyze_ldl(s.lower, order);
    CHECK(is_permutation(sym.perm, s.n));
    // Dense symbolic Cholesky of the permuted pattern by boolean elimination.
    const Index n = s.n;
    std::vector<std::vector<char>> a(to_size(n), std::vector<char>(to_size(n), 0));
    for (Index i = 0; i < n; ++i) {
      for (Index j = 0; j < n; ++j) {
        if (s.dense[to_size(sym.perm[to_size(i)])][to_size(sym.perm[to_size(j)])] != 0.0 || i == j) a[to_size(i)][to_size(j)] = 1;
      }
    }
    std::vector<std::vector<Index>> truth(to_size(n));
    for (Index k = 0; k < n; ++k) {
      std::vector<Index> rows;
      for (Index i = k + 1; i < n; ++i) {
        if (a[to_size(i)][to_size(k)]) rows.push_back(i);
      }
      for (const Index i : rows) {
        for (const Index j : rows) a[to_size(i)][to_size(j)] = 1;
      }
      truth[to_size(k)] = rows;
    }
    const std::vector<std::vector<Index>> pat = ldl_pattern(s.lower, sym);
    long long total = 0;
    bool same = true;
    for (Index k = 0; k < n; ++k) {
      same = same && pat[to_size(k)] == truth[to_size(k)] && sym.col_count[to_size(k)] == static_cast<Index>(truth[to_size(k)].size());
      total += static_cast<long long>(truth[to_size(k)].size());
      // The elimination tree parent is the first row of the column.
      const Index parent = truth[to_size(k)].empty() ? -1 : truth[to_size(k)].front();
      same = same && sym.parent[to_size(k)] == parent;
    }
    CHECK(same);
    CHECK_EQ(sym.nnz_l, total);
    ++checked;
  }
  std::cout << "  symbolic analysis: " << checked << " random patterns (natural, AMD and random orders) agree with a dense symbolic elimination on the column counts, the pattern and the elimination tree\n";
}

TEST_CASE(sparse_ldl_matches_the_dense_oracle_on_random_quasi_definite_matrices) {
  Rng rng(7);
  int solved = 0, refined = 0, regularized = 0;
  double worst_residual = 0.0, worst_diff = 0.0;
  for (int t = 0; t < 600; ++t) {
    const Sym s = random_quasi_definite(rng);
    SparseLdl ldl;
    const SparseLdl::Ordering ord = t % 2 == 0 ? SparseLdl::Ordering::Amd : SparseLdl::Ordering::Natural;
    REQUIRE(ldl.analyze(s.lower, s.sign, ord));
    REQUIRE(ldl.factorize(s.lower));
    std::vector<double> b(to_size(s.n));
    for (double& v : b) v = rng.uniform(-3.0, 3.0);
    std::vector<double> x(to_size(s.n)), truth;
    ldl.solve(b, x);
    std::vector<double> flat(to_size(s.n) * to_size(s.n));
    for (Index i = 0; i < s.n; ++i) {
      for (Index j = 0; j < s.n; ++j) flat[to_size(i) * to_size(s.n) + to_size(j)] = s.dense[to_size(i)][to_size(j)];
    }
    REQUIRE(dense_ldl_solve(s.n, flat, b, &truth));
    double diff = 0.0, xn = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      diff = std::max(diff, std::fabs(x[i] - truth[i]));
      xn = std::max(xn, std::fabs(truth[i]));
    }
    worst_residual = std::max(worst_residual, ldl.stats().residual);
    worst_diff = std::max(worst_diff, diff / (1.0 + xn));
    CHECK(ldl.stats().residual <= 1e-10);
    CHECK(diff <= 1e-8 * (1.0 + xn));
    if (ldl.stats().refinement_steps > 0) ++refined;
    if (ldl.stats().dynamic_regularizations > 0) ++regularized;
    ++solved;
  }
  std::cout << "  sparse LDL^T vs the dense LDL^T oracle: " << solved << " random quasi-definite matrices, worst relative residual " << worst_residual
            << ", worst relative difference to the oracle " << worst_diff << "; refinement took a step on " << refined << ", dynamic regularization on "
            << regularized << "\n";
}

TEST_CASE(sparse_ldl_dynamic_regularization_is_counted_and_refinement_recovers_accuracy) {
  // [[1, 1], [1, 1 - 1e-20]] has a zero pivot after the first elimination step (up to rounding): the second pivot is
  // replaced (counted); the system is still solved through the unregularized residuals as far as it is solvable.
  const std::vector<std::vector<double>> m = {{1.0, 1.0}, {1.0, 1.0}};
  const Sym s = from_dense(m, {1, 1});
  LdlParams exact;  // no static regularization: the zero pivot must be caught by the dynamic one
  exact.rho = exact.delta = 0.0;
  SparseLdl ldl(exact);
  REQUIRE(ldl.analyze(s.lower, s.sign, SparseLdl::Ordering::Natural));
  REQUIRE(ldl.factorize(s.lower));
  CHECK_EQ(ldl.stats().dynamic_regularizations, 1LL);
  // A consistent right-hand side (in the range of K): b = K (1, 2) = (3, 3). The regularized solve plus refinement
  // reaches a small residual.
  std::vector<double> x(2);
  const std::vector<double> rhs = {3.0, 3.0};
  ldl.solve(rhs, x);
  CHECK(ldl.stats().residual < 1e-8);
  // Without any perturbation needed the counter stays at zero.
  const Sym ok = from_dense({{2.0, 1.0}, {1.0, 3.0}}, {1, 1});
  SparseLdl good;
  REQUIRE(good.analyze(ok.lower, ok.sign));
  REQUIRE(good.factorize(ok.lower));
  CHECK_EQ(good.stats().dynamic_regularizations, 0LL);
}

TEST_CASE(sparse_ldl_refactorization_reuses_the_analysis) {
  Rng rng(9);
  Sym s = random_quasi_definite(rng);
  SparseLdl ldl;
  REQUIRE(ldl.analyze(s.lower, s.sign));
  for (int round = 0; round < 5; ++round) {
    for (double& v : s.lower.value) v *= rng.uniform(0.9, 1.1);  // same pattern, new values (keeps definiteness)
    REQUIRE(ldl.factorize(s.lower));
    std::vector<double> b(to_size(s.n), 1.0), x(to_size(s.n));
    ldl.solve(b, x);
    CHECK(ldl.stats().residual < 1e-10);
  }
  CHECK_EQ(ldl.stats().factorizations, 5LL);
  // A different number of entries is refused.
  SparseMatrix other = lower_from_edges(s.n, {});
  CHECK(!ldl.factorize(other));
}

TEST_CASE(amd_fill_versus_natural_order_is_reported) {
  struct Case {
    const char* name;
    SparseMatrix m;
  };
  std::vector<Case> cases;
  {  // arrow-head with the dense row and column FIRST: the natural order fills in completely
    const Index n = 200;
    std::vector<std::pair<Index, Index>> e;
    for (Index i = 1; i < n; ++i) e.push_back({i, 0});
    cases.push_back({"arrow-head (dense node first)", lower_from_edges(n, e)});
  }
  {  // banded
    const Index n = 400;
    std::vector<std::pair<Index, Index>> e;
    for (Index i = 0; i < n; ++i) {
      for (Index k = 1; k <= 4 && i + k < n; ++k) e.push_back({i + k, i});
    }
    cases.push_back({"banded (half-bandwidth 4)", lower_from_edges(n, e)});
  }
  {  // 2-D grid Laplacian, natural (row-major) order
    const Index g = 24, n = g * g;
    std::vector<std::pair<Index, Index>> e;
    for (Index r = 0; r < g; ++r) {
      for (Index c = 0; c < g; ++c) {
        if (c + 1 < g) e.push_back({r * g + c + 1, r * g + c});
        if (r + 1 < g) e.push_back({(r + 1) * g + c, r * g + c});
      }
    }
    cases.push_back({"grid Laplacian 24x24", lower_from_edges(n, e)});
  }
  {  // random sparse pattern
    Rng rng(21);
    const Index n = 300;
    std::vector<std::pair<Index, Index>> e;
    for (Index i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) e.push_back({std::max<Index>(i, rng.range(0, n - 1)), std::min<Index>(i, rng.range(0, n - 1))});
    }
    cases.push_back({"random (about 3 entries per row)", lower_from_edges(n, e)});
  }
  std::cout << "  fill nnz(L), natural order vs AMD (fill table, informational except the arrow-head):\n";
  for (const Case& c : cases) {
    const long long nat = fill_with(c.m, false), amd = fill_with(c.m, true);
    std::cout << "    " << c.name << ": n = " << c.m.n_cols << ", nnz(A lower) = " << c.m.nnz() << ", natural " << nat << ", AMD " << amd << "\n";
    if (std::string(c.name).rfind("arrow-head", 0) == 0) CHECK(amd <= nat);
  }
}
