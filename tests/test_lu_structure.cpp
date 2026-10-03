// Structure of the factorization: pivot classification and sizes. These tests
// only factorize; the solves are tested in test_lu_exact.cpp and test_lu_property.cpp.

#include <string>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "test_harness.hpp"

using namespace shodhan;

namespace {

SparseMatrix make_matrix(Index rows, Index cols, const std::vector<Triplet>& t) {
  SparseMatrix A;
  std::string err;
  if (!SparseMatrix::from_triplets(rows, cols, t, &A, &err)) throw std::runtime_error(err);
  return A;
}

}  // namespace

TEST_CASE(lu_structure_identity_is_all_singletons) {
  std::vector<Triplet> t;
  for (Index i = 0; i < 6; ++i) t.push_back({i, i, 1.0});
  const SparseMatrix A = make_matrix(6, 6, t);
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1, 2, 3, 4, 5}) == FactorStatus::Ok);
  const FactorStats s = f.stats();
  CHECK_EQ(s.m, 6);
  CHECK_EQ(s.nnz_basis, std::size_t{6});
  CHECK_EQ(s.nnz_l, std::size_t{0});
  CHECK_EQ(s.nnz_u, std::size_t{6});
  CHECK_EQ(s.singleton_pivots, 6);
  CHECK_EQ(s.markowitz_pivots, 0);
  CHECK_NEAR(s.fill_ratio, 1.0, 1e-15);
  CHECK_EQ(s.updates, 0);
}

TEST_CASE(lu_structure_logical_basis_counts_singletons) {
  const SparseMatrix A = make_matrix(3, 2, {{0, 0, 1}, {1, 1, 2}, {2, 0, 3}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {2, 3, 4}) == FactorStatus::Ok);
  CHECK_EQ(f.stats().singleton_pivots, 3);
  CHECK_EQ(f.stats().markowitz_pivots, 0);
  CHECK_EQ(f.nnz_l(), std::size_t{0});
}

TEST_CASE(lu_structure_triangular_basis_needs_no_markowitz_pivot_and_no_fill) {
  // Lower bidiagonal-plus: column j has entries in rows j and j+1, j+2 (when present).
  const Index m = 12;
  std::vector<Triplet> t;
  for (Index j = 0; j < m; ++j) {
    t.push_back({j, j, 2.0});
    if (j + 1 < m) t.push_back({j + 1, j, 1.0});
    if (j + 3 < m) t.push_back({j + 3, j, -0.5});
  }
  const SparseMatrix A = make_matrix(m, m, t);
  std::vector<Index> basis(to_size(m));
  for (Index j = 0; j < m; ++j) basis[to_size(j)] = j;
  BasisFactor f;
  REQUIRE(f.factorize(A, basis) == FactorStatus::Ok);
  const FactorStats s = f.stats();
  CHECK_EQ(s.markowitz_pivots, 0);
  CHECK_EQ(s.singleton_pivots, m);
  CHECK_EQ(s.nnz_basis, A.nnz());
  CHECK_EQ(s.nnz_l + s.nnz_u, A.nnz());  // triangular: L and U together hold exactly B, no fill
  CHECK_NEAR(s.fill_ratio, 1.0, 1e-15);
}

TEST_CASE(lu_structure_dense_nucleus_uses_markowitz_pivots) {
  // Symmetric dense 3x3 (no entry cancels during elimination): no singletons at
  // the start. After the first pivot the 2x2 nucleus is dense again; the last 1x1
  // is a singleton.
  const SparseMatrix A = make_matrix(3, 3, {{0, 0, 4}, {1, 0, 1}, {2, 0, 2}, {0, 1, 1}, {1, 1, 5}, {2, 1, 3}, {0, 2, 2}, {1, 2, 3}, {2, 2, 7}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1, 2}) == FactorStatus::Ok);
  const FactorStats s = f.stats();
  CHECK_EQ(s.markowitz_pivots, 2);
  CHECK_EQ(s.singleton_pivots, 1);
  CHECK_EQ(s.nnz_l, std::size_t{3});
  CHECK_EQ(s.nnz_u, std::size_t{6});
  CHECK_NEAR(s.fill_ratio, 1.0, 1e-15);
}

TEST_CASE(lu_structure_threshold_pivoting_avoids_a_tiny_pivot) {
  // Row 0 has the smaller entry in column 0; with the default threshold the
  // pivot of column 0 must be the entry of magnitude 1 and not the one of 1e-3
  // (relative threshold 0.1 against the column maximum). The 2x2 matrix is dense
  // so the Markowitz counts do not decide.
  const SparseMatrix A = make_matrix(2, 2, {{0, 0, 1e-3}, {1, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1}) == FactorStatus::Ok);
  // The multiplier of the eta is a/pivot: with the pivot 1 it is 1e-3, never 1e3.
  const FactorDump d = f.dump();
  double max_abs_real = 0.0;
  for (const double v : d.reals) max_abs_real = std::max(max_abs_real, std::fabs(v));
  CHECK(max_abs_real <= 1.0 + 1e-12);
  // With threshold 0 the tiny entry may be chosen and the multiplier blows up.
  FactorParams loose;
  loose.pivot_threshold = 0.0;
  loose.markowitz_search = 4;
  BasisFactor g(loose);
  REQUIRE(g.factorize(A, {0, 1}) == FactorStatus::Ok);
}

TEST_CASE(lu_structure_exact_cancellation_is_dropped_not_stored) {
  // [[4,1,2],[1,5,3],[2,3,6]]: eliminating with the pivot 6 cancels the entry
  // (0,1) exactly (1 - (2/6)*3 = 0), so L and U together hold fewer than 9 entries.
  const SparseMatrix A = make_matrix(3, 3, {{0, 0, 4}, {1, 0, 1}, {2, 0, 2}, {0, 1, 1}, {1, 1, 5}, {2, 1, 3}, {0, 2, 2}, {1, 2, 3}, {2, 2, 6}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1, 2}) == FactorStatus::Ok);
  CHECK(f.stats().nnz_l + f.stats().nnz_u < 9);
}

TEST_CASE(lu_structure_markowitz_search_parameter_is_accepted) {
  const SparseMatrix A = make_matrix(3, 3, {{0, 0, 4}, {1, 0, 1}, {2, 0, 2}, {0, 1, 1}, {1, 1, 5}, {2, 1, 3}, {0, 2, 2}, {1, 2, 3}, {2, 2, 6}});
  for (const Index search : {1, 2, 4, 10}) {
    FactorParams p;
    p.markowitz_search = search;
    BasisFactor f(p);
    CHECK(f.factorize(A, {0, 1, 2}) == FactorStatus::Ok);
    CHECK_EQ(f.params().markowitz_search, search);
  }
}

TEST_CASE(lu_structure_growth_indicator) {
  // Identity: no growth, every pivot is as large as its column.
  std::vector<Triplet> t;
  for (Index i = 0; i < 4; ++i) t.push_back({i, i, 2.0});
  const SparseMatrix I = make_matrix(4, 4, t);
  BasisFactor f;
  REQUIRE(f.factorize(I, {0, 1, 2, 3}) == FactorStatus::Ok);
  FactorStats s = f.stats();
  CHECK_NEAR(s.max_abs_basis, 2.0, 0.0);
  CHECK_NEAR(s.max_abs_u, 2.0, 0.0);
  CHECK_NEAR(s.growth, 1.0, 1e-15);
  CHECK_NEAR(s.min_pivot_ratio, 1.0, 1e-15);
  // Upper triangular with a large off-diagonal: U holds the entry, so growth is large
  // and the pivot of the second column is small relative to its column maximum.
  const SparseMatrix U = make_matrix(2, 2, {{0, 0, 1.0}, {0, 1, 100.0}, {1, 1, 0.5}});
  REQUIRE(f.factorize(U, {0, 1}) == FactorStatus::Ok);
  s = f.stats();
  CHECK_NEAR(s.max_abs_basis, 100.0, 0.0);
  CHECK_NEAR(s.max_abs_u, 100.0, 0.0);
  CHECK_NEAR(s.growth, 1.0, 1e-15);
  CHECK_NEAR(s.min_pivot_ratio, 0.005, 1e-15);
  // An empty basis has no indicator.
  const SparseMatrix E = make_matrix(0, 0, {});
  REQUIRE(f.factorize(E, {}) == FactorStatus::Ok);
  CHECK_NEAR(f.stats().growth, 0.0, 0.0);
}

TEST_CASE(lu_structure_growth_indicator_follows_updates) {
  // B = I; the entering column (1, 1e3) replaces position 0: U gets the entry 1e3.
  const SparseMatrix A = make_matrix(2, 3, {{0, 0, 1.0}, {1, 1, 1.0}, {0, 2, 1.0}, {1, 2, 1000.0}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1}) == FactorStatus::Ok);
  CHECK_NEAR(f.stats().max_abs_u, 1.0, 0.0);
  SparseWork w(2);
  w.set(0, 1.0);
  w.set(1, 1000.0);
  f.ftran(w, true);
  REQUIRE(f.update(1) == FactorStatus::Ok);
  CHECK(f.stats().max_abs_u >= 1000.0);
  CHECK(f.stats().growth >= 1000.0 - 1e-9);
}
