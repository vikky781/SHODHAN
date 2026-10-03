// Rank-deficient bases: detection of the deficient set and repair().

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "support/lu_testing.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

SparseMatrix make_matrix(Index rows, Index cols, const std::vector<Triplet>& t) {
  SparseMatrix A;
  std::string err;
  if (!SparseMatrix::from_triplets(rows, cols, t, &A, &err)) throw std::runtime_error(err);
  return A;
}

// The reported deficient set is correct when (1) rank(B) = m - k, (2) the basis
// without the deficient columns has the same rank, i.e. every deficient column is
// in the span of the others, and (3) it comes with k unpivoted rows.
void check_deficient_set(const SparseMatrix& A, const std::vector<Index>& basis, const BasisFactor& f,
                         Index expected_k) {
  const Index m = static_cast<Index>(basis.size());
  const std::vector<Index>& def = f.deficient_positions();
  CHECK_EQ(static_cast<Index>(def.size()), expected_k);
  CHECK_EQ(f.unpivoted_rows().size(), def.size());
  CHECK(std::is_sorted(def.begin(), def.end()));
  CHECK(std::is_sorted(f.unpivoted_rows().begin(), f.unpivoted_rows().end()));
  DenseMatrix B = basis_to_dense(A, basis);
  CHECK_EQ(dense_rank(B, 1e-10), m - expected_k);
  for (const Index p : def) {
    for (Index i = 0; i < m; ++i) B.at(i, p) = 0.0;
  }
  CHECK_EQ(dense_rank(B, 1e-10), m - expected_k);
}

// After repair: the basis is nonsingular and ftran/btran pass the residual test.
void check_repaired(const SparseMatrix& A, std::vector<Index> basis, BasisFactor& f) {
  const std::vector<Index> before = basis;
  const std::vector<Index> positions = f.deficient_positions();
  const std::vector<Index> rows = f.unpivoted_rows();
  const std::vector<BasisSubstitution> ch = f.repair(A, basis);
  REQUIRE(f.status() == FactorStatus::Ok);
  REQUIRE(f.valid());
  CHECK_EQ(ch.size(), positions.size());
  for (std::size_t k = 0; k < ch.size() && k < positions.size(); ++k) {
    CHECK_EQ(ch[k].position, positions[k]);
    CHECK_EQ(ch[k].old_var, before[to_size(positions[k])]);
    CHECK_EQ(ch[k].new_var, A.n_cols + rows[k]);  // the logical column of an unpivoted row
    CHECK_EQ(basis[to_size(ch[k].position)], ch[k].new_var);
  }
  CHECK_EQ(dense_rank(basis_to_dense(A, basis), 1e-10), static_cast<Index>(basis.size()));
  Rng rng(4242);
  const SolveCheckResult r = check_solves(f, A, basis, rng);
  CHECK(r.ok());
  if (!r.ok()) std::cerr << "  " << r.failure << "\n";
}

}  // namespace

TEST_CASE(lu_rankdef_zero_column) {
  // Column 1 is empty.
  const SparseMatrix A = make_matrix(3, 3, {{0, 0, 1}, {1, 0, 2}, {2, 2, 3}, {0, 2, 1}});
  const std::vector<Index> basis{0, 1, 2};
  BasisFactor f;
  CHECK(f.factorize(A, basis) == FactorStatus::RankDeficient);
  CHECK(!f.valid());
  CHECK(f.deficient_positions() == (std::vector<Index>{1}));
  check_deficient_set(A, basis, f, 1);
  check_repaired(A, basis, f);
}

TEST_CASE(lu_rankdef_explicit_zero_column) {
  const SparseMatrix A = make_matrix(3, 3, {{0, 0, 1}, {1, 0, 2}, {1, 1, 0.0}, {2, 1, 0.0}, {2, 2, 3}});
  const std::vector<Index> basis{0, 1, 2};
  BasisFactor f;
  CHECK(f.factorize(A, basis) == FactorStatus::RankDeficient);
  CHECK(f.deficient_positions() == (std::vector<Index>{1}));
  check_deficient_set(A, basis, f, 1);
  check_repaired(A, basis, f);
}

TEST_CASE(lu_rankdef_duplicate_columns) {
  // Columns 0 and 2 are equal; exactly one of them is reported.
  const SparseMatrix A = make_matrix(4, 4, {{0, 0, 2}, {1, 0, -1}, {3, 0, 4}, {1, 1, 5}, {2, 1, 1},
                                            {0, 2, 2}, {1, 2, -1}, {3, 2, 4}, {2, 3, 3}, {3, 3, 1}});
  const std::vector<Index> basis{0, 1, 2, 3};
  BasisFactor f;
  CHECK(f.factorize(A, basis) == FactorStatus::RankDeficient);
  REQUIRE(f.deficient_positions().size() == 1);
  const Index d = f.deficient_positions()[0];
  CHECK(d == 0 || d == 2);
  check_deficient_set(A, basis, f, 1);
  check_repaired(A, basis, f);
}

TEST_CASE(lu_rankdef_same_variable_twice_in_the_basis) {
  const SparseMatrix A = make_matrix(3, 3, {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {1, 0, 2}});
  for (const std::vector<Index>& basis : {std::vector<Index>{0, 1, 1}, std::vector<Index>{3, 0, 3}}) {
    BasisFactor f;
    CHECK(f.factorize(A, basis) == FactorStatus::RankDeficient);
    check_deficient_set(A, basis, f, 1);
    check_repaired(A, basis, f);
  }
}

TEST_CASE(lu_rankdef_linear_dependence) {
  // Column 2 = column 0 + 2 * column 1 (small integers: exact in floating point).
  const SparseMatrix A = make_matrix(4, 4, {{0, 0, 1}, {1, 0, 2}, {3, 0, 1},
                                            {1, 1, 1}, {2, 1, 3},
                                            {0, 2, 1}, {1, 2, 4}, {2, 2, 6}, {3, 2, 1},
                                            {0, 3, 1}, {3, 3, 2}});
  const std::vector<Index> basis{0, 1, 2, 3};
  BasisFactor f;
  CHECK(f.factorize(A, basis) == FactorStatus::RankDeficient);
  check_deficient_set(A, basis, f, 1);
  check_repaired(A, basis, f);
}

TEST_CASE(lu_rankdef_several_deficiencies) {
  // A duplicated pair of columns and a repeated logical column: 2 deficiencies.
  const SparseMatrix A = make_matrix(6, 4, {{0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {2, 2, 4}, {3, 3, 1}, {5, 3, 2}});
  // positions: 0: col0, 1: col1 (= col0), 2: col2, 3 and 4: the logical of row 3 twice, 5: col3.
  const std::vector<Index> basis{0, 1, 2, 4 + 3, 4 + 3, 3};
  BasisFactor f;
  CHECK(f.factorize(A, basis) == FactorStatus::RankDeficient);
  const Index rank = dense_rank(basis_to_dense(A, basis), 1e-10);
  check_deficient_set(A, basis, f, 6 - rank);
  CHECK(6 - rank >= 2);
  check_repaired(A, basis, f);
}

TEST_CASE(lu_rankdef_repair_is_deterministic_and_idempotent) {
  const SparseMatrix A = make_matrix(5, 5, {{0, 0, 1}, {1, 1, 1}, {0, 2, 1}, {1, 2, 1}, {2, 3, 2}, {3, 3, 1}});
  // Column 4 is empty and column 2 = column 0 + column 1.
  const std::vector<Index> basis{0, 1, 2, 3, 4};
  BasisFactor f1, f2;
  CHECK(f1.factorize(A, basis) == FactorStatus::RankDeficient);
  CHECK(f2.factorize(A, basis) == FactorStatus::RankDeficient);
  std::vector<Index> b1 = basis, b2 = basis;
  const auto c1 = f1.repair(A, b1);
  const auto c2 = f2.repair(A, b2);
  CHECK(b1 == b2);
  REQUIRE(c1.size() == c2.size());
  for (std::size_t k = 0; k < c1.size(); ++k) {
    CHECK_EQ(c1[k].position, c2[k].position);
    CHECK_EQ(c1[k].old_var, c2[k].old_var);
    CHECK_EQ(c1[k].new_var, c2[k].new_var);
  }
  CHECK(f1.dump().ints == f2.dump().ints);
  CHECK(f1.dump().reals == f2.dump().reals);
  // Repairing a valid factorization changes nothing.
  std::vector<Index> again = b1;
  CHECK(f1.repair(A, again).empty());
  CHECK(again == b1);
  CHECK(f1.valid());
}

TEST_CASE(lu_rankdef_random_structured_matrices) {
  // Random sparse bases with planted dependencies: duplicates and sums of two columns.
  int deficient_seen = 0;
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    Rng rng(seed * 977ULL);
    const Index m = rng.range(4, 30);
    std::vector<std::vector<double>> col(to_size(m), std::vector<double>(to_size(m), 0.0));
    for (Index j = 0; j < m; ++j) {
      col[to_size(j)][to_size(j)] = rng.chance(0.5) ? 1.0 : -2.0;
      for (Index i = 0; i < m; ++i)
        if (i != j && rng.chance(0.15)) col[to_size(j)][to_size(i)] = static_cast<double>(rng.range(-3, 3));
    }
    const Index planted = rng.range(0, 3);
    for (Index k = 0; k < planted; ++k) {
      const Index target = rng.range(0, m - 1), a = rng.range(0, m - 1), b = rng.range(0, m - 1);
      if (target == a || target == b) continue;
      for (Index i = 0; i < m; ++i) col[to_size(target)][to_size(i)] = col[to_size(a)][to_size(i)] + (rng.chance(0.5) ? col[to_size(b)][to_size(i)] : 0.0);
    }
    std::vector<Triplet> t;
    for (Index j = 0; j < m; ++j)
      for (Index i = 0; i < m; ++i)
        if (col[to_size(j)][to_size(i)] != 0.0) t.push_back({i, j, col[to_size(j)][to_size(i)]});
    const SparseMatrix A = make_matrix(m, m, t);
    std::vector<Index> basis(to_size(m));
    for (Index j = 0; j < m; ++j) basis[to_size(j)] = j;
    BasisFactor f;
    const FactorStatus st = f.factorize(A, basis);
    const Index rank = dense_rank(basis_to_dense(A, basis), 1e-10);
    if (st == FactorStatus::Ok) {
      if (rank != m) { std::cerr << "FAILING SEED " << seed << ": Ok but rank " << rank << " of " << m << "\n"; CHECK(false); }
      continue;
    }
    ++deficient_seen;
    if (static_cast<Index>(f.deficient_positions().size()) != m - rank) {
      std::cerr << "FAILING SEED " << seed << ": deficient " << f.deficient_positions().size() << " but rank " << rank << " of " << m << "\n";
      CHECK(false);
      continue;
    }
    check_deficient_set(A, basis, f, m - rank);
    check_repaired(A, basis, f);
  }
  CHECK(deficient_seen > 20);
}
