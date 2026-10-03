#include <cmath>
#include <stdexcept>
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

// Solves B x = a and checks x against `expect`.
void expect_ftran(BasisFactor& f, const std::vector<double>& a, const std::vector<double>& expect,
                  double tol = 1e-13) {
  SparseWork w;
  w.load_dense(a);
  f.ftran(w);
  std::vector<double> x(a.size());
  w.to_dense(x);
  for (std::size_t i = 0; i < a.size(); ++i) CHECK_NEAR(x[i], expect[i], tol);
}

void expect_btran(BasisFactor& f, const std::vector<double>& c, const std::vector<double>& expect,
                  double tol = 1e-13) {
  SparseWork w;
  w.load_dense(c);
  f.btran(w);
  std::vector<double> y(c.size());
  w.to_dense(y);
  for (std::size_t i = 0; i < c.size(); ++i) CHECK_NEAR(y[i], expect[i], tol);
}

// Factorizes and runs the generic checker (residual + oracle) too.
void check_basis(const SparseMatrix& A, const std::vector<Index>& basis, FactorStatus expected = FactorStatus::Ok) {
  BasisFactor f;
  const FactorStatus st = f.factorize(A, basis);
  REQUIRE(st == expected);
  if (st != FactorStatus::Ok) return;
  Rng rng(12345);
  const SolveCheckResult r = check_solves(f, A, basis, rng);
  CHECK(r.ok());
  if (!r.ok()) std::cerr << "  " << r.failure << "\n";
}

}  // namespace

TEST_CASE(lu_identity) {
  const SparseMatrix A = make_matrix(4, 4, {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1, 2, 3}) == FactorStatus::Ok);
  expect_ftran(f, {1, -2, 3, 4}, {1, -2, 3, 4}, 0.0);
  expect_btran(f, {5, 6, 7, 8}, {5, 6, 7, 8}, 0.0);
  const FactorStats s = f.stats();
  CHECK_EQ(s.nnz_l, std::size_t{0});
  CHECK_EQ(s.nnz_u, std::size_t{4});
  CHECK_EQ(s.markowitz_pivots, 0);
  CHECK_EQ(s.singleton_pivots, 4);
  check_basis(A, {0, 1, 2, 3});
}

TEST_CASE(lu_permuted_identity_with_basis_order) {
  const SparseMatrix A = make_matrix(4, 4, {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}});
  const std::vector<Index> basis{2, 0, 3, 1};  // B = [e2 e0 e3 e1]
  BasisFactor f;
  REQUIRE(f.factorize(A, basis) == FactorStatus::Ok);
  // B x = a  =>  x[0]=a[2], x[1]=a[0], x[2]=a[3], x[3]=a[1].
  expect_ftran(f, {10, 20, 30, 40}, {30, 10, 40, 20}, 0.0);
  // B^T y = c  =>  y[2]=c[0], y[0]=c[1], y[3]=c[2], y[1]=c[3].
  expect_btran(f, {10, 20, 30, 40}, {20, 40, 10, 30}, 0.0);
  check_basis(A, basis);
}

TEST_CASE(lu_all_logical_basis_is_minus_identity) {
  const SparseMatrix A = make_matrix(3, 2, {{0, 0, 1}, {1, 1, 2}, {2, 0, 3}});
  const std::vector<Index> basis{2, 3, 4};  // logicals of rows 0, 1, 2 (n = 2)
  BasisFactor f;
  REQUIRE(f.factorize(A, basis) == FactorStatus::Ok);
  expect_ftran(f, {1, 2, 3}, {-1, -2, -3}, 0.0);
  expect_btran(f, {1, 2, 3}, {-1, -2, -3}, 0.0);
  const std::vector<Index> shuffled{4, 2, 3};  // columns -e2, -e0, -e1
  REQUIRE(f.factorize(A, shuffled) == FactorStatus::Ok);
  expect_ftran(f, {1, 2, 3}, {-3, -1, -2}, 0.0);  // x[0]=-a[2], x[1]=-a[0], x[2]=-a[1]
  check_basis(A, shuffled);
}

TEST_CASE(lu_triangular_bases) {
  // Lower triangular [[2,0,0],[1,3,0],[4,5,6]], x = (1,2,3) -> a = (2,7,32).
  const SparseMatrix L = make_matrix(3, 3, {{0, 0, 2}, {1, 0, 1}, {2, 0, 4}, {1, 1, 3}, {2, 1, 5}, {2, 2, 6}});
  BasisFactor f;
  REQUIRE(f.factorize(L, {0, 1, 2}) == FactorStatus::Ok);
  expect_ftran(f, {2, 7, 32}, {1, 2, 3});
  // B^T y = c with y = (1,-1,2): B^T = [[2,1,4],[0,3,5],[0,0,6]] -> c = (2-1+8, -3+10, 12) = (9,7,12).
  expect_btran(f, {9, 7, 12}, {1, -1, 2});
  check_basis(L, {0, 1, 2});
  // Upper triangular [[2,1,4],[0,3,5],[0,0,6]].
  const SparseMatrix U = make_matrix(3, 3, {{0, 0, 2}, {0, 1, 1}, {1, 1, 3}, {0, 2, 4}, {1, 2, 5}, {2, 2, 6}});
  REQUIRE(f.factorize(U, {0, 1, 2}) == FactorStatus::Ok);
  // x = (1,2,3): a = (2+2+12, 6+15, 18) = (16, 21, 18).
  expect_ftran(f, {16, 21, 18}, {1, 2, 3});
  check_basis(U, {0, 1, 2});
  // Fully triangular bases need no Markowitz pivot.
  CHECK_EQ(f.stats().markowitz_pivots, 0);
  CHECK_EQ(f.stats().nnz_l, std::size_t{0});
}

TEST_CASE(lu_3x3_known_inverse) {
  // B = [[2,1,0],[1,3,1],[0,1,4]], det = 2*(12-1) - 1*(4-0) = 18.
  // inverse = 1/18 * [[11,-4,1],[-4,8,-2],[1,-2,5]].
  const SparseMatrix A = make_matrix(3, 3, {{0, 0, 2}, {1, 0, 1}, {0, 1, 1}, {1, 1, 3}, {2, 1, 1}, {1, 2, 1}, {2, 2, 4}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1, 2}) == FactorStatus::Ok);
  const double inv[3][3] = {{11, -4, 1}, {-4, 8, -2}, {1, -2, 5}};
  for (int j = 0; j < 3; ++j) {
    std::vector<double> e(3, 0.0);
    e[static_cast<std::size_t>(j)] = 1.0;
    // B is symmetric, so ftran and btran of e_j both give column j of the inverse.
    const std::vector<double> col{inv[0][j] / 18.0, inv[1][j] / 18.0, inv[2][j] / 18.0};
    expect_ftran(f, e, col, 1e-14);
    expect_btran(f, e, col, 1e-14);
  }
  check_basis(A, {0, 1, 2});
}

TEST_CASE(lu_4x4_with_fill_and_known_solution) {
  // B = [[4,1,0,2],[0,3,1,0],[1,0,2,1],[0,2,0,5]] (needs pivoting choices and fill-in).
  const SparseMatrix A = make_matrix(4, 4,
      {{0, 0, 4}, {2, 0, 1}, {0, 1, 1}, {1, 1, 3}, {3, 1, 2}, {1, 2, 1}, {2, 2, 2}, {0, 3, 2}, {2, 3, 1}, {3, 3, 5}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1, 2, 3}) == FactorStatus::Ok);
  const std::vector<double> x{1, -1, 2, 3};
  // a = B x = (4-1+6, -3+2, 1+4+3, -2+15) = (9, -1, 8, 13).
  expect_ftran(f, {9, -1, 8, 13}, x);
  // B^T y = c for y = (2, 0, -1, 1), with c formed by the product helper.
  const std::vector<double> y{2, 0, -1, 1};
  const std::vector<double> c = basis_multiply_transpose(A, {0, 1, 2, 3}, y);
  expect_btran(f, c, y);
  check_basis(A, {0, 1, 2, 3});
}

TEST_CASE(lu_basis_mixing_structural_and_logical_columns) {
  // A = [[1,2],[3,4],[0,5]], n = 2, m = 3. Basis {col 1, logical row 0, col 0}.
  const SparseMatrix A = make_matrix(3, 2, {{0, 0, 1}, {1, 0, 3}, {0, 1, 2}, {1, 1, 4}, {2, 1, 5}});
  const std::vector<Index> basis{1, 2, 0};
  // B columns: (2,4,5), (-1,0,0), (1,3,0).
  BasisFactor f;
  REQUIRE(f.factorize(A, basis) == FactorStatus::Ok);
  const std::vector<double> x{1.0, 2.0, -1.0};
  const std::vector<double> a = basis_multiply(A, basis, x);
  expect_ftran(f, a, x);
  const std::vector<double> y{0.5, -2.0, 1.0};
  expect_btran(f, basis_multiply_transpose(A, basis, y), y);
  check_basis(A, basis);
}

TEST_CASE(lu_signed_permutation) {
  // B = signed permutation matrix built from structural +/-1 entries and logicals.
  const SparseMatrix A = make_matrix(4, 3, {{1, 0, -1}, {3, 1, 1}, {0, 2, -1}});
  const std::vector<Index> basis{0, 1, 2, 3 + 2};  // -e1... , e3, -e0, logical of row 2 (-e2)
  BasisFactor f;
  REQUIRE(f.factorize(A, basis) == FactorStatus::Ok);
  const std::vector<double> a{1, 2, 3, 4};
  const DenseMatrix B = basis_to_dense(A, basis);
  const DenseLu lu(B);
  expect_ftran(f, a, lu.solve(a), 0.0);
  expect_btran(f, a, lu.solve_transpose(a), 0.0);
  CHECK_EQ(f.stats().markowitz_pivots, 0);
  CHECK_EQ(f.stats().nnz_l, std::size_t{0});
  check_basis(A, basis);
}

TEST_CASE(lu_dimension_one_and_zero) {
  const SparseMatrix A1 = make_matrix(1, 1, {{0, 0, 4.0}});
  BasisFactor f;
  REQUIRE(f.factorize(A1, {0}) == FactorStatus::Ok);
  expect_ftran(f, {8.0}, {2.0}, 0.0);
  expect_btran(f, {8.0}, {2.0}, 0.0);
  REQUIRE(f.factorize(A1, {1}) == FactorStatus::Ok);  // the logical column -e0
  expect_ftran(f, {8.0}, {-8.0}, 0.0);

  const SparseMatrix A0 = make_matrix(0, 0, {});
  REQUIRE(f.factorize(A0, {}) == FactorStatus::Ok);
  SparseWork w(0);
  f.ftran(w);
  f.btran(w);
  CHECK_EQ(w.count(), 0);
}

TEST_CASE(lu_rejects_bad_input) {
  const SparseMatrix A = make_matrix(2, 2, {{0, 0, 1}, {1, 1, 1}});
  BasisFactor f;
  bool threw = false;
  try { f.factorize(A, {0, 1, 2}); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);  // wrong size
  threw = false;
  try { f.factorize(A, {0, 4}); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);  // variable out of range (n + m = 4)
  threw = false;
  try { f.factorize(A, {0, -1}); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
  // Solves need a valid factorization.
  BasisFactor fresh;
  SparseWork w(2);
  threw = false;
  try { fresh.ftran(w); } catch (const std::logic_error&) { threw = true; }
  CHECK(threw);
  REQUIRE(f.factorize(A, {0, 1}) == FactorStatus::Ok);
  SparseWork wrong(3);
  threw = false;
  try { f.btran(wrong); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST_CASE(lu_explicit_zero_entries_are_ignored) {
  // An explicitly stored zero in A must not become a pivot candidate.
  SparseMatrix A = make_matrix(2, 2, {{0, 0, 0.0}, {1, 0, 2.0}, {0, 1, 3.0}, {1, 1, 0.0}});
  BasisFactor f;
  REQUIRE(f.factorize(A, {0, 1}) == FactorStatus::Ok);
  expect_ftran(f, {3.0, 2.0}, {1.0, 1.0}, 0.0);
  CHECK_EQ(f.stats().nnz_basis, std::size_t{2});
}
