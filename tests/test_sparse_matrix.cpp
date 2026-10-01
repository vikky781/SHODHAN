#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "shodhan/sparse_matrix.hpp"
#include "test_harness.hpp"

using namespace shodhan;

namespace {

// 3x4 matrix used throughout:
//   [ 1  0  2  0 ]
//   [ 0  3  0  4 ]
//   [ 5  0  6  0 ]
SparseMatrix sample() {
  std::vector<Triplet> t = {{2, 2, 6.0}, {0, 0, 1.0}, {1, 3, 4.0}, {2, 0, 5.0},
                            {0, 2, 2.0}, {1, 1, 3.0}};
  SparseMatrix m;
  std::string err;
  const bool ok = SparseMatrix::from_triplets(3, 4, t, &m, &err);
  if (!ok) throw std::runtime_error("sample(): " + err);
  return m;
}

std::vector<std::vector<double>> sample_dense() {
  return {{1, 0, 2, 0}, {0, 3, 0, 4}, {5, 0, 6, 0}};
}

}  // namespace

TEST_CASE(sparse_from_triplets_builds_sorted_csc) {
  const SparseMatrix m = sample();
  CHECK_EQ(m.n_rows, 3);
  CHECK_EQ(m.n_cols, 4);
  CHECK_EQ(m.nnz(), std::size_t{6});
  CHECK(m.col_start == (std::vector<Index>{0, 2, 3, 5, 6}));
  CHECK(m.row_index == (std::vector<Index>{0, 2, 1, 0, 2, 1}));
  CHECK(m.value == (std::vector<double>{1, 5, 3, 2, 6, 4}));
  CHECK(m.validate().empty());
}

TEST_CASE(sparse_from_triplets_detects_duplicates_and_names_them) {
  SparseMatrix m;
  std::string err;
  const std::vector<Triplet> t = {{0, 0, 1.0}, {1, 2, 2.0}, {1, 2, 3.0}};
  CHECK(!SparseMatrix::from_triplets(2, 3, t, &m, &err));
  CHECK_CONTAINS(err, "duplicate");
  CHECK_CONTAINS(err, "row=1");
  CHECK_CONTAINS(err, "col=2");
}

TEST_CASE(sparse_from_triplets_rejects_bad_input) {
  SparseMatrix m;
  std::string err;
  CHECK(!SparseMatrix::from_triplets(2, 2, {{2, 0, 1.0}}, &m, &err));
  CHECK_CONTAINS(err, "out of range");
  CHECK(!SparseMatrix::from_triplets(2, 2, {{0, -1, 1.0}}, &m, &err));
  CHECK(!SparseMatrix::from_triplets(2, 2, {{0, 0, std::nan("")}}, &m, &err));
  CHECK_CONTAINS(err, "non-finite");
  CHECK(!SparseMatrix::from_triplets(2, 2, {{0, 0, HUGE_VAL}}, &m, &err));
  CHECK(!SparseMatrix::from_triplets(-1, 2, {}, &m, &err));
}

TEST_CASE(sparse_from_triplets_handles_empty_shapes) {
  SparseMatrix m;
  std::string err;
  CHECK(SparseMatrix::from_triplets(0, 0, {}, &m, &err));
  CHECK(m.validate().empty());
  CHECK(SparseMatrix::from_triplets(3, 2, {}, &m, &err));
  CHECK_EQ(m.nnz(), std::size_t{0});
  CHECK(m.col_start == (std::vector<Index>{0, 0, 0}));
  CHECK(m.validate().empty());
}

TEST_CASE(sparse_transpose_twice_returns_original) {
  const SparseMatrix a = sample();
  const CsrMatrix r = a.to_csr();
  CHECK_EQ(r.n_rows, 3);
  CHECK_EQ(r.n_cols, 4);
  CHECK(r.row_start == (std::vector<Index>{0, 2, 4, 6}));
  CHECK(r.col_index == (std::vector<Index>{0, 2, 1, 3, 0, 2}));
  CHECK(r.value == (std::vector<double>{1, 2, 3, 4, 5, 6}));
  CHECK(r.to_csc() == a);
  CHECK(r.to_csc().to_csr() == r);

  const SparseMatrix empty(5, 2);
  CHECK(empty.to_csr().to_csc() == empty);
}

TEST_CASE(sparse_spmv_matches_dense_reference) {
  const SparseMatrix a = sample();
  const auto dense = sample_dense();
  const std::vector<double> x = {1.0, -2.0, 0.5, 3.0};
  std::vector<double> y(3, 99.0);
  a.multiply(x, y);
  for (std::size_t i = 0; i < 3; ++i) {
    double ref = 0.0;
    for (std::size_t j = 0; j < 4; ++j) ref += dense[i][j] * x[j];
    CHECK_NEAR(y[i], ref, 1e-15);
  }
  // Hand-computed: [1+1, -6+12, 5+3] = [2, 6, 8]
  CHECK_NEAR(y[0], 2.0, 1e-15);
  CHECK_NEAR(y[1], 6.0, 1e-15);
  CHECK_NEAR(y[2], 8.0, 1e-15);

  std::vector<double> y_csr(3, 0.0);
  a.to_csr().multiply(x, y_csr);
  CHECK(y_csr == y);
}

TEST_CASE(sparse_transposed_spmv_matches_dense_reference) {
  const SparseMatrix a = sample();
  const auto dense = sample_dense();
  const std::vector<double> x = {2.0, -1.0, 0.5};
  std::vector<double> y(4, 99.0);
  a.multiply_transpose(x, y);
  for (std::size_t j = 0; j < 4; ++j) {
    double ref = 0.0;
    for (std::size_t i = 0; i < 3; ++i) ref += dense[i][j] * x[i];
    CHECK_NEAR(y[j], ref, 1e-15);
  }
  // Hand-computed: [2+2.5, -3, 4+3, -4] = [4.5, -3, 7, -4]
  CHECK_NEAR(y[0], 4.5, 1e-15);
  CHECK_NEAR(y[1], -3.0, 1e-15);
  CHECK_NEAR(y[2], 7.0, 1e-15);
  CHECK_NEAR(y[3], -4.0, 1e-15);
}

TEST_CASE(sparse_spmv_rejects_wrong_sizes) {
  const SparseMatrix a = sample();
  std::vector<double> x(3, 0.0), y(3, 0.0);
  bool threw = false;
  try {
    a.multiply(x, y);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    std::vector<double> wrong(4, 0.0);
    a.multiply_transpose(wrong, wrong);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE(sparse_counts_and_abs_range) {
  const SparseMatrix a = sample();
  CHECK(a.row_counts() == (std::vector<Index>{2, 2, 2}));
  CHECK(a.col_counts() == (std::vector<Index>{2, 1, 2, 1}));
  CHECK_EQ(a.min_abs(), 1.0);
  CHECK_EQ(a.max_abs(), 6.0);

  SparseMatrix m;
  std::string err;
  REQUIRE(SparseMatrix::from_triplets(2, 2, {{0, 0, -0.25}, {1, 1, 0.0}, {1, 0, 8.0}}, &m, &err));
  CHECK_EQ(m.min_abs(), 0.25);  // explicit zeros are ignored
  CHECK_EQ(m.max_abs(), 8.0);
  CHECK_EQ(SparseMatrix(2, 2).min_abs(), 0.0);
  CHECK_EQ(SparseMatrix(2, 2).max_abs(), 0.0);
}

TEST_CASE(sparse_validate_reports_each_failure_kind) {
  SparseMatrix m = sample();
  CHECK(m.validate().empty());

  SparseMatrix bad = m;
  bad.col_start[2] = 1;  // not monotone relative to col_start[3]? 2,1,5 -> decreasing at col 1
  CHECK(!bad.validate().empty());

  bad = m;
  bad.col_start[0] = 1;
  CHECK(!bad.validate().empty());

  bad = m;
  bad.col_start.pop_back();
  CHECK(!bad.validate().empty());

  bad = m;
  bad.col_start.back() = 5;
  CHECK(!bad.validate().empty());

  bad = m;
  bad.row_index[0] = 7;  // out of range
  CHECK(!bad.validate().empty());

  bad = m;
  std::swap(bad.row_index[0], bad.row_index[1]);  // unsorted within column 0
  CHECK(!bad.validate().empty());

  bad = m;
  bad.row_index[1] = bad.row_index[0];  // duplicate row index
  CHECK(!bad.validate().empty());

  bad = m;
  bad.value[3] = std::nan("");
  CHECK(!bad.validate().empty());

  bad = m;
  bad.value[3] = HUGE_VAL;
  CHECK(!bad.validate().empty());

  bad = m;
  bad.value.pop_back();
  CHECK(!bad.validate().empty());

  bad = m;
  bad.n_rows = -1;
  CHECK(!bad.validate().empty());
}
