#include <cmath>
#include <string>

#include "shodhan/lp_model.hpp"
#include "shodhan/model_stats.hpp"
#include "test_harness.hpp"

using namespace shodhan;

namespace {

// 2 rows, 3 columns:  row0: x0 + x1 <= 4 ; row1: x1 - x2 = 1.
LpModel tiny_model() {
  LpModel m;
  m.name = "tiny";
  m.n_rows = 2;
  m.n_cols = 3;
  std::string err;
  SparseMatrix::from_triplets(2, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, -1.0}}, &m.A,
                              &err);
  m.col_cost = {1.0, 0.0, -2.0};
  m.col_lower = {0.0, -kInf, 0.0};
  m.col_upper = {kInf, kInf, 1.0};
  m.col_type = {ColType::Continuous, ColType::Integer, ColType::Binary};
  m.row_lower = {-kInf, 1.0};
  m.row_upper = {4.0, 1.0};
  return m;
}

}  // namespace

TEST_CASE(model_valid_model_has_no_problems) {
  LpModel m = tiny_model();
  CHECK(m.validate().empty());
  m.row_names = {"r0", "r1"};
  m.col_names = {"a", "b", "c"};
  CHECK(m.validate().empty());
  CHECK(m.is_integer(1));
  CHECK(m.is_integer(2));
  CHECK(!m.is_integer(0));
}

TEST_CASE(model_validate_reports_problems) {
  LpModel m = tiny_model();
  m.col_cost.pop_back();
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.col_lower[0] = 5.0;
  m.col_upper[0] = 1.0;
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.row_lower[1] = std::nan("");
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.col_type[2] = ColType::Binary;
  m.col_upper[2] = 5.0;
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.col_cost[1] = HUGE_VAL;
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.col_names = {"a", "a", "c"};
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.row_names = {"only_one"};
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.A.n_cols = 2;
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.col_lower[0] = kInf;  // +inf lower bound is meaningless
  CHECK(!m.validate().empty());

  m = tiny_model();
  m.objective_offset = std::nan("");
  CHECK(!m.validate().empty());
}

TEST_CASE(model_quadratic_field_is_reserved_and_checked) {
  LpModel m = tiny_model();
  CHECK_EQ(m.quadratic.n_cols, 0);
  CHECK(m.validate().empty());

  std::string err;
  // Lower triangle only: entry (2,0) is fine, (0,2) is not.
  REQUIRE(SparseMatrix::from_triplets(3, 3, {{2, 0, 1.0}}, &m.quadratic, &err));
  CHECK(m.validate().empty());
  REQUIRE(SparseMatrix::from_triplets(3, 3, {{0, 2, 1.0}}, &m.quadratic, &err));
  CHECK(!m.validate().empty());
  m.quadratic = SparseMatrix(2, 2);
  CHECK(!m.validate().empty());  // wrong shape
}

TEST_CASE(model_stats_counts) {
  LpModel m = tiny_model();
  m.row_lower.push_back(-kInf);  // extend with free, ranged, ge rows
  m.row_upper.push_back(kInf);
  m.row_lower.push_back(1.0);
  m.row_upper.push_back(3.0);
  m.row_lower.push_back(2.0);
  m.row_upper.push_back(kInf);
  m.n_rows = 5;
  m.A = SparseMatrix(5, 3);  // keep the model self-consistent
  std::string err;
  REQUIRE(SparseMatrix::from_triplets(
      5, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, -1.0}, {4, 0, 8.0}}, &m.A, &err));
  m.col_lower = {0.0, -kInf, 0.0};
  m.col_upper = {kInf, kInf, 1.0};
  REQUIRE(m.validate().empty());

  const ModelStats s = compute_stats(m);
  CHECK_EQ(s.rows, 5);
  CHECK_EQ(s.cols, 3);
  CHECK_EQ(s.nnz, std::size_t{5});
  CHECK_NEAR(s.density, 5.0 / 15.0, 1e-15);
  CHECK_EQ(s.continuous_cols, 1);
  CHECK_EQ(s.integer_cols, 1);
  CHECK_EQ(s.binary_cols, 1);
  CHECK_EQ(s.rows_le, 1);
  CHECK_EQ(s.rows_eq, 1);
  CHECK_EQ(s.rows_free, 1);
  CHECK_EQ(s.rows_ranged, 1);
  CHECK_EQ(s.rows_ge, 1);
  CHECK_EQ(s.cols_free, 1);       // (-inf, inf)
  CHECK_EQ(s.cols_one_sided, 1);  // [0, inf)
  CHECK_EQ(s.cols_boxed, 1);      // [0, 1]
  CHECK_EQ(s.cols_fixed, 0);
  CHECK(s.coefficient.valid);
  CHECK_EQ(s.coefficient.min, 1.0);
  CHECK_EQ(s.coefficient.max, 8.0);
  CHECK_NEAR(s.coefficient_ratio, 8.0, 1e-15);
  CHECK(s.cost.valid);
  CHECK_EQ(s.cost.min, 1.0);
  CHECK_EQ(s.cost.max, 2.0);
}
