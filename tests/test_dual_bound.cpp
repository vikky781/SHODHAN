#include <cmath>
#include <string>
#include <vector>

#include "shodhan/dual_bound.hpp"
#include "shodhan/lp_solver.hpp"
#include "test_harness.hpp"

using namespace shodhan;

namespace {

// min 2x + 3y  s.t.  x + y >= 4,  x + 3y >= 6,  x, y >= 0   -> x = 3, y = 1, objective 9, y* = (1.5, 0.5).
LpModel tiny() {
  LpModel m;
  m.n_rows = 2;
  m.n_cols = 2;
  std::string err;
  SparseMatrix::from_triplets(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 3.0}}, &m.A, &err);
  m.col_cost = {2.0, 3.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {kInf, kInf};
  m.col_type.assign(2, ColType::Continuous);
  m.row_lower = {4.0, 6.0};
  m.row_upper = {kInf, kInf};
  m.col_names = {"X", "Y"};
  m.row_names = {"R1", "R2"};
  return m;
}

// Same with a free column F in R1 and an equality row R2: f = 0.
LpModel with_free_column() {
  LpModel m;
  m.n_rows = 2;
  m.n_cols = 2;
  std::string err;
  SparseMatrix::from_triplets(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}}, &m.A, &err);
  m.col_cost = {1.0, 0.0};
  m.col_lower = {0.0, -kInf};
  m.col_upper = {kInf, kInf};
  m.col_type.assign(2, ColType::Continuous);
  m.row_lower = {1.0, 0.0};
  m.row_upper = {kInf, 0.0};
  m.col_names = {"X", "F"};
  m.row_names = {"R1", "R2"};
  return m;
}

}  // namespace

TEST_CASE(dual_bound_is_strict_and_rigorous_for_exact_multipliers) {
  LpModel m = tiny();
  m.col_upper = {10.0, 10.0};  // boxed: a computed reduced cost of zero needs no bound, so nothing is uncertain
  const DualBound b = compute_dual_bound(m, {3.0, 1.0}, {1.5, 0.5}, 9.0, 1e-6);
  CHECK(b.finite);
  CHECK(b.rigorous);
  CHECK_EQ(b.dropped, 0);
  CHECK(b.gap_ok);
  CHECK(b.bound <= 9.0);
  CHECK(std::fabs(b.bound - 9.0) < 1e-12);
}

TEST_CASE(dual_bound_of_a_maximization_is_an_upper_bound) {
  LpModel m = tiny();
  m.col_upper = {10.0, 10.0};
  m.sense = Sense::Maximize;
  // max -2x - 3y has the same feasible set; multipliers in minimization form are unchanged.
  m.col_cost = {-2.0, -3.0};
  const DualBound b = compute_dual_bound(m, {3.0, 1.0}, {1.5, 0.5}, -9.0, 1e-6);
  CHECK(b.finite);
  CHECK(b.rigorous);
  CHECK(b.bound >= -9.0);
  CHECK(std::fabs(b.bound + 9.0) < 1e-12);
}

TEST_CASE(dual_bound_a_tiny_wrong_signed_multiplier_is_tolerance_level_not_rigorous) {
  const LpModel m = with_free_column();
  // Exact multipliers (1, -1) give d_F = 0, but a computed zero is never taken as an exact zero on a column
  // without bounds, so the bound exists only tolerance-level. Noise of 1e-14 gives d_F = -1e-14 and the same.
  const DualBound exact = compute_dual_bound(m, {1.0, 0.0}, {1.0, -1.0}, 1.0, 1e-6);
  CHECK(exact.finite);
  CHECK(!exact.rigorous);
  const DualBound noisy = compute_dual_bound(m, {1.0, 0.0}, {1.0, -1.0 + 1e-14}, 1.0, 1e-6);
  CHECK(noisy.finite);
  CHECK(!noisy.rigorous);
  CHECK(noisy.dropped >= 1);
  CHECK(noisy.gap_ok);
  // With the tolerance switched off the same multipliers give no bound at all.
  const DualBound strict_only = compute_dual_bound(m, {1.0, 0.0}, {1.0, -1.0 + 1e-14}, 1.0, 1e-6, 0.0);
  CHECK(!strict_only.finite);
}

TEST_CASE(dual_bound_a_large_wrong_signed_multiplier_gives_no_bound) {
  const LpModel m = with_free_column();
  const DualBound b = compute_dual_bound(m, {1.0, 0.0}, {1.0, -0.5}, 1.0, 1e-6);
  CHECK(!b.finite);
  CHECK(!b.rigorous);
  CHECK(!b.gap_ok);
}

TEST_CASE(dual_bound_gap_detects_a_bound_that_misses_the_objective) {
  const LpModel m = tiny();
  const DualBound b = compute_dual_bound(m, {3.0, 1.0}, {1.0, 0.5}, 9.0, 1e-6);  // dual feasible, bound 4 + 3 = 7 < 9
  CHECK(b.finite);
  CHECK(!b.gap_ok);
}

TEST_CASE(lp_solver_records_a_rigorous_flag_with_the_dual_bound) {
  const LpModel m = tiny();
  const LpResult r = LpSolver().solve(m);
  CHECK(r.status == Status::Optimal);
  CHECK(r.dual_bound.finite);
  CHECK_EQ(r.rigorous, r.dual_bound.rigorous);
  CHECK(r.dual_bound.gap_ok);
}
