// One hand-made model per reduction. Each is small enough to verify by hand; the
// tests assert the shape of the reduced model AND the postsolved solution
// through check_kkt on the original model. Reductions are switched on one at a
// time so that a test exercises exactly the reduction it names.

#include <cmath>
#include <stdexcept>

#include "shodhan/kkt.hpp"
#include "shodhan/presolve.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/test_models.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

constexpr double kTol = 1e-9;

PresolveOptions only(bool PresolveOptions::*flag) {
  PresolveOptions o;
  o.empty_rows = o.empty_columns = o.fixed_columns = o.singleton_rows = false;
  o.redundant_rows = o.forcing_rows = o.doubleton_equations = o.dual_fixing = false;
  o.integer_bounds = false;
  o.mip_propagation = o.coefficient_tightening = o.probing = false;
  o.parallel_rows = o.duplicate_columns = o.clique_table = false;
  o.*flag = true;
  return o;
}

struct Run {
  PresolveResult pr;
  Solution sol;
  KktReport kkt;
  bool solved = false;
};

// presolve -> reference solve of the reduced model -> postsolve -> KKT on the original.
Run run(const LpModel& m, const PresolveOptions& o) {
  Run r;
  r.pr = presolve(m, o);
  Solution reduced;
  if (r.pr.status == PresolveStatus::Reduced) {
    const RefLpResult rr = solve_dense_lp(r.pr.reduced);
    CHECK(rr.status == Status::Optimal);
    if (rr.status != Status::Optimal) return r;
    reduced = rr.solution;
  }
  if (r.pr.status == PresolveStatus::Reduced || r.pr.status == PresolveStatus::SolvedByPresolve) {
    r.sol = postsolve(r.pr.stack, reduced);
    r.kkt = check_kkt(m, r.sol, kTol);
    r.solved = true;
  }
  return r;
}

void expect_kkt(const Run& r, const char* what) {
  CHECK(r.solved);
  if (!r.kkt.ok) std::cerr << "  " << what << ": " << r.kkt.summary() << "\n";
  CHECK(r.kkt.ok);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Empty row
// ---------------------------------------------------------------------------
TEST_CASE(presolve_empty_row_is_removed_when_zero_is_in_range) {
  // min x  s.t.  (empty row) in [-1, 2];  x >= 1;  x >= 0.   Optimum x = 1, y = (0, 1).
  const LpModel m = make_model(2, 1, {{1, 0, 1.0}}, {1.0}, {0.0}, {kInf}, {-1.0, 1.0}, {2.0, kInf});
  const Run r = run(m, only(&PresolveOptions::empty_rows));
  CHECK(r.pr.status == PresolveStatus::Reduced);
  CHECK_EQ(r.pr.reduced.n_rows, 1);
  CHECK_EQ(r.pr.stats.empty_rows, 1);
  CHECK((r.pr.stack.row_map == std::vector<Index>{1}));
  expect_kkt(r, "empty row");
  CHECK_EQ(r.sol.y[0], 0.0);
  CHECK_NEAR(r.sol.y[1], 1.0, 1e-12);
}

TEST_CASE(presolve_empty_row_excluding_zero_is_infeasible) {
  const LpModel m = make_model(2, 1, {{1, 0, 1.0}}, {1.0}, {0.0}, {kInf}, {1.0, 1.0}, {2.0, kInf});
  const PresolveResult pr = presolve(m, only(&PresolveOptions::empty_rows));
  CHECK(pr.status == PresolveStatus::Infeasible);
  CHECK_CONTAINS(pr.note, "empty row 0");
}

// ---------------------------------------------------------------------------
// 2. Empty column
// ---------------------------------------------------------------------------
TEST_CASE(presolve_empty_column_is_fixed_at_the_bound_its_cost_favours) {
  // Columns 0 and 1 have no entries: cost +2 on [1,5] -> 1; cost -3 on [0,4] -> 4.
  // Column 2 is a real column: min x2 s.t. x2 >= 2.
  const LpModel m = make_model(1, 3, {{0, 2, 1.0}}, {2.0, -3.0, 1.0}, {1.0, 0.0, 0.0},
                               {5.0, 4.0, kInf}, {2.0}, {kInf});
  const Run r = run(m, only(&PresolveOptions::empty_columns));
  CHECK(r.pr.status == PresolveStatus::Reduced);
  CHECK_EQ(r.pr.reduced.n_cols, 1);
  CHECK_EQ(r.pr.stats.empty_columns, 2);
  CHECK_NEAR(r.pr.reduced.objective_offset, 2.0 * 1.0 + -3.0 * 4.0, 1e-12);
  expect_kkt(r, "empty column");
  CHECK_EQ(r.sol.x[0], 1.0);
  CHECK_EQ(r.sol.x[1], 4.0);
  CHECK_EQ(r.sol.d[0], 2.0);   // d = c
  CHECK_EQ(r.sol.d[1], -3.0);
}

TEST_CASE(presolve_empty_column_improving_ray_reports_unbounded_or_infeasible_or_unbounded) {
  // min -x, x >= 0, no rows: unbounded, and everything was removed.
  const LpModel alone = make_model(0, 1, {}, {-1.0}, {0.0}, {kInf}, {}, {});
  CHECK(presolve(alone, only(&PresolveOptions::empty_columns)).status == PresolveStatus::Unbounded);

  // Same column next to a row that presolve (with only this reduction) cannot decide:
  // feasibility of the rest is not established.
  const LpModel with_row = make_model(1, 2, {{0, 1, 1.0}}, {-1.0, 1.0}, {0.0, 0.0}, {kInf, kInf},
                                      {1.0}, {kInf});
  CHECK(presolve(with_row, only(&PresolveOptions::empty_columns)).status ==
        PresolveStatus::InfeasibleOrUnbounded);
}

// ---------------------------------------------------------------------------
// 3. Fixed column
// ---------------------------------------------------------------------------
TEST_CASE(presolve_fixed_column_is_substituted_into_rows_and_objective) {
  // min x0 + 3 x1  s.t.  x0 + x1 >= 4,  x0 >= 0,  x1 = 2 (fixed).
  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 3.0}, {0.0, 2.0},
                               {kInf, 2.0}, {4.0}, {kInf});
  const Run r = run(m, only(&PresolveOptions::fixed_columns));
  CHECK(r.pr.status == PresolveStatus::Reduced);
  CHECK_EQ(r.pr.reduced.n_cols, 1);
  CHECK_NEAR(r.pr.reduced.objective_offset, 6.0, 1e-12);   // 3 * 2
  CHECK_NEAR(r.pr.reduced.row_lower[0], 2.0, 1e-12);       // 4 - 2
  expect_kkt(r, "fixed column");
  CHECK_NEAR(r.sol.x[1], 2.0, 1e-12);
  CHECK_NEAR(r.sol.objective, 2.0 + 6.0, 1e-12);
  // d_1 = c_1 - a_1 y_0 = 3 - 1 = 2.
  CHECK_NEAR(r.sol.d[1], 2.0, 1e-12);
}

TEST_CASE(presolve_column_with_crossed_bounds_is_infeasible) {
  LpModel m = make_model(1, 1, {{0, 0, 1.0}}, {1.0}, {0.0}, {1.0}, {0.0}, {kInf});
  m.col_lower[0] = 3.0;  // lower > upper is rejected by validate(), so presolve refuses the model
  bool threw = false;
  try {
    presolve(m);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

// ---------------------------------------------------------------------------
// 4. Singleton row
// ---------------------------------------------------------------------------
TEST_CASE(presolve_singleton_row_becomes_a_bound_and_transfers_its_dual) {
  // min x s.t. 2x >= 6, x >= 0.  Row -> x >= 3.  Reduced optimum x=3 with d' = 1, so the
  // row dual must be y = d'/a = 1/2 and d = 0.
  const LpModel m = make_model(1, 1, {{0, 0, 2.0}}, {1.0}, {0.0}, {kInf}, {6.0}, {kInf});
  const Run r = run(m, only(&PresolveOptions::singleton_rows));
  CHECK_EQ(r.pr.reduced.n_rows, 0);
  CHECK_EQ(r.pr.stats.singleton_rows, 1);
  expect_kkt(r, "singleton row");
  CHECK_NEAR(r.sol.x[0], 3.0, 1e-12);
  CHECK_NEAR(r.sol.y[0], 0.5, 1e-12);
  CHECK_NEAR(r.sol.d[0], 0.0, 1e-12);
}

TEST_CASE(presolve_singleton_row_with_negative_coefficient_binds_the_upper_bound) {
  // min -x s.t. -2x >= -8 (x <= 4).  Reduced: x <= 4, d' = -1 at the upper bound,
  // y = d'/a = 0.5 >= 0 because the row's LOWER side is the active one.
  const LpModel m = make_model(1, 1, {{0, 0, -2.0}}, {-1.0}, {0.0}, {kInf}, {-8.0}, {kInf});
  const Run r = run(m, only(&PresolveOptions::singleton_rows));
  expect_kkt(r, "negative singleton");
  CHECK_NEAR(r.sol.x[0], 4.0, 1e-12);
  CHECK_NEAR(r.sol.y[0], 0.5, 1e-12);
  CHECK_NEAR(r.sol.d[0], 0.0, 1e-12);
}

TEST_CASE(presolve_singleton_row_looser_than_the_column_bound_gets_zero_dual) {
  // 2x >= 2 gives x >= 1, but the column already has x >= 3: the row is inactive.
  const LpModel m = make_model(1, 1, {{0, 0, 2.0}}, {1.0}, {3.0}, {kInf}, {2.0}, {kInf});
  const Run r = run(m, only(&PresolveOptions::singleton_rows));
  expect_kkt(r, "inactive singleton");
  CHECK_EQ(r.sol.y[0], 0.0);
  CHECK_NEAR(r.sol.d[0], 1.0, 1e-12);
}

TEST_CASE(presolve_singleton_rows_conflicting_with_bounds_are_infeasible) {
  const LpModel m = make_model(1, 1, {{0, 0, 1.0}}, {1.0}, {0.0}, {2.0}, {5.0}, {kInf});
  CHECK(presolve(m, only(&PresolveOptions::singleton_rows)).status == PresolveStatus::Infeasible);
}

// ---------------------------------------------------------------------------
// 5. Redundant row
// ---------------------------------------------------------------------------
TEST_CASE(presolve_row_implied_by_bounds_is_removed) {
  // x, y in [0,1]; x + y <= 5 can never bind. A second row keeps the model meaningful.
  const LpModel m = make_model(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, -1.0}},
                               {-1.0, -2.0}, {0.0, 0.0}, {1.0, 1.0}, {-kInf, -kInf}, {5.0, 0.5});
  const Run r = run(m, only(&PresolveOptions::redundant_rows));
  CHECK_EQ(r.pr.stats.redundant_rows, 1);
  CHECK_EQ(r.pr.reduced.n_rows, 1);
  expect_kkt(r, "redundant row");
  CHECK_EQ(r.sol.y[0], 0.0);
}

TEST_CASE(presolve_row_with_one_implied_side_keeps_the_other_side) {
  // x, y in [0,1]: x + y in [0, 1.5]; the lower side is implied, the upper side is not.
  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {-1.0, -1.0}, {0.0, 0.0},
                               {1.0, 1.0}, {0.0}, {1.5});
  const Run r = run(m, only(&PresolveOptions::redundant_rows));
  CHECK_EQ(r.pr.stats.redundant_row_sides, 1);
  CHECK_EQ(r.pr.reduced.n_rows, 1);
  CHECK(is_neg_inf(r.pr.reduced.row_lower[0]));
  CHECK_EQ(r.pr.reduced.row_upper[0], 1.5);
  expect_kkt(r, "one-sided redundancy");
}

TEST_CASE(presolve_row_that_cannot_be_satisfied_is_infeasible) {
  // x, y in [0,1] but x + y >= 3.
  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0},
                               {3.0}, {kInf});
  const PresolveResult pr = presolve(m, only(&PresolveOptions::redundant_rows));
  CHECK(pr.status == PresolveStatus::Infeasible);
  CHECK_CONTAINS(pr.note, "row 0");
}

// ---------------------------------------------------------------------------
// 6. Forcing row
// ---------------------------------------------------------------------------
TEST_CASE(presolve_forcing_row_fixes_its_columns_and_recovers_the_dual) {
  // x, y in [0,2], z in [0,10]:  r0: x + y >= 4 (maximum activity is 4: forcing),
  //                               r1: x + z >= 3.   costs (1, 2, 1).
  // x = y = 2, z = 1, objective 7.  Duals: y1 = 1, y0 = 2, d = (-2, 0, 0).
  const LpModel m = make_model(2, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}},
                               {1.0, 2.0, 1.0}, {0.0, 0.0, 0.0}, {2.0, 2.0, 10.0}, {4.0, 3.0},
                               {kInf, kInf});
  const Run r = run(m, only(&PresolveOptions::forcing_rows));
  CHECK_EQ(r.pr.stats.forcing_rows, 1);
  CHECK_EQ(r.pr.reduced.n_cols, 1);
  expect_kkt(r, "forcing row");
  CHECK_NEAR(r.sol.x[0], 2.0, 1e-12);
  CHECK_NEAR(r.sol.x[1], 2.0, 1e-12);
  CHECK_NEAR(r.sol.x[2], 1.0, 1e-12);
  CHECK_NEAR(r.sol.objective, 7.0, 1e-12);
  CHECK_NEAR(r.sol.y[0], 2.0, 1e-12);
  CHECK_NEAR(r.sol.y[1], 1.0, 1e-12);
  CHECK_NEAR(r.sol.d[0], -2.0, 1e-12);
  CHECK_NEAR(r.sol.d[1], 0.0, 1e-12);
}

TEST_CASE(presolve_min_forcing_row_with_negative_coefficient) {
  // x in [0,3], y in [1,2]:  r0: -x + 2y <= -1  (minimum activity -3 + 2 = -1: forcing at
  // x = 3, y = 1).  min x - y + w, w >= 0 free of rows.
  const LpModel m = make_model(1, 3, {{0, 0, -1.0}, {0, 1, 2.0}}, {1.0, -1.0, 1.0}, {0.0, 1.0, 0.0},
                               {3.0, 2.0, 5.0}, {-kInf}, {-1.0});
  const Run r = run(m, only(&PresolveOptions::forcing_rows));
  CHECK_EQ(r.pr.stats.forcing_rows, 1);
  expect_kkt(r, "min forcing");
  CHECK_NEAR(r.sol.x[0], 3.0, 1e-12);
  CHECK_NEAR(r.sol.x[1], 1.0, 1e-12);
  CHECK(r.sol.y[0] <= 1e-12);  // row is at its upper side
}

TEST_CASE(presolve_forcing_equality_row_may_carry_a_dual_of_either_sign) {
  // x + y = 4 with x, y in [0,2]: forced by both sides.
  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {3.0, 1.0}, {0.0, 0.0}, {2.0, 2.0},
                               {4.0}, {4.0});
  const Run r = run(m, only(&PresolveOptions::forcing_rows));
  CHECK_EQ(r.pr.stats.forcing_rows, 1);
  CHECK_EQ(r.pr.status == PresolveStatus::SolvedByPresolve, true);
  expect_kkt(r, "forcing equality");
}

TEST_CASE(presolve_mip_forcing_is_skipped_when_it_would_fix_an_integer_at_a_fraction) {
  // x integer in [0, 2.5]; x + y >= 3.5 with y in [0,1] continuous is forcing (max activity
  // 3.5) but x would be fixed at the fractional value 2.5, which is not allowed.
  LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 1.0}, {0.0, 0.0}, {2.5, 1.0}, {3.5},
                         {kInf});
  m.col_type = {ColType::Integer, ColType::Continuous};
  PresolveOptions o = only(&PresolveOptions::forcing_rows);
  o.is_mip = true;
  const PresolveResult pr = presolve(m, o);
  CHECK_EQ(pr.stats.forcing_rows, 0);
  CHECK(pr.status == PresolveStatus::Reduced);
}

// ---------------------------------------------------------------------------
// 7. Doubleton equation
// ---------------------------------------------------------------------------
TEST_CASE(presolve_doubleton_equation_substitutes_a_column_out) {
  // min x + 2y + 3z  s.t.  r0: x + y = 4,  r1: x + 2z >= 6;  all in [0, 10].
  // y appears only in r0, so y = 4 - x is substituted: cost x -> 1 - 2 = -1, offset 8.
  const LpModel m = make_model(2, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 2.0}},
                               {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0}, {4.0, 6.0},
                               {4.0, kInf});
  const Run r = run(m, only(&PresolveOptions::doubleton_equations));
  CHECK_EQ(r.pr.stats.doubleton_equations, 1);
  CHECK_EQ(r.pr.reduced.n_rows, 1);
  CHECK_EQ(r.pr.reduced.n_cols, 2);
  CHECK_NEAR(r.pr.reduced.objective_offset, 8.0, 1e-12);
  CHECK_NEAR(r.pr.reduced.col_cost[0], -1.0, 1e-12);
  // x's bounds now also reflect y in [0,10]: 4 - x in [0,10] -> x in [-6, 4] (and x >= 0).
  CHECK_NEAR(r.pr.reduced.col_upper[0], 4.0, 1e-12);
  expect_kkt(r, "doubleton");
  CHECK_NEAR(r.sol.x[0] + r.sol.x[1], 4.0, 1e-12);
}

TEST_CASE(presolve_doubleton_transfers_the_dual_when_the_kept_bound_came_from_the_eliminated_column) {
  // x + y = 4, x, y in [0,3]  =>  x in [1,3] (the bound x >= 1 comes from y <= 3).
  // Costs x:3, y:1  =>  optimum x = 1, y = 3, objective 6, so x rests on the derived bound.
  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {3.0, 1.0}, {0.0, 0.0}, {3.0, 3.0},
                               {4.0}, {4.0});
  const Run r = run(m, only(&PresolveOptions::doubleton_equations));
  CHECK_EQ(r.pr.stats.doubleton_equations, 1);
  expect_kkt(r, "doubleton dual transfer");
  CHECK_NEAR(r.sol.x[0], 1.0, 1e-12);
  CHECK_NEAR(r.sol.x[1], 3.0, 1e-12);
  CHECK_NEAR(r.sol.objective, 6.0, 1e-12);
  // The active bound is y <= 3, so d_y < 0 and d_x = 0:  y_0 = c_x = 3, d_y = 1 - 3 = -2.
  CHECK_NEAR(r.sol.y[0], 3.0, 1e-12);
  CHECK_NEAR(r.sol.d[0], 0.0, 1e-12);
  CHECK_NEAR(r.sol.d[1], -2.0, 1e-12);
}

TEST_CASE(presolve_doubleton_is_refused_when_the_ratio_is_unbalanced) {
  // 2x + y = 4: both pivots give |alpha| outside [1/1.5, 1.5] when growth is limited to 1.5.
  const LpModel m = make_model(2, 2, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}}, {1.0, 1.0},
                               {0.0, 0.0}, {10.0, 10.0}, {4.0, 1.0}, {4.0, kInf});
  PresolveOptions o = only(&PresolveOptions::doubleton_equations);
  o.max_coefficient_growth = 1.5;
  CHECK_EQ(presolve(m, o).stats.doubleton_equations, 0);
  o.max_coefficient_growth = 1e3;
  CHECK_EQ(presolve(m, o).stats.doubleton_equations, 1);
}

TEST_CASE(presolve_doubleton_is_refused_for_a_tiny_pivot) {
  // Column 0 has a tiny entry in the doubleton row relative to its other entry.
  const LpModel m = make_model(2, 2, {{0, 0, 1e-6}, {0, 1, 1e-6}, {1, 0, 1.0}, {1, 1, 1.0}}, {1.0, 1.0},
                               {0.0, 0.0}, {10.0, 10.0}, {1e-6, 1.0}, {1e-6, kInf});
  PresolveOptions o = only(&PresolveOptions::doubleton_equations);
  CHECK_EQ(presolve(m, o).stats.doubleton_equations, 0);  // pivot 1e-6 < 1e-3 * column max
  o.doubleton_pivot_tolerance = 1e-9;
  CHECK_EQ(presolve(m, o).stats.doubleton_equations, 1);
}

TEST_CASE(presolve_mip_doubleton_keeps_integrality) {
  // 2x + y = 5 with x, y integer: eliminating y (unit pivot) is fine, eliminating x is not.
  LpModel m = make_model(2, 2, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}}, {1.0, 1.0},
                         {0.0, 0.0}, {5.0, 5.0}, {5.0, 1.0}, {5.0, kInf});
  m.col_type = {ColType::Integer, ColType::Integer};
  PresolveOptions o = only(&PresolveOptions::doubleton_equations);
  o.is_mip = true;
  const PresolveResult pr = presolve(m, o);
  CHECK_EQ(pr.stats.doubleton_equations, 1);
  REQUIRE(pr.reduced.n_cols == 1);
  CHECK((pr.stack.col_map == std::vector<Index>{0}));  // x stays, y was eliminated

  // x continuous, y integer: eliminating the continuous x is fine (y stays integer),
  // eliminating the integer y would make it a function of a continuous column.
  m.col_type = {ColType::Continuous, ColType::Integer};
  const PresolveResult pr2 = presolve(m, o);
  CHECK_EQ(pr2.stats.doubleton_equations, 1);
  REQUIRE(pr2.reduced.n_cols == 1);
  CHECK((pr2.stack.col_map == std::vector<Index>{1}));  // y stays, x was eliminated

  // Both integer but the pivot is not +-1 in either direction: refuse.
  LpModel hard = make_model(1, 2, {{0, 0, 2.0}, {0, 1, 3.0}}, {1.0, 1.0}, {0.0, 0.0}, {5.0, 5.0}, {6.0},
                            {6.0});
  hard.col_type = {ColType::Integer, ColType::Integer};
  CHECK_EQ(presolve(hard, o).stats.doubleton_equations, 0);
}

// ---------------------------------------------------------------------------
// 8. Dual fixing
// ---------------------------------------------------------------------------
TEST_CASE(presolve_dual_fixing_fixes_a_column_at_the_bound_that_cannot_hurt) {
  // min 2x - y  s.t.  x + y <= 5,  x in [1,4], y in [0,10].  Lowering x never hurts the row
  // and its cost is positive, so x = 1.  Then y = 4, objective 2 - 4 = -2.
  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {2.0, -1.0}, {1.0, 0.0}, {4.0, 10.0},
                               {-kInf}, {5.0});
  const Run r = run(m, only(&PresolveOptions::dual_fixing));
  CHECK_EQ(r.pr.stats.dual_fixed_columns, 1);
  CHECK_EQ(r.pr.reduced.n_cols, 1);
  expect_kkt(r, "dual fixing");
  CHECK_NEAR(r.sol.x[0], 1.0, 1e-12);
  CHECK_NEAR(r.sol.x[1], 4.0, 1e-12);
  CHECK_NEAR(r.sol.objective, -2.0, 1e-12);
  CHECK(r.sol.d[0] >= -1e-12);  // at its lower bound
}

TEST_CASE(presolve_dual_fixing_detects_an_improving_ray) {
  // min x  s.t.  x <= 5, x free: decreasing x is always feasible and improves the objective.
  const LpModel m = make_model(1, 1, {{0, 0, 1.0}}, {1.0}, {-kInf}, {kInf}, {-kInf}, {5.0});
  const PresolveResult pr = presolve(m, only(&PresolveOptions::dual_fixing));
  CHECK_EQ(pr.stats.unbounded_columns, 1);
  CHECK(pr.status == PresolveStatus::Unbounded);
}

TEST_CASE(presolve_dual_fixing_leaves_a_column_alone_when_the_row_could_bind) {
  // x + y >= 2 and cost +1: lowering x can break the row, so x must not be fixed.
  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 1.0}, {0.0, 0.0}, {5.0, 5.0},
                               {2.0}, {kInf});
  CHECK_EQ(presolve(m, only(&PresolveOptions::dual_fixing)).stats.dual_fixed_columns, 0);
}

// ---------------------------------------------------------------------------
// 9. Integer bounds and implied bounds
// ---------------------------------------------------------------------------
TEST_CASE(presolve_mip_rounds_integer_bounds) {
  LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 1.0}, {0.3, 0.0}, {4.7, 3.2}, {0.0},
                         {100.0});
  m.col_type = {ColType::Integer, ColType::Continuous};
  PresolveOptions o = only(&PresolveOptions::integer_bounds);
  o.is_mip = true;
  const PresolveResult pr = presolve(m, o);
  CHECK_EQ(pr.stats.integer_bounds_rounded, 1);
  CHECK_EQ(pr.reduced.col_lower[0], 1.0);
  CHECK_EQ(pr.reduced.col_upper[0], 4.0);
  CHECK_EQ(pr.reduced.col_upper[1], 3.2);  // continuous columns are untouched
}

TEST_CASE(presolve_mip_integer_column_without_an_integer_value_is_infeasible) {
  LpModel m = make_model(1, 1, {{0, 0, 1.0}}, {1.0}, {0.3}, {0.7}, {0.0}, {kInf});
  m.col_type = {ColType::Integer};
  PresolveOptions o = only(&PresolveOptions::integer_bounds);
  o.is_mip = true;
  CHECK(presolve(m, o).status == PresolveStatus::Infeasible);
}

TEST_CASE(presolve_mip_tightens_integer_bounds_from_row_activity) {
  // 2x + 3y <= 7, x, y integer in [0,10]: x <= 3 (7/2 rounded down), y <= 2 (7/3).
  LpModel m = make_model(1, 2, {{0, 0, 2.0}, {0, 1, 3.0}}, {-1.0, -1.0}, {0.0, 0.0}, {10.0, 10.0},
                         {-kInf}, {7.0});
  m.col_type = {ColType::Integer, ColType::Integer};
  PresolveOptions o = only(&PresolveOptions::integer_bounds);
  o.is_mip = true;
  const PresolveResult pr = presolve(m, o);
  CHECK_EQ(pr.stats.integer_bounds_tightened, 2);
  CHECK_EQ(pr.reduced.col_upper[0], 3.0);
  CHECK_EQ(pr.reduced.col_upper[1], 2.0);
}

TEST_CASE(presolve_lp_computes_implied_bounds_but_never_writes_them) {
  // The same model as an LP: bounds must stay [0, 10].
  const LpModel m = make_model(1, 2, {{0, 0, 2.0}, {0, 1, 3.0}}, {-1.0, -1.0}, {0.0, 0.0}, {10.0, 10.0},
                               {-kInf}, {7.0});
  const PresolveResult pr = presolve(m, only(&PresolveOptions::integer_bounds));
  CHECK(pr.stats.implied_bound_checks > 0);
  CHECK_EQ(pr.stats.integer_bounds_tightened, 0);
  CHECK_EQ(pr.reduced.col_upper[0], 10.0);
  CHECK_EQ(pr.reduced.col_upper[1], 10.0);
}

TEST_CASE(presolve_mip_singleton_row_rounds_the_derived_bound) {
  // 2x >= 3 with x integer: x >= 1.5 -> x >= 2.
  LpModel m = make_model(1, 1, {{0, 0, 2.0}}, {1.0}, {0.0}, {10.0}, {3.0}, {kInf});
  m.col_type = {ColType::Integer};
  PresolveOptions o = only(&PresolveOptions::singleton_rows);
  o.is_mip = true;
  const PresolveResult pr = presolve(m, o);
  CHECK_EQ(pr.stats.singleton_rows, 1);
  CHECK_EQ(pr.reduced.col_lower[0], 2.0);
}

// ---------------------------------------------------------------------------
// Whole-pipeline behaviour
// ---------------------------------------------------------------------------
TEST_CASE(presolve_can_solve_a_model_completely) {
  // min x + y s.t. x + y >= 2, x = 1 fixed, y <= 1 via a singleton row.
  const LpModel m = make_model(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}}, {1.0, 1.0}, {1.0, 0.0},
                               {1.0, kInf}, {2.0, -kInf}, {kInf, 1.0});
  const Run r = run(m, PresolveOptions{});
  CHECK(r.pr.status == PresolveStatus::SolvedByPresolve);
  CHECK_EQ(r.pr.reduced.n_rows, 0);
  CHECK_EQ(r.pr.reduced.n_cols, 0);
  expect_kkt(r, "solved by presolve");
  CHECK_NEAR(r.sol.objective, 2.0, 1e-12);
}

TEST_CASE(presolve_converts_maximization_and_reports_in_the_original_sense) {
  // max x + 2y s.t. x + y <= 4, x,y >= 0:  optimum y = 4, objective 8.
  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 2.0}, {0.0, 0.0}, {kInf, kInf},
                               {-kInf}, {4.0}, Sense::Maximize);
  const PresolveResult pr = presolve(m, only(&PresolveOptions::empty_rows));
  CHECK(pr.reduced.sense == Sense::Minimize);
  CHECK_EQ(pr.reduced.col_cost[0], -1.0);
  const Run r = run(m, PresolveOptions{});
  expect_kkt(r, "maximization");
  CHECK_NEAR(r.sol.objective, 8.0, 1e-12);
}

TEST_CASE(presolve_reports_a_counter_per_reduction_and_the_model_sizes) {
  const LpModel m = make_model(2, 2, {{1, 1, 1.0}}, {1.0, 1.0}, {0.0, 0.0}, {kInf, kInf}, {-1.0, 1.0},
                               {1.0, kInf});
  const PresolveResult pr = presolve(m);
  CHECK_EQ(pr.stats.rows_before, 2);
  CHECK_EQ(pr.stats.cols_before, 2);
  CHECK_EQ(pr.stats.nnz_before, std::size_t{1});
  CHECK(pr.stats.passes >= 1);
  const auto counts = pr.stats.reduction_counts();
  CHECK_EQ(counts.size(), std::size_t{23});
  CHECK_EQ(counts[0].first, std::string("empty rows"));
  CHECK(pr.stats.seconds >= 0.0);
}

TEST_CASE(presolve_rejects_invalid_models_and_postsolve_rejects_wrong_sizes) {
  LpModel bad = make_model(1, 1, {{0, 0, 1.0}}, {1.0}, {0.0}, {1.0}, {0.0}, {1.0});
  bad.col_cost.pop_back();
  bool threw = false;
  try {
    presolve(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  const LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 1.0}, {0.0, 0.0}, {5.0, 5.0},
                               {2.0}, {kInf});
  const PresolveResult pr = presolve(m, only(&PresolveOptions::empty_rows));
  Solution wrong;
  wrong.x = {1.0};  // the reduced model has 2 columns
  threw = false;
  try {
    postsolve(pr.stack, wrong);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE(presolve_is_idempotent_on_its_own_output) {
  const LpModel m = make_model(3, 4, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 2.0}, {2, 2, 1.0}, {2, 3, 1.0}},
                               {1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 0.0, 0.0}, {5.0, 5.0, 5.0, 5.0},
                               {1.0, 2.0, 1.0}, {kInf, kInf, kInf});
  const PresolveResult once = presolve(m);
  if (once.status != PresolveStatus::Reduced) return;
  const PresolveResult twice = presolve(once.reduced);
  CHECK_EQ(twice.stats.rows_before, twice.stats.rows_after);
  CHECK_EQ(twice.stats.cols_before, twice.stats.cols_after);
}
