#include <cmath>

#include "shodhan/kkt.hpp"
#include "support/test_models.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using shodhan::testing::make_model;

namespace {

constexpr double kTol = 1e-9;

// min x1 + 2 x2  s.t. x1 + x2 >= 1, x >= 0.   Optimum x=(1,0), y=1, d=(0,1).
LpModel lp_cover() {
  return make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 2.0}, {0.0, 0.0}, {kInf, kInf}, {1.0},
                    {kInf});
}
Solution sol_cover() { return {{1.0, 0.0}, {1.0}, {0.0, 1.0}, 1.0}; }

// min -x1 - x2  s.t. x1 + 2 x2 <= 4, 3 x1 + x2 <= 6, x >= 0.
// Optimum x=(1.6, 1.2), y=(-0.4, -0.2), d=0, objective -2.8.
LpModel lp_two_rows() {
  return make_model(2, 2, {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 1.0}}, {-1.0, -1.0},
                    {0.0, 0.0}, {kInf, kInf}, {-kInf, -kInf}, {4.0, 6.0});
}
Solution sol_two_rows() { return {{1.6, 1.2}, {-0.4, -0.2}, {0.0, 0.0}, -2.8}; }

// min x + 2y  s.t. x + y = 3, 0 <= x <= 2, y free.  Optimum x=2, y=1, row dual 2,
// d=(-1, 0), objective 4.
LpModel lp_eq_free() {
  return make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 2.0}, {0.0, -kInf}, {2.0, kInf}, {3.0},
                    {3.0});
}
Solution sol_eq_free() { return {{2.0, 1.0}, {2.0}, {-1.0, 0.0}, 4.0}; }

// max x  s.t. x <= 3.  Minimization form: min -x, y=-1, d=0; objective (max sense) 3.
LpModel lp_max() {
  return make_model(1, 1, {{0, 0, 1.0}}, {1.0}, {-kInf}, {kInf}, {-kInf}, {3.0}, Sense::Maximize);
}
Solution sol_max() { return {{3.0}, {-1.0}, {0.0}, 3.0}; }

// min 2x + 5  s.t. 1 <= x + 0 <= 4 (ranged row), x free: optimum x=1, y=2.
LpModel lp_ranged_offset() {
  return make_model(1, 1, {{0, 0, 1.0}}, {2.0}, {-kInf}, {kInf}, {1.0}, {4.0}, Sense::Minimize, 5.0);
}
Solution sol_ranged_offset() { return {{1.0}, {2.0}, {0.0}, 7.0}; }

}  // namespace

TEST_CASE(kkt_accepts_known_optimal_pairs) {
  const struct {
    const char* name;
    LpModel model;
    Solution sol;
  } cases[] = {
      {"cover", lp_cover(), sol_cover()},
      {"two_rows", lp_two_rows(), sol_two_rows()},
      {"eq_free", lp_eq_free(), sol_eq_free()},
      {"max", lp_max(), sol_max()},
      {"ranged_offset", lp_ranged_offset(), sol_ranged_offset()},
  };
  for (const auto& c : cases) {
    const KktReport r = check_kkt(c.model, c.sol, kTol);
    if (!r.ok) std::cerr << "  case " << c.name << ": " << r.summary() << "\n";
    CHECK(r.ok);
    CHECK_NEAR(r.gap_abs, 0.0, 1e-12);
  }
}

TEST_CASE(kkt_reports_objectives_of_the_minimization_form) {
  const KktReport r = check_kkt(lp_two_rows(), sol_two_rows(), kTol);
  CHECK_NEAR(r.primal_objective, -2.8, 1e-12);
  CHECK_NEAR(r.dual_objective, -2.8, 1e-12);

  // Max model: min-form objective is the negated maximization value.
  const KktReport rm = check_kkt(lp_max(), sol_max(), kTol);
  CHECK_NEAR(rm.primal_objective, -3.0, 1e-12);
  CHECK_NEAR(rm.dual_objective, -3.0, 1e-12);

  // Offset enters both objectives.
  const KktReport ro = check_kkt(lp_ranged_offset(), sol_ranged_offset(), kTol);
  CHECK_NEAR(ro.primal_objective, 7.0, 1e-12);
  CHECK_NEAR(ro.dual_objective, 7.0, 1e-12);
}

TEST_CASE(kkt_detects_primal_infeasibility) {
  Solution s = sol_cover();
  s.x = {0.5, 0.0};  // violates x1 + x2 >= 1
  s.objective = 0.5;
  const KktReport r = check_kkt(lp_cover(), s, kTol);
  CHECK(!r.ok);
  CHECK_NEAR(r.primal_infeasibility_abs, 0.5, 1e-12);

  s = sol_cover();
  s.x = {1.0, -0.25};  // violates the column lower bound
  const KktReport r2 = check_kkt(lp_cover(), s, kTol);
  CHECK(!r2.ok);
  CHECK_NEAR(r2.primal_infeasibility_abs, 0.25, 1e-12);

  // Row upper bound and column upper bound violations.
  Solution s3 = sol_eq_free();
  s3.x = {2.5, 0.5};  // x1 > 2, row activity 3: only the column bound is violated
  const KktReport r3 = check_kkt(lp_eq_free(), s3, kTol);
  CHECK(!r3.ok);
  CHECK_NEAR(r3.primal_infeasibility_abs, 0.5, 1e-12);
}

TEST_CASE(kkt_detects_dual_sign_violations) {
  Solution s = sol_cover();
  s.y = {-1.0};  // row is a ">=" row: a negative multiplier has no upper bound to push on
  s.d = {2.0, 3.0};
  const KktReport r = check_kkt(lp_cover(), s, kTol);
  CHECK(!r.ok);
  CHECK(r.dual_infeasibility_abs >= 1.0 - 1e-12);

  // Negative reduced cost on a column with an infinite upper bound.
  Solution s2 = sol_cover();
  s2.y = {3.0};  // d = (1-3, 2-3) = (-2, -1)
  s2.d = {-2.0, -1.0};
  const KktReport r2 = check_kkt(lp_cover(), s2, kTol);
  CHECK(!r2.ok);
  CHECK_NEAR(r2.dual_infeasibility_abs, 2.0, 1e-12);
}

TEST_CASE(kkt_detects_wrong_supplied_reduced_costs) {
  Solution s = sol_two_rows();
  s.d = {0.0, 0.5};  // true d is (0, 0)
  const KktReport r = check_kkt(lp_two_rows(), s, kTol);
  CHECK(!r.ok);
  CHECK_NEAR(r.dual_mismatch_abs, 0.5, 1e-12);

  s.d.clear();  // omitting d is allowed: it is recomputed
  CHECK(check_kkt(lp_two_rows(), s, kTol).ok);
}

TEST_CASE(kkt_detects_suboptimal_primal_through_gap_and_complementarity) {
  // Feasible but not optimal: x = (0, 1) for the cover LP has objective 2.
  Solution s = sol_cover();
  s.x = {0.0, 1.0};
  s.objective = 2.0;
  const KktReport r = check_kkt(lp_cover(), s, kTol);
  CHECK(!r.ok);
  CHECK(r.gap_abs > 0.9);
  CHECK(r.complementarity_abs > 0.9);  // d_2 = 1 > 0 but x_2 is away from its lower bound
}

TEST_CASE(kkt_detects_wrong_objective_value_and_wrong_sizes) {
  Solution s = sol_two_rows();
  s.objective = -2.0;
  const KktReport r = check_kkt(lp_two_rows(), s, kTol);
  CHECK(!r.ok);
  CHECK(r.objective_mismatch_rel > 0.1);

  Solution bad = sol_two_rows();
  bad.x.pop_back();
  CHECK(!check_kkt(lp_two_rows(), bad, kTol).ok);
  bad = sol_two_rows();
  bad.y.push_back(0.0);
  CHECK(!check_kkt(lp_two_rows(), bad, kTol).ok);
  bad = sol_two_rows();
  bad.d.pop_back();
  CHECK(!check_kkt(lp_two_rows(), bad, kTol).ok);
}

TEST_CASE(kkt_tolerance_is_relative_to_the_data) {
  // A violation of 1e-8 on a bound of size 1e6 is relatively tiny.
  LpModel m = make_model(1, 1, {{0, 0, 1.0}}, {1.0}, {-kInf}, {kInf}, {1e6}, {kInf});
  Solution s{{1e6 - 1e-8}, {1.0}, {0.0}, 1e6 - 1e-8};
  CHECK(check_kkt(m, s, 1e-9).ok);
  s.x[0] = 1e6 - 1.0;
  s.objective = s.x[0];
  CHECK(!check_kkt(m, s, 1e-9).ok);
}

TEST_CASE(kkt_free_row_and_fixed_column_have_no_sign_requirements) {
  // Free row (no bounds) must carry y = 0; fixed column may carry any d.
  LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, -3.0}, {0.0, 2.0}, {5.0, 2.0},
                         {-kInf}, {kInf});
  Solution s{{0.0, 2.0}, {0.0}, {1.0, -3.0}, -6.0};
  CHECK(check_kkt(m, s, kTol).ok);  // d_2 = -3 on a fixed column is fine

  s.y = {0.5};  // multiplier on a free row is infeasible
  s.d = {0.5, -3.5};
  CHECK(!check_kkt(m, s, kTol).ok);
}
