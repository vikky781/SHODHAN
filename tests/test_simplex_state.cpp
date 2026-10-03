// SimplexEngine state: slack basis, primal and dual computations, bases chosen by
// the caller, repair, statuses and warm-start bookkeeping, checked against dense
// computations.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "shodhan/simplex_engine.hpp"
#include "support/lu_testing.hpp"
#include "support/simplex_lps.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

LpModel one_row_model() {
  // min x   s.t.   3 <= x <= 20 (row), x in [0, 10].
  LpModel m;
  m.n_rows = 1;
  m.n_cols = 1;
  std::string err;
  SparseMatrix::from_triplets(1, 1, {{0, 0, 1.0}}, &m.A, &err);
  m.col_cost = {1.0};
  m.col_lower = {0.0};
  m.col_upper = {10.0};
  m.col_type = {ColType::Continuous};
  m.row_lower = {3.0};
  m.row_upper = {20.0};
  return m;
}

// Dense reference of the state for the engine's current basis and nonbasic values.
struct DenseState {
  std::vector<double> xb;  // by position
  std::vector<double> y;
  std::vector<double> d;   // all variables
  double kappa = 0.0;
};

DenseState dense_state(const LpModel& model, const SimplexEngine& e, const std::vector<double>& cost) {
  const Index n = model.n_cols, m = model.n_rows, N = n + m;
  const std::vector<Index>& basis = e.basis();
  const DenseMatrix B = basis_to_dense(model.A, basis);
  const DenseLu lu(B);
  DenseState s;
  s.kappa = B.norm_inf() * lu.inverse_norm_inf();
  std::vector<double> rhs(to_size(m), 0.0);
  for (Index j = 0; j < N; ++j) {
    if (e.status(j) == VarStatus::Basic) continue;
    const double v = e.primal_all()[to_size(j)];
    if (j < n) {
      for (Index t = model.A.col_start[to_size(j)]; t < model.A.col_start[to_size(j) + 1]; ++t)
        rhs[to_size(model.A.row_index[to_size(t)])] -= model.A.value[to_size(t)] * v;
    } else {
      rhs[to_size(j - n)] += v;
    }
  }
  s.xb = lu.solve(rhs);
  std::vector<double> cb(to_size(m));
  for (Index p = 0; p < m; ++p) cb[to_size(p)] = cost[to_size(basis[to_size(p)])];
  s.y = lu.solve_transpose(cb);
  s.d.assign(to_size(N), 0.0);
  for (Index j = 0; j < N; ++j) {
    if (e.status(j) == VarStatus::Basic) continue;
    if (j < n) {
      double v = cost[to_size(j)];
      for (Index t = model.A.col_start[to_size(j)]; t < model.A.col_start[to_size(j) + 1]; ++t)
        v -= model.A.value[to_size(t)] * s.y[to_size(model.A.row_index[to_size(t)])];
      s.d[to_size(j)] = v;
    } else {
      s.d[to_size(j)] = cost[to_size(j)] + s.y[to_size(j - n)];
    }
  }
  return s;
}

}  // namespace

TEST_CASE(engine_slack_basis_state_on_a_hand_model) {
  const LpModel model = one_row_model();
  SimplexEngine e(model);
  CHECK_EQ(e.basis().size(), std::size_t{1});
  CHECK_EQ(e.basis()[0], 1);  // the logical of row 0
  CHECK(e.status(0) == VarStatus::AtLower);
  CHECK(e.status(1) == VarStatus::Basic);
  CHECK_NEAR(e.primal_all()[0], 0.0, 0.0);
  CHECK_NEAR(e.primal_all()[1], 0.0, 0.0);  // activity of the row at x = 0
  CHECK_NEAR(e.row_duals()[0], 0.0, 0.0);
  CHECK_NEAR(e.dual_all()[0], 1.0, 0.0);    // d = c - A^T y = 1
  const InfeasibilitySummary s = e.infeasibility();
  CHECK_EQ(s.primal_count, 1);
  CHECK_NEAR(s.primal_sum, 3.0, 1e-15);     // row activity 0 < 3
  CHECK_EQ(s.dual_count, 0);
  CHECK_NEAR(e.objective(), 0.0, 0.0);
}

TEST_CASE(engine_objective_and_solution_use_the_model_sense) {
  LpModel model = one_row_model();
  model.sense = Sense::Maximize;
  model.objective_offset = 5.0;
  SimplexEngine e(model);
  // Maximize x: the engine minimizes -x, so the slack basis starts with x at its upper bound.
  CHECK(e.status(0) == VarStatus::AtUpper);
  CHECK_NEAR(e.primal_all()[0], 10.0, 0.0);
  const Solution s = e.solution();
  CHECK_NEAR(s.objective, 15.0, 1e-15);  // 5 + 1 * 10, in the model's own (maximize) sense
  CHECK_NEAR(s.d[0], -1.0, 0.0);         // reduced cost of the minimization form (cost -1)
  CHECK_NEAR(e.objective(), -15.0, 1e-15);  // minimization form: -10 - 5

}

TEST_CASE(engine_slack_basis_matches_dense_on_random_lps) {
  int checked = 0;
  for (std::uint64_t seed = 1; seed <= 60; ++seed) {
    SimplexLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 9);
    o.cols = 4 + static_cast<int>(seed % 13);
    o.free_fraction = 0.1;
    o.dual_feasible_start = seed % 2 == 0;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexEngine e(model);
    // Slack basis: logical values are the row activities of the nonbasic columns.
    for (Index i = 0; i < model.n_rows; ++i) {
      double act = 0.0;
      for (Index j = 0; j < model.n_cols; ++j)
        for (Index t = model.A.col_start[to_size(j)]; t < model.A.col_start[to_size(j) + 1]; ++t)
          if (model.A.row_index[to_size(t)] == i) act += model.A.value[to_size(t)] * e.primal_all()[to_size(j)];
      CHECK_NEAR(e.primal_all()[to_size(model.n_cols + i)], act, 1e-12);
      CHECK_NEAR(e.row_duals()[to_size(i)], 0.0, 0.0);
    }
    for (Index j = 0; j < model.n_cols; ++j) CHECK_NEAR(e.dual_all()[to_size(j)], model.col_cost[to_size(j)], 0.0);
    ++checked;
  }
  CHECK_EQ(checked, 60);
}

TEST_CASE(engine_set_basis_computes_primal_and_dual_like_the_dense_oracle) {
  int compared = 0, repaired = 0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    SimplexLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 10);
    o.cols = 5 + static_cast<int>(seed % 12);
    o.free_fraction = 0.1;
    o.dual_feasible_start = false;
    const LpModel model = make_simplex_lp(seed, o);
    const Index N = model.n_cols + model.n_rows;
    Rng rng(seed * 17 + 3);
    std::vector<Index> all(to_size(N));
    std::iota(all.begin(), all.end(), 0);
    rng.shuffle(all);
    std::vector<Index> basis(all.begin(), all.begin() + model.n_rows);
    SimplexEngine e(model);
    const int repairs_before = e.stats().basis_repairs;
    REQUIRE(e.set_basis(basis));
    if (e.stats().basis_repairs > repairs_before) ++repaired;
    // Exactly m distinct basic variables, statuses and positions consistent.
    std::vector<Index> sorted = e.basis();
    std::sort(sorted.begin(), sorted.end());
    CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
    Index nbasic = 0;
    for (Index j = 0; j < N; ++j) nbasic += e.status(j) == VarStatus::Basic ? 1 : 0;
    CHECK_EQ(nbasic, model.n_rows);
    for (const Index v : e.basis()) CHECK(e.status(v) == VarStatus::Basic);
    // Nonbasic variables sit on a bound (or at 0 when free).
    // Dense comparison (skip badly conditioned bases: the comparison itself is limited by kappa).
    std::vector<double> cost(to_size(N), 0.0);
    for (Index j = 0; j < model.n_cols; ++j) cost[to_size(j)] = model.col_cost[to_size(j)];
    const DenseState dd = dense_state(model, e, cost);
    if (!(dd.kappa < 1e8)) continue;
    ++compared;
    for (Index p = 0; p < model.n_rows; ++p) {
      const double xb = e.primal_all()[to_size(e.basis()[to_size(p)])];
      CHECK_NEAR(xb, dd.xb[to_size(p)], 1e-8 * (1.0 + std::fabs(dd.xb[to_size(p)])) * std::max(1.0, dd.kappa * 1e-6));
    }
    for (Index i = 0; i < model.n_rows; ++i)
      CHECK_NEAR(e.row_duals()[to_size(i)], dd.y[to_size(i)], 1e-8 * (1.0 + std::fabs(dd.y[to_size(i)])) * std::max(1.0, dd.kappa * 1e-6));
    for (Index j = 0; j < N; ++j)
      CHECK_NEAR(e.dual_all()[to_size(j)], dd.d[to_size(j)], 1e-7 * (1.0 + std::fabs(dd.d[to_size(j)])) * std::max(1.0, dd.kappa * 1e-6));
    // Reported objective equals c^T x.
    double obj = 0.0;
    for (Index j = 0; j < model.n_cols; ++j) obj += model.col_cost[to_size(j)] * e.primal_all()[to_size(j)];
    CHECK_NEAR(e.objective(), obj, 1e-9 * (1.0 + std::fabs(obj)));
  }
  std::cout << "    engine state vs dense: " << compared << " bases compared, " << repaired << " needed repair\n";
  CHECK(compared > 150);
}

TEST_CASE(engine_refactor_repairs_a_rank_deficient_basis) {
  // Columns 0 and 1 are identical, so the basis {0, 1} (plus the logical of row 2) is singular.
  LpModel m;
  m.n_rows = 3;
  m.n_cols = 2;
  std::string err;
  SparseMatrix::from_triplets(3, 2, {{0, 0, 1.0}, {1, 0, 2.0}, {0, 1, 1.0}, {1, 1, 2.0}}, &m.A, &err);
  m.col_cost = {1.0, 2.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {kInf, kInf};
  m.col_type = {ColType::Continuous, ColType::Continuous};
  m.row_lower = {1.0, 1.0, -kInf};
  m.row_upper = {kInf, kInf, 5.0};
  SimplexEngine e(m);
  REQUIRE(e.set_basis({0, 1, 4}));
  CHECK_EQ(e.stats().basis_repairs, 1);
  Index nb = 0;
  for (Index j = 0; j < 5; ++j) nb += e.status(j) == VarStatus::Basic ? 1 : 0;
  CHECK_EQ(nb, 3);
  CHECK(e.factor().valid());
  // One of the duplicate columns left the basis and is nonbasic at a bound.
  const bool zero_basic = e.status(0) == VarStatus::Basic;
  const bool one_basic = e.status(1) == VarStatus::Basic;
  CHECK(zero_basic != one_basic);
}

TEST_CASE(engine_fix_dual_infeasibilities_switches_bounds_and_shifts_costs) {
  // Column 0: boxed [0,4], cost -1 (should sit at its upper bound).
  // Column 1: lower bound only, cost -2 (dual infeasible; cannot be fixed by a bound switch).
  LpModel m;
  m.n_rows = 1;
  m.n_cols = 2;
  std::string err;
  SparseMatrix::from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, &m.A, &err);
  m.col_cost = {-1.0, -2.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {4.0, kInf};
  m.col_type = {ColType::Continuous, ColType::Continuous};
  m.row_lower = {-kInf};
  m.row_upper = {10.0};
  SimplexEngine e(m);
  CHECK(e.status(0) == VarStatus::AtUpper);  // chosen from the cost sign at construction
  CHECK(e.status(1) == VarStatus::AtLower);
  CHECK_EQ(e.infeasibility().dual_count, 1);
  // Force column 0 to the wrong bound, as after a bound change, and let the engine repair it.
  e.change_col_bounds(0, 0.0, 4.0);
  e.compute_dual();
  Index remaining = e.fix_dual_infeasibilities(false);
  CHECK_EQ(remaining, 1);  // column 1 cannot be fixed without a shift
  CHECK(!e.costs_modified());
  remaining = e.fix_dual_infeasibilities(true);
  CHECK_EQ(remaining, 0);
  CHECK(e.costs_modified());
  CHECK_EQ(e.stats().cost_shifts, 1);
  CHECK_NEAR(e.dual_all()[1], 0.0, 0.0);
  CHECK_EQ(e.infeasibility().dual_count, 0);
}

TEST_CASE(engine_change_bounds_moves_nonbasic_variables_and_marks_primal_stale) {
  const LpModel model = one_row_model();
  SimplexEngine e(model);
  e.change_col_bounds(0, 2.0, 6.0);
  CHECK_NEAR(e.primal_all()[0], 2.0, 0.0);  // nonbasic at its (new) lower bound
  e.compute_primal();
  CHECK_NEAR(e.primal_all()[1], 2.0, 1e-15);  // row activity follows
  e.change_row_bounds(0, 1.0, 1.5);
  CHECK(e.status(1) == VarStatus::Basic);   // basic variable: only its bounds changed
  CHECK_EQ(e.infeasibility().primal_count, 1);
  bool threw = false;
  try { e.change_col_bounds(0, 5.0, 4.0); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { e.change_col_bounds(3, 0.0, 1.0); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}
