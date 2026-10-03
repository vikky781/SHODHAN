// Dual simplex phase 2 against the dense oracle and the KKT checker.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/simplex_engine.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/simplex_lps.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

struct Tally {
  int total = 0, optimal = 0, infeasible = 0, oracle_inconclusive = 0, failed = 0;
  long long iterations = 0;
  double worst_obj = 0.0;
  double worst_kkt = 0.0;
};

// Runs one LP through the engine and the oracle; returns an empty string when they agree.
std::string compare(const LpModel& model, const SimplexOptions& opt, Tally* tally, bool* optimal_out = nullptr) {
  SimplexEngine e(model, opt);
  const EngineStatus st = e.solve();
  const RefLpResult ref = solve_dense_lp(model);
  ++tally->total;
  tally->iterations += e.stats().iterations;
  if (ref.status == Status::NumericalError) {
    ++tally->oracle_inconclusive;
    return "";
  }
  if (st == EngineStatus::Optimal) {
    if (ref.status != Status::Optimal) return std::string("engine Optimal but oracle ") + to_string(ref.status);
    const Solution sol = e.solution();
    const double rel = std::fabs(sol.objective - ref.solution.objective) / (1.0 + std::fabs(ref.solution.objective));
    tally->worst_obj = std::max(tally->worst_obj, rel);
    if (!(rel <= 1e-7)) return "objective " + std::to_string(sol.objective) + " vs oracle " + std::to_string(ref.solution.objective);
    const KktReport k = check_kkt(model, sol, 1e-6);
    tally->worst_kkt = std::max({tally->worst_kkt, k.primal_infeasibility_rel, k.dual_infeasibility_rel, k.complementarity_rel, k.gap_rel});
    if (!k.ok) return "KKT check failed: " + k.summary();
    ++tally->optimal;
    if (optimal_out != nullptr) *optimal_out = true;
    return "";
  }
  if (st == EngineStatus::Infeasible) {
    if (ref.status != Status::Infeasible) return std::string("engine Infeasible but oracle ") + to_string(ref.status);
    ++tally->infeasible;
    return "";
  }
  return std::string("engine returned ") + to_string(st) + " (oracle " + to_string(ref.status) + ")";
}

SimplexOptions textbook_options() {
  SimplexOptions o;
  o.bound_flipping = false;
  o.harris = false;
  o.perturb = false;
  return o;
}

}  // namespace

TEST_CASE(dual_simplex_solves_a_tiny_lp) {
  // min 2x + 3y  s.t.  x + y >= 4,  x + 3y >= 6,  x, y >= 0  -> x = 3, y = 1, objective 9.
  LpModel m;
  m.n_rows = 2;
  m.n_cols = 2;
  std::string err;
  SparseMatrix::from_triplets(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 3.0}}, &m.A, &err);
  m.col_cost = {2.0, 3.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {kInf, kInf};
  m.col_type = {ColType::Continuous, ColType::Continuous};
  m.row_lower = {4.0, 6.0};
  m.row_upper = {kInf, kInf};
  SimplexEngine e(m, textbook_options());
  REQUIRE(e.solve() == EngineStatus::Optimal);
  const Solution s = e.solution();
  CHECK_NEAR(s.x[0], 3.0, 1e-9);
  CHECK_NEAR(s.x[1], 1.0, 1e-9);
  CHECK_NEAR(s.objective, 9.0, 1e-9);
  CHECK(check_kkt(m, s, 1e-9).ok);
  CHECK(e.stats().iterations >= 2);
}

TEST_CASE(dual_simplex_detects_an_infeasible_row) {
  // x + y <= 1 and x + y >= 3 with x, y in [0, 5].
  LpModel m;
  m.n_rows = 2;
  m.n_cols = 2;
  std::string err;
  SparseMatrix::from_triplets(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}}, &m.A, &err);
  m.col_cost = {1.0, 1.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {5.0, 5.0};
  m.col_type = {ColType::Continuous, ColType::Continuous};
  m.row_lower = {-kInf, 3.0};
  m.row_upper = {1.0, kInf};
  SimplexEngine e(m, textbook_options());
  CHECK(e.solve() == EngineStatus::Infeasible);
  CHECK_EQ(e.farkas_ray().size(), std::size_t{2});
}

TEST_CASE(dual_simplex_matches_the_oracle_on_dual_feasible_starts) {
  Tally t;
  for (std::uint64_t seed = 1; seed <= 600; ++seed) {
    SimplexLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 12);
    o.cols = 4 + static_cast<int>(seed % 17);
    o.density = 0.15 + 0.05 * static_cast<double>(seed % 6);
    o.boxed_fraction = 0.1 + 0.1 * static_cast<double>(seed % 5);
    o.feasible = seed % 7 != 0;
    const LpModel model = make_simplex_lp(seed, o);
    const std::string why = compare(model, textbook_options(), &t);
    if (!why.empty()) {
      ++t.failed;
      std::cerr << "FAILING SEED " << seed << ": " << why << "\n";
      CHECK(false);
    }
  }
  std::cout << "    dual simplex (textbook ratio test): " << t.total << " LPs, " << t.optimal << " optimal, " << t.infeasible
            << " infeasible, " << t.oracle_inconclusive << " oracle inconclusive, " << t.failed << " failed; worst objective error "
            << std::scientific << std::setprecision(1) << t.worst_obj << ", worst KKT residual " << t.worst_kkt << ", total iterations "
            << std::defaultfloat << t.iterations << "\n";
  CHECK_EQ(t.failed, 0);
  CHECK(t.optimal > 300);
}

TEST_CASE(dual_simplex_steepest_edge_and_unit_weights_agree) {
  for (std::uint64_t seed = 1000; seed < 1150; ++seed) {
    SimplexLpOptions o;
    o.rows = 8;
    o.cols = 14;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexOptions a = textbook_options(), b = textbook_options();
    b.dual_steepest_edge = false;
    SimplexEngine ea(model, a), eb(model, b);
    const EngineStatus sa = ea.solve(), sb = eb.solve();
    CHECK(sa == sb);
    if (sa == EngineStatus::Optimal && sb == EngineStatus::Optimal) {
      CHECK_NEAR(ea.solution().objective, eb.solution().objective, 1e-7 * (1.0 + std::fabs(ea.solution().objective)));
    }
  }
}

TEST_CASE(dual_simplex_refactors_often_without_changing_the_answer) {
  // A tiny update limit forces many refactorizations (and NeedRefactor paths).
  for (std::uint64_t seed = 2000; seed < 2060; ++seed) {
    SimplexLpOptions o;
    o.rows = 12;
    o.cols = 20;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexOptions a = textbook_options(), b = textbook_options();
    b.refactor_interval = 2;
    SimplexEngine ea(model, a), eb(model, b);
    const EngineStatus sa = ea.solve(), sb = eb.solve();
    CHECK(sa == sb);
    if (sa == EngineStatus::Optimal && sb == EngineStatus::Optimal) {
      CHECK_NEAR(ea.solution().objective, eb.solution().objective, 1e-7 * (1.0 + std::fabs(ea.solution().objective)));
      if (eb.stats().iterations > 4) CHECK(eb.stats().refactors > ea.stats().refactors);
    }
  }
}

TEST_CASE(dual_simplex_iteration_limit_is_reported) {
  SimplexLpOptions o;
  o.rows = 12;
  o.cols = 20;
  const LpModel model = make_simplex_lp(77, o);
  SimplexOptions opt = textbook_options();
  opt.iteration_limit = 1;
  SimplexEngine e(model, opt);
  const EngineStatus st = e.solve();
  CHECK(st == EngineStatus::IterationLimit || st == EngineStatus::Optimal || st == EngineStatus::Infeasible);
  if (st == EngineStatus::IterationLimit) CHECK_EQ(e.stats().iterations, 1);
}

TEST_CASE(dual_simplex_steepest_edge_weights_track_the_exact_weights) {
  // After some iterations the updated weights must equal ||e_i^T B^-1||^2 (up to the
  // lower clamp of 1e-4 and rounding).
  int checked = 0;
  double worst = 0.0;
  for (std::uint64_t seed = 3000; seed < 3100; ++seed) {
    SimplexLpOptions o;
    o.rows = 12;
    o.cols = 20;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexOptions opt = textbook_options();
    opt.iteration_limit = 4;
    SimplexEngine e(model, opt);
    e.solve();
    if (e.stats().iterations == 0) continue;
    SimplexEngine exact = e;
    exact.compute_exact_weights();
    for (Index i = 0; i < model.n_rows; ++i) {
      const double a = e.weights()[to_size(i)], b = exact.weights()[to_size(i)];
      worst = std::max(worst, std::fabs(a - b) / b);
      CHECK_NEAR(a, b, 1e-8 * b);
    }
    ++checked;
  }
  std::cout << "    DSE weights vs exact after 4 iterations: " << checked << " runs, worst relative difference " << std::scientific
            << std::setprecision(1) << worst << std::defaultfloat << "\n";
  CHECK(checked > 50);
}

TEST_CASE(dual_simplex_set_basis_uses_exact_weights) {
  SimplexLpOptions o;
  o.rows = 6;
  o.cols = 10;
  const LpModel model = make_simplex_lp(5, o);
  SimplexEngine e(model, textbook_options());
  REQUIRE(e.solve() == EngineStatus::Optimal);
  const std::vector<Index> basis = e.basis();
  SimplexEngine f(model, textbook_options());
  REQUIRE(f.set_basis(basis));
  CHECK(f.weights_exact());
  bool all_one = true;
  for (const double w : f.weights()) all_one = all_one && w == 1.0;
  CHECK(!all_one);
}

TEST_CASE(dual_simplex_bound_flipping_ratio_test_matches_the_oracle) {
  Tally t;
  long long flips = 0;
  for (std::uint64_t seed = 1; seed <= 600; ++seed) {
    SimplexLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 12);
    o.cols = 6 + static_cast<int>(seed % 20);
    o.density = 0.15 + 0.05 * static_cast<double>(seed % 6);
    o.boxed_fraction = 0.5 + 0.1 * static_cast<double>(seed % 4);  // many boxed variables: flips happen
    o.feasible = seed % 7 != 0;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexOptions opt = textbook_options();
    opt.bound_flipping = true;
    const std::string why = compare(model, opt, &t);
    {
      SimplexEngine e(model, opt);
      e.solve();
      flips += e.stats().bound_flips;
    }
    if (!why.empty()) {
      ++t.failed;
      std::cerr << "FAILING SEED " << seed << ": " << why << "\n";
      CHECK(false);
    }
  }
  std::cout << "    dual simplex (bound flipping): " << t.total << " LPs, " << t.optimal << " optimal, " << t.infeasible
            << " infeasible, " << t.failed << " failed; " << flips << " bound flips performed, " << t.iterations << " iterations\n";
  CHECK_EQ(t.failed, 0);
  CHECK(flips > 100);
}
