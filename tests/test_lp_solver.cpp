// The LP pipeline (presolve, scaling, simplex, unscale, postsolve, KKT on the original) and
// warm starts of the engine.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/rays.hpp"
#include "shodhan/simplex_engine.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/random_lp.hpp"
#include "support/rng.hpp"
#include "support/simplex_lps.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

LpOptions options(bool presolve, bool scaling) {
  LpOptions o;
  o.presolve = presolve;
  o.scaling = scaling;
  return o;
}

// "" when the pipeline result agrees with the oracle and carries valid evidence.
std::string check_result(const LpModel& model, const LpResult& r, const RefLpResult& ref) {
  if (ref.status == Status::NumericalError) return "";
  if (r.status != ref.status) return std::string("status ") + to_string(r.status) + " but the oracle says " + to_string(ref.status);
  if (r.status == Status::Optimal) {
    const double rel = std::fabs(r.solution.objective - ref.solution.objective) / (1.0 + std::fabs(ref.solution.objective));
    if (!(rel <= 1e-6)) return "objective " + std::to_string(r.solution.objective) + " vs oracle " + std::to_string(ref.solution.objective);
    if (!check_kkt(model, r.solution, 1e-6).ok || !r.kkt.ok) return "KKT check on the original model failed";
  } else if (r.status == Status::Infeasible) {
    if (!check_farkas(model, r.farkas_ray, 1e-9).ok) return "the Farkas certificate does not check";
  } else if (r.status == Status::Unbounded) {
    if (!check_unbounded_ray(model, r.unbounded_ray, 1e-8).ok) return "the unbounded ray does not check";
  }
  return "";
}

}  // namespace

TEST_CASE(lp_solver_hand_model) {
  // min 2x + 3y s.t. x + y >= 4, x + 3y >= 6, x, y >= 0  -> 9 at (3, 1).
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
  for (const bool presolve : {false, true}) {
    for (const bool scaling : {false, true}) {
      const LpResult r = LpSolver(options(presolve, scaling)).solve(m);
      REQUIRE(r.status == Status::Optimal);
      CHECK_NEAR(r.solution.objective, 9.0, 1e-9);
      CHECK_NEAR(r.solution.x[0], 3.0, 1e-9);
      CHECK_NEAR(r.solution.x[1], 1.0, 1e-9);
      CHECK(r.kkt.ok);
      CHECK_EQ(r.attempts, 1);
      CHECK(r.iterations >= 0);
    }
  }
  // Maximization: max -2x - 3y gives -9, with the same x.
  LpModel mx = m;
  mx.sense = Sense::Maximize;
  for (double& c : mx.col_cost) c = -c;
  const LpResult rx = LpSolver().solve(mx);
  REQUIRE(rx.status == Status::Optimal);
  CHECK_NEAR(rx.solution.objective, -9.0, 1e-9);
  CHECK_NEAR(rx.solution.x[0], 3.0, 1e-9);
  CHECK(check_kkt(mx, rx.solution, 1e-9).ok);
}

TEST_CASE(lp_solver_matches_the_oracle_in_every_configuration) {
  int agree = 0, total = 0, optimal = 0, infeasible = 0, unbounded = 0, fallbacks = 0, inconclusive = 0;
  std::map<std::string, int> reasons;
  double worst_kkt = 0.0;
  for (std::uint64_t seed = 1; seed <= 400; ++seed) {
    // Models with planted structure (presolve has work to do), some wide, some degenerate, some max.
    RandomLpOptions ro;
    ro.rows = 4 + static_cast<int>(seed % 10);
    ro.cols = 5 + static_cast<int>(seed % 13);
    ro.density = 0.3;
    ro.degeneracy = seed % 4 == 0 ? 0.5 : 0.0;
    ro.free_col_fraction = seed % 5 == 0 ? 0.15 : 0.0;
    ro.ranged_row_fraction = 0.2;
    ro.free_row_fraction = seed % 6 == 0 ? 0.1 : 0.0;
    ro.wide_coefficients = seed % 7 == 0;
    ro.fixed_cols = static_cast<int>(seed % 3);
    ro.empty_cols = static_cast<int>(seed % 2);
    ro.singleton_rows = static_cast<int>(seed % 3);
    ro.doubleton_eqs = static_cast<int>(seed % 2);
    LpModel model;
    switch (seed % 4) {
      case 0: model = make_random_lp(seed, ro).model; break;
      case 1: model = make_random_infeasible_lp(seed, ro); break;
      case 2: model = make_random_unbounded_lp(seed, ro); break;
      default: {
        SimplexLpOptions so;
        so.rows = ro.rows;
        so.cols = ro.cols;
        so.dual_feasible_start = false;
        so.free_fraction = 0.1;
        so.boxed_fraction = 0.3;
        model = make_simplex_lp(seed, so);
        break;
      }
    }
    if (seed % 3 == 0) {  // maximization: same feasible set, opposite objective sense
      model.sense = Sense::Maximize;
      for (double& c : model.col_cost) c = -c;
      model.objective_offset = -model.objective_offset;
    }
    const RefLpResult ref = solve_dense_lp(model);
    if (ref.status == Status::NumericalError) {
      ++inconclusive;
      continue;
    }
    for (const bool presolve : {true, false}) {
      for (const bool scaling : {true, false}) {
        const LpResult r = LpSolver(options(presolve, scaling)).solve(model);
        ++total;
        if (r.attempts > 1 && !r.message.empty()) {  // presolve statuses are merely confirmed (attempts 2, no message)
          ++fallbacks;
          ++reasons[r.message.substr(0, 90)];
        }
        const std::string why = check_result(model, r, ref);
        if (!why.empty()) {
          std::cerr << "FAILING SEED " << seed << " (presolve " << presolve << ", scaling " << scaling << "): " << why << " [" << r.message << "]\n";
          CHECK(false);
          continue;
        }
        ++agree;
        if (r.status == Status::Optimal) {
          ++optimal;
          worst_kkt = std::max({worst_kkt, r.kkt.primal_infeasibility_rel, r.kkt.dual_infeasibility_rel, r.kkt.complementarity_rel, r.kkt.gap_rel});
        }
        if (r.status == Status::Infeasible) ++infeasible;
        if (r.status == Status::Unbounded) ++unbounded;
      }
    }
  }
  std::cout << "    pipeline vs oracle: " << agree << " of " << total << " agree (" << optimal << " optimal, " << infeasible << " infeasible, " << unbounded
            << " unbounded), " << inconclusive << " oracle-inconclusive models skipped, " << fallbacks << " solves needed a fallback; worst KKT residual "
            << std::scientific << std::setprecision(1) << worst_kkt << std::defaultfloat << "\n";
  for (const auto& kv : reasons) std::cout << "      fallback reason (" << kv.second << " solves): " << kv.first << "\n";
  CHECK_EQ(agree, total);
}

TEST_CASE(lp_solver_reports_limits_and_invalid_input) {
  SimplexLpOptions so;
  so.rows = 25;
  so.cols = 40;
  const LpModel model = make_simplex_lp(3, so);
  LpOptions o;
  o.iteration_limit = 1;
  o.presolve = false;
  const LpResult r = LpSolver(o).solve(model);
  CHECK(r.status == Status::IterationLimit || r.status == Status::Optimal || r.status == Status::Infeasible);
  if (r.status == Status::IterationLimit) CHECK_EQ(r.iterations, 1);
  LpOptions t;
  t.params.time_limit = 0.0;
  t.presolve = false;
  CHECK(LpSolver(t).solve(model).status == Status::TimeLimit);
  LpModel bad = model;
  bad.col_lower[0] = 5.0;
  bad.col_upper[0] = 1.0;
  const LpResult rb = LpSolver().solve(bad);
  CHECK(rb.status == Status::NumericalError);
  CHECK(!rb.message.empty());
}

// ---- warm start -------------------------------------------------------------------------------

// Solve, change one column bound, resolve from the current basis, and compare with a cold solve
// of the modified model: same classification, same objective.
TEST_CASE(engine_warm_start_matches_a_cold_solve_after_a_bound_change) {
  int compared = 0, infeasible_both = 0, failed = 0;
  long long warm_iters = 0, cold_iters = 0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    SimplexLpOptions o;
    o.rows = 5 + static_cast<int>(seed % 12);
    o.cols = 8 + static_cast<int>(seed % 15);
    o.boxed_fraction = 0.4;
    o.density = 0.3;
    o.degenerate = seed % 3 == 0 ? 0.4 : 0.0;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexEngine warm(model);
    if (warm.solve() != EngineStatus::Optimal) continue;
    Rng rng(seed * 101 + 5);
    const Index j = rng.range(0, model.n_cols - 1);
    const double lo = model.col_lower[to_size(j)], hi = model.col_upper[to_size(j)];
    const double xj = warm.primal_all()[to_size(j)];
    double nlo = lo, nhi = hi;
    switch (seed % 4) {
      case 0: nhi = std::max(lo, xj - static_cast<double>(rng.range(1, 8)) / 4.0); break;      // tighten the upper bound (branch down)
      case 1: nlo = std::min(hi, xj + static_cast<double>(rng.range(1, 8)) / 4.0); break;      // tighten the lower bound (branch up)
      case 2: nlo = nhi = std::floor(xj) + static_cast<double>(rng.range(0, 3)); break;        // fix
      default: {                                                                               // relax and shift
        nlo = is_inf(lo) ? lo : lo - 1.0;
        nhi = is_inf(hi) ? hi : hi + 1.0;
        break;
      }
    }
    if (nlo > nhi) std::swap(nlo, nhi);
    LpModel changed = model;
    changed.col_lower[to_size(j)] = nlo;
    changed.col_upper[to_size(j)] = nhi;

    const long long before = warm.stats().iterations;
    warm.change_col_bounds(j, nlo, nhi);
    const EngineStatus sw = warm.solve();
    const long long warm_n = warm.stats().iterations - before;
    SimplexEngine cold(changed);
    const EngineStatus sc = cold.solve();
    const long long cold_n = cold.stats().iterations;
    if (sw != sc) {
      ++failed;
      std::cerr << "FAILING SEED " << seed << ": warm " << to_string(sw) << ", cold " << to_string(sc) << "\n";
      CHECK(false);
      continue;
    }
    if (sw == EngineStatus::Optimal) {
      const double ow = warm.solution().objective, oc = cold.solution().objective;
      if (!(std::fabs(ow - oc) <= 1e-7 * (1.0 + std::fabs(oc)))) {
        ++failed;
        std::cerr << "FAILING SEED " << seed << ": warm objective " << ow << ", cold " << oc << "\n";
        CHECK(false);
        continue;
      }
      CHECK(check_kkt(changed, warm.solution(), 1e-6).ok);
    } else if (sw == EngineStatus::Infeasible) {
      ++infeasible_both;
      CHECK(check_farkas(changed, warm.farkas_ray(), 1e-9).ok);
    }
    ++compared;
    warm_iters += warm_n;
    cold_iters += cold_n;
  }
  std::cout << "    warm start vs cold solve: " << compared << " cases agree (" << infeasible_both << " infeasible after the change), " << failed
            << " failed; average iterations warm " << std::fixed << std::setprecision(2) << static_cast<double>(warm_iters) / std::max(compared, 1) << " vs cold "
            << static_cast<double>(cold_iters) / std::max(compared, 1) << std::defaultfloat << "\n";
  CHECK_EQ(failed, 0);
  CHECK(compared > 200);
}

TEST_CASE(engine_warm_start_sequence_of_bound_changes_and_row_changes) {
  // Several changes in a row without any cold solve in between: the engine keeps its basis.
  for (std::uint64_t seed = 400; seed < 440; ++seed) {
    SimplexLpOptions o;
    o.rows = 10;
    o.cols = 16;
    o.boxed_fraction = 0.5;
    LpModel model = make_simplex_lp(seed, o);
    SimplexEngine e(model);
    if (e.solve() != EngineStatus::Optimal) continue;
    Rng rng(seed);
    for (int step = 0; step < 6; ++step) {
      const bool row = rng.chance(0.3);
      if (row) {
        const Index i = rng.range(0, model.n_rows - 1);
        double lo = model.row_lower[to_size(i)], hi = model.row_upper[to_size(i)];
        if (!is_inf(lo)) lo -= static_cast<double>(rng.range(0, 4)) / 2.0;
        if (!is_inf(hi)) hi += static_cast<double>(rng.range(0, 4)) / 2.0;
        model.row_lower[to_size(i)] = lo;
        model.row_upper[to_size(i)] = hi;
        e.change_row_bounds(i, lo, hi);
      } else {
        const Index j = rng.range(0, model.n_cols - 1);
        double lo = model.col_lower[to_size(j)], hi = model.col_upper[to_size(j)];
        if (!is_inf(hi) && !is_inf(lo) && hi > lo && rng.chance(0.5)) hi = lo + (hi - lo) / 2.0;
        else if (!is_inf(lo) && rng.chance(0.5)) lo += static_cast<double>(rng.range(0, 3)) / 2.0, hi = std::max(hi, lo);
        model.col_lower[to_size(j)] = lo;
        model.col_upper[to_size(j)] = hi;
        e.change_col_bounds(j, lo, hi);
      }
      const EngineStatus st = e.solve();
      SimplexEngine cold(model);
      const EngineStatus sc = cold.solve();
      CHECK(st == sc);
      if (st != sc) break;
      if (st == EngineStatus::Optimal) {
        CHECK_NEAR(e.solution().objective, cold.solution().objective, 1e-7 * (1.0 + std::fabs(cold.solution().objective)));
      } else {
        break;  // after infeasibility the sequence stops
      }
    }
  }
}
