// Dual simplex with all robustness features (bound flipping, Harris, perturbation,
// stall handling, cleanup) against the dense oracle.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/scaling.hpp"
#include "shodhan/simplex_engine.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/simplex_lps.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

// Solves with the engine and compares with the oracle. Returns "" when they agree.
std::string agree(const LpModel& model, const SimplexOptions& opt, long long* iterations, int* optimal, int* infeasible,
                  double* worst_kkt) {
  SimplexEngine e(model, opt);
  const EngineStatus st = e.solve();
  const RefLpResult ref = solve_dense_lp(model);
  if (iterations != nullptr) *iterations += e.stats().iterations;
  if (ref.status == Status::NumericalError) return "";  // oracle inconclusive
  if (st == EngineStatus::Optimal) {
    if (ref.status != Status::Optimal) return std::string("engine Optimal but oracle ") + to_string(ref.status);
    const Solution sol = e.solution();
    const double rel = std::fabs(sol.objective - ref.solution.objective) / (1.0 + std::fabs(ref.solution.objective));
    if (!(rel <= 1e-7)) return "objective " + std::to_string(sol.objective) + " vs oracle " + std::to_string(ref.solution.objective);
    const KktReport k = check_kkt(model, sol, 1e-6);
    if (worst_kkt != nullptr) {
      *worst_kkt = std::max({*worst_kkt, k.primal_infeasibility_rel, k.dual_infeasibility_rel, k.complementarity_rel, k.gap_rel});
    }
    if (!k.ok) return "KKT check failed: " + k.summary();
    if (optimal != nullptr) ++*optimal;
    return "";
  }
  if (st == EngineStatus::Infeasible) {
    if (ref.status != Status::Infeasible) return std::string("engine Infeasible but oracle ") + to_string(ref.status);
    if (infeasible != nullptr) ++*infeasible;
    return "";
  }
  return std::string("engine returned ") + to_string(st) + " (oracle " + to_string(ref.status) + ")";
}

}  // namespace

TEST_CASE(dual_robust_default_options_match_the_oracle) {
  int optimal = 0, infeasible = 0, failed = 0, total = 0;
  long long iters = 0;
  double worst_kkt = 0.0;
  for (std::uint64_t seed = 1; seed <= 800; ++seed) {
    SimplexLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 14);
    o.cols = 5 + static_cast<int>(seed % 22);
    o.density = 0.12 + 0.05 * static_cast<double>(seed % 7);
    o.boxed_fraction = 0.1 + 0.1 * static_cast<double>(seed % 6);
    o.feasible = seed % 8 != 0;
    o.degenerate = seed % 3 == 0 ? 0.5 : 0.0;
    const LpModel model = make_simplex_lp(seed, o);
    const std::string why = agree(model, SimplexOptions{}, &iters, &optimal, &infeasible, &worst_kkt);
    ++total;
    if (!why.empty()) {
      ++failed;
      std::cerr << "FAILING SEED " << seed << ": " << why << "\n";
      CHECK(false);
    }
  }
  std::cout << "    dual simplex (all upgrades): " << total << " LPs, " << optimal << " optimal, " << infeasible << " infeasible, " << failed
            << " failed; worst KKT residual " << std::scientific << std::setprecision(1) << worst_kkt << std::defaultfloat << ", "
            << iters << " iterations\n";
  CHECK_EQ(failed, 0);
}

TEST_CASE(dual_robust_perturbation_on_and_off_give_the_same_optimum_on_degenerate_lps) {
  long long it_on = 0, it_off = 0;
  int compared = 0, perturbed_runs = 0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    SimplexLpOptions o;
    o.rows = 8 + static_cast<int>(seed % 12);
    o.cols = 12 + static_cast<int>(seed % 18);
    o.degenerate = 0.8;
    o.equality_fraction = 0.35;
    o.boxed_fraction = 0.4;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexOptions on, off;
    off.perturb = false;
    SimplexEngine a(model, on), b(model, off);
    const EngineStatus sa = a.solve(), sb = b.solve();
    if (sa != sb) {
      std::cerr << "FAILING SEED " << seed << ": status " << to_string(sa) << " with perturbation, " << to_string(sb) << " without\n";
      CHECK(false);
      continue;
    }
    it_on += a.stats().iterations;
    it_off += b.stats().iterations;
    if (a.stats().perturbed) ++perturbed_runs;
    if (sa == EngineStatus::Optimal) {
      ++compared;
      const double oa = a.solution().objective, ob = b.solution().objective;
      if (!(std::fabs(oa - ob) <= 1e-7 * (1.0 + std::fabs(ob)))) {
        std::cerr << "FAILING SEED " << seed << ": objective " << oa << " with perturbation, " << ob << " without\n";
        CHECK(false);
      }
      CHECK(check_kkt(model, a.solution(), 1e-6).ok);
      CHECK(check_kkt(model, b.solution(), 1e-6).ok);
    }
  }
  std::cout << "    perturbation on/off on degenerate LPs: " << compared << " optimal pairs agree; iterations " << it_on << " (on) vs "
            << it_off << " (off); perturbation used in " << perturbed_runs << " runs\n";
  CHECK(compared > 150);
}

TEST_CASE(dual_robust_stall_detection_triggers_larger_perturbations) {
  // A tiny stall window makes any short plateau count as a stall.
  int triggered = 0;
  for (std::uint64_t seed = 1; seed <= 100; ++seed) {
    SimplexLpOptions o;
    o.rows = 15;
    o.cols = 25;
    o.degenerate = 0.9;
    o.equality_fraction = 0.4;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexOptions opt;
    opt.perturb_at_start = false;  // so that plateaus can occur
    opt.stall_iterations = 1;
    SimplexEngine e(model, opt);
    const EngineStatus st = e.solve();
    const RefLpResult ref = solve_dense_lp(model);
    if (ref.status == Status::NumericalError) continue;
    if (st == EngineStatus::Optimal) {
      CHECK(ref.status == Status::Optimal);
      if (ref.status == Status::Optimal) {
        CHECK_NEAR(e.solution().objective, ref.solution.objective, 1e-7 * (1.0 + std::fabs(ref.solution.objective)));
      }
      CHECK(check_kkt(model, e.solution(), 1e-6).ok);
    } else {
      CHECK(st == EngineStatus::Infeasible && ref.status == Status::Infeasible);
    }
    if (e.stats().perturbation_rounds > 1) ++triggered;
  }
  std::cout << "    stall handling: larger perturbation applied in " << triggered << " of 100 runs\n";
  CHECK(triggered > 0);
}

TEST_CASE(dual_robust_wide_coefficients_after_scaling) {
  int compared = 0, failed = 0;
  double worst = 0.0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    SimplexLpOptions o;
    o.rows = 4 + static_cast<int>(seed % 10);
    o.cols = 6 + static_cast<int>(seed % 16);
    o.wide_coefficients = true;
    o.degenerate = seed % 2 == 0 ? 0.3 : 0.0;
    const LpModel original = make_simplex_lp(seed, o);
    const RefLpResult ref = solve_dense_lp(original);
    if (ref.status == Status::NumericalError) continue;
    const Scaling sc = compute_scaling(original);
    const LpModel scaled = apply_scaling(original, sc);
    SimplexEngine e(scaled);
    const EngineStatus st = e.solve();
    if (st == EngineStatus::Optimal) {
      if (ref.status != Status::Optimal) {
        ++failed;
        std::cerr << "FAILING SEED " << seed << ": engine Optimal, oracle " << to_string(ref.status) << "\n";
        CHECK(false);
        continue;
      }
      const Solution sol = unscale_solution(sc, e.solution());
      const double rel = std::fabs(sol.objective - ref.solution.objective) / (1.0 + std::fabs(ref.solution.objective));
      worst = std::max(worst, rel);
      const KktReport k = check_kkt(original, sol, 1e-6);
      if (!(rel <= 1e-6) || !k.ok) {
        ++failed;
        std::cerr << "FAILING SEED " << seed << ": objective error " << rel << ", KKT " << k.summary() << "\n";
        CHECK(false);
      }
      ++compared;
    } else if (st == EngineStatus::Infeasible) {
      if (ref.status != Status::Infeasible) {
        ++failed;
        std::cerr << "FAILING SEED " << seed << ": engine Infeasible, oracle " << to_string(ref.status) << "\n";
        CHECK(false);
      }
    } else {
      ++failed;
      std::cerr << "FAILING SEED " << seed << ": engine " << to_string(st) << "\n";
      CHECK(false);
    }
  }
  std::cout << "    wide coefficients (1e-4..1e4) after scaling: " << compared << " optimal compared, " << failed
            << " failed, worst objective error " << std::scientific << std::setprecision(1) << worst << std::defaultfloat << "\n";
  CHECK_EQ(failed, 0);
}

TEST_CASE(dual_robust_every_option_combination_agrees) {
  // 16 combinations of the switches on the same LPs: same status and objective.
  for (std::uint64_t seed = 5000; seed < 5060; ++seed) {
    SimplexLpOptions o;
    o.rows = 10;
    o.cols = 16;
    o.degenerate = 0.4;
    const LpModel model = make_simplex_lp(seed, o);
    bool have = false;
    EngineStatus first_status = EngineStatus::Optimal;
    double first_obj = 0.0;
    for (int mask = 0; mask < 16; ++mask) {
      SimplexOptions opt;
      opt.bound_flipping = (mask & 1) != 0;
      opt.harris = (mask & 2) != 0;
      opt.perturb = (mask & 4) != 0;
      opt.dual_steepest_edge = (mask & 8) != 0;
      SimplexEngine e(model, opt);
      const EngineStatus st = e.solve();
      if (!have) {
        have = true;
        first_status = st;
        if (st == EngineStatus::Optimal) first_obj = e.solution().objective;
        continue;
      }
      CHECK(st == first_status);
      if (st == EngineStatus::Optimal && first_status == EngineStatus::Optimal) {
        CHECK_NEAR(e.solution().objective, first_obj, 1e-7 * (1.0 + std::fabs(first_obj)));
      }
    }
  }
}

TEST_CASE(dual_robust_limits_are_reported_honestly) {
  SimplexLpOptions o;
  o.rows = 30;
  o.cols = 45;
  o.density = 0.2;
  const LpModel model = make_simplex_lp(11, o);
  {
    SimplexOptions opt;
    opt.iteration_limit = 2;
    SimplexEngine e(model, opt);
    const EngineStatus st = e.solve();
    CHECK(st == EngineStatus::IterationLimit || st == EngineStatus::Optimal || st == EngineStatus::Infeasible);
    if (st == EngineStatus::IterationLimit) CHECK(e.stats().iterations <= 2);
  }
  {
    SimplexOptions opt;
    opt.time_limit = 0.0;  // already exceeded
    SimplexEngine e(model, opt);
    const EngineStatus st = e.solve();
    CHECK(st == EngineStatus::TimeLimit || st == EngineStatus::Optimal || st == EngineStatus::Infeasible);
    CHECK(st == EngineStatus::TimeLimit);
  }
}

TEST_CASE(dual_robust_refactor_interval_and_growth_limit_do_not_change_the_answer) {
  for (std::uint64_t seed = 6000; seed < 6040; ++seed) {
    SimplexLpOptions o;
    o.rows = 14;
    o.cols = 24;
    o.degenerate = 0.3;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexOptions a, b;
    b.refactor_interval = 1;  // refactor after every pivot
    b.max_growth = 0.5;       // growth indicator is always above this: refactor whenever updates exist
    SimplexEngine ea(model, a), eb(model, b);
    const EngineStatus sa = ea.solve(), sb = eb.solve();
    CHECK(sa == sb);
    if (sa == EngineStatus::Optimal && sb == EngineStatus::Optimal) {
      CHECK_NEAR(ea.solution().objective, eb.solution().objective, 1e-7 * (1.0 + std::fabs(ea.solution().objective)));
      CHECK(check_kkt(model, eb.solution(), 1e-6).ok);
    }
  }
}

TEST_CASE(dual_robust_primal_cleanup_after_a_large_perturbation) {
  // A large perturbation changes the optimal basis, so removing it leaves dual infeasibilities
  // that the primal simplex has to remove.
  int cleaned = 0, optimal = 0, failed = 0;
  long long primal_iters = 0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    SimplexLpOptions o;
    o.rows = 6 + static_cast<int>(seed % 10);
    o.cols = 10 + static_cast<int>(seed % 14);
    o.boxed_fraction = 0.35;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexOptions opt;
    opt.perturb_scale = 0.5;
    SimplexEngine e(model, opt);
    const EngineStatus st = e.solve();
    const RefLpResult ref = solve_dense_lp(model);
    if (ref.status == Status::NumericalError) continue;
    primal_iters += e.stats().primal_iterations;
    if (e.stats().cleanup_rounds > 0) ++cleaned;
    if (st == EngineStatus::Optimal) {
      ++optimal;
      if (ref.status != Status::Optimal ||
          !(std::fabs(e.solution().objective - ref.solution.objective) <= 1e-7 * (1.0 + std::fabs(ref.solution.objective))) ||
          !check_kkt(model, e.solution(), 1e-6).ok) {
        ++failed;
        std::cerr << "FAILING SEED " << seed << ": result differs from the oracle after cleanup\n";
        CHECK(false);
      }
    } else if (st != EngineStatus::Infeasible || ref.status != Status::Infeasible) {
      ++failed;
      std::cerr << "FAILING SEED " << seed << ": engine " << to_string(st) << ", oracle " << to_string(ref.status) << "\n";
      CHECK(false);
    }
  }
  std::cout << "    primal cleanup after a large perturbation: " << optimal << " optimal, cleanup needed in " << cleaned << " runs, " << primal_iters
            << " primal iterations, " << failed << " failed\n";
  CHECK_EQ(failed, 0);
  CHECK(cleaned > 20);
}
