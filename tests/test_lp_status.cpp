// Dual phase 1, infeasible and unbounded classification, certificates and edge cases.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/rays.hpp"
#include "shodhan/simplex_engine.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/random_lp.hpp"
#include "support/simplex_lps.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

LpModel make_model(Index m, Index n, const std::vector<Triplet>& t, std::vector<double> cost, std::vector<double> lo,
                   std::vector<double> hi, std::vector<double> rlo, std::vector<double> rhi) {
  LpModel model;
  model.n_rows = m;
  model.n_cols = n;
  std::string err;
  SparseMatrix::from_triplets(m, n, t, &model.A, &err);
  model.col_cost = std::move(cost);
  model.col_lower = std::move(lo);
  model.col_upper = std::move(hi);
  model.col_type.assign(to_size(n), ColType::Continuous);
  model.row_lower = std::move(rlo);
  model.row_upper = std::move(rhi);
  return model;
}

Status to_status(EngineStatus s) {
  switch (s) {
    case EngineStatus::Optimal: return Status::Optimal;
    case EngineStatus::Infeasible: return Status::Infeasible;
    case EngineStatus::Unbounded: return Status::Unbounded;
    default: return Status::NumericalError;
  }
}

}  // namespace

// ---- certificate checkers -------------------------------------------------------------------

TEST_CASE(farkas_check_accepts_a_valid_certificate_and_rejects_corrupted_ones) {
  // x + y <= 1 and x + y >= 3, x, y in [0, 5].  y = (1, -1) gives (x+y) - (x+y) = 0 with
  // r1 <= 1 and r2 >= 3: the combination  r1 - r2 <= -2 can never be 0.
  const LpModel m = make_model(2, 2, {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}}, {1, 1}, {0, 0}, {5, 5}, {-kInf, 3}, {1, kInf});
  RayCheck ok = check_farkas(m, {1.0, -1.0}, 1e-9);
  CHECK(ok.ok);
  CHECK(check_farkas(m, {-2.0, 2.0}, 1e-9).ok);                  // any positive multiple (or its negative) works
  CHECK(!check_farkas(m, {1.0, 1.0}, 1e-9).ok);                  // wrong signs: the combination can be zero
  CHECK(!check_farkas(m, {0.0, 0.0}, 1e-9).ok);                  // zero multipliers prove nothing
  CHECK(!check_farkas(m, {1.0, -0.2}, 1e-9).ok);                 // corrupted: the columns no longer cancel
  CHECK(!check_farkas(m, {1.0}, 1e-9).ok);                       // wrong length
  // A feasible LP has no certificate: x + y <= 4 and x + y >= 3.
  const LpModel feas = make_model(2, 2, {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}}, {1, 1}, {0, 0}, {5, 5}, {-kInf, 3}, {4, kInf});
  for (const double a : {1.0, -1.0, 0.5, 3.0}) {
    CHECK(!check_farkas(feas, {1.0, -a}, 1e-9).ok);
    CHECK(!check_farkas(feas, {a, -1.0}, 1e-9).ok);
  }
}

TEST_CASE(farkas_check_needs_finite_bounds_where_the_coefficients_point) {
  // x >= 0 (no upper bound), row: x <= -1. Certificate y = (1): x - r with r <= -1: x - r >= 0 + 1 > 0.
  const LpModel m = make_model(1, 1, {{0, 0, 1}}, {1}, {0}, {kInf}, {-kInf}, {-1});
  CHECK(check_farkas(m, {1.0}, 1e-9).ok);
  // With x free the same multipliers prove nothing (x can be very negative).
  const LpModel freex = make_model(1, 1, {{0, 0, 1}}, {1}, {-kInf}, {kInf}, {-kInf}, {-1});
  CHECK(!check_farkas(freex, {1.0}, 1e-9).ok);
}

TEST_CASE(unbounded_ray_check_accepts_a_recession_direction_and_rejects_corrupted_ones) {
  // min -x - y s.t. x - y <= 2 (row), x, y >= 0: the ray (1, 1) is feasible forever and improves.
  const LpModel m = make_model(1, 2, {{0, 0, 1}, {0, 1, -1}}, {-1, -1}, {0, 0}, {kInf, kInf}, {-kInf}, {2});
  CHECK(check_unbounded_ray(m, {1.0, 1.0}, 1e-9).ok);
  CHECK(check_unbounded_ray(m, {3.0, 3.0}, 1e-9).ok);
  CHECK(!check_unbounded_ray(m, {1.0, 0.0}, 1e-9).ok);    // x - y grows past the row bound
  CHECK(!check_unbounded_ray(m, {-1.0, -1.0}, 1e-9).ok);  // leaves the column bounds, and worsens the objective
  CHECK(!check_unbounded_ray(m, {0.0, 0.0}, 1e-9).ok);    // zero ray
  CHECK(!check_unbounded_ray(m, {1.0, 1.0, 1.0}, 1e-9).ok);  // wrong length
  // Not improving: with cost +1 the same direction makes the objective worse.
  LpModel worse = m;
  worse.col_cost = {1, 1};
  CHECK(!check_unbounded_ray(worse, {1.0, 1.0}, 1e-9).ok);
  // For a maximization model improving means increasing.
  worse.sense = Sense::Maximize;
  CHECK(check_unbounded_ray(worse, {1.0, 1.0}, 1e-9).ok);
  // A finite upper bound on a column that moves up breaks the ray.
  LpModel boxed = m;
  boxed.col_upper = {kInf, 5.0};
  CHECK(!check_unbounded_ray(boxed, {1.0, 1.0}, 1e-9).ok);
}

// ---- edge cases -----------------------------------------------------------------------------

TEST_CASE(lp_edge_zero_cost_lp_is_feasibility_only) {
  const LpModel m = make_model(2, 3, {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 2, -1}}, {0, 0, 0}, {0, 0, 0}, {10, 10, 10}, {4, -2}, {kInf, 3});
  SimplexEngine e(m);
  REQUIRE(e.solve() == EngineStatus::Optimal);
  CHECK_NEAR(e.solution().objective, 0.0, 1e-12);
  CHECK(check_kkt(m, e.solution(), 1e-9).ok);
}

TEST_CASE(lp_edge_no_rows) {
  // min x - 2y, x in [1, 4], y in [0, 3], no constraints: x = 1, y = 3.
  const LpModel m = make_model(0, 2, {}, {1, -2}, {1, 0}, {4, 3}, {}, {});
  SimplexEngine e(m);
  REQUIRE(e.solve() == EngineStatus::Optimal);
  CHECK_NEAR(e.solution().x[0], 1.0, 0.0);
  CHECK_NEAR(e.solution().x[1], 3.0, 0.0);
  CHECK_NEAR(e.solution().objective, -5.0, 1e-12);
  // And unbounded without rows: a free improving column.
  const LpModel u = make_model(0, 1, {}, {-1}, {0}, {kInf}, {}, {});
  SimplexEngine eu(u);
  CHECK(eu.solve() == EngineStatus::Unbounded);
  CHECK(check_unbounded_ray(u, eu.unbounded_ray(), 1e-9).ok);
}

TEST_CASE(lp_edge_single_variable) {
  // min x s.t. 3 <= 2x <= 8, x free: x = 1.5.
  const LpModel m = make_model(1, 1, {{0, 0, 2}}, {1}, {-kInf}, {kInf}, {3}, {8});
  SimplexEngine e(m);
  REQUIRE(e.solve() == EngineStatus::Optimal);
  CHECK_NEAR(e.solution().x[0], 1.5, 1e-12);
  CHECK(check_kkt(m, e.solution(), 1e-9).ok);
  // Infeasible single variable: 2x >= 3 with x <= 1.
  const LpModel inf = make_model(1, 1, {{0, 0, 2}}, {1}, {0}, {1}, {3}, {kInf});
  SimplexEngine ei(inf);
  CHECK(ei.solve() == EngineStatus::Infeasible);
  CHECK(check_farkas(inf, ei.farkas_ray(), 1e-9).ok);
}

TEST_CASE(lp_edge_fixed_variables_only) {
  // Every column fixed: the LP is a feasibility check of A x against the rows.
  const LpModel feasible = make_model(2, 2, {{0, 0, 1}, {1, 1, 1}}, {5, -3}, {2, 4}, {2, 4}, {1, 0}, {3, 4});
  SimplexEngine e(feasible);
  REQUIRE(e.solve() == EngineStatus::Optimal);
  CHECK_NEAR(e.solution().objective, 5 * 2 - 3 * 4, 1e-12);
  const LpModel infeasible = make_model(2, 2, {{0, 0, 1}, {1, 1, 1}}, {5, -3}, {2, 4}, {2, 4}, {3, 0}, {4, 4});
  SimplexEngine ei(infeasible);
  CHECK(ei.solve() == EngineStatus::Infeasible);
  CHECK(check_farkas(infeasible, ei.farkas_ray(), 1e-9).ok);
}

TEST_CASE(lp_edge_free_variables) {
  // min x + y s.t. x - y = 1, x + y >= 3, x, y free: optimum 3 at any point of the face x + y = 3.
  const LpModel m = make_model(2, 2, {{0, 0, 1}, {0, 1, -1}, {1, 0, 1}, {1, 1, 1}}, {1, 1}, {-kInf, -kInf}, {kInf, kInf}, {1, 3}, {1, kInf});
  SimplexEngine e(m);
  REQUIRE(e.solve() == EngineStatus::Optimal);
  CHECK_NEAR(e.solution().objective, 3.0, 1e-9);
  CHECK(check_kkt(m, e.solution(), 1e-9).ok);
  CHECK(e.stats().phase1_iterations >= 0);
  // Free and improving: min -x - y with the same rows is unbounded along (1, 1)? x - y = 1 fixes the
  // difference, so the ray is (1, 1).
  LpModel u = m;
  u.col_cost = {-1, -1};
  SimplexEngine eu(u);
  REQUIRE(eu.solve() == EngineStatus::Unbounded);
  CHECK(check_unbounded_ray(u, eu.unbounded_ray(), 1e-9).ok);
}

TEST_CASE(lp_edge_free_variable_with_no_improvement_direction_is_not_unbounded) {
  // min x with x free and the row x >= 2 is bounded: the dual infeasibility of the start basis is resolved.
  const LpModel m = make_model(1, 1, {{0, 0, 1}}, {1}, {-kInf}, {kInf}, {2}, {kInf});
  SimplexEngine e(m);
  REQUIRE(e.solve() == EngineStatus::Optimal);
  CHECK_NEAR(e.solution().x[0], 2.0, 1e-12);
  CHECK(e.stats().phase1_iterations > 0 || e.stats().iterations > 0);
}

// ---- phase 1 and classification against the oracle -----------------------------------------

TEST_CASE(phase1_lps_with_arbitrary_costs_match_the_oracle) {
  int optimal = 0, infeasible = 0, unbounded = 0, inconclusive = 0, failed = 0, with_phase1 = 0;
  for (std::uint64_t seed = 1; seed <= 700; ++seed) {
    SimplexLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 10);
    o.cols = 4 + static_cast<int>(seed % 14);
    o.dual_feasible_start = false;  // costs of any sign: unbounded LPs occur
    o.free_fraction = 0.12;
    o.upper_only_fraction = 0.12;
    o.boxed_fraction = 0.25;
    o.feasible = seed % 6 != 0;
    o.degenerate = seed % 4 == 0 ? 0.5 : 0.0;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexEngine e(model);
    const EngineStatus st = e.solve();
    const RefLpResult ref = solve_dense_lp(model);
    if (e.stats().phase1_iterations > 0) ++with_phase1;
    if (ref.status == Status::NumericalError) {
      ++inconclusive;
      continue;
    }
    bool good = to_status(st) == ref.status;
    std::string why = good ? "" : std::string("engine ") + to_string(st) + ", oracle " + to_string(ref.status);
    if (good && st == EngineStatus::Optimal) {
      const double rel = std::fabs(e.solution().objective - ref.solution.objective) / (1.0 + std::fabs(ref.solution.objective));
      if (!(rel <= 1e-7)) { good = false; why = "objective differs by " + std::to_string(rel); }
      if (!check_kkt(model, e.solution(), 1e-6).ok) { good = false; why = "KKT failed"; }
      ++optimal;
    } else if (good && st == EngineStatus::Infeasible) {
      if (!check_farkas(model, e.farkas_ray(), 1e-9).ok) { good = false; why = "Farkas check failed"; }
      ++infeasible;
    } else if (good && st == EngineStatus::Unbounded) {
      if (!check_unbounded_ray(model, e.unbounded_ray(), 1e-8).ok) { good = false; why = "ray check failed"; }
      ++unbounded;
    }
    if (!good) {
      ++failed;
      std::cerr << "FAILING SEED " << seed << ": " << why << "\n";
      CHECK(false);
    }
  }
  std::cout << "    arbitrary costs: " << optimal << " optimal, " << infeasible << " infeasible, " << unbounded << " unbounded, " << inconclusive
            << " oracle inconclusive, " << failed << " failed; dual phase 1 ran in " << with_phase1 << " solves\n";
  CHECK_EQ(failed, 0);
  CHECK(unbounded > 30);
  CHECK(with_phase1 > 100);
}

TEST_CASE(planted_infeasible_and_unbounded_lps_are_classified_correctly) {
  int correct = 0, total = 0, farkas_ok = 0, farkas_total = 0, ray_ok = 0, ray_total = 0;
  for (std::uint64_t seed = 1; seed <= 500; ++seed) {
    RandomLpOptions o;
    o.rows = 4 + static_cast<int>(seed % 9);
    o.cols = 6 + static_cast<int>(seed % 12);
    o.density = 0.35;
    o.free_col_fraction = seed % 3 == 0 ? 0.15 : 0.0;
    o.ranged_row_fraction = 0.2;
    o.degeneracy = seed % 5 == 0 ? 0.4 : 0.0;
    o.wide_coefficients = seed % 11 == 0;
    const bool want_infeasible = seed % 2 == 0;
    const LpModel model = want_infeasible ? make_random_infeasible_lp(seed, o) : make_random_unbounded_lp(seed, o);
    const RefLpResult ref = solve_dense_lp(model);
    SimplexEngine e(model);
    const EngineStatus st = e.solve();
    ++total;
    const Status expected = want_infeasible ? Status::Infeasible : Status::Unbounded;
    const bool is_correct = to_status(st) == expected;
    if (is_correct) ++correct;
    else std::cerr << "MISCLASSIFIED SEED " << seed << ": planted " << to_string(expected) << ", engine " << to_string(st) << ", oracle " << to_string(ref.status) << "\n";
    if (ref.status != Status::NumericalError && ref.status != expected) {
      std::cerr << "NOTE seed " << seed << ": the oracle says " << to_string(ref.status) << " for a planted " << to_string(expected) << " model\n";
    }
    if (st == EngineStatus::Infeasible) {
      ++farkas_total;
      if (check_farkas(model, e.farkas_ray(), 1e-9).ok) ++farkas_ok;
    }
    if (st == EngineStatus::Unbounded) {
      ++ray_total;
      if (check_unbounded_ray(model, e.unbounded_ray(), 1e-8).ok) ++ray_ok;
    }
  }
  std::cout << "    planted infeasible/unbounded LPs: " << correct << " of " << total << " classified correctly (" << std::fixed << std::setprecision(1)
            << 100.0 * correct / total << "%); Farkas certificates passing the checker " << farkas_ok << "/" << farkas_total << ", unbounded rays passing "
            << ray_ok << "/" << ray_total << std::defaultfloat << "\n";
  CHECK(correct >= 495);  // at least 99%
  CHECK_EQ(farkas_ok, farkas_total);
  CHECK_EQ(ray_ok, ray_total);
}
