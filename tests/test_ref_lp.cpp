#include <cmath>
#include <string>

#include "shodhan/kkt.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/random_lp.hpp"
#include "support/test_models.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

// Solves, requires Optimal, and requires the result to pass check_kkt.
RefLpResult solve_and_verify(const LpModel& m, const char* what, double tol = 1e-8) {
  const RefLpResult r = solve_dense_lp(m);
  CHECK(r.status == Status::Optimal);
  if (r.status == Status::Optimal) {
    const KktReport k = check_kkt(m, r.solution, tol);
    if (!k.ok) std::cerr << "  " << what << ": " << k.summary() << "\n";
    CHECK(k.ok);
  }
  return r;
}

}  // namespace

TEST_CASE(ref_lp_solves_hand_made_lps) {
  // min x1 + 2 x2, x1 + x2 >= 1, x >= 0
  LpModel cover = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 2.0}, {0.0, 0.0}, {kInf, kInf},
                             {1.0}, {kInf});
  RefLpResult r = solve_and_verify(cover, "cover");
  CHECK_NEAR(r.solution.objective, 1.0, 1e-10);

  // min -x1 - x2, x1 + 2 x2 <= 4, 3 x1 + x2 <= 6
  LpModel two = make_model(2, 2, {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 1.0}}, {-1.0, -1.0},
                           {0.0, 0.0}, {kInf, kInf}, {-kInf, -kInf}, {4.0, 6.0});
  r = solve_and_verify(two, "two_rows");
  CHECK_NEAR(r.solution.objective, -2.8, 1e-10);
  CHECK_NEAR(r.solution.x[0], 1.6, 1e-10);
  CHECK_NEAR(r.solution.x[1], 1.2, 1e-10);

  // equality row, free variable, boxed variable
  LpModel eqfree = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 2.0}, {0.0, -kInf}, {2.0, kInf},
                              {3.0}, {3.0});
  r = solve_and_verify(eqfree, "eq_free");
  CHECK_NEAR(r.solution.objective, 4.0, 1e-10);

  // maximization with offset
  LpModel mx = make_model(1, 1, {{0, 0, 1.0}}, {1.0}, {-kInf}, {kInf}, {-kInf}, {3.0},
                          Sense::Maximize, 10.0);
  r = solve_and_verify(mx, "max");
  CHECK_NEAR(r.solution.objective, 13.0, 1e-10);
  CHECK_NEAR(r.solution.y[0], -1.0, 1e-10);

  // ranged row
  LpModel ranged = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 3.0}, {0.0, 0.0}, {kInf, kInf},
                              {2.0}, {5.0});
  r = solve_and_verify(ranged, "ranged");
  CHECK_NEAR(r.solution.objective, 2.0, 1e-10);

  // variable bounded only above
  LpModel upper_only = make_model(1, 1, {{0, 0, 1.0}}, {-1.0}, {-kInf}, {7.0}, {-kInf}, {kInf});
  r = solve_and_verify(upper_only, "upper_only");
  CHECK_NEAR(r.solution.x[0], 7.0, 1e-10);

  // no rows at all
  LpModel no_rows = make_model(0, 2, {}, {1.0, -1.0}, {1.0, 0.0}, {4.0, 3.0}, {}, {});
  r = solve_and_verify(no_rows, "no_rows");
  CHECK_NEAR(r.solution.objective, -2.0, 1e-10);
}

TEST_CASE(ref_lp_handles_degenerate_and_redundant_rows) {
  // Highly degenerate: several constraints pass through the optimum.
  LpModel deg = make_model(
      4, 2,
      {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, -1.0}, {2, 0, 2.0}, {2, 1, 1.0}, {3, 0, 1.0},
       {3, 1, 1.0}},
      {-1.0, -2.0}, {0.0, 0.0}, {kInf, kInf}, {-kInf, -kInf, -kInf, 2.0}, {2.0, 0.0, 3.0, 2.0});
  solve_and_verify(deg, "degenerate");

  // Linearly dependent equality rows.
  LpModel dep = make_model(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 2.0}, {1, 1, 2.0}}, {1.0, 2.0},
                           {0.0, 0.0}, {kInf, kInf}, {3.0, 6.0}, {3.0, 6.0});
  const RefLpResult r = solve_and_verify(dep, "dependent_rows");
  CHECK_NEAR(r.solution.objective, 3.0, 1e-9);
}

TEST_CASE(ref_lp_classifies_infeasible_and_unbounded) {
  // x >= 2 and x <= 1.
  LpModel infeas = make_model(2, 1, {{0, 0, 1.0}, {1, 0, 1.0}}, {1.0}, {-kInf}, {kInf}, {2.0, -kInf},
                              {kInf, 1.0});
  CHECK(solve_dense_lp(infeas).status == Status::Infeasible);

  // Bounds that cannot hold with the row: x, y in [0,1], x + y >= 5.
  LpModel infeas2 = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0},
                               {5.0}, {kInf});
  CHECK(solve_dense_lp(infeas2).status == Status::Infeasible);

  // min -x, x >= 0, no rows.
  LpModel unb = make_model(0, 1, {}, {-1.0}, {0.0}, {kInf}, {}, {});
  CHECK(solve_dense_lp(unb).status == Status::Unbounded);

  // min -x - y, x - y <= 1, x,y >= 0: ray (1,1).
  LpModel unb2 = make_model(1, 2, {{0, 0, 1.0}, {0, 1, -1.0}}, {-1.0, -1.0}, {0.0, 0.0},
                            {kInf, kInf}, {-kInf}, {1.0});
  CHECK(solve_dense_lp(unb2).status == Status::Unbounded);

  // Free variable going to -infinity.
  LpModel unb3 = make_model(0, 1, {}, {1.0}, {-kInf}, {kInf}, {}, {});
  CHECK(solve_dense_lp(unb3).status == Status::Unbounded);

  // Infeasible takes precedence over unbounded.
  LpModel both = make_model(1, 2, {{0, 1, 1.0}}, {-1.0, 0.0}, {0.0, 0.0}, {kInf, 1.0}, {5.0}, {kInf});
  CHECK(solve_dense_lp(both).status == Status::Infeasible);
}

TEST_CASE(random_lp_generator_produces_valid_known_optimal_pairs) {
  int checked = 0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    RandomLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 12);
    o.cols = 3 + static_cast<int>((seed * 7) % 14);
    o.density = 0.2 + 0.1 * static_cast<double>(seed % 5);
    o.degeneracy = (seed % 3 == 0) ? 0.6 : 0.0;
    o.free_col_fraction = (seed % 4 == 0) ? 0.2 : 0.0;
    o.ranged_row_fraction = (seed % 5 == 0) ? 0.3 : 0.0;
    o.free_row_fraction = (seed % 7 == 0) ? 0.2 : 0.0;
    o.wide_coefficients = seed % 2 == 0;
    o.fixed_cols = static_cast<int>(seed % 3);
    o.empty_cols = static_cast<int>((seed / 3) % 2);
    o.empty_rows = static_cast<int>((seed / 5) % 2);
    o.singleton_rows = static_cast<int>((seed / 2) % 3);
    o.doubleton_eqs = static_cast<int>((seed / 7) % 3);
    o.forcing_rows = static_cast<int>((seed / 11) % 2);
    const RandomLp lp = make_random_lp(seed, o);
    const auto problems = lp.model.validate();
    if (!problems.empty()) {
      std::cerr << "  seed " << seed << ": " << problems.front() << "\n";
      CHECK(problems.empty());
      continue;
    }
    const KktReport k = check_kkt(lp.model, lp.known, 1e-9);
    if (!k.ok) std::cerr << "  seed " << seed << ": " << k.summary() << "\n";
    CHECK(k.ok);
    ++checked;
  }
  CHECK_EQ(checked, 300);
}

TEST_CASE(ref_lp_solves_random_lps_and_matches_the_known_optimum) {
  int solved = 0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    RandomLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 14);
    o.cols = 3 + static_cast<int>((seed * 5) % 16);
    o.density = 0.2 + 0.1 * static_cast<double>(seed % 5);
    o.degeneracy = (seed % 3 == 0) ? 0.6 : 0.0;
    o.active_fraction = (seed % 3 == 0) ? 0.9 : 0.5;
    o.free_col_fraction = (seed % 4 == 0) ? 0.25 : 0.0;
    o.ranged_row_fraction = (seed % 5 == 0) ? 0.3 : 0.0;
    o.wide_coefficients = seed % 2 == 0;
    o.fixed_cols = static_cast<int>(seed % 3);
    o.singleton_rows = static_cast<int>(seed % 2);
    o.doubleton_eqs = static_cast<int>((seed / 3) % 2);
    const RandomLp lp = make_random_lp(seed, o);
    const RefLpResult r = solve_dense_lp(lp.model);
    if (r.status != Status::Optimal) {
      std::cerr << "  seed " << seed << ": status " << to_string(r.status) << "\n";
      CHECK(r.status == Status::Optimal);
      continue;
    }
    const KktReport k = check_kkt(lp.model, r.solution, 1e-7);
    if (!k.ok) std::cerr << "  seed " << seed << ": " << k.summary() << "\n";
    CHECK(k.ok);
    const double known = lp.known.objective;
    const double tol = 1e-7 * (1.0 + std::fabs(known));
    if (std::fabs(r.solution.objective - known) > tol) {
      std::cerr << "  seed " << seed << ": objective " << r.solution.objective << " vs known "
                << known << "\n";
    }
    CHECK(std::fabs(r.solution.objective - known) <= tol);
    ++solved;
  }
  CHECK_EQ(solved, 300);
}

TEST_CASE(ref_lp_classifies_random_infeasible_and_unbounded_constructions) {
  for (std::uint64_t seed = 1; seed <= 120; ++seed) {
    RandomLpOptions o;
    o.rows = 3 + static_cast<int>(seed % 8);
    o.cols = 3 + static_cast<int>(seed % 9);
    o.ranged_row_fraction = 0.2;
    o.wide_coefficients = seed % 3 == 0;
    std::string kind;
    const LpModel inf = make_random_infeasible_lp(seed, o, &kind);
    CHECK(inf.validate().empty());
    const RefLpResult ri = solve_dense_lp(inf);
    if (ri.status != Status::Infeasible) {
      std::cerr << "  infeasible seed " << seed << " (" << kind << ")\n";
    }
    CHECK(ri.status == Status::Infeasible);

    const LpModel unb = make_random_unbounded_lp(seed, o, &kind);
    CHECK(unb.validate().empty());
    const RefLpResult ru = solve_dense_lp(unb);
    if (ru.status != Status::Unbounded) {
      std::cerr << "  unbounded seed " << seed << " (" << kind << ")\n";
    }
    CHECK(ru.status == Status::Unbounded);
  }
}
