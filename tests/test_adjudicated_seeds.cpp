// Disputed LPs decided by exact arithmetic: the pipeline must agree with the exact answer
// (tests/support/adjudicated_seeds.hpp), not with the tolerance-based dense oracle.

#include <cmath>
#include <iostream>

#include "shodhan/kkt.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/rays.hpp"
#include "support/adjudicated_seeds.hpp"
#include "support/lp_families.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

TEST_CASE(pipeline_agrees_with_exact_arithmetic_on_adjudicated_seeds) {
  for (const AdjudicatedSeed& s : kAdjudicatedWideSeeds) {
    const LpModel model = make_family_instance(kWide, s.seed);
    const LpResult r = LpSolver().solve(model);
    std::cout << "    wide seed " << s.seed << ": " << to_string(r.status) << "\n";
    if (s.exact_status == 0) {
      CHECK(r.status == Status::Optimal);
      if (r.status != Status::Optimal) continue;
      const double rel = std::fabs(r.solution.objective - s.exact_objective) / (1.0 + std::fabs(s.exact_objective));
      CHECK(rel <= s.max_rel_error);
      CHECK(check_kkt(model, r.solution, 1e-6).ok);
    } else if (s.pipeline_certifies) {
      CHECK(r.status == Status::Unbounded);
      CHECK(check_unbounded_ray(model, r.unbounded_ray, 1e-7).ok);
      CHECK(max_relative_violation(model, r.unbounded_point) <= 1e-6);
    } else {
      // Known limitation: the pipeline must not claim a wrong answer.
      CHECK(r.status == Status::Unbounded || r.status == Status::NumericalError);
    }
  }
}
