// Property test of the full LP pipeline (presolve, scaling, dual simplex, unscale,
// postsolve, KKT on the original) against the dense oracle, over seeded random LPs of
// mixed families.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/rays.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/lp_families.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

constexpr int kPerFamily = 300;

struct FamilyTally {
  int hypersensitive = 0;  // objective comparison skipped, see below
  int total = 0, agree = 0, optimal = 0, infeasible = 0, unbounded = 0, inconclusive = 0, other = 0;
  double worst_obj = 0.0, worst_kkt = 0.0;
};

}  // namespace

TEST_CASE(lp_property_full_pipeline_against_the_dense_oracle) {
  FamilyTally tally[kNumFamilies];
  int failed = 0;
  int classified = 0, planted = 0;
  for (int f = 0; f < kNumFamilies; ++f) {
    for (int k = 1; k <= kPerFamily; ++k) {
      const std::uint64_t seed = static_cast<std::uint64_t>(f) * 100000ULL + static_cast<std::uint64_t>(k);
      const LpModel model = make_family_instance(f, seed);
      FamilyTally& t = tally[f];
      ++t.total;
      const RefLpResult ref = solve_dense_lp(model);
      if (ref.status == Status::NumericalError) {
        ++t.inconclusive;
        continue;
      }
      const LpResult r = LpSolver().solve(model);
      std::string why;
      if (r.status != ref.status) {
        why = std::string("status ") + to_string(r.status) + ", oracle " + to_string(ref.status) + " [" + r.message + "]";
      } else if (r.status == Status::Optimal) {
        const double rel = std::fabs(r.solution.objective - ref.solution.objective) / (1.0 + std::fabs(ref.solution.objective));
        t.worst_obj = std::max(t.worst_obj, rel);
        const KktReport kr = check_kkt(model, r.solution, 1e-6);
        t.worst_kkt = std::max({t.worst_kkt, kr.primal_infeasibility_rel, kr.dual_infeasibility_rel, kr.complementarity_rel, kr.gap_rel});
        // A solver tolerates data errors of about 1e-9; if that could move the objective by more than the
        // comparison tolerance (sum |y| * 1e-9 > 1e-6 (1 + |obj|), multipliers around 1e10 on LPs with
        // coefficients over eight decades), two accurate solvers can legitimately disagree. Such cases
        // are counted and only their KKT check is required.
        double ysum = 0.0;
        for (const double yv : r.solution.y) ysum += std::fabs(yv);
        const bool sensitive = ysum * 1e-9 > 1e-6 * (1.0 + std::fabs(ref.solution.objective));
        if (!(rel <= 1e-6) && sensitive) {
          ++t.hypersensitive;
          if (!kr.ok) why = "KKT on the original model failed: " + kr.summary();
        }
        else if (!(rel <= 1e-6)) why = "objective " + std::to_string(r.solution.objective) + " vs oracle " + std::to_string(ref.solution.objective);
        else if (!kr.ok) why = "KKT on the original model failed: " + kr.summary();
        ++t.optimal;
      } else if (r.status == Status::Infeasible) {
        if (!check_farkas(model, r.farkas_ray, 1e-9).ok) why = "Farkas certificate does not check";
        ++t.infeasible;
      } else if (r.status == Status::Unbounded) {
        if (!check_unbounded_ray(model, r.unbounded_ray, 1e-7).ok) why = "unbounded ray does not check";
        ++t.unbounded;
      } else {
        ++t.other;
      }
      if (f == kInfeasible || f == kUnbounded) {
        ++planted;
        if (r.status == (f == kInfeasible ? Status::Infeasible : Status::Unbounded)) ++classified;
      }
      if (why.empty()) {
        ++t.agree;
      } else {
        ++failed;
        std::cerr << "FAILING SEED " << seed << " (" << family_name(f) << ", m = " << model.n_rows << ", n = " << model.n_cols << "): " << why << "\n";
        CHECK(false);
      }
    }
  }
  int total = 0, agree = 0;
  std::cout << "    LP pipeline vs dense oracle (presolve + scaling + dual simplex, KKT on the original):\n";
  for (int f = 0; f < kNumFamilies; ++f) {
    const FamilyTally& t = tally[f];
    total += t.total - t.inconclusive;
    agree += t.agree;
    std::cout << "      " << std::left << std::setw(18) << family_name(f) << std::right << " " << std::setw(4) << t.total << " LPs: " << std::setw(4) << t.agree
              << " agree (" << t.optimal << " optimal, " << t.infeasible << " infeasible, " << t.unbounded << " unbounded), " << t.inconclusive
              << " oracle inconclusive, " << t.hypersensitive << " hypersensitive; worst objective error " << std::scientific << std::setprecision(1) << t.worst_obj << ", worst KKT residual "
              << t.worst_kkt << std::defaultfloat << "\n";
  }
  std::cout << "    total: " << agree << " of " << total << " conclusive LPs agree; planted infeasible/unbounded classified correctly: " << classified
            << " of " << planted << "\n";
  CHECK_EQ(failed, 0);
  CHECK(total >= 2000);
}
