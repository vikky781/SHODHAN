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
#include "support/random_lp.hpp"
#include "support/simplex_lps.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

constexpr int kPerFamily = 300;

enum Family { kDegenerate, kFreeVars, kRangedRows, kBoxed, kWide, kInfeasible, kUnbounded, kNumFamilies };

const char* family_name(int f) {
  static const char* const names[kNumFamilies] = {"degenerate", "free variables", "ranged rows", "boxed", "wide coefficients", "infeasible", "unbounded"};
  return names[f];
}

LpModel make_family_model(int family, std::uint64_t seed) {
  const int rows = 3 + static_cast<int>(seed % 14);
  const int cols = 4 + static_cast<int>(seed % 19);
  RandomLpOptions ro;
  ro.rows = rows;
  ro.cols = cols;
  ro.density = 0.2 + 0.05 * static_cast<double>(seed % 6);
  SimplexLpOptions so;
  so.rows = rows;
  so.cols = cols;
  so.density = ro.density;
  so.dual_feasible_start = seed % 3 != 0;
  switch (family) {
    case kDegenerate:
      ro.degeneracy = 0.6;
      ro.active_fraction = 0.9;
      so.degenerate = 0.9;
      so.equality_fraction = 0.4;
      return seed % 2 == 0 ? make_random_lp(seed, ro).model : make_simplex_lp(seed, so);
    case kFreeVars:
      ro.free_col_fraction = 0.3;
      so.free_fraction = 0.25;
      so.upper_only_fraction = 0.1;
      return seed % 2 == 0 ? make_random_lp(seed, ro).model : make_simplex_lp(seed, so);
    case kRangedRows:
      ro.ranged_row_fraction = 0.6;
      ro.free_row_fraction = 0.1;
      so.ranged_fraction = 0.6;
      so.free_row_fraction = 0.1;
      return seed % 2 == 0 ? make_random_lp(seed, ro).model : make_simplex_lp(seed, so);
    case kBoxed:
      so.boxed_fraction = 0.8;
      return make_simplex_lp(seed, so);
    case kWide:
      ro.wide_coefficients = true;
      so.wide_coefficients = true;
      return seed % 2 == 0 ? make_random_lp(seed, ro).model : make_simplex_lp(seed, so);
    case kInfeasible:
      ro.free_col_fraction = seed % 3 == 0 ? 0.15 : 0.0;
      ro.ranged_row_fraction = 0.2;
      return make_random_infeasible_lp(seed, ro);
    default:
      ro.ranged_row_fraction = 0.2;
      return make_random_unbounded_lp(seed, ro);
  }
}

struct FamilyTally {
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
      LpModel model = make_family_model(f, seed);
      if (k % 5 == 0) {  // a fifth of the models are maximizations
        model.sense = Sense::Maximize;
        for (double& c : model.col_cost) c = -c;
        model.objective_offset = -model.objective_offset;
      }
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
        if (!(rel <= 1e-6)) why = "objective " + std::to_string(r.solution.objective) + " vs oracle " + std::to_string(ref.solution.objective);
        else if (!kr.ok) why = "KKT on the original model failed: " + kr.summary();
        ++t.optimal;
      } else if (r.status == Status::Infeasible) {
        if (!check_farkas(model, r.farkas_ray, 1e-9).ok) why = "Farkas certificate does not check";
        ++t.infeasible;
      } else if (r.status == Status::Unbounded) {
        if (!check_unbounded_ray(model, r.unbounded_ray, 1e-8).ok) why = "unbounded ray does not check";
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
              << " oracle inconclusive; worst objective error " << std::scientific << std::setprecision(1) << t.worst_obj << ", worst KKT residual "
              << t.worst_kkt << std::defaultfloat << "\n";
  }
  std::cout << "    total: " << agree << " of " << total << " conclusive LPs agree; planted infeasible/unbounded classified correctly: " << classified
            << " of " << planted << "\n";
  CHECK_EQ(failed, 0);
  CHECK(total >= 2000);
}
