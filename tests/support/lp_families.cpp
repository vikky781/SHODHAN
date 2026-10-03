#include "support/lp_families.hpp"

#include <cstring>

#include "support/random_lp.hpp"
#include "support/simplex_lps.hpp"

namespace shodhan::testing {

const char* family_name(int f) {
  static const char* const names[kNumFamilies] = {"degenerate", "free variables", "ranged rows", "boxed", "wide coefficients", "infeasible", "unbounded"};
  return f >= 0 && f < kNumFamilies ? names[f] : "unknown";
}

int family_from_name(const char* name) {
  static const char* const keys[kNumFamilies] = {"degenerate", "free", "ranged", "boxed", "wide", "infeasible", "unbounded"};
  for (int f = 0; f < kNumFamilies; ++f) {
    if (std::strcmp(name, keys[f]) == 0) return f;
  }
  return -1;
}

namespace {

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

}  // namespace

LpModel make_family_instance(int family, std::uint64_t seed) {
  LpModel model = make_family_model(family, seed);
  if (seed % 5 == 0) {
    model.sense = Sense::Maximize;
    for (double& c : model.col_cost) c = -c;
    model.objective_offset = -model.objective_offset;
  }
  return model;
}

}  // namespace shodhan::testing
