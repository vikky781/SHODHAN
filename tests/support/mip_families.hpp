#pragma once

// Seeded small MILP families for the tests (test-only code). Each instance is small enough for the
// brute-force oracle in mip_oracle.hpp (integer columns have finite bounds), except the unbounded family.

#include <cstdint>

#include "shodhan/lp_model.hpp"

namespace shodhan::testing {

enum MipFamily {
  kMipKnapsack,        // multi-dimensional knapsack, binaries
  kMipSetCover,        // set cover, binaries
  kMipAssignmentSide,  // assignment with a side constraint (may be infeasible)
  kMipFixedCharge,     // fixed-charge network flow with a weak big-M
  kMipFacility,        // capacitated facility location
  kMipLotSizing,       // small lot sizing
  kMipGeneralInt,      // general integers 0..5
  kMipMixed,           // integers 0..3 plus continuous columns
  kMipParity,          // infeasible by parity / rounding
  kMipUnbounded,       // unbounded (LP relaxation unbounded, an integer feasible point exists)
  kNumMipFamilies
};

const char* mip_family_name(int family);

/// The model of a family for a seed. About a third of the seeds are maximization problems where that makes
/// sense. The seed fixes everything.
LpModel make_mip_instance(int family, std::uint64_t seed);

/// True for the family whose instances are intended to be unbounded (no brute-force oracle applies).
inline bool mip_family_is_unbounded(int family) { return family == kMipUnbounded; }

}  // namespace shodhan::testing
