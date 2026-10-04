#pragma once

// Seeded small MILP families tuned for the cutting-plane tests (test-only code): the root LP relaxation is
// fractional for most seeds, and every family has structure that one of the separators works on. Each instance is
// small enough to enumerate its integer points.

#include <cstdint>

#include "shodhan/lp_model.hpp"

namespace shodhan::testing {

enum CutFamily {
  kCutKnapsackHard,     // 2-3 correlated knapsack rows, maximization (covers)
  kCutSetCoverHard,     // each element covered by 2-3 sets (odd-cycle structure)
  kCutIndependentSet,   // maximum weight independent set with edge rows (cliques)
  kCutSetPartition,     // set partitioning rows with costs (cliques, may be infeasible)
  kCutFixedChargeVub,   // y_i <= M_i x_i with a demand row (implied bounds, MIR)
  kCutGeneralIntEq,     // general integers with an equality and inequalities (Gomory)
  kCutMixedBinary,      // binaries and continuous columns in mixed rows (MIR, Gomory)
  kCutLotSizingBigM,    // lot sizing with setup binaries and big-M linking rows
  kNumCutFamilies
};

const char* cut_family_name(int family);

LpModel make_cut_instance(int family, std::uint64_t seed);

}  // namespace shodhan::testing
