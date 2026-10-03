#pragma once

// The LP families shared by the property test and the stress harness (test-only code).

#include <cstdint>

#include "shodhan/lp_model.hpp"

namespace shodhan::testing {

enum Family { kDegenerate, kFreeVars, kRangedRows, kBoxed, kWide, kInfeasible, kUnbounded, kNumFamilies };

const char* family_name(int family);

/// Family index from a name given to the stress harness ("degenerate", "free", "ranged", "boxed",
/// "wide", "infeasible", "unbounded"); -1 if unknown.
int family_from_name(const char* name);

/// The model of a family for a seed. A fifth of the seeds (seed % 5 == 0) are turned into
/// maximization problems (same feasible set, opposite objective).
LpModel make_family_instance(int family, std::uint64_t seed);

}  // namespace shodhan::testing
