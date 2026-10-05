#pragma once

// Test-only generators for the structure analysis (docs/STRUCTURE.md): models with PLANTED variable-upper-bound,
// flow-balance and set rows (plus decoys that must not be recognised), and small fixed-charge, scheduling-style and
// lot-sizing MIPs with a deliberately loose big-M whose tighter value is implied by the other rows only.

#include <cstdint>
#include <vector>

#include "shodhan/lp_model.hpp"
#include "shodhan/structure.hpp"

namespace shodhan::testing {

struct PlantedStructure {
  LpModel model;
  std::vector<VubRow> vubs;          ///< sorted by row
  std::vector<Index> balance_rows;   ///< sorted
  std::vector<SetRow> set_rows;      ///< sorted by row
  int decoys = 0;                    ///< rows that look similar and must be left alone
};

PlantedStructure make_planted_structure(std::uint64_t seed);

enum BigMKind { kBigMFixedCharge, kBigMScheduling, kBigMLotSizing, kNumBigMKinds };
const char* bigm_kind_name(int kind);

/// A minimization MIP with at most 10 binaries and no integer column with an infinite bound, so the brute-force oracle
/// applies. The big-M coefficients are loose; the other rows imply much smaller values.
LpModel make_bigm_instance(int kind, std::uint64_t seed);

}  // namespace shodhan::testing
