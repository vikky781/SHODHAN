#pragma once

// Wide-coefficient seeds on which the pipeline and the dense test oracle disagreed, decided by solving the
// LP exactly (rational arithmetic on the exact value of every double in the model file) and verifying that
// exact answer with KASAUTI in strict exact mode (see docs/KASAUTI.md, "Adjudication of disputed LPs").
//
// exact_status: 0 optimal, 1 unbounded. exact_objective is the exact optimum rounded to a double (model
// sense). max_rel_error is the bound the test enforces on |pipeline - exact| / (1 + |exact|); it is the
// measured error rounded up to one significant digit, not a general tolerance.

#include <cstdint>

namespace shodhan::testing {

struct AdjudicatedSeed {
  std::uint64_t seed;  // seed of the wide-coefficient family
  int exact_status;
  double exact_objective;
  double max_rel_error;
  bool pipeline_certifies;  // false: the pipeline reports NumericalError (never a wrong answer), see the note
};

inline constexpr AdjudicatedSeed kAdjudicatedWideSeeds[] = {
    {453507, 0, -18266017920876808.0, 1e-15, true},  // the oracle said Unbounded: the oracle was wrong
    {450741, 0, -11.437507350760493, 1e-5, true},    // sum|y| about 6e9: conditioning limit of double precision
    {450835, 0, -9.3281250009935022, 1e-9, true},
    {453961, 0, 20.123667009608404, 1e-8, true},
    {451287, 1, 0.0, 0.0, true},   // unbounded; certified since the point is taken from a fresh factorization
    {450165, 1, 0.0, 0.0, false},  // unbounded; the engine's point is infeasible by 2e-6 relative: NumericalError
};

}  // namespace shodhan::testing
