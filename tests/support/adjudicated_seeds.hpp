#pragma once

// Wide-coefficient seeds on which the pipeline and the dense test oracle disagreed, decided by solving the
// LP exactly (rational arithmetic on the exact value of every double in the model file) and verifying that
// exact answer with KASAUTI in strict exact mode (see docs/KASAUTI.md, "Adjudication of disputed LPs").
//
// exact_status: 0 optimal, 1 unbounded, 2 infeasible. exact_objective is the exact optimum rounded to a double (model
// sense). max_rel_error is the bound the test enforces on |pipeline - exact| / (1 + |exact|); it is the
// measured error rounded up to one significant digit, not a general tolerance.

#include <cstdint>

namespace shodhan::testing {

struct AdjudicatedSeed {
  std::uint64_t seed;  // seed of the wide-coefficient family
  int exact_status;
  double exact_objective;
  double max_rel_error;
  // false: the pipeline may report NumericalError (an honest failure, never a wrong answer); if it reports
  // Optimal or Unbounded the result must still match the exact one. The outcome of such a seed depends on
  // floating-point details of the platform (fused multiply-add, library rounding).
  bool pipeline_certifies;
};

inline constexpr AdjudicatedSeed kAdjudicatedWideSeeds[] = {
    {453507, 0, -18266017920876808.0, 1e-15, true},  // the oracle said Unbounded: the oracle was wrong
    {450741, 0, -11.437507350760493, 1e-5, false},   // sum|y| about 6e9: Optimal on x86-64, NumericalError on Apple arm64
    {450835, 0, -9.3281250009935022, 1e-9, true},
    {453961, 0, 20.123667009608404, 1e-8, true},
    {451287, 1, 0.0, 0.0, true},   // unbounded; certified since the point is taken from a fresh factorization
    {450165, 1, 0.0, 0.0, true},   // unbounded; point from a tight feasibility solve, ray cleaned of noise
    // Found on seeds 1..2000 when the dual-bound acceptance check was added (open findings, see docs/KASAUTI.md):
    {433, 0, 7.4272773546400339, 1e-11, true},  // was 3.9e-6 off: polishing left a dual infeasibility of 2e-7
    {1999, 2, 0.0, 0.0, false},                // exactly infeasible by far less than any tolerance; reported Optimal within tolerance
};

}  // namespace shodhan::testing
