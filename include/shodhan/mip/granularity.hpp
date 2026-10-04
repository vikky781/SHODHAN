#pragma once

#include "shodhan/lp_model.hpp"

namespace shodhan::mip {

/// If every feasible solution's objective is `offset + k * g` for an integer k (all objective coefficients are
/// multiples of g on integer columns and zero on continuous columns), returns g > 0, else 0. Coefficients
/// that are integer multiples of a simple decimal fraction (0.5, 0.25, 0.1, ...) are detected by trying a few
/// scale factors; the detection is conservative (it returns 0 when in doubt).
double objective_granularity(const LpModel& model);

}  // namespace shodhan::mip
