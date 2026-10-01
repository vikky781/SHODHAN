#pragma once

#include <iosfwd>

#include "shodhan/lp_model.hpp"
#include "shodhan/presolve.hpp"
#include "shodhan/scaling.hpp"

namespace shodhan::cli {

/// Presolve options implied by a model: integer columns switch on MIP-safe mode.
PresolveOptions presolve_options_for(const LpModel& model, bool need_duals);

/// Prints the presolve statistics block (status, sizes, per-reduction counts, time).
void print_presolve_report(const PresolveResult& result, std::ostream& out);

/// Prints the scaling report (coefficient ratio before/after) of `model`.
void print_scaling_report(const LpModel& model, std::ostream& out);

}  // namespace shodhan::cli
