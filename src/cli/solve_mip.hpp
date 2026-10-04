#pragma once

#include <string>

#include "shodhan/lp_model.hpp"
#include "shodhan/mip/options.hpp"

namespace shodhan::cli {

/// Solves a model with integer columns by branch and bound and prints the report. Exit codes: 0 optimal within the
/// gap, 1 write error, 3 infeasible, unbounded or numerical, 4 stopped at a limit (with or without incumbent).
int run_solve_mip(const LpModel& model, const std::string& model_path, const mip::MipOptions& options, const std::string& sol_path,
                  const std::string& cert_path);

}  // namespace shodhan::cli
