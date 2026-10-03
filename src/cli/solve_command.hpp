#pragma once

#include <string>
#include <vector>

namespace shodhan::cli {

/// `shodhan solve <file> [options]`: runs the LP pipeline on pure LPs and reports
/// NotImplemented for models with integer columns. args[0] is the command name.
/// Returns the exit code: 0 optimal, 1 usage or read error, 2 not implemented,
/// 3 infeasible, unbounded, limit reached or numerical status.
int run_solve(const std::vector<std::string>& args);

}  // namespace shodhan::cli
