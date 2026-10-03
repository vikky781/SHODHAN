#pragma once

#include <string>
#include <vector>

namespace shodhan::cli {

/// `shodhan factor-bench <file> [--threshold u] [--max-updates k]`: crash basis,
/// factorization statistics, random solves and a short Forrest-Tomlin run. A
/// developer diagnostic, not a benchmark. args[0] is the command name. Returns
/// the process exit code (0 ok, 1 usage or read error).
int run_factor_bench(const std::vector<std::string>& args);

}  // namespace shodhan::cli
