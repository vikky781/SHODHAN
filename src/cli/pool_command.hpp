#pragma once

#include <string>
#include <vector>

namespace shodhan::cli {

/// `shodhan pool <spec.pool> [--method recursion|slp] [--starts N] [--tol t] [--seed s] [--damping a]
/// [--max-iter n] [--no-bound] [--write-sol path] [--write-cert path] [--verbose]`: the pooling driver
/// (docs/POOLING.md). Returns 0 converged and verified, 1 usage or read error, 3 failure (no verified point, cycling,
/// numerical), 4 stopped at an iteration or time limit.
int run_pool(const std::vector<std::string>& args);

}  // namespace shodhan::cli
