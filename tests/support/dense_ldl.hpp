#pragma once

// TEST ORACLE ONLY: dense LDL^T of a symmetric matrix without pivoting (valid for quasi-definite matrices) and the
// matching solve. Never linked into shodhan_core.

#include <vector>

namespace shodhan::testing {

/// Solves K x = b for the dense symmetric K (row-major n x n). Returns false when a pivot is zero or not finite.
bool dense_ldl_solve(int n, const std::vector<double>& K, const std::vector<double>& b, std::vector<double>* x);

}  // namespace shodhan::testing
