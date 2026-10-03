#pragma once

// Helpers for testing the basis factorization: structured random bases, the
// residual measure and basis products (test-only code).

#include <cstdint>
#include <string>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "shodhan/sparse_matrix.hpp"
#include "shodhan/sparse_work.hpp"
#include "support/dense_lu.hpp"
#include "support/rng.hpp"
#include "support/work_check.hpp"

namespace shodhan::testing {

/// A structural matrix A (m x n) and a basis (m variable indices in the
/// computational form: structural 0..n-1, logical n..n+m-1 = column -e_i).
struct TestBasis {
  SparseMatrix A;
  std::vector<Index> basis;
  std::string family;
  Index m() const { return A.n_rows; }
};

constexpr int kNumBasisFamilies = 11;
const char* basis_family_name(int family);

/// Basis of dimension m from one of the structured families (0..kNumBasisFamilies-1).
/// Rows and columns are randomly permuted, so no structure is visible in the
/// index order. Some families are nonsingular by construction, others (random
/// sparse, bases of random LPs) are often singular.
TestBasis make_family_basis(int family, Index m, std::uint64_t seed);

/// Family and size picked from the seed (sizes 1..200, mostly small).
TestBasis make_seeded_basis(std::uint64_t seed);

/// Dense copy of the basis matrix B (column p = column of basis[p]).
DenseMatrix basis_to_dense(const SparseMatrix& A, const std::vector<Index>& basis);
/// y = B x (x indexed by position, y by row).
std::vector<double> basis_multiply(const SparseMatrix& A, const std::vector<Index>& basis,
                                   const std::vector<double>& x);
/// y = B^T x (x indexed by row, y by position).
std::vector<double> basis_multiply_transpose(const SparseMatrix& A, const std::vector<Index>& basis,
                                             const std::vector<double>& x);

double norm_inf(const std::vector<double>& v);

/// ||b_times_x - a|| / (||B|| ||x|| + ||a||) in infinity norms, where
/// `b_times_x` is the product B x computed by the caller and `b_norm` = ||B||.
double relative_residual(const std::vector<double>& b_times_x, double b_norm,
                         const std::vector<double>& x, const std::vector<double>& a);

/// Random right-hand sides: dense, or sparse with `nnz` nonzeros.
std::vector<double> random_dense_vector(Index n, Rng& rng);
std::vector<double> random_sparse_vector(Index n, Index nnz, Rng& rng);

struct SolveCheckOptions {
  /// Required relative residual (target 1e-10).
  double residual_tol = 1e-10;
  int dense_rhs = 2;
  int sparse_rhs = 3;
  /// Safety factor in the forward-error bound |x - x_oracle| <= factor * kappa * eps * |x_oracle|.
  /// The sparse code and the oracle both carry errors of order kappa * eps (growth
  /// and rounding are covered by the factor), so they may differ by about twice that.
  double forward_factor = 1e3;
};

struct SolveCheckResult {
  double worst_residual = 0.0;
  /// Largest ratio of the observed difference to the bound; at most 1 passes.
  double worst_forward_ratio = 0.0;
  double kappa = 0.0;
  int solves = 0;
  /// Empty when everything passed, otherwise a description of the first failure.
  std::string failure;
  bool ok() const { return failure.empty(); }
};

/// Runs ftran and btran of an already valid factorization on dense, sparse and
/// unit right-hand sides, and checks (1) the relative residual, (2) agreement
/// with the dense LU oracle within a condition-aware bound, (3) the SparseWork
/// invariants of the results.
SolveCheckResult check_solves(BasisFactor& f, const SparseMatrix& A, const std::vector<Index>& basis,
                              Rng& rng, const SolveCheckOptions& opt = {});

}  // namespace shodhan::testing
