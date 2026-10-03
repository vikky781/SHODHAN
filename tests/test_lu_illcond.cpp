// Ill-conditioned bases: the factorization must either return accurate solutions
// or flag the basis as rank deficient; it must never return garbage silently.
//
// What "accurate" can mean depends on the condition number kappa of the basis.
// An LU factorization with threshold pivoting is backward stable, so the relative
// residual ||B x - a|| / (||B|| ||x|| + ||a||) is of the order of the unit
// roundoff however ill-conditioned B is: that is required with the usual 1e-10.
// The forward error is only controlled up to kappa * eps, so the solution is
// compared with the dense LU oracle (whose own error is also about kappa * eps)
// within  1e3 * kappa * eps * ||x||  (the factor covers growth and rounding; see
// SolveCheckOptions). Once kappa * eps is of order one nothing can be said about
// the digits of x, and that comparison cannot fail; those cases are counted
// below so that the report does not overstate what was verified.
//
// The drop tolerance of the solves is absolute (default 1e-14), so it is meant for
// scaled models, whose coefficients are near one. For a basis with coefficients up
// to 1e6, an entry of the solution of size 1e-14 that is dropped contributes up to
// 1e-8 to the residual (observed: 1.9e-9 on 11 of 300 of the bases below with the
// default; 1.8e-16 with drop_tol = 0). The tests with wide coefficient ranges
// therefore set drop_tol = 1e-14 / max|B|, which is what the default amounts to
// after scaling the matrix to unit maximum.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "support/lu_testing.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

SparseMatrix from_dense_columns(const std::vector<std::vector<double>>& cols) {
  const Index n = static_cast<Index>(cols.size());
  const Index m = n == 0 ? 0 : static_cast<Index>(cols[0].size());
  std::vector<Triplet> t;
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < m; ++i)
      if (cols[to_size(j)][to_size(i)] != 0.0) t.push_back({i, j, cols[to_size(j)][to_size(i)]});
  SparseMatrix A;
  std::string err;
  if (!SparseMatrix::from_triplets(m, n, t, &A, &err)) throw std::runtime_error(err);
  return A;
}

std::vector<Index> identity_basis(Index m) {
  std::vector<Index> b(to_size(m));
  for (Index j = 0; j < m; ++j) b[to_size(j)] = j;
  return b;
}

struct Outcome {
  bool flagged = false;
  bool accurate = false;
  double kappa = 0.0;
  double residual = 0.0;
  std::string failure;
};

// Factorize: either RankDeficient (flagged), or Ok with passing solves.
Outcome classify(const SparseMatrix& A, const std::vector<Index>& basis, std::uint64_t seed,
                 double drop_tol = SparseWork::kDefaultDropTol) {
  Outcome o;
  FactorParams params;
  params.drop_tol = drop_tol;
  BasisFactor f(params);
  Rng rng(seed);
  const FactorStatus st = f.factorize(A, basis);
  if (st != FactorStatus::Ok) {
    o.flagged = true;
    // A flagged basis must be repairable into a basis that passes the same test.
    std::vector<Index> repaired = basis;
    f.repair(A, repaired);
    if (f.status() != FactorStatus::Ok) { o.failure = "repair of a flagged basis failed"; return o; }
    const SolveCheckResult r = check_solves(f, A, repaired, rng);
    o.residual = r.worst_residual;
    o.kappa = r.kappa;
    if (!r.ok()) o.failure = "after repair: " + r.failure;
    return o;
  }
  const SolveCheckResult r = check_solves(f, A, basis, rng);
  o.kappa = r.kappa;
  o.residual = r.worst_residual;
  o.accurate = r.ok();
  if (!r.ok()) o.failure = r.failure;
  return o;
}

}  // namespace

TEST_CASE(lu_illcond_hilbert_matrices) {
  std::cout << "    lu illcond, Hilbert matrices:";
  for (Index n = 2; n <= 8; ++n) {
    std::vector<std::vector<double>> cols(to_size(n), std::vector<double>(to_size(n)));
    for (Index j = 0; j < n; ++j)
      for (Index i = 0; i < n; ++i) cols[to_size(j)][to_size(i)] = 1.0 / static_cast<double>(i + j + 1);
    const SparseMatrix A = from_dense_columns(cols);
    const std::vector<Index> basis = identity_basis(n);
    const Outcome o = classify(A, basis, 100 + static_cast<std::uint64_t>(n));
    CHECK(o.failure.empty());
    if (!o.failure.empty()) std::cerr << "  n = " << n << ": " << o.failure << "\n";
    std::cout << " n=" << n << (o.flagged ? " flagged" : " ok") << " (kappa " << std::scientific << std::setprecision(1)
              << o.kappa << ")" << std::defaultfloat;

    // Against the exact solution x = (1, ..., 1): the right-hand side a = H x is
    // formed in floating point, so the comparison is limited by kappa * eps.
    BasisFactor f;
    if (f.factorize(A, basis) == FactorStatus::Ok) {
      const std::vector<double> ones(to_size(n), 1.0);
      const std::vector<double> a = basis_multiply(A, basis, ones);
      SparseWork w;
      w.load_dense(a);
      f.ftran(w);
      std::vector<double> x(to_size(n));
      w.to_dense(x);
      double err = 0.0;
      for (const double v : x) err = std::max(err, std::fabs(v - 1.0));
      const double bound = 1e3 * o.kappa * 2.220446049250313e-16;
      CHECK(err <= std::max(bound, 1e-13));
      if (!(err <= std::max(bound, 1e-13))) std::cerr << "  n = " << n << ": error against the exact solution " << err << " exceeds " << bound << "\n";
    }
  }
  std::cout << "\n";
}

TEST_CASE(lu_illcond_nearly_dependent_columns) {
  int flagged = 0, accurate = 0, vacuous = 0;
  std::cout << "    lu illcond, nearly dependent columns (delta -> outcome):";
  const double deltas[] = {1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 0.0};
  std::uint64_t seed = 5000;
  for (const double delta : deltas) {
    for (int rep = 0; rep < 12; ++rep, ++seed) {
      Rng rng(seed);
      const Index m = rng.range(3, 12);
      std::vector<std::vector<double>> cols(to_size(m), std::vector<double>(to_size(m), 0.0));
      for (Index j = 0; j < m; ++j)
        for (Index i = 0; i < m; ++i)
          if (i == j || rng.chance(0.4)) cols[to_size(j)][to_size(i)] = rng.uniform(0.5, 2.0) * (rng.chance(0.5) ? 1 : -1);
      // Last column = col0 + col1 + delta * random.
      for (Index i = 0; i < m; ++i)
        cols[to_size(m - 1)][to_size(i)] = cols[0][to_size(i)] + cols[1][to_size(i)] + delta * rng.uniform(-1.0, 1.0);
      const SparseMatrix A = from_dense_columns(cols);
      const Outcome o = classify(A, identity_basis(m), seed);
      if (!o.failure.empty()) {
        std::cerr << "FAILING SEED " << seed << " (delta " << delta << ", m = " << m << "): " << o.failure << "\n";
        CHECK(false);
      }
      if (o.flagged) ++flagged; else { ++accurate; if (o.kappa * 2.220446049250313e-16 > 0.1) ++vacuous; }
    }
    std::cout << " " << std::scientific << std::setprecision(0) << delta << ":" << std::defaultfloat << flagged << "f/" << accurate << "a";
  }
  std::cout << "\n      flagged rank deficient " << flagged << ", accepted with passing checks " << accurate << " (of which "
            << vacuous << " have kappa*eps > 0.1, where only the residual check is meaningful)\n";
  CHECK(flagged > 0);   // delta = 0 is exactly dependent
  CHECK(accurate > 0);  // delta = 1e-2 is a perfectly good basis
}

TEST_CASE(lu_illcond_wide_coefficient_ranges) {
  // Entries log-uniform in 1e-6 .. 1e6, sparse, with a nonzero diagonal.
  int flagged = 0, accurate = 0, vacuous = 0;
  double worst_residual = 0.0;
  for (std::uint64_t seed = 7000; seed < 7300; ++seed) {
    Rng rng(seed);
    const Index m = rng.range(2, 40);
    auto wide = [&rng]() { return std::pow(10.0, rng.uniform(-6.0, 6.0)) * (rng.chance(0.5) ? 1.0 : -1.0); };
    std::vector<std::vector<double>> cols(to_size(m), std::vector<double>(to_size(m), 0.0));
    const double density = rng.uniform(0.05, 0.4);
    for (Index j = 0; j < m; ++j)
      for (Index i = 0; i < m; ++i)
        if (i == j || rng.chance(density)) cols[to_size(j)][to_size(i)] = wide();
    const SparseMatrix A = from_dense_columns(cols);
    const Outcome o = classify(A, identity_basis(m), seed, SparseWork::kDefaultDropTol / A.max_abs());
    worst_residual = std::max(worst_residual, o.residual);
    if (!o.failure.empty()) {
      std::cerr << "FAILING SEED " << seed << " (m = " << m << "): " << o.failure << "\n";
      CHECK(false);
    }
    if (o.flagged) ++flagged; else { ++accurate; if (o.kappa * 2.220446049250313e-16 > 0.1) ++vacuous; }
  }
  std::cout << "    lu illcond, coefficients 1e-6..1e6: " << accurate << " accepted with passing checks (" << vacuous
            << " with kappa*eps > 0.1), " << flagged << " flagged rank deficient and repaired; worst residual "
            << std::scientific << std::setprecision(2) << worst_residual << std::defaultfloat << "\n";
  CHECK(accurate > 100);
  CHECK(worst_residual < 1e-10);
}

TEST_CASE(lu_illcond_badly_scaled_columns_and_rows) {
  // A well-conditioned matrix times diagonal scalings spanning 12 orders of
  // magnitude: kappa is huge only because of the scaling.
  int accurate = 0, flagged = 0;
  for (std::uint64_t seed = 9000; seed < 9100; ++seed) {
    Rng rng(seed);
    const Index m = rng.range(2, 30);
    std::vector<std::vector<double>> cols(to_size(m), std::vector<double>(to_size(m), 0.0));
    std::vector<double> rs(to_size(m)), cs(to_size(m));
    for (Index i = 0; i < m; ++i) { rs[to_size(i)] = std::pow(10.0, rng.uniform(-6.0, 6.0)); cs[to_size(i)] = std::pow(10.0, rng.uniform(-6.0, 6.0)); }
    for (Index j = 0; j < m; ++j)
      for (Index i = 0; i < m; ++i)
        if (i == j) cols[to_size(j)][to_size(i)] = 4.0 * rs[to_size(i)] * cs[to_size(j)];
        else if (rng.chance(0.2)) cols[to_size(j)][to_size(i)] = rng.uniform(-1.0, 1.0) * rs[to_size(i)] * cs[to_size(j)];
    const SparseMatrix A = from_dense_columns(cols);
    const Outcome o = classify(A, identity_basis(m), seed);
    if (!o.failure.empty()) {
      std::cerr << "FAILING SEED " << seed << " (m = " << m << "): " << o.failure << "\n";
      CHECK(false);
    }
    if (o.flagged) ++flagged; else ++accurate;
  }
  std::cout << "    lu illcond, badly scaled: " << accurate << " accepted with passing checks, " << flagged << " flagged\n";
  CHECK(accurate > 50);
}
