// Hypersparse versus dense triangular solves.

#include <algorithm>
#include <chrono>
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

// A large, nearly triangular basis: lower triangular with short chains (column j
// has an entry below the diagonal unless j ends a chain of `chain` columns), a few
// 2x2 dense blocks as a nucleus, rows and columns randomly permuted.
TestBasis nearly_triangular(Index m, Index chain, std::uint64_t seed) {
  Rng rng(seed);
  std::vector<Index> rp(to_size(m)), cp(to_size(m));
  for (Index i = 0; i < m; ++i) rp[to_size(i)] = cp[to_size(i)] = i;
  rng.shuffle(rp);
  rng.shuffle(cp);
  std::vector<Triplet> t;
  for (Index j = 0; j < m; ++j) {
    t.push_back({rp[to_size(j)], cp[to_size(j)], 2.0 + 0.001 * static_cast<double>(j % 7)});
    if (j % chain != chain - 1 && j + 1 < m) t.push_back({rp[to_size(j + 1)], cp[to_size(j)], 1.0});
  }
  // Nucleus: couple column j+1 back to row j in a few chains (a 2x2 block).
  for (Index k = 0; k < 5; ++k) {
    const Index j = 3 * chain * (k + 1);
    if (j + 1 < m && j % chain != chain - 1) t.push_back({rp[to_size(j)], cp[to_size(j + 1)], 0.25});
  }
  TestBasis tb;
  tb.family = "nearly-triangular";
  std::string err;
  if (!SparseMatrix::from_triplets(m, m, t, &tb.A, &err)) throw std::runtime_error(err);
  tb.basis.resize(to_size(m));
  for (Index j = 0; j < m; ++j) tb.basis[to_size(j)] = j;
  return tb;
}

FactorParams with_threshold(double thr) {
  FactorParams p;
  p.hyper_threshold = thr;
  return p;
}

std::vector<double> solve_dense(BasisFactor& f, const std::vector<double>& a, bool transpose) {
  SparseWork w;
  w.load_dense(a);
  if (transpose) f.btran(w); else f.ftran(w);
  std::vector<double> x(a.size());
  w.to_dense(x);
  return x;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

}  // namespace

TEST_CASE(lu_hypersparse_path_is_taken_and_matches_the_dense_path) {
  const Index m = 20000;
  const TestBasis tb = nearly_triangular(m, 8, 77);
  BasisFactor f(with_threshold(0.10));  // the defaults
  BasisFactor dense(with_threshold(0.0));
  REQUIRE(f.factorize(tb.A, tb.basis) == FactorStatus::Ok);
  REQUIRE(dense.factorize(tb.A, tb.basis) == FactorStatus::Ok);
  CHECK(f.stats().markowitz_pivots <= 20);  // nearly triangular: almost everything is a singleton

  Rng rng(5);
  double worst = 0.0;
  const std::size_t hyper_before = f.stats().hyper_solves;
  for (int k = 0; k < 40; ++k) {
    const std::vector<double> a = random_sparse_vector(m, 1 + k % 3, rng);
    for (const bool transpose : {false, true}) {
      const std::vector<double> xh = solve_dense(f, a, transpose);
      const std::vector<double> xd = solve_dense(dense, a, transpose);
      worst = std::max(worst, max_abs_diff(xh, xd) / std::max(1.0, norm_inf(xd)));
    }
  }
  const FactorStats s = f.stats();
  CHECK(s.hyper_solves - hyper_before == 160);  // 80 solves, two triangular stages each
  CHECK_EQ(s.dense_solves, std::size_t{0});     // never fell back to a dense loop
  CHECK_EQ(dense.stats().hyper_solves, std::size_t{0});
  CHECK(dense.stats().dense_solves == 160);
  CHECK(worst <= 1e-13);
  std::cout << "    hypersparse vs dense, m = " << m << ": " << s.hyper_solves << " hypersparse stages, worst relative difference "
            << std::scientific << std::setprecision(2) << worst << std::defaultfloat << "\n";

  // A dense right-hand side takes the dense path.
  const std::size_t dense_before = f.stats().dense_solves;
  const std::vector<double> a = random_dense_vector(m, rng);
  const std::vector<double> xf = solve_dense(f, a, false);
  CHECK(f.stats().dense_solves > dense_before);
  const std::vector<double> xd = solve_dense(dense, a, false);
  CHECK(max_abs_diff(xf, xd) / std::max(1.0, norm_inf(xd)) <= 1e-13);
  // And the residual is small.
  const std::vector<double> bx = basis_multiply(tb.A, tb.basis, xf);
  double r = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) r = std::max(r, std::fabs(bx[i] - a[i]));
  CHECK(r <= 1e-10 * (4.0 * norm_inf(xf) + norm_inf(a)));
}

TEST_CASE(lu_hypersparse_threshold_parameter_selects_the_path) {
  const TestBasis tb = nearly_triangular(400, 8, 3);
  Rng rng(9);
  const std::vector<double> a = random_sparse_vector(400, 2, rng);
  BasisFactor never(with_threshold(0.0));
  BasisFactor always(with_threshold(2.0));
  REQUIRE(never.factorize(tb.A, tb.basis) == FactorStatus::Ok);
  REQUIRE(always.factorize(tb.A, tb.basis) == FactorStatus::Ok);
  const std::vector<double> x1 = solve_dense(never, a, false);
  const std::vector<double> x2 = solve_dense(always, a, false);
  CHECK_EQ(never.stats().hyper_solves, std::size_t{0});
  CHECK_EQ(never.stats().dense_solves, std::size_t{2});
  CHECK_EQ(always.stats().hyper_solves, std::size_t{2});
  CHECK_EQ(always.stats().dense_solves, std::size_t{0});
  CHECK(max_abs_diff(x1, x2) <= 1e-13 * std::max(1.0, norm_inf(x1)));
  never.reset_counters();
  CHECK_EQ(never.stats().dense_solves, std::size_t{0});
}

// Forced hypersparse and forced dense solves on the random bases of the property
// test must agree. Both paths do the same floating-point operations on every
// entry in a different summation order, so they may differ by rounding errors
// amplified by the condition number: the bound is 100 * kappa * eps (at least 1e-13).
TEST_CASE(lu_hypersparse_and_dense_paths_agree_on_random_bases) {
  constexpr double kEps = 2.220446049250313e-16;
  int compared = 0;
  double worst_ratio = 0.0;
  for (std::uint64_t seed = 1; seed <= 500; ++seed) {
    TestBasis tb = make_seeded_basis(seed);
    BasisFactor hyper(with_threshold(2.0)), dense(with_threshold(0.0));
    const FactorStatus s1 = hyper.factorize(tb.A, tb.basis);
    const FactorStatus s2 = dense.factorize(tb.A, tb.basis);
    CHECK(s1 == s2);
    if (s1 != FactorStatus::Ok) {
      hyper.repair(tb.A, tb.basis);
      if (hyper.status() != FactorStatus::Ok) continue;
      if (dense.factorize(tb.A, tb.basis) != FactorStatus::Ok) continue;
    }
    const DenseMatrix B = basis_to_dense(tb.A, tb.basis);
    const DenseLu lu(B);
    if (lu.singular()) continue;
    const double kappa = std::max(B.norm_inf() * lu.inverse_norm_inf(), 1.0);
    Rng rng(seed * 13ULL);
    for (int k = 0; k < 4; ++k) {
      const std::vector<double> a = k < 1 ? random_dense_vector(tb.m(), rng) : random_sparse_vector(tb.m(), k, rng);
      for (const bool transpose : {false, true}) {
        const std::vector<double> xh = solve_dense(hyper, a, transpose);
        const std::vector<double> xd = solve_dense(dense, a, transpose);
        const double diff = max_abs_diff(xh, xd);
        const double bound = std::max(1e-13, 100.0 * kappa * kEps) * std::max(1.0, norm_inf(xd));
        worst_ratio = std::max(worst_ratio, diff / bound);
        if (!(diff <= bound)) {
          std::cerr << "FAILING SEED " << seed << " (" << tb.family << ", m = " << tb.m() << "): hypersparse and dense differ by " << diff
                    << " (bound " << bound << ")\n";
          CHECK(false);
        }
        ++compared;
      }
    }
  }
  std::cout << "    hypersparse vs dense on random bases: " << compared << " solve pairs, worst difference / bound = " << std::scientific
            << std::setprecision(2) << worst_ratio << std::defaultfloat << "\n";
  CHECK(compared > 1000);
}

// Quick timing of the two paths on the largest test basis. Printed for the
// record; nothing is asserted about speed (it depends on the machine and on the
// structure of this particular basis).
TEST_CASE(lu_hypersparse_quick_timing_report) {
  const Index m = 60000;
  const TestBasis tb = nearly_triangular(m, 8, 21);
  BasisFactor hyper(with_threshold(2.0)), dense(with_threshold(0.0));
  REQUIRE(hyper.factorize(tb.A, tb.basis) == FactorStatus::Ok);
  REQUIRE(dense.factorize(tb.A, tb.basis) == FactorStatus::Ok);
  Rng rng(1);
  const int solves = 300;
  std::vector<std::vector<Index>> where(to_size(solves));
  for (auto& w : where) w = {rng.range(0, m - 1), rng.range(0, m - 1)};
  auto run = [&](BasisFactor& f) {
    SparseWork w(m);
    const auto t0 = std::chrono::steady_clock::now();
    double sink = 0.0;
    for (int k = 0; k < solves; ++k) {
      w.clear();
      w.set(where[to_size(k)][0], 1.0);
      w.set(where[to_size(k)][1], -0.5);
      f.ftran(w);
      sink += w.norm_inf();
    }
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(sink > 0.0);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };
  const double t_hyper = run(hyper);
  const double t_dense = run(dense);
  std::cout << "    timing, m = " << m << ", " << solves << " ftran with 2 nonzeros: hypersparse " << std::fixed << std::setprecision(1)
            << t_hyper << " ms, dense loops " << t_dense << " ms ("
            << (t_hyper < t_dense ? "hypersparse faster" : "hypersparse NOT faster") << " here); not a benchmark\n"
            << std::defaultfloat;
}
