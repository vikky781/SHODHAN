// Determinism: identical inputs give bit-identical factors and solutions.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "support/lu_testing.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

bool bits_equal(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

bool factors_identical(const BasisFactor& a, const BasisFactor& b) {
  const FactorDump da = a.dump();
  const FactorDump db = b.dump();
  return da.ints == db.ints && bits_equal(da.reals, db.reals);
}

std::vector<double> solve(BasisFactor& f, const std::vector<double>& a, bool transpose, bool spike = false) {
  SparseWork w;
  w.load_dense(a);
  if (transpose) f.btran(w); else f.ftran(w, spike);
  std::vector<double> x(a.size());
  w.to_dense(x);
  return x;
}

}  // namespace

TEST_CASE(lu_determinism_factorize_twice_is_bit_identical) {
  int identical = 0;
  const int kSeeds = 300;
  for (std::uint64_t seed = 1; seed <= kSeeds; ++seed) {
    TestBasis tb = make_seeded_basis(seed);
    BasisFactor f1, f2;
    const FactorStatus s1 = f1.factorize(tb.A, tb.basis);
    const FactorStatus s2 = f2.factorize(tb.A, tb.basis);
    CHECK(s1 == s2);
    CHECK(f1.deficient_positions() == f2.deficient_positions());
    CHECK(f1.unpivoted_rows() == f2.unpivoted_rows());
    if (s1 != FactorStatus::Ok) {
      std::vector<Index> b1 = tb.basis, b2 = tb.basis;
      const auto c1 = f1.repair(tb.A, b1);
      const auto c2 = f2.repair(tb.A, b2);
      CHECK(b1 == b2);
      CHECK_EQ(c1.size(), c2.size());
      tb.basis = b1;
      if (f1.status() != FactorStatus::Ok || f2.status() != FactorStatus::Ok) continue;
    }
    // Refactorizing the same object (reused buffers) must reproduce a fresh object.
    BasisFactor reused;
    const TestBasis other = make_family_basis(1, 7, seed);
    REQUIRE(reused.factorize(other.A, other.basis) == FactorStatus::Ok);
    REQUIRE(reused.factorize(tb.A, tb.basis) == FactorStatus::Ok);
    BasisFactor fresh;
    REQUIRE(fresh.factorize(tb.A, tb.basis) == FactorStatus::Ok);
    bool ok = factors_identical(reused, fresh) && factors_identical(f1, fresh);
    CHECK(ok);
    Rng rng(seed);
    for (int k = 0; k < 3; ++k) {
      const std::vector<double> a = k == 0 ? random_dense_vector(tb.m(), rng) : random_sparse_vector(tb.m(), k, rng);
      for (const bool tr : {false, true}) {
        const bool same = bits_equal(solve(reused, a, tr), solve(fresh, a, tr)) && bits_equal(solve(f2, a, tr), solve(fresh, a, tr));
        CHECK(same);
        ok = ok && same;
      }
    }
    if (ok) ++identical;
    else std::cerr << "FAILING SEED " << seed << ": factors or solutions are not bit-identical\n";
  }
  CHECK_EQ(identical, kSeeds);
}

TEST_CASE(lu_determinism_update_sequences_are_bit_identical) {
  for (std::uint64_t seed = 1; seed <= 40; ++seed) {
    TestBasis tb = make_family_basis(static_cast<int>(1 + seed % 7), 20 + static_cast<Index>(seed % 15), seed);
    const Index m = tb.m();
    const Index total = tb.A.n_cols + m;
    BasisFactor fa, fb;
    REQUIRE(fa.factorize(tb.A, tb.basis) == FactorStatus::Ok);
    REQUIRE(fb.factorize(tb.A, tb.basis) == FactorStatus::Ok);
    std::vector<Index> basis_a = tb.basis, basis_b = tb.basis;
    Rng ra(seed), rb(seed);
    auto step = [&](BasisFactor& f, std::vector<Index>& basis, Rng& rng) {
      std::vector<double> col(to_size(m), 0.0);
      const Index q = rng.range(0, total - 1);
      if (std::find(basis.begin(), basis.end(), q) != basis.end()) return;
      if (q < tb.A.n_cols) {
        for (Index t = tb.A.col_start[to_size(q)]; t < tb.A.col_start[to_size(q) + 1]; ++t) col[to_size(tb.A.row_index[to_size(t)])] = tb.A.value[to_size(t)];
      } else {
        col[to_size(q - tb.A.n_cols)] = -1.0;
      }
      const std::vector<double> x = solve(f, col, false, true);
      Index p = 0;
      for (Index i = 1; i < m; ++i) if (std::fabs(x[to_size(i)]) > std::fabs(x[to_size(p)])) p = i;
      if (std::fabs(x[to_size(p)]) < 1e-6) return;
      if (f.update(p) == FactorStatus::Ok) {
        basis[to_size(p)] = q;
      } else {
        basis[to_size(p)] = q;
        f.factorize(tb.A, basis);
      }
    };
    for (int k = 0; k < 120; ++k) {
      step(fa, basis_a, ra);
      step(fb, basis_b, rb);
    }
    CHECK(basis_a == basis_b);
    CHECK(factors_identical(fa, fb));
    Rng rng(seed + 99);
    const std::vector<double> a = random_dense_vector(m, rng);
    CHECK(bits_equal(solve(fa, a, false), solve(fb, a, false)));
    CHECK(bits_equal(solve(fa, a, true), solve(fb, a, true)));
  }
}
