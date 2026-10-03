// Property tests of the basis factorization against the dense oracle.

#include <algorithm>
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

constexpr int kSeeds = 1500;

struct FamilyTally {
  int total = 0;
  int ok = 0;
  int deficient = 0;
  int repaired = 0;
};

}  // namespace

// For every seed: build a structured basis, factorize, and check ftran/btran on
// dense, sparse and unit right-hand sides (relative residual and agreement with
// the dense oracle). A rank deficient basis must be reported with the number of
// deficient columns the dense rank computation predicts; after repair() the basis
// must be nonsingular and pass the same checks.
//
// "The" numerical rank is only well defined when the matrix is far from
// singular. Bases of LPs with coefficients over eight decades can be
// within rounding of singular (condition numbers beyond 1e18), where the rank
// computed at tolerance 1e-8 differs from the one at 1e-14. The oracle therefore
// gives an interval [rank(1e-8), rank(1e-14)] (on the row/column-equilibrated
// matrix) and the factorization's rank m - (deficient count) must lie in it; for
// well-conditioned or exactly singular matrices the interval is a single number.
TEST_CASE(lu_property_random_bases) {
  FamilyTally tally[kNumBasisFamilies];
  double worst_residual = 0.0;
  double worst_forward = 0.0;
  int passes = 0;
  int solves = 0;
  for (std::uint64_t seed = 1; seed <= kSeeds; ++seed) {
    TestBasis tb = make_seeded_basis(seed);
    const int family = static_cast<int>(seed % static_cast<std::uint64_t>(kNumBasisFamilies));
    FamilyTally& t = tally[family];
    ++t.total;
    BasisFactor f;
    Rng rng(seed * 31ULL + 7ULL);
    const FactorStatus st = f.factorize(tb.A, tb.basis);
    bool seed_ok = true;
    auto report = [&](const std::string& what) {
      seed_ok = false;
      std::cerr << "FAILING SEED " << seed << " (family " << tb.family << ", m = " << tb.m() << "): " << what << "\n";
      CHECK(false);
    };
    auto rank_interval_ok = [&](const std::vector<Index>& basis, Index factor_rank, std::string* why) {
      const DenseMatrix B = basis_to_dense(tb.A, basis);
      const Index lo = dense_rank(B, 1e-8);
      const Index hi = dense_rank(B, 1e-14);
      if (factor_rank >= lo && factor_rank <= hi) return true;
      *why = "factorization rank " + std::to_string(factor_rank) + " outside the oracle interval [" + std::to_string(lo) +
             ", " + std::to_string(hi) + "] of " + std::to_string(tb.m());
      return false;
    };
    std::string why;
    if (st == FactorStatus::Ok) {
      ++t.ok;
      if (!rank_interval_ok(tb.basis, tb.m(), &why)) report(why);
      const SolveCheckResult r = check_solves(f, tb.A, tb.basis, rng);
      worst_residual = std::max(worst_residual, r.worst_residual);
      worst_forward = std::max(worst_forward, r.worst_forward_ratio);
      solves += r.solves;
      if (!r.ok()) report(r.failure);
    } else {
      ++t.deficient;
      const Index ndef = static_cast<Index>(f.deficient_positions().size());
      if (!rank_interval_ok(tb.basis, tb.m() - ndef, &why)) report(why);
      if (f.unpivoted_rows().size() != f.deficient_positions().size()) report("unpivoted rows / deficient positions differ in length");
      std::vector<Index> basis = tb.basis;
      const std::vector<BasisSubstitution> changes = f.repair(tb.A, basis);
      if (f.status() != FactorStatus::Ok) {
        report("repair did not give a valid factorization");
      } else {
        ++t.repaired;
        if (changes.size() != static_cast<std::size_t>(ndef)) report("repair returned the wrong number of substitutions");
        for (const BasisSubstitution& c : changes) {
          if (basis[to_size(c.position)] != c.new_var || tb.basis[to_size(c.position)] != c.old_var ||
              c.new_var < tb.A.n_cols) report("inconsistent substitution triple");
        }
        if (!rank_interval_ok(basis, tb.m(), &why)) report("repaired basis: " + why);
        const SolveCheckResult r = check_solves(f, tb.A, basis, rng);
        worst_residual = std::max(worst_residual, r.worst_residual);
        worst_forward = std::max(worst_forward, r.worst_forward_ratio);
        solves += r.solves;
        if (!r.ok()) report("after repair: " + r.failure);
      }
    }
    if (seed_ok) ++passes;
  }
  std::cout << "    lu property: " << passes << "/" << kSeeds << " seeds passed, " << solves
            << " solves, worst relative residual " << std::scientific << std::setprecision(2) << worst_residual
            << ", worst forward-error ratio " << worst_forward << " (1 = bound)\n";
  for (int k = 0; k < kNumBasisFamilies; ++k) {
    std::cout << "      " << std::left << std::setw(20) << basis_family_name(k) << std::right << " bases " << std::setw(4)
              << tally[k].total << "  ok " << std::setw(4) << tally[k].ok << "  rank deficient " << std::setw(4)
              << tally[k].deficient << "  repaired " << std::setw(4) << tally[k].repaired << "\n";
  }
  std::cout << std::defaultfloat;
  CHECK_EQ(passes, kSeeds);
}
