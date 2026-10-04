// Basis snapshots of the simplex engine: restoring one and resolving matches a cold solve.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "shodhan/kkt.hpp"
#include "shodhan/rays.hpp"
#include "shodhan/simplex_engine.hpp"
#include "support/rng.hpp"
#include "support/simplex_lps.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

TEST_CASE(basis_snapshot_restores_the_optimal_basis_without_iterations) {
  int checked = 0;
  for (std::uint64_t seed = 1; seed <= 100; ++seed) {
    SimplexLpOptions o;
    o.rows = 4 + static_cast<int>(seed % 10);
    o.cols = 6 + static_cast<int>(seed % 12);
    o.boxed_fraction = 0.4;
    o.degenerate = seed % 3 == 0 ? 0.4 : 0.0;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexEngine first(model);
    if (first.solve() != EngineStatus::Optimal) continue;
    const BasisSnapshot snap = first.get_basis_snapshot();
    CHECK_EQ(snap.status.size(), static_cast<std::size_t>(model.n_cols + model.n_rows));
    SimplexEngine second(model);
    CHECK(second.set_basis(snap));
    CHECK(second.get_basis_snapshot() == snap);
    const long long before = second.stats().iterations;
    CHECK(second.solve() == EngineStatus::Optimal);
    CHECK_EQ(second.stats().iterations - before, 0LL);
    CHECK(std::fabs(second.solution().objective - first.solution().objective) <= 1e-9 * (1.0 + std::fabs(first.solution().objective)));
    ++checked;
  }
  CHECK(checked > 60);
}

TEST_CASE(basis_snapshot_resolve_matches_a_cold_solve_on_300_seeded_lps) {
  int compared = 0, infeasible = 0, failed = 0;
  long long warm_iters = 0, cold_iters = 0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    SimplexLpOptions o;
    o.rows = 5 + static_cast<int>(seed % 12);
    o.cols = 8 + static_cast<int>(seed % 15);
    o.boxed_fraction = 0.4;
    o.degenerate = seed % 3 == 0 ? 0.4 : 0.0;
    const LpModel model = make_simplex_lp(seed, o);
    SimplexEngine parent(model);
    if (parent.solve() != EngineStatus::Optimal) continue;
    const BasisSnapshot snap = parent.get_basis_snapshot();
    Rng rng(seed * 977 + 3);
    const Index j = rng.range(0, model.n_cols - 1);
    const double lo = model.col_lower[to_size(j)], hi = model.col_upper[to_size(j)];
    const double xj = parent.primal_all()[to_size(j)];
    double nlo = lo, nhi = hi;
    if (seed % 2 == 0) nhi = std::max(lo, xj - static_cast<double>(rng.range(1, 8)) / 4.0);
    else nlo = std::min(hi, xj + static_cast<double>(rng.range(1, 8)) / 4.0);
    if (nlo > nhi) std::swap(nlo, nhi);
    LpModel changed = model;
    changed.col_lower[to_size(j)] = nlo;
    changed.col_upper[to_size(j)] = nhi;

    // The child engine starts from the slack basis on the changed model, then restores the parent's basis.
    SimplexEngine child(changed);
    if (!child.set_basis(snap)) {
      ++failed;
      std::cerr << "FAILING SEED " << seed << ": set_basis failed\n";
      CHECK(false);
      continue;
    }
    const EngineStatus sw = child.solve();
    SimplexEngine cold(changed);
    const EngineStatus sc = cold.solve();
    if (sw != sc) {
      ++failed;
      std::cerr << "FAILING SEED " << seed << ": restored " << to_string(sw) << ", cold " << to_string(sc) << "\n";
      CHECK(false);
      continue;
    }
    if (sw == EngineStatus::Optimal) {
      const double ow = child.solution().objective, oc = cold.solution().objective;
      if (!(std::fabs(ow - oc) <= 1e-7 * (1.0 + std::fabs(oc)))) {
        ++failed;
        std::cerr << "FAILING SEED " << seed << ": restored objective " << ow << ", cold " << oc << "\n";
        CHECK(false);
        continue;
      }
      CHECK(check_kkt(changed, child.solution(), 1e-6).ok);
    } else if (sw == EngineStatus::Infeasible) {
      ++infeasible;
      CHECK(check_farkas(changed, child.farkas_ray(), 1e-9).ok);
    }
    ++compared;
    warm_iters += child.stats().iterations;
    cold_iters += cold.stats().iterations;
  }
  std::cout << "    basis snapshot restore vs cold solve: " << compared << " cases agree (" << infeasible << " infeasible after the change), " << failed
            << " failed; average iterations restored " << std::fixed << std::setprecision(2) << static_cast<double>(warm_iters) / std::max(compared, 1)
            << " vs cold " << static_cast<double>(cold_iters) / std::max(compared, 1) << std::defaultfloat << "\n";
  CHECK_EQ(failed, 0);
  CHECK(compared > 200);
}

TEST_CASE(basis_snapshot_rejects_a_wrong_size_or_wrong_basic_count) {
  const LpModel model = make_simplex_lp(7, SimplexLpOptions{});
  SimplexEngine e(model);
  BasisSnapshot bad;
  bad.status.assign(3, 0);
  CHECK(!e.set_basis(bad));
  BasisSnapshot all_nonbasic = e.get_basis_snapshot();
  std::fill(all_nonbasic.status.begin(), all_nonbasic.status.end(), static_cast<std::uint8_t>(VarStatus::AtLower));
  CHECK(!e.set_basis(all_nonbasic));
}
