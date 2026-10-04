// Engine probes: save_state / restore_state / EngineProbe leave the engine exactly as it was.

#include <cmath>
#include <iostream>
#include <vector>

#include "shodhan/simplex_engine.hpp"
#include "support/rng.hpp"
#include "support/simplex_lps.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

struct Snapshot {
  BasisSnapshot basis;
  std::vector<Index> order;
  std::vector<double> x, d, y, lo, hi;
  long long iterations;
  double objective;
};

Snapshot take(const SimplexEngine& e) {
  Snapshot s;
  s.basis = e.get_basis_snapshot();
  s.order = e.basis();
  s.x = e.primal_all();
  s.d = e.dual_all();
  s.y = e.row_duals();
  for (Index j = 0; j < e.n_structural(); ++j) {
    s.lo.push_back(e.col_lower(j));
    s.hi.push_back(e.col_upper(j));
  }
  s.iterations = e.stats().iterations;
  s.objective = e.objective();
  return s;
}

bool same(const Snapshot& a, const Snapshot& b) {
  return a.basis == b.basis && a.order == b.order && a.x == b.x && a.d == b.d && a.y == b.y && a.lo == b.lo && a.hi == b.hi && a.iterations == b.iterations && a.objective == b.objective;
}

}  // namespace

TEST_CASE(engine_probe_restores_the_state_exactly) {
  int checked = 0;
  for (std::uint64_t seed = 1; seed <= 60; ++seed) {
    SimplexLpOptions o;
    o.rows = 6 + static_cast<int>(seed % 10);
    o.cols = 10 + static_cast<int>(seed % 12);
    o.boxed_fraction = 0.5;
    const LpModel m = make_simplex_lp(seed, o);
    SimplexEngine e(m);
    if (e.solve() != EngineStatus::Optimal) continue;
    const Snapshot before = take(e);
    Rng rng(seed * 13 + 1);
    {
      EngineProbe probe(e);
      for (int k = 0; k < 3; ++k) {
        const Index j = rng.range(0, m.n_cols - 1);
        const double v = e.primal_all()[to_size(j)];
        e.change_col_bounds(j, e.col_lower(j), std::max(e.col_lower(j), std::floor(v)));
        e.solve_limited(20);
      }
      CHECK(probe.iterations() >= 0);
    }
    CHECK(same(before, take(e)));
    CHECK(!e.factor_valid());  // rebuilt lazily
    ++checked;
  }
  CHECK(checked > 40);
}

TEST_CASE(main_engine_solves_identically_after_1000_random_probes) {
  int probes = 0, compared = 0;
  for (std::uint64_t seed = 1; seed <= 25; ++seed) {
    SimplexLpOptions o;
    o.rows = 6 + static_cast<int>(seed % 12);
    o.cols = 10 + static_cast<int>(seed % 14);
    o.boxed_fraction = 0.5;
    o.degenerate = seed % 3 == 0 ? 0.4 : 0.0;
    const LpModel m = make_simplex_lp(seed, o);
    SimplexEngine probed(m), twin(m);
    if (probed.solve() != EngineStatus::Optimal || twin.solve() != EngineStatus::Optimal) continue;
    Rng rng(seed * 977 + 5);
    for (int k = 0; k < 40; ++k) {
      EngineProbe probe(probed);
      const int depth = rng.range(1, 3);  // nested probes too
      for (int t = 0; t < depth; ++t) {
        const Index j = rng.range(0, m.n_cols - 1);
        const double v = probed.primal_all()[to_size(j)];
        if (rng.chance(0.5)) probed.change_col_bounds(j, probed.col_lower(j), std::max(probed.col_lower(j), std::floor(v)));
        else probed.change_col_bounds(j, std::min(probed.col_upper(j), std::ceil(v)), probed.col_upper(j));
        {
          EngineProbe inner(probed);
          probed.solve_limited(rng.range(1, 30));
        }
        probed.solve_limited(rng.range(1, 30));
      }
      ++probes;
    }
    CHECK(same(take(twin), take(probed)));
    // Now do the same real change on both and solve fully: identical results, bit for bit.
    const Index j = rng.range(0, m.n_cols - 1);
    const double v = twin.primal_all()[to_size(j)];
    const double nhi = std::max(twin.col_lower(j), std::floor(v));
    twin.change_col_bounds(j, twin.col_lower(j), nhi);
    probed.change_col_bounds(j, probed.col_lower(j), nhi);
    const EngineStatus st_twin = twin.solve(), st_probed = probed.solve();
    CHECK(st_twin == st_probed);
    CHECK(same(take(twin), take(probed)));
    ++compared;
  }
  std::cout << "    engine probes: " << probes << " random (partly nested) probes on " << compared << " LPs; the probed engine equals the twin that never probed, bit for bit, and then solves identically\n";
  CHECK(probes >= 1000);
  CHECK(compared >= 20);
}
