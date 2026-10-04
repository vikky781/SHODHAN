// SimplexEngine::add_rows / remove_rows: a warm resolve after appending cuts agrees with a cold solve of the
// explicitly extended model; removing the new rows (or only the slack ones) restores the old behaviour.

#include <cmath>
#include <iostream>
#include <vector>

#include "shodhan/simplex_engine.hpp"
#include "support/random_lp.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

LpModel with_rows(const LpModel& m, const std::vector<RowSpec>& rows) {
  LpModel out = m;
  std::vector<Triplet> t;
  for (Index j = 0; j < m.n_cols; ++j) {
    for (Index p = m.A.col_start[to_size(j)]; p < m.A.col_start[to_size(j) + 1]; ++p) {
      t.push_back({m.A.row_index[to_size(p)], j, m.A.value[to_size(p)]});
    }
  }
  Index r = m.n_rows;
  for (const RowSpec& rs : rows) {
    for (std::size_t q = 0; q < rs.idx.size(); ++q) t.push_back({r, rs.idx[q], rs.val[q]});
    out.row_lower.push_back(rs.lo);
    out.row_upper.push_back(rs.hi);
    ++r;
  }
  out.n_rows = r;
  std::string err;
  SparseMatrix::from_triplets(out.n_rows, out.n_cols, std::move(t), &out.A, &err);
  return out;
}

std::vector<RowSpec> random_cuts(const LpModel& m, const std::vector<double>& x, Rng& rng, int count) {
  std::vector<RowSpec> rows;
  for (int k = 0; k < count; ++k) {
    RowSpec r;
    double act = 0.0;
    for (Index j = 0; j < m.n_cols; ++j) {
      if (!rng.chance(0.5)) continue;
      const double a = static_cast<double>(rng.range(-3, 3));
      if (a == 0.0) continue;
      r.idx.push_back(j);
      r.val.push_back(a);
      act += a * x[to_size(j)];
    }
    if (r.idx.empty()) continue;
    if (rng.chance(0.6)) {
      r.hi = act - 0.25 - rng.unit();  // cuts off x
    } else {
      r.hi = act + 1.0 + rng.unit();  // slack at x
    }
    rows.push_back(std::move(r));
  }
  return rows;
}

}  // namespace

TEST_CASE(engine_add_rows_warm_resolve_matches_a_cold_solve) {
  int compared = 0, optimal = 0, infeasible = 0, warm_cheaper = 0;
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    RandomLpOptions o;
    o.rows = 6 + static_cast<int>(seed % 5);
    o.cols = 8 + static_cast<int>(seed % 7);
    const RandomLp lp = make_random_lp(seed, o);
    SimplexEngine e(lp.model);
    if (e.solve() != EngineStatus::Optimal) continue;
    Rng rng(seed * 31 + 5);
    const std::vector<double> x(e.primal_all().begin(), e.primal_all().begin() + lp.model.n_cols);
    const std::vector<RowSpec> cuts = random_cuts(lp.model, x, rng, 1 + static_cast<int>(seed % 4));
    if (cuts.empty()) continue;
    const long long it0 = e.stats().iterations;
    e.add_rows(cuts);
    CHECK_EQ(e.n_rows(), lp.model.n_rows + static_cast<Index>(cuts.size()));
    const EngineStatus warm = e.solve();
    const long long warm_iters = e.stats().iterations - it0;
    const LpModel full = with_rows(lp.model, cuts);
    SimplexEngine cold(full);
    const EngineStatus cs = cold.solve();
    ++compared;
    if (warm != cs) {
      std::cerr << "  FAILING SEED " << seed << ": warm " << to_string(warm) << " cold " << to_string(cs) << "\n";
      CHECK(false);
      continue;
    }
    if (cs == EngineStatus::Optimal) {
      ++optimal;
      const double a = e.objective(), b = cold.objective();
      if (std::fabs(a - b) > 1e-6 * (1.0 + std::fabs(b))) {
        std::cerr << "  FAILING SEED " << seed << ": warm objective " << a << " cold " << b << "\n";
        CHECK(false);
      }
      if (warm_iters < cold.stats().iterations) ++warm_cheaper;
      // Drop the slack cut rows; the optimum must stay.
      std::vector<Index> slack;
      for (Index i = lp.model.n_rows; i < e.n_rows(); ++i) {
        if (e.status(e.n_structural() + i) == VarStatus::Basic) slack.push_back(i);
      }
      REQUIRE(e.remove_rows(slack));
      CHECK_EQ(e.n_rows(), lp.model.n_rows + static_cast<Index>(cuts.size()) - static_cast<Index>(slack.size()));
      CHECK(e.solve() == EngineStatus::Optimal);
      CHECK_NEAR(e.objective(), b, 1e-6 * (1.0 + std::fabs(b)));
    } else if (cs == EngineStatus::Infeasible) {
      ++infeasible;
    }
  }
  std::cout << "  add_rows: " << compared << " models, " << optimal << " optimal, " << infeasible
            << " infeasible after the cuts, warm start needed fewer iterations than cold in " << warm_cheaper << "\n";
  CHECK(optimal > 150);
}

TEST_CASE(engine_remove_rows_refuses_tight_rows_and_add_then_remove_is_the_identity) {
  for (std::uint64_t seed = 1; seed <= 60; ++seed) {
    RandomLpOptions o;
    const RandomLp lp = make_random_lp(seed + 1000, o);
    SimplexEngine e(lp.model);
    if (e.solve() != EngineStatus::Optimal) continue;
    const double base = e.objective();
    Rng rng(seed);
    const std::vector<double> x(e.primal_all().begin(), e.primal_all().begin() + lp.model.n_cols);
    const std::vector<RowSpec> cuts = random_cuts(lp.model, x, rng, 3);
    if (cuts.empty()) continue;
    e.add_rows(cuts);
    std::vector<Index> all;
    for (Index i = lp.model.n_rows; i < e.n_rows(); ++i) all.push_back(i);
    REQUIRE(e.remove_rows(all));  // freshly added rows are basic
    CHECK_EQ(e.n_rows(), lp.model.n_rows);
    CHECK(e.solve() == EngineStatus::Optimal);
    CHECK_NEAR(e.objective(), base, 1e-7 * (1.0 + std::fabs(base)));
    // A tight row cannot be removed.
    e.add_rows(cuts);
    e.solve();
    for (Index i = lp.model.n_rows; i < e.n_rows(); ++i) {
      if (e.status(e.n_structural() + i) != VarStatus::Basic) {
        const Index before = e.n_rows();
        CHECK(!e.remove_rows({i}));
        CHECK_EQ(e.n_rows(), before);
        break;
      }
    }
  }
}
