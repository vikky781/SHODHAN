// Primal heuristics: every proposal is verified by the incumbent manager on the ORIGINAL model; a heuristic
// that returns an infeasible point is rejected and counted, never accepted.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/mip/mip_solver.hpp"
#include "support/mip_families.hpp"
#include "support/mip_harness.hpp"
#include "support/mip_oracle.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::mip;
using namespace shodhan::testing;

namespace shodhan::mip {
void register_heuristics_simple(PluginRegistry&);  // defined in src/mip/heuristics_simple.cpp
}

namespace {

struct Rate {
  int runs = 0, found = 0, bad = 0;  // bad: accepted but not feasible on the original model (must stay 0)
};

// Runs heuristic `name` at the root LP of every instance of a family; counts instances with a fractional LP.
Rate run_lp_heuristic(const std::string& name, int family, std::uint64_t first, std::uint64_t last, bool require_feasible) {
  Rate r;
  for (std::uint64_t seed = first; seed <= last; ++seed) {
    const LpModel m = make_mip_instance(family, seed);
    if (require_feasible) {
      const MipRefResult ref = solve_mip_brute_force(m);
      if (ref.status != Status::Optimal) continue;
    }
    MipHarness h(m);
    if (!h.lp_ok || h.state.lp.fractional.empty()) continue;
    auto heur = plugin_registry().heuristics.make(name);
    ++r.runs;
    heur->run(h.state, HeuristicWhen::AfterRootLp);
    if (h.incumbent->has_incumbent()) {
      ++r.found;
      if (!check_point(m, h.incumbent->x()).ok(1e-6, 0.0)) ++r.bad;
    }
  }
  return r;
}

class RogueHeuristic final : public PrimalHeuristic {
 public:
  const char* name() const override { return "trivial"; }  // replaces the trivial heuristic in the end-to-end test
  bool wants(const MipOptions&, HeuristicWhen when, long long, Index) const override { return when == HeuristicWhen::BeforeRootLp || when == HeuristicWhen::AfterRootLp; }
  void run(SearchState& s, HeuristicWhen) override {
    const Index n = s.model->n_cols;
    std::vector<double> bad(to_size(n), 1e6);  // violates bounds and rows
    s.submit(bad, name());
    std::vector<double> low(to_size(n), -1e6);  // below every lower bound (fractional values would just be snapped)
    s.submit(low, name());
    s.submit(std::vector<double>(to_size(n) + 7, 0.0), name());  // wrong size, whatever the dimension
  }
};

}  // namespace

TEST_CASE(simple_rounding_succeeds_where_a_locked_direction_exists) {
  const Rate sc = run_lp_heuristic("rounding", kMipSetCover, 1, 1500, false);
  const Rate kn = run_lp_heuristic("rounding", kMipKnapsack, 1, 600, false);
  std::cout << "    simple rounding: set cover " << sc.found << " of " << sc.runs << " fractional LPs, knapsack " << kn.found << " of " << kn.runs << "\n";
  CHECK(sc.runs + kn.runs >= 200);
  CHECK_EQ(sc.found, sc.runs);  // covering rows (>=): rounding up never violates one
  CHECK_EQ(kn.found, kn.runs);  // packing rows (<=): rounding down never violates one
  CHECK_EQ(sc.bad + kn.bad, 0);
}

TEST_CASE(simple_rounding_does_not_pretend_when_columns_are_locked_both_ways) {
  // Assignment rows are equalities: every column is locked both ways, so rounding cannot help.
  int attempts = 0, found = 0;
  for (std::uint64_t seed = 1; seed <= 80; ++seed) {
    const LpModel m = make_mip_instance(kMipAssignmentSide, seed);
    MipHarness h(m);
    if (!h.lp_ok || h.state.lp.fractional.empty()) continue;
    ++attempts;
    plugin_registry().heuristics.make("rounding")->run(h.state, HeuristicWhen::AfterRootLp);
    if (h.incumbent->has_incumbent()) {
      ++found;
      CHECK(check_point(m, h.incumbent->x()).ok(1e-6, 0.0));
    }
  }
  std::cout << "    simple rounding on assignment+side (both-way locks): " << found << " of " << attempts << " fractional LPs\n";
  CHECK_EQ(found, 0);
}

TEST_CASE(trivial_heuristic_finds_the_all_lower_or_all_upper_point) {
  int runs = 0, found = 0;
  for (const int family : {kMipKnapsack, kMipSetCover}) {
    for (std::uint64_t seed = 1; seed <= 120; ++seed) {
      const LpModel m = make_mip_instance(family, seed);
      MipHarness h(m, false);
      ++runs;
      plugin_registry().heuristics.make("trivial")->run(h.state, HeuristicWhen::BeforeRootLp);
      if (h.incumbent->has_incumbent()) {
        ++found;
        CHECK(check_point(m, h.incumbent->x()).ok(1e-6, 0.0));
      }
    }
  }
  std::cout << "    trivial heuristic: " << found << " of " << runs << " (knapsack: all lower, set cover: all upper)\n";
  CHECK(runs >= 200);
  CHECK_EQ(found, runs);
}

TEST_CASE(diving_heuristics_find_feasible_points_and_leave_the_search_untouched) {
  for (const char* name : {"diving-fractional", "diving-coefficient"}) {
    struct Fam { int family; bool feasible_only; };
    int total_runs = 0, total_found = 0, bad = 0;
    std::cout << "    " << name << ":";
    for (const Fam f : {Fam{kMipSetCover, false}, Fam{kMipKnapsack, false}, Fam{kMipAssignmentSide, true}, Fam{kMipFacility, false}}) {
      const Rate r = run_lp_heuristic(name, f.family, 1, f.family == kMipSetCover ? 900 : 200, f.feasible_only);
      total_runs += r.runs;
      total_found += r.found;
      bad += r.bad;
      std::cout << " " << mip_family_name(f.family) << " " << r.found << "/" << r.runs << ";";
    }
    std::cout << " total " << total_found << " of " << total_runs << "\n";
    CHECK(total_runs >= 200);
    CHECK_EQ(bad, 0);
    CHECK(total_found * 10 >= total_runs * 6);  // at least 60% (a target, reported above)
  }
  // The engine of the search is not changed by a dive.
  const LpModel m = make_mip_instance(kMipKnapsack, 5);
  MipHarness h(m);
  if (h.lp_ok && !h.state.lp.fractional.empty()) {
    const BasisSnapshot before = h.engine->get_basis_snapshot();
    const std::vector<double> x_before = h.engine->primal_all();
    plugin_registry().heuristics.make("diving-fractional")->run(h.state, HeuristicWhen::AfterRootLp);
    CHECK(h.engine->get_basis_snapshot() == before);
    CHECK(h.engine->primal_all() == x_before);
  }
}

TEST_CASE(feasibility_jump_finds_feasible_points_without_an_lp) {
  struct Fam { int family; const char* label; bool feasible_only; };
  int total = 0, found = 0;
  std::cout << "    feasibility jump (no LP):";
  for (const Fam f : {Fam{kMipSetCover, "set cover", false}, Fam{kMipKnapsack, "knapsack", false}, Fam{kMipAssignmentSide, "assignment+side", true}, Fam{kMipGeneralInt, "general int", true},
                      Fam{kMipFacility, "facility", false}, Fam{kMipLotSizing, "lot sizing", false}}) {
    int runs = 0, ok = 0;
    for (std::uint64_t seed = 1; seed <= 100; ++seed) {
      const LpModel m = make_mip_instance(f.family, seed);
      if (f.feasible_only && solve_mip_brute_force(m).status != Status::Optimal) continue;
      MipHarness h(m, false);
      ++runs;
      plugin_registry().heuristics.make("feasibility-jump")->run(h.state, HeuristicWhen::BeforeRootLp);
      if (h.incumbent->has_incumbent()) {
        ++ok;
        CHECK(check_point(m, h.incumbent->x()).ok(1e-6, 0.0));
      }
    }
    std::cout << " " << f.label << " " << ok << "/" << runs << ";";
    total += runs;
    found += ok;
  }
  std::cout << " overall " << found << " of " << total << " (" << std::fixed << std::setprecision(1) << 100.0 * found / std::max(total, 1) << "%)" << std::defaultfloat << "\n";
  CHECK(total >= 200);
  CHECK(found * 10 >= total * 7);  // stated fraction: at least 70% on these easy satisfiable instances
}

TEST_CASE(a_heuristic_that_returns_infeasible_points_is_rejected_and_counted) {
  const LpModel m = make_mip_instance(kMipKnapsack, 4);
  MipHarness h(m, false);
  RogueHeuristic rogue;
  rogue.run(h.state, HeuristicWhen::BeforeRootLp);
  CHECK(!h.incumbent->has_incumbent());
  CHECK_EQ(h.incumbent->submitted(), 3);
  CHECK_EQ(h.incumbent->rejections().total(), 3);
  CHECK_EQ(h.incumbent->found(), 0);
}

TEST_CASE(a_rogue_heuristic_inside_a_search_never_corrupts_the_result) {
  PluginRegistry& reg = plugin_registry();
  int rejected_runs = 0, solved_by_presolve = 0, checked = 0;
  reg.heuristics.add("trivial", [] { return std::make_unique<RogueHeuristic>(); });
  for (std::uint64_t seed = 1; seed <= 30; ++seed) {
    const LpModel m = make_mip_instance(static_cast<int>(seed % 4), seed);
    const MipRefResult ref = solve_mip_brute_force(m);
    if (ref.status != Status::Optimal) continue;
    MipOptions o;
    o.mip_gap = 1e-9;
    o.mip_abs_gap = 1e-9;
    o.params.verbosity = 0;
    const MipResult r = MipSolver(o).solve(m);
    ++checked;
    CHECK(r.status == Status::Optimal);
    CHECK(std::fabs(r.objective - ref.objective) <= 1e-6 * (1 + std::fabs(ref.objective)));
    CHECK(check_point(m, r.solution.x).ok(1e-6, 0.0));
    if (r.rejections.total() >= 3) ++rejected_runs;
    else if (r.message == "solved by presolve") ++solved_by_presolve;  // no heuristic runs when presolve decides everything
  }
  // Restore the real trivial heuristic for the tests that run after this one.
  register_heuristics_simple(reg);  // restore the real trivial (and rounding) heuristic for the tests that follow
  std::cout << "    rogue heuristic in " << checked << " searches: results correct, rejections counted in " << rejected_runs << "\n";
  CHECK(checked > 15);
  CHECK_EQ(rejected_runs + solved_by_presolve, checked);
}

TEST_CASE(heuristics_on_and_off_give_the_same_optimum) {
  int compared = 0, failed = 0;
  long long nodes_on = 0, nodes_off = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    if (mip_family_is_unbounded(f)) continue;
    for (std::uint64_t seed = 201; seed <= 235; ++seed) {
      const LpModel m = make_mip_instance(f, seed);
      MipOptions on, off;
      on.mip_gap = off.mip_gap = 1e-9;
      on.mip_abs_gap = off.mip_abs_gap = 1e-9;
      on.params.verbosity = off.params.verbosity = 0;
      off.heuristics = false;
      const MipResult a = MipSolver(on).solve(m);
      const MipResult b = MipSolver(off).solve(m);
      ++compared;
      const bool same = a.status == b.status && (a.status != Status::Optimal || std::fabs(a.objective - b.objective) <= 1e-6 * (1 + std::fabs(b.objective)));
      if (!same) {
        ++failed;
        std::cerr << "FAILING " << mip_family_name(f) << " seed " << seed << ": on " << to_string(a.status) << " " << a.objective << ", off " << to_string(b.status) << " " << b.objective << "\n";
        CHECK(false);
      }
      nodes_on += a.nodes_processed;
      nodes_off += b.nodes_processed;
    }
  }
  std::cout << "    heuristics on vs off: " << compared << " instances, " << failed << " different; total nodes on " << nodes_on << ", off " << nodes_off << " (informational)\n";
  CHECK_EQ(failed, 0);
  CHECK(compared >= 300);
}

TEST_CASE(search_reports_per_heuristic_statistics) {
  const LpModel m = make_mip_instance(kMipSetCover, 7);
  MipOptions o;
  o.params.verbosity = 0;
  const MipResult r = MipSolver(o).solve(m);
  CHECK_EQ(r.heuristics.size(), 5u);
  CHECK_EQ(r.heuristics[0].name, std::string("trivial"));
  CHECK_EQ(r.heuristics[4].name, std::string("feasibility-jump"));
  long long successes = 0;
  for (const HeuristicStats& h : r.heuristics) successes += h.successes;
  CHECK(successes <= r.incumbents_found);
  CHECK(r.heuristics[0].calls >= 1);
}
