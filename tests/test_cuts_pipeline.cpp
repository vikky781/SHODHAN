// Cutting planes in the whole pipeline: agreement with brute force, early termination, determinism, bound
// monotonicity, and an informational ablation on generated instances.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "shodhan/mip/incumbent.hpp"
#include "shodhan/mip/mip_solver.hpp"
#include "support/cut_families.hpp"
#include "support/mip_oracle.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::mip;
using namespace shodhan::testing;

namespace {

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == 0) return fallback;
  return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
}

}  // namespace

// 1000 seeded MIPs against brute force; half of them with presolve on.
TEST_CASE(cuts_full_pipeline_agrees_with_brute_force) {
  const std::uint64_t per_family = env_u64("SHODHAN_CUT_PIPE_SEEDS", 125);
  long long total = 0, agree = 0, with_cuts = 0, cuts_added = 0, abandoned = 0, bound_ok = 0, bound_checked = 0;
  for (int f = 0; f < kNumCutFamilies; ++f) {
    for (std::uint64_t k = 1; k <= per_family; ++k) {
      const std::uint64_t seed = 90000ULL + static_cast<std::uint64_t>(f) * 1000ULL + k;
      const LpModel m = make_cut_instance(f, seed);
      const MipRefResult ref = solve_mip_brute_force(m);
      if (ref.too_large || (ref.status != Status::Optimal && ref.status != Status::Infeasible)) continue;
      ++total;
      MipOptions o;
      o.params.verbosity = 0;
      o.mip_gap = 1e-9;
      o.mip_abs_gap = 1e-9;
      o.presolve = (k % 2 == 0);
      const MipResult r = MipSolver(o).solve(m);
      std::string why;
      if (ref.status == Status::Infeasible) {
        if (r.status != Status::Infeasible) why = std::string("status ") + to_string(r.status) + ", oracle Infeasible";
      } else if (r.status != Status::Optimal) {
        why = std::string("status ") + to_string(r.status) + " [" + r.message + "], oracle Optimal";
      } else if (std::fabs(r.objective - ref.objective) > 1e-6 * (1.0 + std::fabs(ref.objective))) {
        why = "objective " + std::to_string(r.objective) + " vs oracle " + std::to_string(ref.objective);
      } else if (!check_point(m, r.solution.x).ok(1e-6, 0.0)) {
        why = "incumbent infeasible on the original model";
      }
      if (r.cuts.ran) {
        ++with_cuts;
        cuts_added += r.cuts.cuts_added;
        if (r.cuts.abandoned) ++abandoned;
        if (r.cuts.has_bounds && ref.status == Status::Optimal && !r.cuts.infeasible) {
          ++bound_checked;
          // The root bound with cuts is a valid bound on the optimum and no weaker than without cuts.
          const double sg = m.sense == Sense::Maximize ? -1.0 : 1.0;
          const double with = sg * r.cuts.root_bound_with_cuts, without = sg * r.cuts.root_bound_without_cuts;
          const double opt = sg * ref.objective;
          const double tol = 1e-6 * (1.0 + std::fabs(opt));
          if (with <= opt + tol && with >= without - tol) ++bound_ok;
          else if (why.empty()) why = "root bound with cuts " + std::to_string(with) + ", without " + std::to_string(without) + ", optimum " + std::to_string(opt);
        }
      }
      if (why.empty()) ++agree;
      else std::cerr << "  FAILING SEED " << seed << " (" << cut_family_name(f) << "): " << why << "\n";
      CHECK(why.empty());
    }
  }
  std::cout << "    cuts + presolve on/off vs brute force: " << agree << " of " << total << " agree; the cut loop ran on " << with_cuts
            << ", " << cuts_added << " cuts added, " << abandoned << " loops abandoned; root bound valid and monotone in " << bound_ok
            << " of " << bound_checked << "\n";
  CHECK(total >= 900);
  CHECK_EQ(agree, total);
  CHECK(with_cuts > 300);
}

// After at most 5 nodes the reported bound must still be a valid bound on the optimum.
TEST_CASE(cuts_early_termination_keeps_the_bound_valid) {
  int runs = 0, stopped = 0, valid = 0, inc_valid = 0, with_inc = 0;
  const int fams[] = {kCutKnapsackHard, kCutSetCoverHard, kCutIndependentSet, kCutSetPartition};
  for (int fi = 0; fi < 4; ++fi) {
    for (int k = 1; k <= 50; ++k) {
      const LpModel m = make_cut_instance(fams[fi], 123000ULL + static_cast<std::uint64_t>(fi) * 100ULL + static_cast<std::uint64_t>(k));
      const MipRefResult ref = solve_mip_brute_force(m);
      if (ref.status != Status::Optimal) continue;
      MipOptions o;
      o.params.verbosity = 0;
      o.node_limit = 1 + k % 5;
      o.cut_rounds = (k % 2 == 0) ? 1 : 20;  // a weak loop leaves a tree behind
      o.presolve = false;
      o.heuristics = (k % 3 == 0);
      const MipResult r = MipSolver(o).solve(m);
      ++runs;
      if (r.status == Status::NodeLimit) ++stopped;
      const double sg = m.sense == Sense::Maximize ? -1.0 : 1.0;
      const double tol = 1e-6 * (1.0 + std::fabs(ref.objective));
      bool ok = true;
      if (r.has_bound && sg * r.best_bound > sg * ref.objective + tol) ok = false;
      if (r.has_solution) {
        ++with_inc;
        if (sg * r.objective < sg * ref.objective - tol) ok = false;
        else ++inc_valid;
      }
      if (r.status == Status::Optimal && std::fabs(r.objective - ref.objective) > tol) ok = false;
      if (ok) ++valid;
      else std::cerr << "  FAILING early-termination run " << cut_family_name(fams[fi]) << " k=" << k << ": bound " << r.best_bound << " objective " << r.objective << " optimum " << ref.objective << "\n";
      CHECK(ok);
    }
  }
  std::cout << "    early termination with cuts: " << runs << " node-limited runs (limit 1..5), " << stopped
            << " stopped at the node limit; bound and incumbent valid in " << valid << " (incumbents " << inc_valid << " of " << with_inc << ")\n";
  CHECK(runs >= 150);
  CHECK_EQ(valid, runs);
}

TEST_CASE(cuts_are_deterministic) {
  int compared = 0, same = 0;
  for (int f = 0; f < kNumCutFamilies; ++f) {
    for (int k = 1; k <= 25; ++k) {
      const LpModel m = make_cut_instance(f, 150000ULL + static_cast<std::uint64_t>(f) * 100ULL + static_cast<std::uint64_t>(k));
      MipOptions o;
      o.params.verbosity = 0;
      o.params.seed = 7;
      const MipResult a = MipSolver(o).solve(m);
      const MipResult b = MipSolver(o).solve(m);
      ++compared;
      const bool eq = a.status == b.status && a.nodes_processed == b.nodes_processed && a.lp_iterations == b.lp_iterations &&
                      a.objective == b.objective && a.best_bound == b.best_bound && a.cuts.cuts_added == b.cuts.cuts_added &&
                      a.cuts.cuts_kept == b.cuts.cuts_kept && a.cuts.rounds == b.cuts.rounds &&
                      a.cuts.root_bound_with_cuts == b.cuts.root_bound_with_cuts && a.solution.x == b.solution.x;
      if (eq) ++same;
      else std::cerr << "  NONDETERMINISTIC: " << cut_family_name(f) << " k=" << k << "\n";
      CHECK(eq);
    }
  }
  std::cout << "    determinism with cuts on: " << compared << " instances solved twice, " << same
            << " identical (status, nodes, iterations, objective, bound, cuts, solution)\n";
}

// Informational (not asserted): share of the root gap closed per configuration, on GENERATED instances.
TEST_CASE(cuts_ablation_on_generated_instances_is_informational) {
  struct Cfg {
    const char* name;
    bool g, m, c, q, i;
  };
  const Cfg cfgs[] = {{"all separators", true, true, true, true, true}, {"no gomory", false, true, true, true, true},
                      {"no mir", true, false, true, true, true},       {"no cover", true, true, false, true, true},
                      {"no clique", true, true, true, false, true},    {"no implied bound", true, true, true, true, false},
                      {"no cuts at all", false, false, false, false, false}};
  std::cout << "    ablation (generated instances, 8 families x 30 seeds, presolve off; informational):\n";
  for (const Cfg& c : cfgs) {
    double closed = 0.0;
    int counted = 0;
    long long nodes = 0;
    for (int f = 0; f < kNumCutFamilies; ++f) {
      for (int k = 1; k <= 30; ++k) {
        const LpModel m = make_cut_instance(f, 170000ULL + static_cast<std::uint64_t>(f) * 100ULL + static_cast<std::uint64_t>(k));
        const MipRefResult ref = solve_mip_brute_force(m);
        if (ref.status != Status::Optimal) continue;
        MipOptions o;
        o.params.verbosity = 0;
        o.presolve = false;
        o.cuts = c.g || c.m || c.c || c.q || c.i;
        o.cut_gomory = c.g;
        o.cut_mir = c.m;
        o.cut_cover = c.c;
        o.cut_clique = c.q;
        o.cut_implied_bound = c.i;
        const MipResult r = MipSolver(o).solve(m);
        nodes += r.nodes_processed;
        if (!r.cuts.ran) continue;
        const double sg = m.sense == Sense::Maximize ? -1.0 : 1.0;
        const double lp = sg * r.cuts.root_bound_without_cuts, with = sg * r.cuts.root_bound_with_cuts, opt = sg * ref.objective;
        if (opt - lp > 1e-6) {
          closed += (with - lp) / (opt - lp);
          ++counted;
        }
      }
    }
    std::cout << "      " << c.name << ": mean root gap closed " << (counted ? 100.0 * closed / counted : 0.0) << "% over " << counted
              << " instances with a gap, " << nodes << " nodes in total\n";
  }
}
