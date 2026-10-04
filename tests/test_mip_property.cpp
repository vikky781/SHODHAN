// Property test of the MILP solver against the brute-force oracle over seeded small MIPs of ten families, with
// the early-termination bound check and a determinism test.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/mip/incumbent.hpp"
#include "shodhan/mip/mip_solver.hpp"
#include "support/mip_families.hpp"
#include "support/mip_oracle.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::mip;
using namespace shodhan::testing;

namespace {

constexpr int kSeedsPerFamily = 105;  // 10 families x 105 = 1050 instances

MipOptions tight_options() {
  MipOptions o;
  o.cuts = false;  // these tests exercise the tree search, see test_cuts.cpp for cuts
  o.mip_gap = 1e-9;
  o.mip_abs_gap = 1e-9;
  o.params.verbosity = 0;
  return o;
}

struct FamilyTally {
  int total = 0, agree = 0, skipped = 0;
  int optimal = 0, infeasible = 0, unbounded = 0;
  int early_checked = 0, early_valid = 0, early_stopped = 0, branched = 0;
  double worst_obj_error = 0.0;
  double worst_final_gap = 0.0;
};

}  // namespace

TEST_CASE(mip_property_against_the_brute_force_oracle) {
  FamilyTally tally[kNumMipFamilies];
  int failed = 0, early_total = 0, early_valid_total = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    for (int k = 1; k <= kSeedsPerFamily; ++k) {
      const std::uint64_t seed = 5000ULL + static_cast<std::uint64_t>(f) * 1000ULL + static_cast<std::uint64_t>(k);
      const LpModel m = make_mip_instance(f, seed);
      FamilyTally& t = tally[f];
      ++t.total;
      const MipResult r = MipSolver(tight_options()).solve(m);
      if (r.nodes_processed > 1) ++t.branched;
      std::string why;
      MipRefResult ref;
      if (mip_family_is_unbounded(f)) {
        if (r.status == Status::InfeasibleOrUnbounded || r.status == Status::Unbounded) ++t.unbounded;
        else why = std::string("status ") + to_string(r.status) + ", expected InfeasibleOrUnbounded";
        if (r.has_solution) why = "an unbounded model must not report a solution";
      } else {
        ref = solve_mip_brute_force(m);
        if (ref.too_large || ref.status == Status::NumericalError) {
          ++t.skipped;
          continue;
        }
        if (ref.status == Status::Infeasible) {
          ++t.infeasible;
          if (r.status != Status::Infeasible) why = std::string("status ") + to_string(r.status) + " [" + r.message + "], oracle Infeasible";
          else if (r.has_solution) why = "an infeasible model must not report a solution";
        } else if (ref.status == Status::Unbounded) {
          ++t.unbounded;
          if (r.status != Status::InfeasibleOrUnbounded && r.status != Status::Unbounded) why = std::string("status ") + to_string(r.status) + ", oracle Unbounded";
        } else {
          ++t.optimal;
          if (r.status != Status::Optimal) {
            why = std::string("status ") + to_string(r.status) + " [" + r.message + "], oracle Optimal";
          } else {
            const double err = std::fabs(r.objective - ref.objective) / (1.0 + std::fabs(ref.objective));
            t.worst_obj_error = std::max(t.worst_obj_error, err);
            t.worst_final_gap = std::max(t.worst_final_gap, r.rel_gap);
            if (!(err <= 1e-6)) why = "objective " + std::to_string(r.objective) + " vs oracle " + std::to_string(ref.objective);
            else if (!r.has_solution) why = "Optimal without a solution";
            else if (!check_point(m, r.solution.x).ok(1e-6, 0.0)) why = "the incumbent is not feasible on the original model";
            else if (!(r.rel_gap <= 1e-6)) why = "Optimal but the reported gap is " + std::to_string(r.rel_gap);
          }
        }
        // Early termination: node limit 5 on a share of the instances with an optimum. The reported bound must be
        // a valid bound on the true optimum and the incumbent (if any) must not beat it.
        if (why.empty() && ref.status == Status::Optimal && (k % 5 == 0)) {
          MipOptions o = tight_options();
          o.node_limit = 5;
          const MipResult e = MipSolver(o).solve(m);
          ++t.early_checked;
          if (e.status == Status::NodeLimit && e.nodes_open > 0) ++t.early_stopped;
          ++early_total;
          const double tol = 1e-6 * (1.0 + std::fabs(ref.objective));
          bool ok = true;
          if (e.status != Status::NodeLimit && e.status != Status::Optimal) ok = false;
          if (e.has_bound) {
            if (m.sense == Sense::Maximize ? e.best_bound < ref.objective - tol : e.best_bound > ref.objective + tol) ok = false;
          }
          if (e.has_solution) {
            if (m.sense == Sense::Maximize ? e.objective > ref.objective + tol : e.objective < ref.objective - tol) ok = false;
            if (!check_point(m, e.solution.x).ok(1e-6, 0.0)) ok = false;
          }
          if (ok) {
            ++t.early_valid;
            ++early_valid_total;
          } else {
            why = std::string("early termination (node limit 5) reported an invalid bound or incumbent: status ") + to_string(e.status) + ", bound " +
                  std::to_string(e.best_bound) + ", objective " + std::to_string(e.objective) + ", true optimum " + std::to_string(ref.objective);
          }
        }
      }
      if (why.empty()) {
        ++t.agree;
      } else {
        ++failed;
        std::cerr << "FAILING SEED " << seed << " (" << mip_family_name(f) << "): " << why << "\n";
        CHECK(false);
      }
    }
  }
  int total = 0, agree = 0, skipped = 0;
  std::cout << "    MILP vs brute force (reliability branching, best bound, heuristics on):\n";
  for (int f = 0; f < kNumMipFamilies; ++f) {
    const FamilyTally& t = tally[f];
    total += t.total;
    agree += t.agree;
    skipped += t.skipped;
    std::cout << "      " << std::left << std::setw(20) << mip_family_name(f) << std::right << std::setw(4) << t.total << " seeds: " << std::setw(4) << t.agree << " agree ("
              << t.optimal << " optimal, " << t.infeasible << " infeasible, " << t.unbounded << " unbounded, " << t.skipped << " skipped, " << t.branched << " branched); early-stop bounds valid " << t.early_valid
              << " of " << t.early_checked << " (" << t.early_stopped << " really stopped with open nodes); worst objective error " << std::scientific << std::setprecision(1) << t.worst_obj_error << ", worst final gap " << t.worst_final_gap
              << std::defaultfloat << "\n";
  }
  std::cout << "    total: " << agree << " of " << total << " seeds agree; early termination bound valid in " << early_valid_total << " of " << early_total << " node-limited runs\n";
  CHECK_EQ(failed, 0);
  CHECK(total >= 1000);
  CHECK(early_total >= 100);
  CHECK(skipped <= total / 20);
}

TEST_CASE(mip_search_is_deterministic_for_a_seed) {
  int checked = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    for (std::uint64_t seed = 9001; seed <= 9020; ++seed) {
      const LpModel m = make_mip_instance(f, seed);
      MipOptions o;
      o.cuts = false;  // these tests exercise the tree search, see test_cuts.cpp for cuts
      o.params.verbosity = 0;
      o.params.seed = 17;
      const MipResult a = MipSolver(o).solve(m);
      const MipResult b = MipSolver(o).solve(m);
      ++checked;
      const bool same = a.status == b.status && a.nodes_processed == b.nodes_processed && a.lp_iterations == b.lp_iterations && a.objective == b.objective &&
                        a.solution.x == b.solution.x && a.best_bound == b.best_bound && a.incumbent_source == b.incumbent_source;
      if (!same) {
        std::cerr << "NONDETERMINISTIC " << mip_family_name(f) << " seed " << seed << ": nodes " << a.nodes_processed << " vs " << b.nodes_processed << ", objective " << a.objective
                  << " vs " << b.objective << "\n";
        CHECK(false);
      }
    }
  }
  std::cout << "    determinism: " << checked << " instances solved twice with the same seed, identical node count, iterations, objective, bound and incumbent\n";
  CHECK(checked >= 200);
}

TEST_CASE(mip_optimum_does_not_depend_on_the_seed) {
  int compared = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    if (mip_family_is_unbounded(f)) continue;
    for (std::uint64_t seed = 9101; seed <= 9112; ++seed) {
      const LpModel m = make_mip_instance(f, seed);
      MipOptions a = tight_options(), b = tight_options();
      a.params.seed = 1;
      b.params.seed = 99;
      const MipResult ra = MipSolver(a).solve(m), rb = MipSolver(b).solve(m);
      ++compared;
      CHECK(ra.status == rb.status);
      if (ra.status == Status::Optimal && rb.status == Status::Optimal) CHECK(std::fabs(ra.objective - rb.objective) <= 1e-6 * (1 + std::fabs(ra.objective)));
    }
  }
  CHECK(compared >= 100);
}

// The same comparison with the search made harder: no heuristics (the tree must find everything), simple
// branching and depth-first selection, so that many more nodes are processed.
TEST_CASE(mip_property_with_heuristics_off_and_simple_search_rules) {
  int compared = 0, failed = 0, branched = 0;
  long long nodes = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    if (mip_family_is_unbounded(f)) continue;
    for (int k = 1; k <= 40; ++k) {
      const std::uint64_t seed = 7000ULL + static_cast<std::uint64_t>(f) * 1000ULL + static_cast<std::uint64_t>(k);
      const LpModel m = make_mip_instance(f, seed);
      const MipRefResult ref = solve_mip_brute_force(m);
      if (ref.too_large || ref.status == Status::NumericalError) continue;
      for (const int config : {0, 1}) {
        MipOptions o = tight_options();
        o.heuristics = false;
        o.presolve = config == 0;
        o.branching = config == 0 ? BranchingKind::MostFractional : BranchingKind::FirstIndex;
        o.node_select = config == 0 ? NodeSelectKind::DepthFirst : NodeSelectKind::BestEstimate;
        const MipResult r = MipSolver(o).solve(m);
        ++compared;
        nodes += r.nodes_processed;
        if (r.nodes_processed > 1) ++branched;
        bool ok;
        if (ref.status == Status::Infeasible) ok = r.status == Status::Infeasible;
        else ok = r.status == Status::Optimal && std::fabs(r.objective - ref.objective) <= 1e-6 * (1 + std::fabs(ref.objective)) && check_point(m, r.solution.x).ok(1e-6, 0.0);
        if (!ok) {
          ++failed;
          std::cerr << "FAILING SEED " << seed << " (" << mip_family_name(f) << ", config " << config << "): " << to_string(r.status) << " " << r.objective << " vs oracle " << ref.objective << "\n";
          CHECK(false);
        }
      }
    }
  }
  std::cout << "    hard configuration (no heuristics, no presolve for half, simple rules): " << compared << " solves, " << failed << " failed; " << branched << " branched, "
            << nodes << " nodes in total\n";
  CHECK_EQ(failed, 0);
  CHECK(compared >= 500);
}

namespace {

// Correlated multi-dimensional knapsack with 14-16 items: brute force still works (2^16 points) but the tree needs
// many nodes, so a small node limit really stops the search with open nodes.
LpModel hard_knapsack(std::uint64_t seed, bool maximize) {
  Rng r(seed * 31ULL + 7ULL);
  const int n = r.range(14, 16), rows = r.range(2, 3);
  LpModel m;
  m.n_rows = rows;
  m.n_cols = n;
  std::vector<Triplet> t;
  std::vector<double> cap(static_cast<std::size_t>(rows), 0.0);
  std::vector<double> value(static_cast<std::size_t>(n), 0.0);
  for (int j = 0; j < n; ++j) {
    double wsum = 0;
    for (int i = 0; i < rows; ++i) {
      const int w = r.range(10, 60);
      t.push_back({i, j, static_cast<double>(w)});
      cap[static_cast<std::size_t>(i)] += w;
      wsum += w;
    }
    value[static_cast<std::size_t>(j)] = std::floor(wsum / rows) + r.range(-3, 3);  // value correlated with the weights
  }
  std::string err;
  SparseMatrix::from_triplets(rows, n, t, &m.A, &err);
  for (int j = 0; j < n; ++j) {
    m.col_cost.push_back(maximize ? value[static_cast<std::size_t>(j)] : -value[static_cast<std::size_t>(j)]);
    m.col_lower.push_back(0.0);
    m.col_upper.push_back(1.0);
    m.col_type.push_back(ColType::Binary);
  }
  for (int i = 0; i < rows; ++i) {
    m.row_lower.push_back(-kInf);
    m.row_upper.push_back(std::floor(0.5 * cap[static_cast<std::size_t>(i)]));
  }
  m.sense = maximize ? Sense::Maximize : Sense::Minimize;
  return m;
}

}  // namespace

TEST_CASE(mip_early_termination_bound_is_valid_when_the_search_really_stops) {
  int runs = 0, stopped = 0, bound_valid = 0, with_incumbent = 0, incumbent_valid = 0, failed = 0;
  for (int k = 1; k <= 200; ++k) {
    const LpModel m = hard_knapsack(static_cast<std::uint64_t>(k), k % 2 == 0);
    const MipRefResult ref = solve_mip_brute_force(m);
    if (ref.status != Status::Optimal) continue;
    MipOptions o = tight_options();
    o.node_limit = 1 + (k % 5);  // 1..5
    if (k % 3 == 0) o.heuristics = false;
    const MipResult e = MipSolver(o).solve(m);
    ++runs;
    const double tol = 1e-6 * (1.0 + std::fabs(ref.objective));
    const bool really_stopped = e.status == Status::NodeLimit && e.nodes_open > 0;
    if (really_stopped) ++stopped;
    bool ok = e.status == Status::NodeLimit || e.status == Status::Optimal;
    if (e.has_bound) {
      const bool valid = m.sense == Sense::Maximize ? e.best_bound >= ref.objective - tol : e.best_bound <= ref.objective + tol;
      if (valid) ++bound_valid;
      else ok = false;
    } else if (really_stopped) {
      ok = false;  // a stopped search after at least one LP must have a bound
    }
    if (e.has_solution) {
      ++with_incumbent;
      const bool valid = (m.sense == Sense::Maximize ? e.objective <= ref.objective + tol : e.objective >= ref.objective - tol) && check_point(m, e.solution.x).ok(1e-6, 0.0);
      if (valid) ++incumbent_valid;
      else ok = false;
    }
    if (!ok) {
      ++failed;
      std::cerr << "FAILING hard knapsack " << k << ": status " << to_string(e.status) << ", bound " << e.best_bound << ", objective " << e.objective << ", optimum " << ref.objective << "\n";
      CHECK(false);
    }
  }
  std::cout << "    early termination on hard knapsacks: " << runs << " node-limited runs (limit 1..5), " << stopped << " really stopped with open nodes; bound valid in " << bound_valid
            << ", incumbent valid in " << incumbent_valid << " of " << with_incumbent << " with an incumbent, " << failed << " failed\n";
  CHECK_EQ(failed, 0);
  CHECK(runs >= 150);
  CHECK(stopped >= 100);  // the point of this test: most runs really are stopped early
}
