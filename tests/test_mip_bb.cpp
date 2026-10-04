// Branch and bound: hand-made MIPs, statuses, limits, objective granularity, agreement with the oracles.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/mip/granularity.hpp"
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

MipOptions exact_options() {
  MipOptions o;
  o.cuts = false;  // these tests exercise the tree search, see test_cuts.cpp for cuts
  o.mip_gap = 1e-9;
  o.mip_abs_gap = 1e-9;
  o.params.verbosity = 0;
  o.branching = BranchingKind::MostFractional;
  return o;
}

// max 5a + 4b + 3c + y   s.t.  2a + 3b + c + y <= 5,  a + b + c <= 2,  a,b,c binary, 0 <= y <= 1.5.
LpModel small_mip() {
  LpModel m;
  m.sense = Sense::Maximize;
  m.n_rows = 2;
  m.n_cols = 4;
  std::string err;
  SparseMatrix::from_triplets(2, 4, {{0, 0, 2.0}, {0, 1, 3.0}, {0, 2, 1.0}, {0, 3, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}}, &m.A, &err);
  m.col_cost = {5.0, 4.0, 3.0, 1.0};
  m.col_lower = {0.0, 0.0, 0.0, 0.0};
  m.col_upper = {1.0, 1.0, 1.0, 1.5};
  m.col_type = {ColType::Binary, ColType::Binary, ColType::Binary, ColType::Continuous};
  m.row_lower = {-kInf, -kInf};
  m.row_upper = {5.0, 2.0};
  return m;
}

// Classic 0-1 knapsack, max 60a + 100b + 120c  s.t.  10a + 20b + 30c <= 50: optimum 220 (b, c).
LpModel knapsack3() {
  LpModel m;
  m.sense = Sense::Maximize;
  m.n_rows = 1;
  m.n_cols = 3;
  std::string err;
  SparseMatrix::from_triplets(1, 3, {{0, 0, 10.0}, {0, 1, 20.0}, {0, 2, 30.0}}, &m.A, &err);
  m.col_cost = {60.0, 100.0, 120.0};
  m.col_lower = {0, 0, 0};
  m.col_upper = {1, 1, 1};
  m.col_type.assign(3, ColType::Binary);
  m.row_lower = {-kInf};
  m.row_upper = {50.0};
  return m;
}

}  // namespace

TEST_CASE(objective_granularity_is_detected_and_conservative) {
  LpModel m = knapsack3();
  CHECK_EQ(objective_granularity(m), 20.0);  // gcd(60, 100, 120)
  m.col_cost = {0.5, 1.5, 2.5};
  CHECK(std::fabs(objective_granularity(m) - 0.5) < 1e-12);
  m.col_cost = {0.1, 0.3, 0.7};
  CHECK(std::fabs(objective_granularity(m) - 0.1) < 1e-12);
  m.col_cost = {2.5, 5.0, 7.5};
  CHECK(std::fabs(objective_granularity(m) - 2.5) < 1e-12);
  m.col_cost = {1.0 / 3.0, 1.0, 2.0};
  CHECK_EQ(objective_granularity(m), 0.0);  // not on a decimal grid: no strengthening
  m.col_cost = {1.0, 2.0, 3.0};
  m.col_type[1] = ColType::Continuous;  // a continuous column with a cost: no grid
  CHECK_EQ(objective_granularity(m), 0.0);
  m.col_cost = {1.0, 0.0, 3.0};         // zero cost on the continuous column: fine
  CHECK_EQ(objective_granularity(m), 1.0);
  m.col_cost = {0.0, 0.0, 0.0};
  CHECK_EQ(objective_granularity(m), 0.0);
}

TEST_CASE(branch_and_bound_solves_the_classic_knapsack) {
  const MipResult r = MipSolver(exact_options()).solve(knapsack3());
  CHECK(r.status == Status::Optimal);
  CHECK(r.has_solution);
  CHECK_EQ(r.objective, 220.0);
  CHECK_EQ(r.solution.x[0], 0.0);
  CHECK_EQ(r.solution.x[1], 1.0);
  CHECK_EQ(r.solution.x[2], 1.0);
  CHECK(r.has_bound);
  CHECK(r.best_bound >= r.objective - 1e-9);  // an upper bound for a maximization
  CHECK(r.abs_gap <= 1e-6);
  CHECK(r.nodes_processed >= 1);
}

TEST_CASE(branch_and_bound_handles_continuous_columns_and_a_max_model) {
  const MipResult r = MipSolver(exact_options()).solve(small_mip());
  CHECK(r.status == Status::Optimal);
  const MipRefResult ref = solve_mip_brute_force(small_mip());
  CHECK(ref.status == Status::Optimal);
  CHECK(std::fabs(r.objective - ref.objective) < 1e-7);
  const PointCheck pc = check_point(small_mip(), r.solution.x);
  CHECK(pc.ok(1e-6, 0.0));
}

TEST_CASE(branch_and_bound_reports_infeasible_by_parity_without_a_certificate) {
  // 2a + 2b = 3 with integers: the LP relaxation is feasible, there is no integer point.
  LpModel m;
  m.n_rows = 1;
  m.n_cols = 2;
  std::string err;
  SparseMatrix::from_triplets(1, 2, {{0, 0, 2.0}, {0, 1, 2.0}}, &m.A, &err);
  m.col_cost = {1.0, 1.0};
  m.col_lower = {0, 0};
  m.col_upper = {5, 5};
  m.col_type.assign(2, ColType::Integer);
  m.row_lower = {3.0};
  m.row_upper = {3.0};
  MipOptions o = exact_options();
  o.presolve = false;
  const MipResult r = MipSolver(o).solve(m);
  CHECK(r.status == Status::Infeasible);
  CHECK(!r.has_solution);
  CHECK(!r.lp_infeasible_certified);  // proved by branching: no certificate
}

TEST_CASE(branch_and_bound_certifies_an_infeasible_lp_relaxation_only_when_it_is_one) {
  LpModel m = knapsack3();
  // 10a + 20b + 30c <= 50 and the same expression >= 60 in a second row: the LP relaxation is infeasible.
  m.n_rows = 2;
  std::string err;
  SparseMatrix::from_triplets(2, 3, {{0, 0, 10.0}, {0, 1, 20.0}, {0, 2, 30.0}, {1, 0, 10.0}, {1, 1, 20.0}, {1, 2, 30.0}}, &m.A, &err);
  m.row_lower = {-kInf, 60.0};
  m.row_upper = {50.0, kInf};
  MipOptions o = exact_options();
  o.presolve = false;
  const MipResult r = MipSolver(o).solve(m);
  CHECK(r.status == Status::Infeasible);
  CHECK(r.lp_infeasible_certified);
  CHECK(!r.lp_farkas.empty());
}

TEST_CASE(branch_and_bound_reports_infeasible_or_unbounded_for_an_unbounded_relaxation) {
  const LpModel m = make_mip_instance(kMipUnbounded, 4);
  const MipResult r = MipSolver(exact_options()).solve(m);
  CHECK(r.status == Status::InfeasibleOrUnbounded);
  CHECK(!r.has_solution);
}

TEST_CASE(branch_and_bound_stops_at_the_node_limit_with_a_valid_bound) {
  int limited = 0, checked = 0;
  for (std::uint64_t seed = 1; seed <= 60; ++seed) {
    const LpModel m = make_mip_instance(kMipKnapsack, seed);
    const MipRefResult ref = solve_mip_brute_force(m);
    if (ref.status != Status::Optimal) continue;
    MipOptions o = exact_options();
    o.node_limit = 3;
    const MipResult r = MipSolver(o).solve(m);
    ++checked;
    if (r.status == Status::NodeLimit) ++limited;
    if (r.has_bound) {
      // The bound is valid: a lower bound for min, an upper bound for max.
      if (m.sense == Sense::Maximize) CHECK(r.best_bound >= ref.objective - 1e-6 * (1 + std::fabs(ref.objective)));
      else CHECK(r.best_bound <= ref.objective + 1e-6 * (1 + std::fabs(ref.objective)));
    }
    if (r.has_solution) {
      if (m.sense == Sense::Maximize) CHECK(r.objective <= ref.objective + 1e-6 * (1 + std::fabs(ref.objective)));
      else CHECK(r.objective >= ref.objective - 1e-6 * (1 + std::fabs(ref.objective)));
    }
  }
  CHECK(checked > 30);
  std::cout << "    node limit 3 on knapsacks: " << limited << " of " << checked << " stopped at the limit\n";
}

TEST_CASE(branch_and_bound_agrees_with_the_brute_force_oracle) {
  struct Tally { int total = 0, agree = 0, skipped = 0; double worst = 0.0; };
  Tally tally[kNumMipFamilies];
  int failed = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    for (std::uint64_t seed = 1; seed <= 30; ++seed) {
      const LpModel m = make_mip_instance(f, seed);
      Tally& t = tally[f];
      ++t.total;
      const MipResult r = MipSolver(exact_options()).solve(m);
      if (mip_family_is_unbounded(f)) {
        if (r.status == Status::InfeasibleOrUnbounded || r.status == Status::Unbounded) ++t.agree;
        else {
          ++failed;
          std::cerr << "FAILING " << mip_family_name(f) << " seed " << seed << ": status " << to_string(r.status) << "\n";
          CHECK(false);
        }
        continue;
      }
      const MipRefResult ref = solve_mip_brute_force(m);
      if (ref.too_large || ref.status == Status::NumericalError) {
        ++t.skipped;
        continue;
      }
      std::string why;
      if (ref.status == Status::Infeasible) {
        if (r.status != Status::Infeasible) why = std::string("status ") + to_string(r.status) + ", oracle Infeasible";
      } else if (ref.status == Status::Unbounded) {
        if (r.status != Status::InfeasibleOrUnbounded && r.status != Status::Unbounded) why = std::string("status ") + to_string(r.status) + ", oracle Unbounded";
      } else {
        if (r.status != Status::Optimal) why = std::string("status ") + to_string(r.status) + " [" + r.message + "], oracle Optimal";
        else {
          const double rel = std::fabs(r.objective - ref.objective) / (1.0 + std::fabs(ref.objective));
          t.worst = std::max(t.worst, rel);
          if (!(rel <= 1e-6)) why = "objective " + std::to_string(r.objective) + " vs oracle " + std::to_string(ref.objective);
          else if (!check_point(m, r.solution.x).ok(1e-6, 0.0)) why = "incumbent is not feasible on the original model";
        }
      }
      if (why.empty()) ++t.agree;
      else {
        ++failed;
        std::cerr << "FAILING " << mip_family_name(f) << " seed " << seed << ": " << why << "\n";
        CHECK(false);
      }
    }
  }
  std::cout << "    branch and bound (most fractional, best bound) vs brute force:\n";
  int total = 0, agree = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    const Tally& t = tally[f];
    total += t.total - t.skipped;
    agree += t.agree;
    std::cout << "      " << std::left << std::setw(20) << mip_family_name(f) << std::right << std::setw(3) << t.agree << " of " << std::setw(3) << t.total - t.skipped
              << " agree (" << t.skipped << " skipped), worst objective error " << std::scientific << std::setprecision(1) << t.worst << std::defaultfloat << "\n";
  }
  CHECK_EQ(failed, 0);
  CHECK(total > 200);
  CHECK_EQ(agree, total);
}

TEST_CASE(branch_and_bound_agrees_across_node_selectors_and_simple_branching_rules) {
  int compared = 0, failed = 0;
  for (std::uint64_t seed = 1; seed <= 40; ++seed) {
    const int f = static_cast<int>(seed % 8);  // the bounded families
    const LpModel m = make_mip_instance(f, seed);
    const MipRefResult ref = solve_mip_brute_force(m);
    if (ref.status != Status::Optimal) continue;
    for (const NodeSelectKind sel : {NodeSelectKind::BestBound, NodeSelectKind::DepthFirst, NodeSelectKind::BestEstimate}) {
      for (const BranchingKind br : {BranchingKind::MostFractional, BranchingKind::FirstIndex}) {
        MipOptions o = exact_options();
        o.node_select = sel;
        o.branching = br;
        const MipResult r = MipSolver(o).solve(m);
        ++compared;
        if (r.status != Status::Optimal || std::fabs(r.objective - ref.objective) > 1e-6 * (1 + std::fabs(ref.objective))) {
          ++failed;
          std::cerr << "FAILING seed " << seed << " " << to_string(sel) << "/" << to_string(br) << ": " << to_string(r.status) << " " << r.objective << " vs " << ref.objective << "\n";
          CHECK(false);
        }
      }
    }
  }
  std::cout << "    selector x simple branching rule: " << compared << " runs, " << failed << " failed\n";
  CHECK_EQ(failed, 0);
  CHECK(compared > 100);
}

TEST_CASE(branch_and_bound_is_deterministic) {
  for (const std::uint64_t seed : {3ULL, 11ULL, 29ULL}) {
    const LpModel m = make_mip_instance(kMipKnapsack, seed);
    const MipResult a = MipSolver(exact_options()).solve(m);
    const MipResult b = MipSolver(exact_options()).solve(m);
    CHECK_EQ(a.nodes_processed, b.nodes_processed);
    CHECK_EQ(a.objective, b.objective);
    CHECK(a.solution.x == b.solution.x);
    CHECK_EQ(a.lp_iterations, b.lp_iterations);
  }
}

TEST_CASE(dense_branch_and_bound_oracle_agrees_with_brute_force_enumeration) {
  int compared = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    if (mip_family_is_unbounded(f)) continue;
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
      const LpModel m = make_mip_instance(f, seed);
      const MipRefResult a = solve_mip_brute_force(m);
      const MipRefResult b = solve_mip_dense_bb(m);
      if (a.too_large || a.status == Status::NumericalError || b.status == Status::NumericalError) continue;
      ++compared;
      CHECK(a.status == b.status);
      if (a.status == Status::Optimal && b.status == Status::Optimal) CHECK(std::fabs(a.objective - b.objective) <= 1e-6 * (1 + std::fabs(a.objective)));
    }
  }
  std::cout << "    dense branch-and-bound oracle vs brute force: " << compared << " instances compared\n";
  CHECK(compared > 80);
}

TEST_CASE(search_releases_finished_nodes) {
  const LpModel m = [] {
    // 30-item knapsack with 3 rows (the same family as the generator in bench/gen_mip.py): needs many nodes.
    Rng r(11);
    LpModel k;
    k.n_rows = 3;
    k.n_cols = 30;
    std::vector<Triplet> t;
    std::vector<double> cap(3, 0.0);
    for (int j = 0; j < 30; ++j) {
      double wsum = 0;
      for (int i = 0; i < 3; ++i) {
        const int w = r.range(10, 99);
        t.push_back({i, j, static_cast<double>(w)});
        cap[static_cast<std::size_t>(i)] += w;
        wsum += w;
      }
      k.col_cost.push_back(-std::floor(wsum / 3) - r.range(-10, 10));
      k.col_lower.push_back(0.0);
      k.col_upper.push_back(1.0);
      k.col_type.push_back(ColType::Binary);
    }
    std::string err;
    SparseMatrix::from_triplets(3, 30, t, &k.A, &err);
    for (int i = 0; i < 3; ++i) {
      k.row_lower.push_back(-kInf);
      k.row_upper.push_back(std::floor(0.4 * cap[static_cast<std::size_t>(i)]));
    }
    return k;
  }();
  MipOptions o = exact_options();
  o.heuristics = false;
  o.node_select = NodeSelectKind::DepthFirst;  // best-bound search keeps a large open set by nature; depth first does not
  const MipResult r = MipSolver(o).solve(m);
  CHECK(r.status == Status::Optimal);
  std::cout << "    node memory (depth first): " << r.nodes_created << " nodes created, at most " << r.peak_live_nodes << " alive at the same time\n";
  CHECK(r.nodes_created > 50);
  CHECK(r.peak_live_nodes * 2 < r.nodes_created);
}
