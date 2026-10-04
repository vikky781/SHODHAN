// Pseudocosts, reliability branching and strong branching; every branching rule x node selector combination.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "shodhan/mip/mip_solver.hpp"
#include "shodhan/mip/plugins.hpp"
#include "support/mip_families.hpp"
#include "support/mip_oracle.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::mip;
using namespace shodhan::testing;

namespace {

// A SearchState around an engine that has just solved the LP relaxation of `model` (no presolve, no scaling).
struct Harness {
  LpModel model;
  Scaling scaling;
  std::unique_ptr<SimplexEngine> engine;
  MipOptions options;
  Pseudocosts pseudo;
  Random rng{1};
  std::vector<Index> int_cols;
  double incumbent = kInf;
  long long strong_iterations = 0, strong_solves = 0;
  SearchState state;

  explicit Harness(const LpModel& m) : model(m), pseudo(m.n_cols) {
    scaling.row_scale.assign(to_size(m.n_rows), 1.0);
    scaling.col_scale.assign(to_size(m.n_cols), 1.0);
    engine = std::make_unique<SimplexEngine>(model);
    ok = engine->solve() == EngineStatus::Optimal;
    for (Index j = 0; j < m.n_cols; ++j) {
      if (m.is_integer(j)) int_cols.push_back(j);
    }
    state.model = &model;
    state.scaling = &scaling;
    state.engine = engine.get();
    state.options = &options;
    state.pseudo = &pseudo;
    state.rng = &rng;
    state.int_cols = &int_cols;
    lo = model.col_lower;
    hi = model.col_upper;
    state.lo = &lo;
    state.hi = &hi;
    state.lp.objective = engine->objective();
    state.lp.x.assign(engine->primal_all().begin(), engine->primal_all().begin() + model.n_cols);
    for (const Index j : int_cols) {
      const double v = state.lp.x[to_size(j)];
      if (std::fabs(v - std::round(v)) > 1e-5) state.lp.fractional.push_back(j);
    }
    state.submit = [](const std::vector<double>&, const std::string&) { return false; };
    state.cutoff = [this]() { return incumbent; };
    state.time_up = []() { return false; };
    state.lp_iterations = []() { return 0LL; };
    state.add_iterations = [this](long long n, bool) { strong_iterations += n; };
    state.count_strong_solve = [this]() { ++strong_solves; };
  }
  bool ok = false;
  std::vector<double> lo, hi;
};

LpModel one_integer(double row_lo, double row_hi, double cost) {
  LpModel m;
  m.n_rows = 1;
  m.n_cols = 1;
  std::string err;
  SparseMatrix::from_triplets(1, 1, {{0, 0, 1.0}}, &m.A, &err);
  m.col_cost = {cost};
  m.col_lower = {0.0};
  m.col_upper = {5.0};
  m.col_type = {ColType::Integer};
  m.row_lower = {row_lo};
  m.row_upper = {row_hi};
  return m;
}

}  // namespace

TEST_CASE(pseudocosts_average_update_and_reliability) {
  Pseudocosts pc(3);
  CHECK_EQ(pc.value(0, -1), 1.0);  // no data anywhere: default 1
  pc.update(0, -1, 2.0, 0.5);      // 4 per unit
  pc.update(0, -1, 1.0, 0.5);      // 2 per unit
  CHECK_EQ(pc.count(0, -1), 2);
  CHECK(std::fabs(pc.value(0, -1) - 3.0) < 1e-12);
  CHECK_EQ(pc.count(0, +1), 0);
  CHECK(std::fabs(pc.value(1, -1) - 3.0) < 1e-12);  // a column without data uses the global average of that direction
  CHECK_EQ(pc.value(1, +1), 1.0);
  CHECK(!pc.reliable(0, 1));  // down observed, up not
  pc.update(0, +1, 1.0, 1.0);
  CHECK(pc.reliable(0, 1));
  CHECK(!pc.reliable(0, 2));
  pc.update(0, -1, -1.0, 0.5);  // negative gains and zero deltas are ignored
  pc.update(0, -1, 1.0, 0.0);
  CHECK_EQ(pc.count(0, -1), 2);
  // product score: down = 3 * f, up = 1 * (1 - f)
  CHECK(std::fabs(pc.score(0, 0.5) - (3.0 * 0.5) * (1.0 * 0.5)) < 1e-12);
}

TEST_CASE(reliability_branching_one_infeasible_child_tightens_the_bound) {
  // min -x with 0 <= x <= 0.6 as a row: the LP solution x = 0.6 is fractional; the up child (x >= 1) is infeasible.
  Harness h(one_integer(-kInf, 0.6, -1.0));
  CHECK(h.ok);
  CHECK_EQ(h.state.lp.fractional.size(), 1u);
  auto rule = plugin_registry().branching.make("reliability");
  CHECK(rule != nullptr);
  const BranchDecision d = rule->select(h.state);
  CHECK(d.kind == BranchDecision::Kind::Tighten);
  CHECK_EQ(d.tightenings.size(), 1u);
  CHECK_EQ(d.tightenings[0].col, 0);
  CHECK_EQ(d.tightenings[0].hi, 0.0);  // x <= floor(0.6)
  CHECK_EQ(h.strong_solves, 2);
}

TEST_CASE(reliability_branching_two_infeasible_children_prune_the_node) {
  // 0.3 <= x <= 0.7: both x <= 0 and x >= 1 are infeasible.
  Harness h(one_integer(0.3, 0.7, 1.0));
  CHECK(h.ok);
  auto rule = plugin_registry().branching.make("reliability");
  const BranchDecision d = rule->select(h.state);
  CHECK(d.kind == BranchDecision::Kind::Prune);
}

TEST_CASE(reliability_branching_a_cutoff_child_counts_as_cut) {
  // min x with 0.4 <= x: LP x = 0.4. Down child infeasible; up child costs 1. An incumbent of 1 cuts the up child too.
  Harness h(one_integer(0.4, kInf, 1.0));
  CHECK(h.ok);
  h.incumbent = 1.0;
  auto rule = plugin_registry().branching.make("reliability");
  const BranchDecision d = rule->select(h.state);
  CHECK(d.kind == BranchDecision::Kind::Prune);
}

TEST_CASE(strong_branching_leaves_the_engine_state_unchanged_bit_for_bit) {
  int checked = 0;
  for (std::uint64_t seed = 1; seed <= 40; ++seed) {
    const LpModel m = make_mip_instance(static_cast<int>(seed % 8), seed);
    Harness h(m);
    if (!h.ok || h.state.lp.fractional.empty()) continue;
    const BasisSnapshot basis_before = h.engine->get_basis_snapshot();
    const std::vector<double> x_before = h.engine->primal_all();
    const std::vector<double> d_before = h.engine->dual_all();
    const std::vector<double> y_before = h.engine->row_duals();
    const std::vector<Index> order_before = h.engine->basis();
    const long long iters_before = h.engine->stats().iterations;
    std::vector<double> lo_before, hi_before;
    for (Index j = 0; j < m.n_cols; ++j) {
      lo_before.push_back(h.engine->col_lower(j));
      hi_before.push_back(h.engine->col_upper(j));
    }
    auto rule = plugin_registry().branching.make("reliability");
    (void)rule->select(h.state);
    CHECK(h.engine->get_basis_snapshot() == basis_before);
    CHECK(h.engine->primal_all() == x_before);
    CHECK(h.engine->dual_all() == d_before);
    CHECK(h.engine->row_duals() == y_before);
    CHECK(h.engine->basis() == order_before);
    CHECK_EQ(h.engine->stats().iterations, iters_before);
    for (Index j = 0; j < m.n_cols; ++j) {
      CHECK_EQ(h.engine->col_lower(j), lo_before[to_size(j)]);
      CHECK_EQ(h.engine->col_upper(j), hi_before[to_size(j)]);
    }
    ++checked;
  }
  std::cout << "    strong branching state restoration: " << checked << " instances, engine state identical before and after\n";
  CHECK(checked > 15);
}

TEST_CASE(every_branching_rule_and_node_selector_reach_the_same_optimum_on_300_mips) {
  struct Avg { long long nodes = 0; int runs = 0; };
  std::map<std::string, Avg> per_rule, per_selector;
  int compared = 0, failed = 0, instances = 0;
  for (int f = 0; f < kNumMipFamilies; ++f) {
    if (mip_family_is_unbounded(f)) continue;
    for (std::uint64_t seed = 101; seed <= 140 && instances < 300; ++seed) {
      const LpModel m = make_mip_instance(f, seed);
      const MipRefResult ref = solve_mip_brute_force(m);
      if (ref.too_large || ref.status == Status::NumericalError) continue;
      ++instances;
      for (const BranchingKind br : {BranchingKind::Reliability, BranchingKind::Pseudocost, BranchingKind::MostFractional, BranchingKind::FirstIndex}) {
        for (const NodeSelectKind sel : {NodeSelectKind::BestBound, NodeSelectKind::DepthFirst, NodeSelectKind::BestEstimate}) {
          MipOptions o;
          o.mip_gap = 1e-9;
          o.mip_abs_gap = 1e-9;
          o.params.verbosity = 0;
          o.branching = br;
          o.node_select = sel;
          const MipResult r = MipSolver(o).solve(m);
          ++compared;
          bool ok;
          if (ref.status == Status::Infeasible) ok = r.status == Status::Infeasible;
          else ok = r.status == Status::Optimal && std::fabs(r.objective - ref.objective) <= 1e-6 * (1 + std::fabs(ref.objective));
          if (!ok) {
            ++failed;
            std::cerr << "FAILING " << mip_family_name(f) << " seed " << seed << " " << to_string(br) << "/" << to_string(sel) << ": " << to_string(r.status) << " " << r.objective
                      << " vs oracle " << ref.objective << "\n";
            CHECK(false);
          } else {
            per_rule[to_string(br)].nodes += r.nodes_processed;
            ++per_rule[to_string(br)].runs;
            per_selector[to_string(sel)].nodes += r.nodes_processed;
            ++per_selector[to_string(sel)].runs;
          }
        }
      }
    }
  }
  std::cout << "    " << instances << " MIPs x 4 branching rules x 3 node selectors = " << compared << " runs, " << failed << " failed\n";
  std::cout << "    average nodes per run (informational, not asserted):\n";
  for (const auto& kv : per_rule) std::cout << "      rule " << std::left << std::setw(14) << kv.first << std::right << std::fixed << std::setprecision(2) << static_cast<double>(kv.second.nodes) / std::max(kv.second.runs, 1) << "\n";
  for (const auto& kv : per_selector) std::cout << "      selector " << std::left << std::setw(14) << kv.first << std::right << std::fixed << std::setprecision(2) << static_cast<double>(kv.second.nodes) / std::max(kv.second.runs, 1) << "\n";
  std::cout << std::defaultfloat;
  CHECK_EQ(failed, 0);
  CHECK_EQ(instances, 300);
}
