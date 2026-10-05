// Structure analysis (docs/STRUCTURE.md): detection of planted variable-upper-bound, flow-balance and set rows (exact
// counts, decoys left alone), and validity of the big-M tightening with implied bounds on seeded fixed-charge,
// scheduling-style and lot-sizing MIPs against the brute-force oracle.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/lp_solver.hpp"
#include "shodhan/mip/mip_solver.hpp"
#include "shodhan/presolve.hpp"
#include "shodhan/structure.hpp"
#include "support/mip_oracle.hpp"
#include "support/structure_families.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == 0) return fallback;
  return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
}

LpModel hand_model(Index n_cols, const std::vector<ColType>& types, const std::vector<double>& lo, const std::vector<double>& hi) {
  LpModel m;
  m.n_cols = n_cols;
  m.col_type = types;
  m.col_lower = lo;
  m.col_upper = hi;
  m.col_cost.assign(to_size(n_cols), 0.0);
  return m;
}

void add_row(LpModel& m, double lo, double hi, const std::vector<std::pair<Index, double>>& e, std::vector<Triplet>& t) {
  for (const auto& c : e) t.push_back({m.n_rows, c.first, c.second});
  m.row_lower.push_back(lo);
  m.row_upper.push_back(hi);
  ++m.n_rows;
}

}  // namespace

TEST_CASE(vub_shape_matcher_on_hand_made_rows) {
  double u = 0.0;
  bool comp = false;
  CHECK(match_vub_shape(1.0, -10.0, 0.0, &u, &comp));
  CHECK_EQ(u, 10.0);
  CHECK(!comp);
  CHECK(match_vub_shape(2.0, 20.0, 20.0, &u, &comp));
  CHECK_EQ(u, 10.0);
  CHECK(comp);
  CHECK(!match_vub_shape(1.0, -10.0, 1.0, &u, &comp));  // x <= 1 + 10 y
  CHECK(!match_vub_shape(1.0, 10.0, 0.0, &u, &comp));   // x + 10 y <= 0
  CHECK(!match_vub_shape(-1.0, -10.0, 0.0, &u, &comp)); // the x coefficient must be positive after normalization
  CHECK(!match_vub_shape(1.0, 0.0, 0.0, &u, &comp));
}

TEST_CASE(structure_detection_on_a_hand_made_model) {
  // columns: 0 x (continuous), 1 y (binary), 2 z (continuous), 3 w (continuous), 4 v (binary)
  LpModel m = hand_model(5, {ColType::Continuous, ColType::Binary, ColType::Continuous, ColType::Continuous, ColType::Binary},
                         {0, 0, 0, 0, 0}, {kInf, 1, kInf, kInf, 1});
  std::vector<Triplet> t;
  add_row(m, -kInf, 0.0, {{0, 1.0}, {1, -7.0}}, t);        // 0: x <= 7 y
  add_row(m, 0.0, kInf, {{0, -1.0}, {1, 7.0}}, t);         // 1: the same, written with >=
  add_row(m, -kInf, 7.0, {{0, 1.0}, {1, 7.0}}, t);         // 2: x <= 7 (1 - y)
  add_row(m, 2.0, 2.0, {{0, 1.0}, {2, -1.0}, {3, 1.0}}, t);  // 3: balance
  add_row(m, -kInf, 1.0, {{1, 1.0}, {4, 1.0}}, t);         // 4: packing row (not a VUB: both binary)
  add_row(m, 1.0, 1.0, {{1, 1.0}, {4, 1.0}}, t);           // 5: partitioning
  add_row(m, -kInf, 1.0, {{0, 1.0}, {1, -7.0}}, t);        // 6: decoy x <= 1 + 7 y
  std::string err;
  CHECK(SparseMatrix::from_triplets(m.n_rows, m.n_cols, t, &m.A, &err));
  CHECK(m.validate().empty());
  const StructureInfo s = detect_structure(m);
  CHECK_EQ(s.n_vubs(), std::size_t{3});
  CHECK_EQ(s.vubs[0].row, 0);
  CHECK_EQ(s.vubs[1].row, 1);
  CHECK_EQ(s.vubs[2].row, 2);
  CHECK(!s.vubs[0].complemented && !s.vubs[1].complemented && s.vubs[2].complemented);
  CHECK_EQ(s.vubs[2].u, 7.0);
  CHECK_EQ(s.n_balance(), std::size_t{1});
  CHECK_EQ(s.balance_rows[0], 3);
  CHECK_EQ(s.n_set(SetRowKind::Packing), std::size_t{1});
  CHECK_EQ(s.n_set(SetRowKind::Partition), std::size_t{1});
  CHECK_EQ(s.n_set(SetRowKind::Covering), std::size_t{0});
}

TEST_CASE(planted_structure_is_found_exactly_and_decoys_are_left_alone) {
  const std::uint64_t count = env_u64("SHODHAN_STRUCT_SEEDS", 400);
  long long planted_vub = 0, planted_bal = 0, planted_set = 0, decoys = 0, found_vub = 0, found_bal = 0, found_set = 0;
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const PlantedStructure p = make_planted_structure(seed);
    REQUIRE(p.model.validate().empty());
    const StructureInfo s = detect_structure(p.model);
    std::vector<VubRow> want = p.vubs;
    std::sort(want.begin(), want.end(), [](const VubRow& a, const VubRow& b) { return a.row < b.row; });
    bool ok = s.vubs.size() == want.size();
    for (std::size_t k = 0; ok && k < want.size(); ++k) {
      ok = s.vubs[k].row == want[k].row && s.vubs[k].x == want[k].x && s.vubs[k].y == want[k].y &&
           std::fabs(s.vubs[k].u - want[k].u) <= 1e-12 * want[k].u && s.vubs[k].complemented == want[k].complemented;
    }
    if (!ok) std::cout << "  seed " << seed << ": VUB rows differ (found " << s.vubs.size() << ", planted " << want.size() << ")\n";
    CHECK(ok);
    std::vector<Index> bal = p.balance_rows;
    std::sort(bal.begin(), bal.end());
    CHECK(s.balance_rows == bal);
    std::vector<SetRow> sets = p.set_rows;
    std::sort(sets.begin(), sets.end(), [](const SetRow& a, const SetRow& b) { return a.row < b.row; });
    bool sets_ok = s.set_rows.size() == sets.size();
    for (std::size_t k = 0; sets_ok && k < sets.size(); ++k) sets_ok = s.set_rows[k].row == sets[k].row && s.set_rows[k].kind == sets[k].kind;
    CHECK(sets_ok);
    planted_vub += static_cast<long long>(want.size());
    planted_bal += static_cast<long long>(bal.size());
    planted_set += static_cast<long long>(sets.size());
    found_vub += static_cast<long long>(s.vubs.size());
    found_bal += static_cast<long long>(s.balance_rows.size());
    found_set += static_cast<long long>(s.set_rows.size());
    decoys += p.decoys;
  }
  std::cout << "  planted structure, seeds 1.." << count << ": VUB rows " << found_vub << " found of " << planted_vub << " planted, balance rows "
            << found_bal << " of " << planted_bal << ", set rows " << found_set << " of " << planted_set << "; " << decoys
            << " decoy rows (none recognised, since the counts above are exact)\n";
  CHECK_EQ(found_vub, planted_vub);
  CHECK_EQ(found_bal, planted_bal);
  CHECK_EQ(found_set, planted_set);
}

namespace {

// Bound of the LP relaxation of a reduced model (offset included), or the optimum when presolve solved it.
bool lp_bound(const PresolveResult& pr, double* bound) {
  if (pr.status != PresolveStatus::Reduced) return false;
  LpModel relaxed = pr.reduced;
  relaxed.col_type.assign(to_size(relaxed.n_cols), ColType::Continuous);
  const LpResult r = LpSolver().solve(relaxed);
  if (r.status != Status::Optimal) return false;
  *bound = r.solution.objective;
  return true;
}

}  // namespace

TEST_CASE(implied_bound_tightening_keeps_the_optimum_and_never_weakens_the_root_bound) {
  const std::uint64_t count = env_u64("SHODHAN_BIGM_SEEDS", 510);
  int checked = 0, changed_rows = 0, bound_improved = 0, gap_instances = 0;
  double closed_sum = 0.0;
  long long tightenings = 0;
  int by_kind[kNumBigMKinds] = {0, 0, 0}, improved_by_kind[kNumBigMKinds] = {0, 0, 0};
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const int kind = static_cast<int>(seed % kNumBigMKinds);
    const LpModel m = make_bigm_instance(kind, seed);
    REQUIRE(m.validate().empty());
    const MipRefResult truth = solve_mip_brute_force(m);
    REQUIRE(!truth.too_large);
    REQUIRE(truth.status == Status::Optimal);
    PresolveOptions off;
    off.is_mip = true;
    off.need_duals = false;
    off.implied_bound_tightening = false;
    PresolveOptions on = off;
    on.implied_bound_tightening = true;
    double bounds[2] = {0.0, 0.0};
    bool have[2] = {false, false};
    for (int v = 0; v < 2; ++v) {
      const PresolveResult pr = presolve(m, v == 0 ? off : on);
      REQUIRE(pr.status == PresolveStatus::Reduced || pr.status == PresolveStatus::SolvedByPresolve);
      if (pr.status == PresolveStatus::Reduced) {
        const MipRefResult rb = solve_mip_brute_force(pr.reduced);
        REQUIRE(!rb.too_large);
        CHECK(rb.status == Status::Optimal);
        const bool same = std::fabs(rb.objective - truth.objective) <= 1e-7 * (1.0 + std::fabs(truth.objective));
        if (!same) std::cout << "  seed " << seed << " kind " << kind << (v ? " with" : " without") << " tightening: optimum " << rb.objective << " vs " << truth.objective << "\n";
        CHECK(same);
        have[v] = lp_bound(pr, &bounds[v]);
      } else {
        const Solution sol = postsolve(pr.stack, Solution{});
        CHECK(std::fabs(sol.objective - truth.objective) <= 1e-7 * (1.0 + std::fabs(truth.objective)));
        bounds[v] = truth.objective;
        have[v] = true;
      }
      if (v == 1) tightenings += pr.stats.implied_bound_tightenings;
      if (v == 1 && pr.stats.implied_bound_tightenings > 0) ++changed_rows;
    }
    ++checked;
    ++by_kind[kind];
    if (have[0] && have[1]) {
      const double tol = 1e-7 * (1.0 + std::fabs(truth.objective));
      CHECK(bounds[1] >= bounds[0] - tol);
      CHECK(bounds[0] <= truth.objective + tol);
      CHECK(bounds[1] <= truth.objective + tol);
      if (bounds[1] > bounds[0] + tol) {
        ++bound_improved;
        ++improved_by_kind[kind];
      }
      const double gap = truth.objective - bounds[0];
      if (gap > tol) {
        ++gap_instances;
        closed_sum += std::min(1.0, std::max(0.0, (bounds[1] - bounds[0]) / gap));
      }
    }
  }
  std::cout << "  big-M tightening on " << checked << " MIPs (fixed-charge " << by_kind[0] << ", sequencing " << by_kind[1] << ", lot sizing " << by_kind[2]
            << "): optimum unchanged on all; implied-bound tightening applied in " << changed_rows << " models (" << tightenings
            << " rows), root LP bound improved in " << bound_improved << " (" << improved_by_kind[0] << " / " << improved_by_kind[1] << " / "
            << improved_by_kind[2] << "); mean share of the root gap closed by it over the " << gap_instances << " models with a root gap: "
            << (gap_instances > 0 ? closed_sum / gap_instances : 0.0) << " (informational)\n";
  CHECK(changed_rows > 0);  // the test is only meaningful if the tightening fires on these families
}

TEST_CASE(structure_aware_cuts_keep_the_optimum_and_the_effect_is_measured) {
  const std::uint64_t count = env_u64("SHODHAN_STRUCT_CUT_SEEDS", 300);
  int checked = 0, better = 0, worse = 0, equal = 0, with_cuts = 0;
  long long nodes_on = 0, nodes_off = 0;
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const int kind = static_cast<int>(seed % kNumBigMKinds);
    const LpModel m = make_bigm_instance(kind, seed + 100000);
    const MipRefResult truth = solve_mip_brute_force(m);
    REQUIRE(truth.status == Status::Optimal);
    double root[2] = {0.0, 0.0};
    bool ran[2] = {false, false};
    for (int v = 0; v < 2; ++v) {
      mip::MipOptions mo;
      mo.mip_gap = 1e-9;
      mo.mip_abs_gap = 1e-9;
      mo.params.verbosity = 0;
      mo.cut_structure_aware = v == 1;
      const mip::MipResult r = mip::MipSolver(mo).solve(m);
      const bool ok = r.status == Status::Optimal && std::fabs(r.objective - truth.objective) <= 1e-6 * (1.0 + std::fabs(truth.objective));
      if (!ok) std::cout << "  seed " << seed << " kind " << kind << " structure-aware " << v << ": status " << to_string(r.status) << " objective " << r.objective << " vs " << truth.objective << "\n";
      CHECK(ok);
      (v == 1 ? nodes_on : nodes_off) += r.nodes_processed;
      ran[v] = r.cuts.ran;
      root[v] = r.cuts.root_bound_with_cuts;
    }
    ++checked;
    if (ran[0] && ran[1]) {
      ++with_cuts;
      const double tol = 1e-7 * (1.0 + std::fabs(truth.objective));
      if (root[1] > root[0] + tol) ++better;
      else if (root[1] < root[0] - tol) ++worse;
      else ++equal;
    }
  }
  std::cout << "  structure-aware MIR cuts on " << checked << " MIPs: optimum equal to brute force in every run; root bound with cuts: higher with the structure in "
            << better << ", lower in " << worse << ", equal in " << equal << " (of " << with_cuts << " where the cut loop ran); B&B nodes " << nodes_on << " with, "
            << nodes_off << " without (informational)\n";
}
