// MIP presolve: equivalence with brute force on the ORIGINAL model over seeded small MIPs that are built from the
// structures the reductions work on (set packing, big-M links, loose knapsacks, parallel rows, duplicate and
// dominated columns, implication chains). Also validity of the cliques and implications on every feasible
// integer point of the reduced model.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "shodhan/mip/incumbent.hpp"
#include "shodhan/mip/mip_solver.hpp"
#include "shodhan/presolve.hpp"
#include "support/mip_oracle.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == 0) return fallback;
  return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
}

struct Builder {
  LpModel m;
  std::vector<Triplet> t;
  int rows = 0;

  int col(ColType type, double lo, double up, double cost) {
    m.col_type.push_back(type);
    m.col_lower.push_back(lo);
    m.col_upper.push_back(up);
    m.col_cost.push_back(cost);
    return static_cast<int>(m.col_cost.size()) - 1;
  }
  int row(const std::vector<std::pair<int, double>>& e, double lo, double up) {
    for (const auto& p : e) t.push_back({rows, p.first, p.second});
    m.row_lower.push_back(lo);
    m.row_upper.push_back(up);
    return rows++;
  }
  LpModel finish() {
    m.n_rows = rows;
    m.n_cols = static_cast<Index>(m.col_cost.size());
    std::string err;
    SparseMatrix::from_triplets(m.n_rows, m.n_cols, std::move(t), &m.A, &err);
    return std::move(m);
  }
};

struct RowCopy {
  std::vector<std::pair<int, double>> e;
  double lo, up;
};

LpModel make_presolve_mip(std::uint64_t seed) {
  Rng rng(seed * 104729 + 7);
  Builder b;
  const bool maximize = rng.chance(1.0 / 3.0);
  const double sgn = maximize ? -1.0 : 1.0;  // cost sign that keeps the "min-form" costs positive
  const int nb = rng.range(3, 6), ni = rng.range(0, 2), nc = rng.chance(0.3) ? rng.range(1, 2) : 0;
  std::vector<int> bins, ints, conts;
  for (int k = 0; k < nb; ++k) bins.push_back(b.col(ColType::Binary, 0.0, 1.0, rng.range(-6, 6)));
  for (int k = 0; k < ni; ++k) ints.push_back(b.col(ColType::Integer, 0.0, 3.0, rng.range(-6, 6)));
  for (int k = 0; k < nc; ++k) {
    conts.push_back(b.col(ColType::Continuous, 0.0, static_cast<double>(rng.range(3, 5)), rng.range(-6, 6)));
  }
  std::vector<int> all = bins;
  all.insert(all.end(), ints.begin(), ints.end());
  all.insert(all.end(), conts.begin(), conts.end());
  std::vector<RowCopy> made;
  auto subset = [&](const std::vector<int>& pool, int lo_k, int hi_k) {
    std::vector<int> v = pool;
    rng.shuffle(v);
    v.resize(static_cast<std::size_t>(std::min<int>(rng.range(lo_k, hi_k), static_cast<int>(v.size()))));
    return v;
  };
  auto add = [&](std::vector<std::pair<int, double>> e, double lo, double up) {
    b.row(e, lo, up);
    made.push_back({std::move(e), lo, up});
  };

  if (rng.chance(0.65)) {  // set packing
    const int reps = rng.range(1, 2);
    for (int r = 0; r < reps; ++r) {
      std::vector<std::pair<int, double>> e;
      for (const int j : subset(bins, 2, std::min<int>(4, nb))) e.push_back({j, 1.0});
      if (e.size() >= 2) add(e, -kInf, 1.0);
    }
  }
  if (!conts.empty() && rng.chance(0.8)) {  // big-M link y <= M x
    const double big[3] = {10.0, 20.0, 50.0};
    add({{conts[0], 1.0}, {bins[0], -big[rng.range(0, 2)]}}, -kInf, 0.0);
  }
  if (rng.chance(0.75)) {  // loose knapsack
    std::vector<std::pair<int, double>> e;
    double mx = 0.0;
    for (const int j : subset(all, 2, 5)) {
      const double a = rng.range(1, 9);
      e.push_back({j, a});
      mx += a * b.m.col_upper[static_cast<std::size_t>(j)];
    }
    add(e, -kInf, std::floor(mx * (0.35 + 0.5 * rng.unit())));
  }
  if (rng.chance(0.55)) {  // covering
    std::vector<std::pair<int, double>> e;
    for (const int j : subset(all, 2, 4)) e.push_back({j, static_cast<double>(rng.range(1, 3))});
    add(e, static_cast<double>(rng.range(1, 3)), kInf);
  }
  if (rng.chance(0.5)) {  // implication x_a <= x_b
    const std::vector<int> ab = subset(bins, 2, 2);
    if (ab.size() == 2) add({{ab[0], 1.0}, {ab[1], -1.0}}, -kInf, 0.0);
  }
  if (nb >= 3 && rng.chance(0.4)) {  // a + b <= 1, b + c >= 1, a + c >= 1: c = 1 by probing
    const std::vector<int> abc = subset(bins, 3, 3);
    add({{abc[0], 1.0}, {abc[1], 1.0}}, -kInf, 1.0);
    add({{abc[1], 1.0}, {abc[2], 1.0}}, 1.0, kInf);
    add({{abc[0], 1.0}, {abc[2], 1.0}}, 1.0, kInf);
  }
  if (rng.chance(0.2)) {  // equality
    std::vector<std::pair<int, double>> e;
    for (const int j : subset(bins, 2, 4)) e.push_back({j, 1.0});
    if (e.size() >= 2) add(e, 1.0, 1.0);
  }
  if (!made.empty() && rng.chance(0.55)) {  // a parallel copy of a row, scaled, with its own side
    const RowCopy& src = made[static_cast<std::size_t>(rng.range(0, static_cast<int>(made.size()) - 1))];
    const double lams[4] = {2.0, -1.0, 3.0, 0.5};
    const double lam = lams[rng.range(0, 3)];
    std::vector<std::pair<int, double>> e = src.e;
    for (auto& p : e) p.second *= lam;
    const double k = rng.range(0, 2);
    double lo = -kInf, up = kInf;
    if (!is_inf(src.up)) {
      (lam > 0 ? up : lo) = lam * (src.up + k);
    } else {
      (lam > 0 ? lo : up) = lam * (src.lo - k);
    }
    b.row(e, lo, up);
  }
  if (rng.chance(0.6)) {  // duplicate column: same entries, same cost (merge) or a different cost
    const int j = all[static_cast<std::size_t>(rng.range(0, static_cast<int>(all.size()) - 1))];
    const std::size_t sj = static_cast<std::size_t>(j);
    const int nj = b.col(b.m.col_type[sj], b.m.col_lower[sj], b.m.col_upper[sj],
                         b.m.col_cost[sj] + (rng.chance(0.3) ? static_cast<double>(rng.range(1, 2)) : 0.0));
    std::vector<Triplet> extra;
    for (const Triplet& tr : b.t) {
      if (tr.col == j) extra.push_back({tr.row, nj, tr.value});
    }
    b.t.insert(b.t.end(), extra.begin(), extra.end());
  }
  if (rng.chance(0.4)) {  // dominated pair: two continuous columns with identical entries, the cheap one unbounded
    const int c1 = b.col(ColType::Continuous, 0.0, kInf, sgn * 1.0);
    const int c2 = b.col(ColType::Continuous, 0.0, 4.0, sgn * 2.0);
    b.row({{c1, 1.0}, {c2, 1.0}}, 2.5, kInf);
    b.row({{c1, 1.0}, {c2, 1.0}, {all[0], 1.0}}, -kInf, 9.0);
  }
  b.m.sense = maximize ? Sense::Maximize : Sense::Minimize;
  b.m.objective_offset = static_cast<double>(rng.range(-3, 3));
  b.m.name = "pm" + std::to_string(seed);
  return b.finish();
}

// Enumerates all integer points of a pure-integer model with finite bounds; calls f(x) for each feasible one.
// Returns false when the model is not enumerable (continuous columns, infinite bounds, too many points).
template <typename F>
bool for_each_feasible(const LpModel& m, F&& f) {
  const std::size_t n = static_cast<std::size_t>(m.n_cols);
  long long combos = 1;
  for (std::size_t j = 0; j < n; ++j) {
    if (m.col_type[j] == ColType::Continuous || is_inf(m.col_lower[j]) || is_inf(m.col_upper[j])) return false;
    combos *= static_cast<long long>(m.col_upper[j] - m.col_lower[j]) + 1;
    if (combos > 300000) return false;
  }
  std::vector<double> x(n);
  for (std::size_t j = 0; j < n; ++j) x[j] = m.col_lower[j];
  std::vector<double> act(static_cast<std::size_t>(m.n_rows));
  for (long long c = 0; c < combos; ++c) {
    std::fill(act.begin(), act.end(), 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      for (Index k = m.A.col_start[j]; k < m.A.col_start[j + 1]; ++k) {
        act[static_cast<std::size_t>(m.A.row_index[static_cast<std::size_t>(k)])] +=
            m.A.value[static_cast<std::size_t>(k)] * x[j];
      }
    }
    bool ok = true;
    for (std::size_t i = 0; ok && i < act.size(); ++i) {
      ok = act[i] >= m.row_lower[i] - 1e-9 && act[i] <= m.row_upper[i] + 1e-9;
    }
    if (ok) f(x);
    for (std::size_t j = 0; j < n; ++j) {  // odometer
      if (x[j] < m.col_upper[j]) {
        x[j] += 1.0;
        break;
      }
      x[j] = m.col_lower[j];
    }
  }
  return true;
}

struct Tally {
  int seeds = 0, feasible = 0, infeasible = 0, skipped = 0, reduced = 0, solved = 0, proved_infeasible = 0;
  int bb_checked = 0, fired[11] = {0};
  long long total[11] = {0};
  int validity_models = 0, validity_skipped = 0;
  long long cliques_checked = 0, implications_checked = 0, points_checked = 0;
};

const char* const kNames[11] = {"propagated bounds", "coefficients tightened", "probing fixings", "probing bounds",
                                "implications",      "cliques",                "parallel rows",     "duplicate columns",
                                "dominated columns", "fixed columns",          "dual-fixed columns"};

// Presolve with `opt` and compare with the truth; empty string when everything agrees.
std::string check_presolve(const LpModel& m, const MipRefResult& truth, const PresolveOptions& opt, Tally& ty,
                           bool count_stats, PresolveResult* out) {
  const bool feas = truth.status == Status::Optimal;
  const double truth_min = m.sense == Sense::Maximize ? -truth.objective : truth.objective;
  const PresolveResult pr = presolve(m, opt);
    std::string why;
    if (pr.status == PresolveStatus::Unbounded || pr.status == PresolveStatus::InfeasibleOrUnbounded) {
      why = std::string("bounded model reported ") + to_string(pr.status);
    } else if (pr.status == PresolveStatus::Infeasible) {
      if (count_stats) ++ty.proved_infeasible;
      if (feas) why = "presolve claims infeasible (" + pr.note + ") but a feasible point exists";
    } else {
      Solution reduced_sol;
      if (pr.status == PresolveStatus::Reduced) {
        if (count_stats) ++ty.reduced;
        const MipRefResult rb = solve_mip_brute_force(pr.reduced);
        if (rb.too_large || rb.status == Status::NumericalError) {
          why = "reduced model could not be solved by the oracle";
        } else if ((rb.status == Status::Optimal) != feas) {
          why = "feasibility changed by presolve";
        } else if (feas) {
          if (std::fabs(rb.objective - truth_min) > 1e-7 * (1.0 + std::fabs(truth_min))) {
            why = "optimal value changed: reduced " + std::to_string(rb.objective) + " vs " + std::to_string(truth_min);
          }
          reduced_sol.x = rb.x;
        }
      } else {
        if (count_stats) ++ty.solved;
        if (!feas) why = "SolvedByPresolve for an infeasible model";
      }
      if (why.empty() && feas) {
        const Solution sol = postsolve(pr.stack, reduced_sol);
        if (!mip::check_point(m, sol.x).ok(1e-7, 1e-7)) {
          why = "postsolved point is not feasible for the original MIP";
        } else if (std::fabs(sol.objective - truth.objective) > 1e-7 * (1.0 + std::fabs(truth.objective))) {
          why = "postsolved objective " + std::to_string(sol.objective) + " differs from the optimum " +
                std::to_string(truth.objective);
        }
      }
    }
  if (out != nullptr) *out = pr;
  return why;
}

}  // namespace

TEST_CASE(presolve_mip_equivalence_with_brute_force_on_the_original_model) {
  const std::uint64_t first = env_u64("SHODHAN_SEED_FIRST", 1);
  const std::uint64_t count = env_u64("SHODHAN_MIP_SEED_COUNT", 1000);
  PresolveOptions o;
  o.is_mip = true;
  o.need_duals = false;
  Tally ty;
  for (std::uint64_t seed = first; seed < first + count; ++seed) {
    const LpModel m = make_presolve_mip(seed);
    REQUIRE(m.validate().empty());
    const MipRefResult truth = solve_mip_brute_force(m);
    if (truth.too_large || (truth.status != Status::Optimal && truth.status != Status::Infeasible)) {
      ++ty.skipped;
      continue;
    }
    ++ty.seeds;
    const bool feas = truth.status == Status::Optimal;
    feas ? ++ty.feasible : ++ty.infeasible;
    PresolveResult pr;
    std::string why = check_presolve(m, truth, o, ty, true, &pr);
    // Each MIP reduction alone (plus the fixed-column pass that turns its bound changes into removals).
    for (int variant = 0; variant < 5 && why.empty(); ++variant) {
      PresolveOptions one = o;
      one.empty_rows = one.empty_columns = one.singleton_rows = one.redundant_rows = false;
      one.forcing_rows = one.doubleton_equations = one.dual_fixing = one.integer_bounds = false;
      one.mip_propagation = one.coefficient_tightening = one.probing = false;
      one.parallel_rows = one.duplicate_columns = one.clique_table = false;
      (variant == 0 ? one.mip_propagation : variant == 1 ? one.coefficient_tightening : variant == 2 ? one.probing
       : variant == 3 ? one.parallel_rows : one.duplicate_columns) = true;
      Tally scratch;
      why = check_presolve(m, truth, one, scratch, false, nullptr);
      if (!why.empty()) why = "[only " + std::string(variant == 0 ? "propagation" : variant == 1 ? "coefficient tightening" : variant == 2 ? "probing" : variant == 3 ? "parallel rows" : "duplicate columns") + "] " + why;
    }
    // The branch and bound with the presolve inside, against the same truth.
    if (why.empty()) {
      mip::MipOptions mo;
      mo.mip_gap = 1e-9;
      mo.mip_abs_gap = 1e-9;
      mo.params.verbosity = 0;
      const mip::MipResult r = mip::MipSolver(mo).solve(m);
      ++ty.bb_checked;
      if (feas) {
        if (r.status != Status::Optimal) why = std::string("B&B status ") + to_string(r.status) + ", oracle Optimal";
        else if (std::fabs(r.objective - truth.objective) > 1e-6 * (1.0 + std::fabs(truth.objective))) {
          why = "B&B objective " + std::to_string(r.objective) + " vs " + std::to_string(truth.objective);
        }
      } else if (r.status != Status::Infeasible) {
        why = std::string("B&B status ") + to_string(r.status) + ", oracle Infeasible";
      }
    }
    // Validity of cliques and implications on every feasible point of the reduced model.
    if (why.empty() && pr.status == PresolveStatus::Reduced) {
      bool bad = false;
      std::string detail;
      const bool enumerable = for_each_feasible(pr.reduced, [&](const std::vector<double>& x) {
        ++ty.points_checked;
        for (const auto& cl : pr.mip.cliques.cliques) {
          int on = 0;
          for (const int l : cl) {
            const double v = x[static_cast<std::size_t>(l >> 1)];
            on += ((l & 1) ? 1.0 - v : v) > 0.5 ? 1 : 0;
          }
          if (on > 1) {
            bad = true;
            detail = "clique violated";
          }
        }
        for (const Implication& im : pr.mip.implications) {
          if (static_cast<int>(std::lround(x[static_cast<std::size_t>(im.var)])) != im.var_value) continue;
          const double v = x[static_cast<std::size_t>(im.other)];
          if (im.is_upper ? v > im.bound + 1e-9 : v < im.bound - 1e-9) {
            bad = true;
            detail = "implication violated";
          }
        }
      });
      if (enumerable) {
        ++ty.validity_models;
        ty.cliques_checked += static_cast<long long>(pr.mip.cliques.cliques.size());
        ty.implications_checked += static_cast<long long>(pr.mip.implications.size());
        if (bad) why = detail;
      } else {
        ++ty.validity_skipped;
      }
    }
    if (!why.empty()) std::cerr << "  FAILING SEED " << seed << " (MIP presolve): " << why << "\n";
    CHECK(why.empty());
    const PresolveStats& s = pr.stats;
    const int v[11] = {s.propagated_bounds, s.coefficients_tightened, s.probing_fixings, s.probing_bounds,
                       s.implications,      s.cliques,                s.parallel_rows,    s.duplicate_columns,
                       s.dominated_columns, s.fixed_columns,          s.dual_fixed_columns};
    for (int k = 0; k < 11; ++k) {
      ty.total[k] += v[k];
      if (v[k] > 0) ++ty.fired[k];
    }
  }
  std::cout << "  MIP presolve: seeds " << first << ".." << (first + count - 1) << ", checked " << ty.seeds
            << " (feasible " << ty.feasible << ", infeasible " << ty.infeasible << ", skipped " << ty.skipped
            << "), presolve reduced " << ty.reduced << ", solved " << ty.solved << ", proved infeasible "
            << ty.proved_infeasible << ", B&B checked " << ty.bb_checked << "\n";
  std::cout << "  validity: " << ty.validity_models << " reduced models enumerated (" << ty.points_checked
            << " feasible points), " << ty.cliques_checked << " cliques and " << ty.implications_checked
            << " implications checked, " << ty.validity_skipped << " models not enumerable\n";
  // Each reduction must fire on at least this many models, otherwise the equivalence above says nothing about it.
  const int minimum = 25;
  for (int k = 0; k < 11; ++k) {
    std::cout << "  " << kNames[k] << ": " << ty.total[k] << " in " << ty.fired[k] << " models";
    if (ty.fired[k] < minimum) std::cout << "  INCONCLUSIVE for this reduction (fired on fewer than " << minimum << " models)";
    std::cout << "\n";
  }
  CHECK(ty.seeds >= 1000);
  CHECK(ty.feasible > 200);
  CHECK(ty.infeasible > 20);
}

TEST_CASE(presolve_mip_parallel_rows_keep_the_tighter_bound_of_each_side) {
  // r0: x + y <= 5;  r1: 2x + 2y in [4, 8] (= x + y in [2, 4]);  r2: -x - y >= -3 (= x + y <= 3).
  Builder b;
  const int x = b.col(ColType::Integer, 0.0, 10.0, 1.0);
  const int y = b.col(ColType::Integer, 0.0, 10.0, 2.0);
  b.row({{x, 1.0}, {y, 1.0}}, -kInf, 5.0);
  b.row({{x, 2.0}, {y, 2.0}}, 4.0, 8.0);
  b.row({{x, -1.0}, {y, -1.0}}, -3.0, kInf);
  const LpModel m = b.finish();
  PresolveOptions o;
  o.is_mip = true;
  o.need_duals = false;
  o.empty_rows = o.empty_columns = o.fixed_columns = o.singleton_rows = o.redundant_rows = false;
  o.forcing_rows = o.doubleton_equations = o.dual_fixing = o.integer_bounds = false;
  o.mip_propagation = o.coefficient_tightening = o.probing = o.duplicate_columns = o.clique_table = false;
  o.parallel_rows = true;
  const PresolveResult pr = presolve(m, o);
  CHECK(pr.status == PresolveStatus::Reduced);
  CHECK_EQ(pr.stats.parallel_rows, 2);
  REQUIRE(pr.reduced.n_rows == 1);
  // Which of the three rows survives is an implementation detail; its range must be the intersection [2, 3]
  // expressed in the scale of the surviving row.
  const double scale = std::fabs(pr.reduced.A.value[0]);
  const bool negated = pr.reduced.A.value[0] < 0.0;
  const double lo = negated ? -pr.reduced.row_upper[0] : pr.reduced.row_lower[0];
  const double up = negated ? -pr.reduced.row_lower[0] : pr.reduced.row_upper[0];
  CHECK_NEAR(lo / scale, 2.0, 1e-12);
  CHECK_NEAR(up / scale, 3.0, 1e-12);
}
