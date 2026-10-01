#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/mps.hpp"
#include "shodhan/presolve.hpp"
#include "shodhan/scaling.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/random_lp.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

// Seed range for the property tests. Override with SHODHAN_SEED_FIRST and
// SHODHAN_SEED_COUNT to stress a larger range; the defaults are what CI runs.
std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == 0) return fallback;
  return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
}

// Wide variety of structure from the seed, including every planted reduction.
RandomLpOptions options_for(std::uint64_t seed) {
  RandomLpOptions o;
  o.rows = 3 + static_cast<int>((seed * 3) % 20);
  o.cols = 4 + static_cast<int>((seed * 7) % 24);
  o.density = 0.12 + 0.08 * static_cast<double>(seed % 6);
  o.degeneracy = (seed % 3 == 0) ? 0.6 : 0.0;
  o.active_fraction = (seed % 4 == 0) ? 0.85 : 0.5;
  o.free_col_fraction = (seed % 5 == 0) ? 0.25 : 0.0;
  o.ranged_row_fraction = (seed % 4 == 1) ? 0.3 : 0.0;
  o.free_row_fraction = (seed % 9 == 0) ? 0.15 : 0.0;
  o.wide_coefficients = seed % 3 == 1;
  o.fixed_cols = static_cast<int>(seed % 4);
  o.empty_cols = static_cast<int>((seed / 2) % 3);
  o.empty_rows = static_cast<int>((seed / 3) % 3);
  o.singleton_rows = static_cast<int>((seed / 4) % 4);
  o.doubleton_eqs = static_cast<int>((seed / 5) % 4);
  o.forcing_rows = static_cast<int>((seed / 7) % 3);
  return o;
}

struct Failure {
  bool failed = false;
  /// The dense oracle reported NumericalError on this model, so nothing can be
  /// concluded about presolve. Counted and capped, never silently ignored.
  bool inconclusive = false;
  std::string message;
};

void fail(Failure* f, const std::string& msg) {
  if (!f->failed) {
    f->failed = true;
    f->message = msg;
  }
}

LpModel to_maximization(const LpModel& m) {
  LpModel out = m;
  out.sense = Sense::Maximize;
  for (double& c : out.col_cost) c = -c;
  out.objective_offset = -out.objective_offset;
  return out;
}

// Full presolve -> reference solve -> postsolve cycle on a model with a known
// optimum (given in minimization form; `maximize` flips the model's sense).
Failure run_lp_seed(std::uint64_t seed, const PresolveOptions& popt, bool through_scaling,
                    bool maximize) {
  Failure f;
  const RandomLp lp = make_random_lp(seed, options_for(seed));
  const LpModel model = maximize ? to_maximization(lp.model) : lp.model;
  const double known = maximize ? -lp.known.objective : lp.known.objective;

  PresolveResult pr = presolve(model, popt);
  if (pr.status == PresolveStatus::Infeasible || pr.status == PresolveStatus::Unbounded ||
      pr.status == PresolveStatus::InfeasibleOrUnbounded) {
    fail(&f, std::string("presolve returned ") + to_string(pr.status) +
                 " for a feasible bounded LP: " + pr.note);
    return f;
  }
  if (!pr.reduced.validate().empty()) {
    fail(&f, "reduced model is invalid: " + pr.reduced.validate().front());
    return f;
  }
  Solution reduced_sol;
  if (pr.status == PresolveStatus::Reduced) {
    LpModel to_solve = pr.reduced;
    Scaling sc;
    if (through_scaling) {
      sc = compute_scaling(pr.reduced);
      to_solve = apply_scaling(pr.reduced, sc);
    }
    const RefLpResult rr = solve_dense_lp(to_solve);
    if (rr.status == Status::NumericalError) {
      f.inconclusive = true;
      f.message = "oracle reported NumericalError on the reduced model";
      return f;
    }
    if (rr.status != Status::Optimal) {
      fail(&f, std::string("reduced model solve status ") + to_string(rr.status));
      return f;
    }
    reduced_sol = through_scaling ? unscale_solution(sc, rr.solution) : rr.solution;
    const KktReport kr = check_kkt(pr.reduced, reduced_sol, 1e-6);
    if (!kr.ok) {
      fail(&f, "reduced KKT failed: " + kr.summary());
      return f;
    }
  }
  const Solution sol = postsolve(pr.stack, reduced_sol);
  const KktReport k = check_kkt(model, sol, 1e-6);
  if (!k.ok) {
    fail(&f, "original KKT failed after postsolve: " + k.summary());
    return f;
  }
  if (std::fabs(sol.objective - known) > 1e-7 * (1.0 + std::fabs(known))) {
    std::ostringstream os;
    os.precision(15);
    os << "objective " << sol.objective << " != known optimum " << known;
    fail(&f, os.str());
    return f;
  }
  // Also compare with the dense solve of the original model.
  const RefLpResult direct = solve_dense_lp(model);
  if (direct.status == Status::Optimal &&
      std::fabs(sol.objective - direct.solution.objective) > 1e-7 * (1.0 + std::fabs(known))) {
    std::ostringstream os;
    os.precision(15);
    os << "objective " << sol.objective << " != dense solve of the original " << direct.solution.objective;
    fail(&f, os.str());
  }
  return f;
}

void run_range(const char* label, const PresolveOptions& popt, bool through_scaling, bool maximize,
               std::uint64_t first, std::uint64_t count) {
  int passed = 0;
  std::vector<std::uint64_t> inconclusive;
  const std::uint64_t last = first + count - 1;
  for (std::uint64_t seed = first; seed <= last; ++seed) {
    const Failure f = run_lp_seed(seed, popt, through_scaling, maximize);
    if (f.failed) {
      std::cerr << "  FAILING SEED " << seed << " (" << label << "): " << f.message << "\n";
      CHECK(!f.failed);
    } else if (f.inconclusive) {
      inconclusive.push_back(seed);
    } else {
      ++passed;
    }
  }
  std::cout << "  " << label << ": seeds " << first << ".." << last << ", passed " << passed << " of "
            << count << ", oracle inconclusive " << inconclusive.size();
  if (!inconclusive.empty()) {
    std::cout << " (seeds";
    for (const std::uint64_t sd : inconclusive) std::cout << " " << sd;
    std::cout << ")";
  }
  std::cout << "\n";
  // Every seed either passed or was inconclusive, and inconclusive stays rare.
  CHECK_EQ(passed + static_cast<int>(inconclusive.size()), static_cast<int>(count));
  CHECK(static_cast<double>(inconclusive.size()) <= 0.005 * static_cast<double>(count) + 1.0);
}

}  // namespace

TEST_CASE(presolve_property_lp_with_dense_reference) {
  run_range("presolve property (LP)", PresolveOptions{}, false, false, env_u64("SHODHAN_SEED_FIRST", 1),
            env_u64("SHODHAN_SEED_COUNT", 600));
}

TEST_CASE(presolve_property_lp_pipeline_presolve_scale_solve_unscale_postsolve) {
  run_range("pipeline (presolve+scale)", PresolveOptions{}, true, false,
            env_u64("SHODHAN_SEED_FIRST", 1) + 10000, env_u64("SHODHAN_SEED_COUNT", 300));
}

TEST_CASE(presolve_property_maximization_models) {
  run_range("maximization", PresolveOptions{}, false, true, env_u64("SHODHAN_SEED_FIRST", 1) + 20000,
            env_u64("SHODHAN_SEED_COUNT", 150));
}

TEST_CASE(presolve_property_without_duals_still_returns_the_primal_optimum) {
  PresolveOptions o;
  o.need_duals = false;
  int passed = 0;
  int inconclusive = 0;
  for (std::uint64_t seed = 1; seed <= 120; ++seed) {
    const RandomLp lp = make_random_lp(seed + 30000, options_for(seed));
    PresolveResult pr = presolve(lp.model, o);
    REQUIRE(pr.status == PresolveStatus::Reduced || pr.status == PresolveStatus::SolvedByPresolve);
    Solution rs;
    if (pr.status == PresolveStatus::Reduced) {
      const RefLpResult rr = solve_dense_lp(pr.reduced);
      if (rr.status == Status::NumericalError) {
        ++inconclusive;
        continue;
      }
      REQUIRE(rr.status == Status::Optimal);
      rs = rr.solution;
    }
    const Solution sol = postsolve(pr.stack, rs);
    CHECK(sol.y.empty() && sol.d.empty());
    CHECK_NEAR(sol.objective, lp.known.objective, 1e-7 * (1.0 + std::fabs(lp.known.objective)));
    // Primal feasibility of the mapped point (the dual part is not requested).
    std::vector<double> act(to_size(lp.model.n_rows), 0.0);
    lp.model.A.multiply(sol.x, act);
    bool feasible = true;
    for (std::size_t i = 0; i < act.size(); ++i) {
      const double tol = 1e-6 * (1.0 + std::fabs(act[i]));
      if (act[i] < lp.model.row_lower[i] - tol || act[i] > lp.model.row_upper[i] + tol) feasible = false;
    }
    for (std::size_t j = 0; j < sol.x.size(); ++j) {
      const double tol = 1e-6 * (1.0 + std::fabs(sol.x[j]));
      if (sol.x[j] < lp.model.col_lower[j] - tol || sol.x[j] > lp.model.col_upper[j] + tol) feasible = false;
    }
    CHECK(feasible);
    ++passed;
  }
  CHECK_EQ(passed + inconclusive, 120);
  CHECK(inconclusive <= 2);
}

TEST_CASE(presolve_property_infeasible_and_unbounded_are_never_reported_optimal) {
  int infeasible_seen = 0;
  int unbounded_seen = 0;
  const std::uint64_t first = env_u64("SHODHAN_SEED_FIRST", 1);
  const std::uint64_t count = env_u64("SHODHAN_SEED_COUNT", 300);
  for (std::uint64_t seed = first; seed < first + count; ++seed) {
    RandomLpOptions o = options_for(seed);
    o.rows = 3 + static_cast<int>(seed % 9);
    o.cols = 4 + static_cast<int>(seed % 10);
    std::string kind;

    // ---- infeasible by construction ----
    {
      const LpModel m = make_random_infeasible_lp(seed, o, &kind);
      const PresolveResult pr = presolve(m);
      bool good = false;
      switch (pr.status) {
        case PresolveStatus::Infeasible:
        case PresolveStatus::InfeasibleOrUnbounded:
          good = true;
          break;
        case PresolveStatus::Reduced: {
          const RefLpResult rr = solve_dense_lp(pr.reduced);
          good = rr.status == Status::Infeasible || rr.status == Status::Unbounded;
          break;
        }
        default:
          good = false;  // SolvedByPresolve or Unbounded would claim feasibility
          break;
      }
      if (!good) std::cerr << "  FAILING SEED " << seed << " (infeasible: " << kind << "): status " << to_string(pr.status) << "\n";
      CHECK(good);
      if (pr.status == PresolveStatus::Infeasible) ++infeasible_seen;
    }

    // ---- unbounded by construction ----
    {
      const LpModel m = make_random_unbounded_lp(seed, o, &kind);
      const PresolveResult pr = presolve(m);
      bool good = false;
      switch (pr.status) {
        case PresolveStatus::Unbounded:
        case PresolveStatus::InfeasibleOrUnbounded:
          good = true;
          break;
        case PresolveStatus::Reduced: {
          const RefLpResult rr = solve_dense_lp(pr.reduced);
          good = rr.status == Status::Unbounded;  // the reduced model must stay unbounded
          break;
        }
        default:
          good = false;  // Infeasible or SolvedByPresolve would be wrong
          break;
      }
      if (!good) std::cerr << "  FAILING SEED " << seed << " (unbounded: " << kind << "): status " << to_string(pr.status) << "\n";
      CHECK(good);
      if (pr.status == PresolveStatus::Unbounded) ++unbounded_seen;
    }
  }
  std::cout << "  infeasible/unbounded: seeds " << first << ".." << (first + count - 1)
            << "; presolve alone proved Infeasible " << infeasible_seen << " times and Unbounded "
            << unbounded_seen << " times\n";
}

// ---------------------------------------------------------------------------
// MIP safety against brute force
// ---------------------------------------------------------------------------
namespace {

LpModel random_small_mip(std::uint64_t seed) {
  Rng rng(seed * 7919 + 13);
  const int n = rng.range(3, 6);
  const int m = rng.range(2, 6);
  LpModel model;
  model.n_rows = m;
  model.n_cols = n;
  model.name = "mip" + std::to_string(seed);
  std::vector<int> x0(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    const bool binary = rng.chance(0.3);
    double lo = binary ? 0.0 : static_cast<double>(rng.range(-2, 1));
    double up = binary ? 1.0 : lo + static_cast<double>(rng.range(0, 3));
    if (rng.chance(0.1)) up = lo;  // fixed
    model.col_lower.push_back(lo);
    model.col_upper.push_back(up);
    model.col_type.push_back(binary ? ColType::Binary : ColType::Integer);
    model.col_cost.push_back(static_cast<double>(rng.range(-5, 5)));
    x0[static_cast<std::size_t>(j)] = static_cast<int>(lo) + rng.range(0, static_cast<int>(up - lo));
  }
  const bool empty_col = rng.chance(0.2);
  const int empty_j = rng.range(0, n - 1);
  std::vector<Triplet> t;
  for (int i = 0; i < m; ++i) {
    const double kind = rng.unit();
    std::vector<int> cols;
    if (kind < 0.25) {  // singleton
      cols = {rng.range(0, n - 1)};
    } else if (kind < 0.5) {  // doubleton
      const int a = rng.range(0, n - 1);
      int b = rng.range(0, n - 1);
      if (b == a) b = (a + 1) % n;
      cols = {a, b};
    } else {
      for (int j = 0; j < n; ++j) {
        if (rng.chance(0.5)) cols.push_back(j);
      }
      if (cols.empty() && rng.chance(0.7)) cols.push_back(rng.range(0, n - 1));
    }
    double act = 0.0;
    std::vector<int> used;
    for (const int j : cols) {
      if (empty_col && j == empty_j) continue;
      bool dup = false;
      for (const int u : used) dup = dup || u == j;
      if (dup) continue;
      used.push_back(j);
      double a = static_cast<double>(rng.range(1, 3)) * (rng.chance(0.5) ? 1.0 : -1.0);
      if (kind >= 0.25 && kind < 0.5) a = rng.chance(0.5) ? 1.0 : -1.0;  // unit doubletons
      t.push_back({i, j, a});
      act += a * x0[static_cast<std::size_t>(j)];
    }
    double lo = act;
    double up = act;
    const double rk = rng.unit();
    if (kind >= 0.25 && kind < 0.5 && used.size() == 2) {
      // equality doubleton
    } else if (rk < 0.3) {
      lo = -kInf;
      up = act + rng.range(0, 2);
    } else if (rk < 0.6) {
      lo = act - rng.range(0, 2);
      up = kInf;
    } else if (rk < 0.8) {
      lo = act - rng.range(0, 2);
      up = act + rng.range(0, 2);
    }
    if (rng.chance(0.08)) {  // possibly make the model infeasible
      lo = act + 1.0;
      up = act + 1.0 + rng.range(0, 1);
    }
    if (rng.chance(0.07) && used.size() >= 2) {  // forcing row: activity at its maximum
      double mx = 0.0;
      for (const auto& tr : t) {
        if (tr.row != i) continue;
        mx += tr.value > 0 ? tr.value * model.col_upper[static_cast<std::size_t>(tr.col)]
                           : tr.value * model.col_lower[static_cast<std::size_t>(tr.col)];
      }
      lo = mx;
      up = kInf;
    }
    model.row_lower.push_back(lo);
    model.row_upper.push_back(up);
  }
  std::string err;
  SparseMatrix::from_triplets(m, n, std::move(t), &model.A, &err);
  model.objective_offset = static_cast<double>(rng.range(-3, 3));
  for (std::size_t j = 0; j < model.col_type.size(); ++j) {
    if (model.col_type[j] == ColType::Binary) {
      model.col_lower[j] = 0.0;
      model.col_upper[j] = 1.0;
    }
  }
  return model;
}

struct BruteResult {
  bool feasible = false;
  double objective = 0.0;  // minimization form including offset
  std::vector<double> x;
};

// Enumerates every integer point within the (finite, integral) column bounds.
BruteResult brute_force(const LpModel& m) {
  BruteResult best;
  const int n = m.n_cols;
  std::vector<double> x(static_cast<std::size_t>(n));
  const double sgn = m.sense == Sense::Maximize ? -1.0 : 1.0;
  std::vector<double> act(to_size(m.n_rows));
  auto rec = [&](auto&& self, int j) -> void {
    if (j == n) {
      m.A.multiply(x, act);
      for (std::size_t i = 0; i < act.size(); ++i) {
        if (act[i] < m.row_lower[i] - 1e-9 || act[i] > m.row_upper[i] + 1e-9) return;
      }
      double obj = sgn * m.objective_offset;
      for (std::size_t k = 0; k < x.size(); ++k) obj += sgn * m.col_cost[k] * x[k];
      if (!best.feasible || obj < best.objective - 1e-12) {
        best.feasible = true;
        best.objective = obj;
        best.x = x;
      }
      return;
    }
    const std::size_t sj = static_cast<std::size_t>(j);
    for (double v = std::ceil(m.col_lower[sj] - 1e-9); v <= std::floor(m.col_upper[sj] + 1e-9); v += 1.0) {
      x[sj] = v;
      self(self, j + 1);
    }
  };
  rec(rec, 0);
  return best;
}

bool feasible_point(const LpModel& m, const std::vector<double>& x) {
  std::vector<double> act(to_size(m.n_rows), 0.0);
  m.A.multiply(x, act);
  for (std::size_t i = 0; i < act.size(); ++i) {
    if (act[i] < m.row_lower[i] - 1e-7 || act[i] > m.row_upper[i] + 1e-7) return false;
  }
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (x[j] < m.col_lower[j] - 1e-7 || x[j] > m.col_upper[j] + 1e-7) return false;
    if (m.col_type[j] != ColType::Continuous && std::fabs(x[j] - std::round(x[j])) > 1e-7) return false;
  }
  return true;
}

}  // namespace

TEST_CASE(presolve_mip_safety_never_removes_the_optimal_integer_solution) {
  PresolveOptions o;
  o.is_mip = true;
  int feasible_models = 0;
  int infeasible_models = 0;
  int reduced_models = 0;
  int solved_by_presolve = 0;
  const std::uint64_t first = env_u64("SHODHAN_SEED_FIRST", 1);
  const std::uint64_t count = env_u64("SHODHAN_SEED_COUNT", 600);
  for (std::uint64_t seed = first; seed < first + count; ++seed) {
    const LpModel m = random_small_mip(seed);
    REQUIRE(m.validate().empty());
    const BruteResult truth = brute_force(m);
    const PresolveResult pr = presolve(m, o);
    bool good = true;
    std::string why;
    if (pr.status == PresolveStatus::Infeasible) {
      good = !truth.feasible;
      why = "presolve claims infeasible (" + pr.note + ") but a feasible integer point exists";
    } else if (pr.status == PresolveStatus::Unbounded || pr.status == PresolveStatus::InfeasibleOrUnbounded) {
      good = false;
      why = std::string("bounded integer model reported ") + to_string(pr.status);
    } else {
      Solution reduced_sol;
      if (pr.status == PresolveStatus::Reduced) {
        ++reduced_models;
        const BruteResult rb = brute_force(pr.reduced);
        if (rb.feasible != truth.feasible) {
          good = false;
          why = "feasibility changed by presolve";
        } else if (truth.feasible) {
          if (std::fabs(rb.objective - truth.objective) > 1e-9) {
            good = false;
            why = "optimal value changed: reduced " + std::to_string(rb.objective) + " vs " +
                  std::to_string(truth.objective);
          }
          reduced_sol.x = rb.x;
        }
      } else {
        ++solved_by_presolve;
        if (!truth.feasible) {
          good = false;
          why = "SolvedByPresolve for an infeasible model";
        }
      }
      if (good && truth.feasible) {
        const Solution sol = postsolve(pr.stack, reduced_sol);
        if (!feasible_point(m, sol.x)) {
          good = false;
          why = "postsolved point is not feasible for the original MIP";
        } else if (std::fabs(sol.objective - (m.sense == Sense::Maximize ? -truth.objective : truth.objective)) > 1e-9) {
          good = false;
          why = "postsolved objective " + std::to_string(sol.objective) + " differs from the optimum " +
                std::to_string(truth.objective);
        }
      }
    }
    if (!good) std::cerr << "  FAILING SEED " << seed << " (MIP): " << why << "\n";
    CHECK(good);
    truth.feasible ? ++feasible_models : ++infeasible_models;
  }
  std::cout << "  MIP safety: seeds " << first << ".." << (first + count - 1) << ", feasible "
            << feasible_models << ", infeasible " << infeasible_models << ", reduced " << reduced_models
            << ", solved by presolve " << solved_by_presolve << "\n";
  CHECK(feasible_models > 50);
}

TEST_CASE(presolve_is_deterministic_byte_for_byte) {
  for (std::uint64_t seed = 1; seed <= 60; ++seed) {
    const RandomLp lp = make_random_lp(seed, options_for(seed));
    PresolveOptions o;
    o.is_mip = (seed % 4 == 0);
    LpModel m = lp.model;
    if (o.is_mip) {
      for (std::size_t j = 0; j < m.col_type.size(); j += 2) m.col_type[j] = ColType::Integer;
    }
    const PresolveResult a = presolve(m, o);
    const PresolveResult b = presolve(m, o);
    CHECK(a.status == b.status);
    std::ostringstream sa;
    std::ostringstream sb;
    std::string ea;
    std::string eb;
    const bool wa = write_mps(a.reduced, sa, &ea);
    const bool wb = write_mps(b.reduced, sb, &eb);
    CHECK_EQ(wa, wb);
    if (wa) CHECK_EQ(sa.str(), sb.str());
    CHECK_EQ(a.stack.records.size(), b.stack.records.size());
    CHECK(a.stack.row_map == b.stack.row_map);
    CHECK(a.stack.col_map == b.stack.col_map);
    CHECK_EQ(a.stats.reduction_counts() == b.stats.reduction_counts(), true);
  }
}
