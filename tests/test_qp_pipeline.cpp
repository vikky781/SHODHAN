// The continuous solver facade on QPs: QP-safe presolve, scaling, the interior-point method, the fallback ladder, the
// statuses (NonConvex, NotImplemented, Infeasible with a Farkas ray, InfeasibleOrUnbounded) and determinism.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/quadratic.hpp"
#include "shodhan/rays.hpp"
#include "support/qp_families.hpp"
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

}  // namespace

TEST_CASE(qp_pipeline_solves_planted_qps_with_presolve_and_scaling) {
  const std::uint64_t count = env_u64("SHODHAN_QP_PIPE_SEEDS", 1500);
  const int nv = static_cast<int>(QpVariant::kCount);
  int total = 0, ok = 0, with_presolve_reductions = 0, fallbacks = 0;
  long long fixed = 0, singleton = 0, redundant = 0, empty_rows = 0;
  double worst_obj = 0.0;
  std::vector<std::string> failures;
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const QpVariant v = static_cast<QpVariant>(seed % static_cast<std::uint64_t>(nv));
    const PlantedQp p = make_planted_qp(40000 + seed, v);
    ++total;
    LpOptions o;
    o.presolve = seed % 5 != 0;  // a fifth without presolve
    const LpResult r = LpSolver(o).solve(p.model);
    if (r.status != Status::Optimal) {
      failures.push_back("seed " + std::to_string(seed) + " (" + qp_variant_name(v) + "): " + to_string(r.status) + " " + r.message);
      continue;
    }
    const double err = std::fabs(r.solution.objective - p.objective) / (1.0 + std::fabs(p.objective));
    worst_obj = std::max(worst_obj, err);
    const KktReport k = check_kkt(p.model, r.solution, 1e-6);
    if (err <= 1e-6 && k.ok && r.kkt.ok) ++ok;
    else failures.push_back("seed " + std::to_string(seed) + " (" + qp_variant_name(v) + "): objective error " + std::to_string(err) + " KKT " + k.summary());
    if (r.attempts > 1) ++fallbacks;
    if (r.presolve_ran) {
      fixed += r.presolve_stats.fixed_columns;
      singleton += r.presolve_stats.singleton_rows;
      redundant += r.presolve_stats.redundant_rows;
      empty_rows += r.presolve_stats.empty_rows;
      // Reductions that are not valid with a quadratic term must not have fired.
      if (p.model.quadratic.nnz() > 0) {
        CHECK_EQ(r.presolve_stats.dual_fixed_columns, 0);
        CHECK_EQ(r.presolve_stats.doubleton_equations, 0);
        CHECK_EQ(r.presolve_stats.forcing_rows, 0);
        CHECK_EQ(r.presolve_stats.unbounded_columns, 0);
      }
      if (r.presolve_stats.fixed_columns + r.presolve_stats.singleton_rows + r.presolve_stats.redundant_rows + r.presolve_stats.empty_rows > 0) ++with_presolve_reductions;
    }
  }
  std::cout << "  QP pipeline (LpSolver, presolve on for four fifths): " << ok << " of " << total << " solved to the planted optimum with the KKT check passed on the original model, worst objective error "
            << worst_obj << "; " << fallbacks << " needed a fallback; presolve reductions over all runs: " << fixed << " fixed columns, " << singleton << " singleton rows, " << redundant
            << " redundant rows, " << empty_rows << " empty rows (" << with_presolve_reductions << " models reduced)\n";
  for (std::size_t i = 0; i < failures.size() && i < 20; ++i) std::cerr << "  FAILING " << failures[i] << "\n";
  CHECK_EQ(ok, total);
  CHECK(with_presolve_reductions > 50 || count < 500);
}

TEST_CASE(qp_pipeline_reports_the_right_statuses) {
  // A non-convex objective: indefinite Q.
  PlantedQp p = make_planted_qp(77, QpVariant::PositiveDefinite);
  LpModel indef = p.model;
  indef.quadratic.value[0] = -10.0;  // the first stored entry is the diagonal of column 0
  {
    const LpResult r = LpSolver().solve(indef);
    CHECK(r.status == Status::NonConvex);
    CHECK(r.message.find("not positive semidefinite") != std::string::npos);
    CHECK(r.solution.x.empty());
  }
  // The simplex cannot solve a QP.
  {
    LpOptions o;
    o.method = LpMethod::Simplex;
    const LpResult r = LpSolver(o).solve(p.model);
    CHECK(r.status == Status::NotImplemented);
    CHECK(r.message.find("cannot solve a quadratic program") != std::string::npos);
  }
  // Method names.
  LpMethod m;
  CHECK(parse_method("ipm", &m) && m == LpMethod::Ipm);
  CHECK(parse_method("auto", &m) && m == LpMethod::Auto);
  CHECK(!parse_method("barrier", &m));
  // An LP with the IPM method gives the same optimum as the simplex.
  const PlantedQp lp = make_planted_qp(78, QpVariant::Lp);
  LpOptions oi;
  oi.method = LpMethod::Ipm;
  const LpResult ri = LpSolver(oi).solve(lp.model);
  const LpResult rs = LpSolver().solve(lp.model);
  CHECK(ri.status == Status::Optimal && rs.status == Status::Optimal);
  CHECK_NEAR(ri.solution.objective, rs.solution.objective, 1e-6 * (1.0 + std::fabs(rs.solution.objective)));
  CHECK(ri.method_used.find("interior point") != std::string::npos);
}

TEST_CASE(qp_pipeline_infeasible_gets_a_farkas_ray_and_unbounded_gets_no_certificate) {
  int infeasible_ok = 0, infeasible_total = 0, unbounded_total = 0, unbounded_ok = 0;
  for (std::uint64_t seed = 1; seed <= 120; ++seed) {
    {
      const LpModel m = make_infeasible_qp(seed);
      ++infeasible_total;
      const LpResult r = LpSolver().solve(m);
      if (r.status == Status::Infeasible && check_farkas(m, r.farkas_ray, 1e-9).ok) ++infeasible_ok;
      else std::cerr << "  infeasible QP seed " << seed << ": " << to_string(r.status) << " " << r.message << "\n";
    }
    {
      const LpModel m = make_unbounded_qp(seed);
      ++unbounded_total;
      const LpResult r = LpSolver().solve(m);
      // Never Optimal; without a certificate the honest statuses are InfeasibleOrUnbounded (or a numerical failure).
      if (r.status != Status::Optimal && r.status != Status::Infeasible) ++unbounded_ok;
      else std::cerr << "  unbounded QP seed " << seed << ": " << to_string(r.status) << "\n";
    }
  }
  std::cout << "  QP statuses: " << infeasible_ok << " of " << infeasible_total << " infeasible QPs reported Infeasible with verified Farkas multipliers; " << unbounded_ok << " of "
            << unbounded_total << " unbounded QPs reported neither Optimal nor Infeasible\n";
  CHECK_EQ(infeasible_ok, infeasible_total);
  CHECK_EQ(unbounded_ok, unbounded_total);
}

TEST_CASE(qp_pipeline_is_deterministic) {
  int same = 0, total = 0;
  for (std::uint64_t seed = 1; seed <= 120; ++seed) {
    const PlantedQp p = make_planted_qp(90000 + seed, static_cast<QpVariant>(seed % static_cast<std::uint64_t>(QpVariant::kCount)));
    const LpResult a = LpSolver().solve(p.model);
    const LpResult b = LpSolver().solve(p.model);
    ++total;
    if (a.status == b.status && a.iterations == b.iterations && a.solution.x == b.solution.x && a.solution.y == b.solution.y && a.solution.d == b.solution.d &&
        a.solution.objective == b.solution.objective && a.ipm_nnz_l == b.ipm_nnz_l && a.attempts == b.attempts) {
      ++same;
    } else {
      std::cerr << "  NONDETERMINISTIC seed " << seed << "\n";
    }
  }
  std::cout << "  determinism: " << same << " of " << total << " QPs solved twice with bit-identical x, y, d, objective and iteration counts\n";
  CHECK_EQ(same, total);
}
