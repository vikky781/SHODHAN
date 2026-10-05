// Interior-point method: planted convex QPs and LPs with a known optimum, agreement with the dual simplex on LPs,
// and a tiny-QP active-set enumerator (docs/IPM.md).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/ipm.hpp"
#include "shodhan/kkt.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/quadratic.hpp"
#include "shodhan/scaling.hpp"
#include "support/qp_families.hpp"
#include "support/random_lp.hpp"
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

// Solves `model` the way the pipeline does: scaled, then unscaled for the checks. Returns the status of the IPM and the
// solution in the original space.
IpmResult solve_scaled(const LpModel& model, const IpmOptions& opt, bool scaled) {
  if (!scaled) return solve_ipm(model, opt);
  const Scaling sc = compute_scaling(model);
  const LpModel sm = apply_scaling(model, sc);
  IpmResult r = solve_ipm(sm, opt);
  if (r.status == Status::Optimal) r.solution = unscale_solution(sc, r.solution);
  return r;
}

}  // namespace

TEST_CASE(ipm_solves_planted_qps_to_the_known_optimum) {
  const std::uint64_t count = env_u64("SHODHAN_QP_SEEDS", 1600);
  const int nv = static_cast<int>(QpVariant::kCount);
  std::vector<int> total(nv, 0), solved(nv, 0), kkt_ok(nv, 0), raw_ok(nv, 0);
  std::vector<double> worst_obj(nv, 0.0), worst_res(nv, 0.0);
  std::vector<std::string> failures;
  long long iterations = 0;
  int max_iterations = 0;
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const QpVariant v = static_cast<QpVariant>(seed % static_cast<std::uint64_t>(nv));
    const PlantedQp p = make_planted_qp(seed, v);
    REQUIRE(p.model.validate().empty());
    const int vi = static_cast<int>(v);
    ++total[vi];
    {  // informational: the same problem without scaling
      const IpmResult raw = solve_scaled(p.model, IpmOptions(), false);
      if (raw.status == Status::Optimal && check_kkt(p.model, raw.solution, 1e-6).ok) ++raw_ok[vi];
    }
    const IpmResult r = solve_scaled(p.model, IpmOptions(), true);
    if (r.status != Status::Optimal) {
      failures.push_back("seed " + std::to_string(seed) + " (" + qp_variant_name(v) + "): " + to_string(r.status) + " " + r.message + " after " + std::to_string(r.iterations) + " iterations");
      continue;
    }
    ++solved[vi];
    iterations += r.iterations;
    max_iterations = std::max(max_iterations, r.iterations);
    const double err = std::fabs(r.solution.objective - p.objective) / (1.0 + std::fabs(p.objective));
    worst_obj[vi] = std::max(worst_obj[vi], err);
    const KktReport k = check_kkt(p.model, r.solution, 1e-6);
    worst_res[vi] = std::max(worst_res[vi], std::max(k.primal_infeasibility_rel, std::max(k.dual_infeasibility_rel, k.gap_rel)));
    if (err <= 1e-6 && k.ok) {
      ++kkt_ok[vi];
    } else {
      failures.push_back("seed " + std::to_string(seed) + " (" + qp_variant_name(v) + "): objective error " + std::to_string(err) + ", KKT " + k.summary());
    }
  }
  int all_total = 0, all_ok = 0, all_raw = 0;
  std::cout << "  planted QPs (seeds 1.." << count << "), scaled like the pipeline, IPM tol 1e-8, objective within 1e-6 and check_kkt at 1e-6 on the original model:\n";
  for (int i = 0; i < nv; ++i) {
    std::cout << "    " << qp_variant_name(static_cast<QpVariant>(i)) << ": " << kkt_ok[i] << " of " << total[i] << " pass (solved " << solved[i]
              << "), worst objective error " << worst_obj[i] << ", worst KKT residual " << worst_res[i] << "; without scaling " << raw_ok[i] << " pass\n";
    all_total += total[i];
    all_ok += kkt_ok[i];
    all_raw += raw_ok[i];
  }
  std::cout << "    total " << all_ok << " of " << all_total << " (without scaling " << all_raw << "); mean iterations " << (all_ok ? static_cast<double>(iterations) / all_ok : 0.0)
            << ", maximum " << max_iterations << "\n";
  for (std::size_t k = 0; k < failures.size() && k < 25; ++k) std::cerr << "  FAILING " << failures[k] << "\n";
  CHECK_EQ(all_ok, all_total);
}
