// The pooling driver (docs/POOLING.md), SYNTHETIC instances only: parser, residual check, the fixed-quality LP, the
// two local methods and the McCormick bound against an independent grid reference, the cycling detection.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/pooling.hpp"
#include "support/pooling_families.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::pooling;
using namespace shodhan::testing;

namespace {

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == 0) return fallback;
  return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
}

const char* kGood =
    "# a comment\n"
    "name demo\n"
    "synthetic yes\n"
    "qualities 2 S N   # two named qualities\n"
    "source A 5 inf 4 1\n"
    "source B 9 50 1 2\n"
    "pool P 80\n"
    "terminal T 12 60 2.5 inf\n"
    "arc A P\n"
    "arc B P\n"
    "arc P T\n"
    "arc B T\n";

PoolOptions options(PoolMethod m, int starts) {
  PoolOptions o;
  o.method = m;
  o.starts = starts;
  o.seed = 7;
  return o;
}

}  // namespace

TEST_CASE(pool_spec_parser_reads_a_good_file) {
  const PoolParseResult r = parse_pool_string(kGood, "good.pool");
  REQUIRE(r.ok);
  const PoolProblem& p = r.problem;
  CHECK_EQ(p.name, std::string("demo"));
  CHECK(p.synthetic);
  CHECK_EQ(p.n_qualities, 2);
  CHECK_EQ(p.quality_names[1], std::string("N"));
  CHECK_EQ(p.sources.size(), std::size_t{2});
  CHECK(is_inf(p.sources[0].supply));
  CHECK_EQ(p.sources[1].supply, 50.0);
  CHECK_EQ(p.pools[0].capacity, 80.0);
  CHECK(is_inf(p.terminals[0].spec[1]));
  CHECK_EQ(p.arcs_sp.size(), std::size_t{2});
  CHECK_EQ(p.arcs_pt.size(), std::size_t{1});
  CHECK_EQ(p.arcs_st.size(), std::size_t{1});
  CHECK_EQ(p.n_flows(), std::size_t{4});
  CHECK_EQ(p.arc_name_sp(1), std::string("B>P"));
}

TEST_CASE(pool_spec_parser_reports_line_numbers) {
  struct Bad {
    const char* text;
    const char* expect;  // substring of the error, including the line number
  };
  const Bad bad[] = {
      {"name x\nqualities 1\nsource A 5 inf\n", "x.pool:3: 'source' needs"},
      {"qualities 1\nsource A 5 inf 1\nsource A 6 inf 2\n", "x.pool:3: duplicate name 'A'"},
      {"qualities 1\nsource A 5 inf 1\npool P 10\narc A Z\n", "x.pool:4: unknown name 'Z'"},
      {"qualities 1\nsource A 5 inf 1\npool P 10\narc P A\n", "x.pool:4: an arc goes from"},
      {"qualities 1\nsource A five inf 1\n", "x.pool:2: not a number: 'five'"},
      {"source A 5 inf 1\n", "x.pool:1: 'qualities' must come before 'source'"},
      {"qualities 1\nfoo bar\n", "x.pool:2: unknown keyword 'foo'"},
      {"qualities 1\nsource A 5 inf 1\nterminal T 9 10 2\npool P 5\narc A T\narc A P\n", "pool 'P' needs an incoming and an outgoing arc"},
      {"qualities 1\nsource A>B 5 inf 1\n", "x.pool:2: a name must not contain"},
      {"qualities 1\nsource A 5 inf 1\n", "at least one source and one terminal"},
      {"qualities 1\nsource A 5 inf 1\nterminal T 9 10 2\narc A T\narc A T\n", "x.pool:5: duplicate arc A T"},
  };
  for (const Bad& b : bad) {
    const PoolParseResult r = parse_pool_string(b.text, "x.pool");
    const bool found = !r.ok && r.error.find(b.expect) != std::string::npos;
    if (!found) std::cout << "  text: " << b.text << "  error: " << r.error << "  (expected '" << b.expect << "')\n";
    CHECK(found);
  }
  const PoolParseResult none = parse_pool_file("no_such_file.pool");
  CHECK(!none.ok);
}

TEST_CASE(pool_check_detects_every_kind_of_violation) {
  const PoolProblem p = make_single_pool(5);
  PoolOptions o = options(PoolMethod::Slp, 4);
  o.mccormick = false;
  const PoolResult r = solve_pool(p, o);
  REQUIRE(!r.point.flow.empty());
  CHECK(check_pool_point(p, r.point).ok(1e-6));
  // Perturb one thing at a time by a clearly visible amount.
  {
    PoolPoint q = r.point;
    q.flow[0] += 5.0;  // breaks the pool material balance (and possibly more)
    CHECK(check_pool_point(p, q).material > 1e-3);
  }
  {
    PoolPoint q = r.point;
    q.q[0] += 0.5;  // the pool quality no longer matches the mix
    CHECK(check_pool_point(p, q).quality_balance > 1e-4 || check_pool_point(p, q).bounds > 1e-4);
  }
  {
    PoolPoint q = r.point;
    q.flow[0] = -1.0;
    CHECK(check_pool_point(p, q).bounds > 0.1);
  }
  {
    PoolProblem tight = p;
    for (Terminal& t : tight.terminals) t.demand = 1e-3;  // the flows now exceed every demand
    double total = 0.0;
    for (const double v : r.point.flow) total += v;
    if (total > 1.0) CHECK(check_pool_point(tight, r.point).demand > 1e-3);
  }
  {
    PoolProblem strict = p;
    for (Terminal& t : strict.terminals) t.spec[0] = -100.0;  // impossible specification
    CHECK(check_pool_point(strict, r.point).terminal_spec > 1e-3);
  }
}

TEST_CASE(pool_fixed_quality_lp_is_a_relaxation_of_the_enforced_lp) {
  const std::uint64_t count = env_u64("SHODHAN_POOL_LP_SEEDS", 100);
  int checked = 0;
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const PoolProblem p = seed % 2 == 0 ? make_single_pool(seed) : make_two_pool(seed);
    std::vector<double> q(p.n_q());
    for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
      for (int k = 0; k < p.n_qualities; ++k) {
        const auto r = pool_quality_range(p, pool, k);
        q[p.qidx(pool, k)] = r.first + (r.second - r.first) * (0.2 + 0.6 * static_cast<double>((seed + pool + static_cast<std::uint64_t>(k)) % 5) / 4.0);
      }
    }
    std::vector<double> f;
    double obj = 0.0;
    REQUIRE(solve_fixed_quality_lp(p, q, &f, &obj));
    const double enforced = fixed_q_enforced_lp(p, q);
    // enforcing the pool quality equality can only lower the maximum
    if (enforced > -kInf / 2) {
      CHECK(obj >= enforced - 1e-7 * (1.0 + std::fabs(enforced)));
    }
    // the LP solution is feasible for the linear rows: no supply, capacity, demand or material violation
    PoolPoint pt{f, implied_pool_qualities(p, f, q)};
    const PoolCheck c = check_pool_point(p, pt);
    CHECK(c.supply <= 1e-7 && c.capacity <= 1e-7 && c.demand <= 1e-7 && c.material <= 1e-7 && c.bounds <= 1e-7);
    ++checked;
  }
  std::cout << "  fixed-quality LP vs enforced LP: " << checked << " instances, the relaxation property holds on all\n";
}

TEST_CASE(pool_driver_against_the_single_pool_grid_reference) {
  const std::uint64_t count = env_u64("SHODHAN_POOL_SEEDS", 300);
  const int grid = static_cast<int>(env_u64("SHODHAN_POOL_GRID", 2001));
  const int starts = static_cast<int>(env_u64("SHODHAN_POOL_STARTS", 8));
  struct Tally {
    int feasible = 0, within = 0, exceeded = 0, no_point = 0;
    double worst_excess = 0.0, worst_residual = 0.0;
    long long converged_starts = 0, all_starts = 0;
  } t[2];
  int bound_ok = 0, bound_bad = 0, bound_tight = 0;
  double gap_sum = 0.0;
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const PoolProblem p = make_single_pool(seed);
    const double ref = single_pool_reference(p, grid);
    const double tol_ref = 1e-6 * (1.0 + std::fabs(ref));
    for (int m = 0; m < 2; ++m) {
      const PoolResult r = solve_pool(p, options(m == 0 ? PoolMethod::Recursion : PoolMethod::Slp, starts));
      t[m].converged_starts += r.starts_converged;
      t[m].all_starts += r.starts_run;
      if (r.point.flow.empty()) {
        ++t[m].no_point;
        continue;
      }
      const PoolCheck c = check_pool_point(p, r.point);
      CHECK(c.ok(1e-6));
      t[m].worst_residual = std::max(t[m].worst_residual, c.worst);
      ++t[m].feasible;
      const double excess = r.objective - ref;
      if (excess > tol_ref) {
        ++t[m].exceeded;
        t[m].worst_excess = std::max(t[m].worst_excess, excess / (1.0 + std::fabs(ref)));
        std::cout << "  seed " << seed << " method " << (m == 0 ? "recursion" : "slp") << ": driver " << r.objective << " exceeds the reference " << ref << "\n";
      }
      if (r.objective >= ref - 1e-4 * (1.0 + std::fabs(ref))) ++t[m].within;
      // McCormick validity: the bound is at least every feasible objective
      if (r.has_bound) {
        if (r.mccormick_bound >= std::max(ref, r.objective) - tol_ref) ++bound_ok;
        else ++bound_bad;
        if (m == 1) {
          gap_sum += (r.mccormick_bound - ref) / std::max(1.0, std::fabs(r.mccormick_bound));
          if (r.mccormick_bound - ref <= 1e-4 * (1.0 + std::fabs(ref))) ++bound_tight;
        }
      }
    }
  }
  for (int m = 0; m < 2; ++m) {
    std::cout << "  " << (m == 0 ? "recursion" : "slp      ") << ": " << t[m].feasible << " of " << count << " multi-start results feasible for the nonlinear model (worst residual " << t[m].worst_residual
              << "), " << t[m].within << " within 1e-4 relative of the grid reference, " << t[m].exceeded << " above the reference by more than 1e-6; starts converged " << t[m].converged_starts
              << " of " << t[m].all_starts << "\n";
    CHECK_EQ(t[m].exceeded, 0);
    CHECK_EQ(t[m].no_point, 0);
  }
  std::cout << "  single pool, grid of " << grid << " values refined by ternary search, " << starts << " starts per method; McCormick bound valid in " << bound_ok << " of " << bound_ok + bound_bad
            << " checks, equal to the reference within 1e-4 in " << bound_tight << " instances, mean relative gap to the reference " << (count > 0 ? gap_sum / static_cast<double>(count) : 0.0) << "\n";
  CHECK_EQ(bound_bad, 0);
}

TEST_CASE(pool_driver_on_two_pool_two_quality_instances) {
  const std::uint64_t count = env_u64("SHODHAN_POOL2_SEEDS", 30);
  const int points = static_cast<int>(env_u64("SHODHAN_POOL2_GRID", 6));
  const int starts = static_cast<int>(env_u64("SHODHAN_POOL2_STARTS", 12));
  int ge[2] = {0, 0}, above[2] = {0, 0}, feasible[2] = {0, 0}, bound_ok = 0, bound_bad = 0;
  double best_ratio_sum[2] = {0.0, 0.0};
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const PoolProblem p = make_two_pool(seed);
    const double ref = grid_reference(p, points);
    for (int m = 0; m < 2; ++m) {
      const PoolResult r = solve_pool(p, options(m == 0 ? PoolMethod::Recursion : PoolMethod::Slp, starts));
      if (r.point.flow.empty()) continue;
      CHECK(check_pool_point(p, r.point).ok(1e-6));
      ++feasible[m];
      if (r.objective >= ref - 1e-6 * (1.0 + std::fabs(ref))) ++ge[m];
      if (r.objective > ref + 1e-6 * (1.0 + std::fabs(ref))) ++above[m];
      best_ratio_sum[m] += ref > 1e-9 ? r.objective / ref : 1.0;
      if (r.has_bound) {
        if (r.mccormick_bound >= std::max(ref, r.objective) - 1e-6 * (1.0 + std::fabs(ref))) ++bound_ok;
        else ++bound_bad;
      }
    }
  }
  for (int m = 0; m < 2; ++m) {
    std::cout << "  " << (m == 0 ? "recursion" : "slp      ") << " on " << count << " two-pool instances: " << feasible[m] << " feasible, objective >= the " << points << "^4-point grid reference in " << ge[m]
              << " (strictly above it in " << above[m] << "), mean objective/reference " << best_ratio_sum[m] / static_cast<double>(std::max<std::uint64_t>(1, count)) << "\n";
    CHECK_EQ(feasible[m], static_cast<int>(count));
  }
  std::cout << "  McCormick bound at least max(driver, grid reference) in " << bound_ok << " of " << bound_ok + bound_bad << " checks (the grid is coarse: it is a reference, not a proof)\n";
  CHECK_EQ(bound_bad, 0);
}

TEST_CASE(pool_undamped_recursion_cycles_and_never_returns_an_unverified_point) {
  // single_pool_58 (SYNTHETIC) oscillates between two pool qualities when the recursion is not damped.
  const PoolProblem p = make_single_pool(58);
  PoolOptions o = options(PoolMethod::Recursion, 1);
  o.damping = 1.0;
  o.mccormick = false;
  const auto range = pool_quality_range(p, 0, 0);
  const PoolRun run = solve_recursion(p, {0.5 * (range.first + range.second)}, o);
  CHECK(run.status == PoolStatus::Cycling);
  CHECK(!run.message.empty());
  // whatever point accompanies the status satisfies the nonlinear model
  if (!run.point.flow.empty()) CHECK(check_pool_point(p, run.point).ok(1e-6));
  const PoolResult res = solve_pool(p, o);
  CHECK(res.status == PoolStatus::Cycling);
  CHECK_EQ(res.starts_converged, 0);
  CHECK_EQ(res.starts_cycled, 1);
  if (!res.point.flow.empty()) CHECK(check_pool_point(p, res.point).ok(1e-6));
  // damping resolves the same instance
  o.damping = 0.5;
  const PoolResult damped = solve_pool(p, o);
  CHECK(damped.status == PoolStatus::Converged);
  CHECK(check_pool_point(p, damped.point).ok(1e-6));
  std::cout << "  single_pool_58: undamped recursion " << to_string(res.status) << " (" << res.message << "); damping 0.5: " << to_string(damped.status) << ", objective " << damped.objective << "\n";
}

TEST_CASE(pool_results_are_deterministic_and_the_unverified_never_leave_the_driver) {
  const std::uint64_t count = env_u64("SHODHAN_POOL_DET_SEEDS", 40);
  int same = 0;
  for (std::uint64_t seed = 1; seed <= count; ++seed) {
    const PoolProblem p = seed % 2 == 0 ? make_single_pool(seed) : make_two_pool(seed);
    for (const PoolMethod m : {PoolMethod::Recursion, PoolMethod::Slp}) {
      const PoolResult a = solve_pool(p, options(m, 5)), b = solve_pool(p, options(m, 5));
      const bool eq = a.status == b.status && a.objective == b.objective && a.point.flow == b.point.flow && a.point.q == b.point.q && a.mccormick_bound == b.mccormick_bound;
      CHECK(eq);
      if (eq) ++same;
      if (!a.point.flow.empty()) CHECK(check_pool_point(p, a.point).ok(1e-6));
    }
  }
  std::cout << "  determinism: " << same << " of " << 2 * count << " pairs of runs bit-identical\n";
}
