#include <cmath>
#include <cstdint>

#include "shodhan/kkt.hpp"
#include "shodhan/scaling.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/random_lp.hpp"
#include "support/test_models.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

bool is_power_of_two(double v) {
  if (!(v > 0.0) || !std::isfinite(v)) return false;
  int e = 0;
  return std::frexp(v, &e) == 0.5;
}

RandomLpOptions options_for(std::uint64_t seed) {
  RandomLpOptions o;
  o.rows = 3 + static_cast<int>(seed % 12);
  o.cols = 3 + static_cast<int>((seed * 5) % 14);
  o.density = 0.25 + 0.1 * static_cast<double>(seed % 5);
  o.degeneracy = (seed % 3 == 0) ? 0.5 : 0.0;
  o.free_col_fraction = (seed % 4 == 0) ? 0.2 : 0.0;
  o.ranged_row_fraction = (seed % 5 == 0) ? 0.3 : 0.0;
  o.wide_coefficients = seed % 2 == 0;
  o.fixed_cols = static_cast<int>(seed % 2);
  return o;
}

}  // namespace

TEST_CASE(scaling_factors_are_powers_of_two_and_scaling_is_exact) {
  for (std::uint64_t seed = 1; seed <= 40; ++seed) {
    RandomLpOptions o = options_for(seed);
    o.wide_coefficients = true;
    const RandomLp lp = make_random_lp(seed, o);
    const Scaling sc = compute_scaling(lp.model);
    for (const double f : sc.row_scale) CHECK(is_power_of_two(f));
    for (const double f : sc.col_scale) CHECK(is_power_of_two(f));
    CHECK(is_power_of_two(sc.obj_scale));

    const LpModel scaled = apply_scaling(lp.model, sc);
    // Power-of-two factors: dividing back recovers every coefficient bit for bit.
    for (Index j = 0; j < lp.model.n_cols; ++j) {
      for (Index p = lp.model.A.col_start[to_size(j)]; p < lp.model.A.col_start[to_size(j) + 1]; ++p) {
        const double back = scaled.A.value[to_size(p)] / sc.col_scale[to_size(j)] /
                            sc.row_scale[to_size(lp.model.A.row_index[to_size(p)])];
        CHECK_EQ(back, lp.model.A.value[to_size(p)]);
      }
    }
    CHECK(scaled.validate().empty());
  }
}

TEST_CASE(scaling_improves_a_badly_scaled_matrix) {
  // Entries from 1e-4 to 1e4: ratio 1e8.
  LpModel m = make_model(2, 2, {{0, 0, 1e4}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1e-4}}, {1.0, 1.0},
                         {0.0, 0.0}, {kInf, kInf}, {-kInf, -kInf}, {1.0, 1.0});
  ScalingReport rep;
  const Scaling sc = compute_scaling(m, {}, &rep);
  CHECK(rep.before.valid);
  CHECK_NEAR(rep.before.ratio, 1e8, 1.0);
  CHECK(rep.after.ratio < rep.before.ratio);
  CHECK(rep.after.ratio < 10.0);
  CHECK(rep.geometric_passes >= 1);
  const CoefficientRange direct = coefficient_range(apply_scaling(m, sc).A);
  CHECK_EQ(direct.ratio, rep.after.ratio);
}

TEST_CASE(scaling_never_scales_integer_columns) {
  LpModel m = make_model(2, 3, {{0, 0, 1e4}, {0, 1, 1.0}, {1, 1, 1e-3}, {1, 2, 5.0}}, {1.0, 2.0, 3.0},
                         {0.0, 0.0, 0.0}, {5.0, 5.0, 1.0}, {-kInf, -kInf}, {10.0, 10.0});
  m.col_type = {ColType::Integer, ColType::Continuous, ColType::Binary};
  const Scaling sc = compute_scaling(m);
  CHECK_EQ(sc.col_scale[0], 1.0);
  CHECK_EQ(sc.col_scale[2], 1.0);
  const LpModel scaled = apply_scaling(m, sc);
  CHECK_EQ(scaled.col_lower[0], 0.0);
  CHECK_EQ(scaled.col_upper[0], 5.0);
  CHECK_EQ(scaled.col_upper[2], 1.0);
  CHECK(scaled.col_type == m.col_type);
}

TEST_CASE(scaling_transforms_bounds_costs_and_offset_consistently) {
  LpModel m = make_model(1, 2, {{0, 0, 8.0}, {0, 1, 0.25}}, {3.0, 0.0}, {-kInf, 1.0}, {4.0, kInf},
                         {-2.0}, {kInf}, Sense::Minimize, 5.0);
  Scaling sc;
  sc.row_scale = {0.5};
  sc.col_scale = {2.0, 4.0};
  sc.obj_scale = 0.25;
  const LpModel s = apply_scaling(m, sc);
  CHECK_EQ(s.A.value[0], 0.5 * 8.0 * 2.0);
  CHECK_EQ(s.A.value[1], 0.5 * 0.25 * 4.0);
  CHECK_EQ(s.col_cost[0], 0.25 * 2.0 * 3.0);
  CHECK_EQ(s.col_cost[1], 0.0);
  CHECK(is_neg_inf(s.col_lower[0]));  // infinity stays infinity
  CHECK_EQ(s.col_upper[0], 4.0 / 2.0);
  CHECK_EQ(s.col_lower[1], 1.0 / 4.0);
  CHECK(is_pos_inf(s.col_upper[1]));
  CHECK_EQ(s.row_lower[0], 0.5 * -2.0);
  CHECK(is_pos_inf(s.row_upper[0]));
  CHECK_EQ(s.objective_offset, 0.25 * 5.0);
}

TEST_CASE(scaling_objective_factor_is_skipped_for_zero_costs) {
  LpModel m = make_model(1, 2, {{0, 0, 100.0}, {0, 1, 0.01}}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0}, {0.0},
                         {1.0});
  ScalingReport rep;
  const Scaling sc = compute_scaling(m, {}, &rep);
  CHECK_EQ(sc.obj_scale, 1.0);
  CHECK(!rep.objective_scaled);

  m.col_cost = {1e6, 1e4};
  const Scaling sc2 = compute_scaling(m, {}, &rep);
  CHECK(is_power_of_two(sc2.obj_scale));
}

TEST_CASE(scaling_empty_and_degenerate_models_get_identity_factors) {
  LpModel none = make_model(0, 0, {}, {}, {}, {}, {}, {});
  const Scaling s0 = compute_scaling(none);
  CHECK(s0.row_scale.empty() && s0.col_scale.empty());
  CHECK_EQ(s0.obj_scale, 1.0);

  LpModel nomatrix = make_model(2, 2, {}, {1.0, 2.0}, {0.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0});
  const Scaling s1 = compute_scaling(nomatrix);
  for (const double f : s1.row_scale) CHECK_EQ(f, 1.0);
  for (const double f : s1.col_scale) CHECK_EQ(f, 1.0);
}

TEST_CASE(scaling_is_deterministic) {
  const RandomLp lp = make_random_lp(77, options_for(77));
  const Scaling a = compute_scaling(lp.model);
  const Scaling b = compute_scaling(lp.model);
  CHECK(a.row_scale == b.row_scale);
  CHECK(a.col_scale == b.col_scale);
  CHECK_EQ(a.obj_scale, b.obj_scale);
}

TEST_CASE(unscale_formulas_hold_for_known_optimal_pairs) {
  // Derivation check independent of any solver: map a known KKT pair into the
  // scaled space with the inverse formulas, require KKT of the scaled model,
  // then map back and require the original pair.
  for (std::uint64_t seed = 1; seed <= 120; ++seed) {
    const RandomLp lp = make_random_lp(seed, options_for(seed));
    ScalingReport rep;
    Scaling sc = compute_scaling(lp.model, {}, &rep);
    if (seed % 3 == 0) sc.obj_scale = 8.0;  // exercise a non-trivial objective factor
    const LpModel scaled = apply_scaling(lp.model, sc);
    const Solution sp = scale_solution(sc, lp.known);
    const KktReport k = check_kkt(scaled, sp, 1e-9);
    if (!k.ok) std::cerr << "  seed " << seed << ": " << k.summary() << "\n";
    CHECK(k.ok);

    const Solution back = unscale_solution(sc, sp);
    for (std::size_t j = 0; j < back.x.size(); ++j) CHECK_NEAR(back.x[j], lp.known.x[j], 1e-12 * (1 + std::fabs(lp.known.x[j])));
    for (std::size_t i = 0; i < back.y.size(); ++i) CHECK_NEAR(back.y[i], lp.known.y[i], 1e-12 * (1 + std::fabs(lp.known.y[i])));
    for (std::size_t j = 0; j < back.d.size(); ++j) CHECK_NEAR(back.d[j], lp.known.d[j], 1e-12 * (1 + std::fabs(lp.known.d[j])));
    CHECK_NEAR(back.objective, lp.known.objective, 1e-12 * (1 + std::fabs(lp.known.objective)));
  }
}

TEST_CASE(unscale_formulas_hold_for_maximization) {
  // max 3x + 2y, x + y <= 4, x <= 3 (via bound), y <= 3: optimum x=3, y=1, objective 11.
  LpModel m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {3.0, 2.0}, {0.0, 0.0}, {3.0, 3.0},
                         {-kInf}, {4.0}, Sense::Maximize);
  const RefLpResult r = solve_dense_lp(m);
  REQUIRE(r.status == Status::Optimal);
  CHECK_NEAR(r.solution.objective, 11.0, 1e-10);
  Scaling sc;
  sc.row_scale = {2.0};
  sc.col_scale = {0.5, 4.0};
  sc.obj_scale = 0.125;
  const LpModel s = apply_scaling(m, sc);
  CHECK(s.sense == Sense::Maximize);
  const RefLpResult rs = solve_dense_lp(s);
  REQUIRE(rs.status == Status::Optimal);
  const Solution back = unscale_solution(sc, rs.solution);
  const KktReport k = check_kkt(m, back, 1e-9);
  if (!k.ok) std::cerr << "  " << k.summary() << "\n";
  CHECK(k.ok);
  CHECK_NEAR(back.objective, 11.0, 1e-9);
}

TEST_CASE(scale_solve_unscale_passes_kkt_on_the_original_model) {
  int ok = 0;
  int inconclusive = 0;  // the dense oracle reported NumericalError: nothing to judge
  for (std::uint64_t seed = 1; seed <= 250; ++seed) {
    RandomLpOptions o = options_for(seed);
    o.singleton_rows = static_cast<int>(seed % 2);
    o.doubleton_eqs = static_cast<int>((seed / 2) % 2);
    const RandomLp lp = make_random_lp(seed, o);
    const Scaling sc = compute_scaling(lp.model);
    const LpModel scaled = apply_scaling(lp.model, sc);
    const RefLpResult r = solve_dense_lp(scaled);
    if (r.status == Status::NumericalError) {
      ++inconclusive;
      continue;
    }
    if (r.status != Status::Optimal) {
      std::cerr << "  seed " << seed << ": scaled solve status " << to_string(r.status) << "\n";
      CHECK(r.status == Status::Optimal);
      continue;
    }
    const Solution back = unscale_solution(sc, r.solution);
    const KktReport k = check_kkt(lp.model, back, 1e-6);
    if (!k.ok) std::cerr << "  seed " << seed << ": " << k.summary() << "\n";
    CHECK(k.ok);
    CHECK_NEAR(back.objective, lp.known.objective, 1e-6 * (1 + std::fabs(lp.known.objective)));
    ++ok;
  }
  CHECK_EQ(ok + inconclusive, 250);
  CHECK(inconclusive <= 2);
}
