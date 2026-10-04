// Cutting planes: validity of every candidate cut (all separators) against every feasible integer point of
// seeded small MIPs, with presolve OFF; bound monotonicity; safety filters. See docs/CUTS.md.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/mip/cuts.hpp"
#include "support/cut_families.hpp"
#include "support/cut_harness.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::mip;
using namespace shodhan::testing;

namespace {

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == 0) return fallback;
  return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
}

MipOptions test_options() {
  MipOptions o;
  o.params.verbosity = 0;
  return o;
}

}  // namespace

TEST_CASE(cuts_are_valid_for_every_feasible_integer_point_presolve_off) {
  const std::uint64_t per_family = env_u64("SHODHAN_CUT_SEEDS", 90);
  const MipOptions opt = test_options();
  long long models = 0, ran = 0, fractional = 0, enumerated = 0, too_large = 0, with_cuts = 0;
  long long candidates[kNumSeparators] = {0};
  long long models_with[kNumSeparators] = {0};
  long long checks = 0, points = 0, inconclusive = 0, violations = 0;
  double worst = 0.0;
  long long fam_ran[kNumCutFamilies] = {0}, fam_frac[kNumCutFamilies] = {0};
  for (int f = 0; f < kNumCutFamilies; ++f) {
    for (std::uint64_t k = 1; k <= per_family; ++k) {
      const std::uint64_t seed = 70000ULL + static_cast<std::uint64_t>(f) * 1000ULL + k;
      const LpModel m = make_cut_instance(f, seed);
      ++models;
      REQUIRE(m.validate().empty());
      const CutRunResult r = run_cut_loop_on(m, opt, true, true);
      if (!r.ran) continue;
      ++ran;
      if (r.root_fractional) ++fractional;
      ++fam_ran[f];
      if (r.root_fractional) ++fam_frac[f];
      if (r.loop.candidates.empty()) continue;
      ++with_cuts;
      const ValidityResult v = check_cuts_valid(r.pm, r.loop.candidates);
      if (!v.enumerated) {
        ++too_large;
        continue;
      }
      ++enumerated;
      checks += v.checks;
      points += v.feasible_points;
      inconclusive += v.inconclusive;
      worst = std::max(worst, v.worst_violation);
      if (v.violated_cut >= 0) {
        ++violations;
        const Cut& c = r.loop.candidates[static_cast<std::size_t>(v.violated_cut)];
        std::cerr << "  INVALID CUT: family " << cut_family_name(f) << " seed " << seed << " separator "
                  << separator_name(c.separator) << " (" << v.detail << ") cut:";
        for (std::size_t q = 0; q < c.idx.size(); ++q) std::cerr << " " << c.val[q] << "*x" << c.idx[q];
        std::cerr << " <= " << c.rhs << "\n";
      }
      bool seen[kNumSeparators] = {false};
      for (const Cut& c : r.loop.candidates) {
        ++candidates[c.separator];
        seen[c.separator] = true;
      }
      for (int s = 0; s < kNumSeparators; ++s) models_with[s] += seen[s] ? 1 : 0;
    }
  }
  std::cout << "  cut validity (presolve off): " << models << " models, root LP optimal " << ran << ", root LP fractional "
            << fractional << " (" << (ran ? 100.0 * static_cast<double>(fractional) / static_cast<double>(ran) : 0.0)
            << "% of those solved), cuts found on " << with_cuts << ", enumerated " << enumerated << " (too large "
            << too_large << ")\n";
  std::cout << "  " << points << " feasible integer points, " << checks << " (cut, point) checks, " << inconclusive
            << " LP checks undecided, worst relative violation " << worst << "\n";
  for (int f = 0; f < kNumCutFamilies; ++f) {
    std::cout << "    family " << cut_family_name(f) << ": root LP solved " << fam_ran[f] << ", fractional " << fam_frac[f] << "\n";
  }
  const long long min_candidates = 50;
  for (int s = 0; s < kNumSeparators; ++s) {
    std::cout << "  " << separator_name(s) << ": " << candidates[s] << " candidate cuts checked on " << models_with[s] << " models";
    if (candidates[s] < min_candidates) std::cout << "  inconclusive for this separator (fewer than " << min_candidates << " cuts)";
    std::cout << "\n";
  }
  CHECK_EQ(violations, 0);
  CHECK(enumerated >= 500);
  CHECK(ran > 0 && 100 * fractional >= 60 * ran);
}

// ---------------------------------------------------------------------------------------------------------
// The validity checker itself: it must flag an invalid cut and accept valid ones.
TEST_CASE(cut_validity_checker_flags_an_invalid_cut) {
  // x0 + x1 >= 1 (set cover), binaries: the point (1, 0) is feasible.
  LpModel m;
  m.n_rows = 1;
  m.n_cols = 2;
  m.col_cost = {1.0, 1.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {1.0, 1.0};
  m.col_type = {ColType::Binary, ColType::Binary};
  m.row_lower = {1.0};
  m.row_upper = {kInf};
  std::string err;
  SparseMatrix::from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, &m.A, &err);
  Cut valid;
  valid.idx = {0, 1};
  valid.val = {1.0, 1.0};
  valid.rhs = 2.0;  // x0 + x1 <= 2
  Cut invalid;
  invalid.idx = {0, 1};
  invalid.val = {1.0, 1.0};
  invalid.rhs = 1.0 - 1e-3;  // x0 + x1 <= 0.999 cuts off (1, 0)
  const ValidityResult ok = check_cuts_valid(m, {valid});
  CHECK(ok.enumerated);
  CHECK_EQ(ok.violated_cut, -1);
  CHECK_EQ(ok.feasible_points, 3LL);
  const ValidityResult bad = check_cuts_valid(m, {valid, invalid});
  CHECK_EQ(bad.violated_cut, 1);
}

// ---------------------------------------------------------------------------------------------------------
// Safety filters.
TEST_CASE(cut_cleaning_relaxes_tiny_coefficients_with_the_bounds_and_rejects_bad_cuts) {
  MipOptions opt = test_options();
  const std::vector<double> lo = {0.0, 0.0, 0.0}, hi = {1.0, 1.0, 1.0}, x = {0.9, 0.9, 0.5};
  SeparatorStats ss;
  // 1.0 x0 + 1.0 x1 + 1e-12 x2 <= 1.5: the tiny term is dropped by relaxing the right-hand side with
  // min(g x2) = 0 (g > 0, lower bound 0), which keeps the cut valid.
  Cut c;
  c.idx = {0, 1, 2};
  c.val = {1.0, 1.0, 1e-12};
  c.rhs = 1.5;
  CHECK(clean_cut(c, lo, hi, x, opt, ss));
  CHECK_EQ(c.idx.size(), std::size_t{2});
  CHECK(c.rhs >= 1.5 / 1.0);  // relaxed, never tightened
  CHECK(c.efficacy > 0.0);
  // A tiny NEGATIVE coefficient needs the upper bound: with an infinite one the cut is rejected.
  const std::vector<double> hi_inf = {1.0, 1.0, kInf};
  Cut d;
  d.idx = {0, 1, 2};
  d.val = {1.0, 1.0, -1e-12};
  d.rhs = 1.5;
  CHECK(!clean_cut(d, lo, hi_inf, x, opt, ss));
  CHECK(ss.rejected_numerics >= 1);
  // Dynamism: coefficients 1 and 1e-7 are too far apart (the default limit is 1e6, 1e-7 > 1e-9 is kept as a term).
  Cut e;
  e.idx = {0, 1};
  e.val = {1.0, 1e-7};
  e.rhs = 0.5;
  opt.cut_max_dynamism = 1e5;
  SeparatorStats s2;
  CHECK(!clean_cut(e, lo, hi, x, opt, s2));
  CHECK_EQ(s2.rejected_numerics, 1LL);
  // Efficacy: a cut that the LP point satisfies is rejected.
  Cut f;
  f.idx = {0};
  f.val = {1.0};
  f.rhs = 5.0;
  SeparatorStats s3;
  CHECK(!clean_cut(f, lo, hi, x, test_options(), s3));
  CHECK_EQ(s3.rejected_efficacy, 1LL);
}

TEST_CASE(cut_selection_drops_parallel_cuts_and_keeps_the_most_efficacious) {
  MipOptions opt = test_options();
  std::vector<Cut> cands(3);
  for (Cut& c : cands) {
    c.idx = {0, 1};
    c.separator = kSepMir;
  }
  cands[0].val = {1.0, 1.0};
  cands[0].efficacy = 0.5;
  cands[1].val = {1.0, 1.01};  // almost parallel to cands[0], less efficacious
  cands[1].efficacy = 0.3;
  cands[2].val = {1.0, -1.0};  // orthogonal
  cands[2].efficacy = 0.1;
  CutStats stats;
  const std::vector<std::size_t> chosen = select_cuts(cands, 2, 10, opt, stats);
  CHECK_EQ(chosen.size(), std::size_t{2});
  CHECK_EQ(chosen[0], std::size_t{0});
  CHECK_EQ(chosen[1], std::size_t{2});
  CHECK_EQ(stats.sep[kSepMir].rejected_parallel, 1LL);
  // The limit applies.
  CutStats s2;
  CHECK_EQ(select_cuts(cands, 2, 1, opt, s2).size(), std::size_t{1});
}
