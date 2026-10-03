// Opt-in stress harness for the LP pipeline (build with -DSHODHAN_BUILD_STRESS=ON; it is NOT part of
// the default ctest run).
//
//   shodhan_stress <family> <first_seed> <last_seed> [--emit-mps DIR] [--quiet]
//
// family: degenerate, free, ranged, boxed, wide, infeasible, unbounded, all
//
// For every seed the LP of that family is solved by the full pipeline and compared with the dense
// oracle (statuses, objective within 1e-6 relative, KKT on the original model, certificates checked).
// Mismatches are printed with their seeds. An objective that differs while both points pass the
// KKT check is counted as "hypersensitive" when sum |y| * 1e-9 exceeds the comparison tolerance (see
// docs/SIMPLEX.md); the harness prints those seeds too, so that they can be adjudicated with KASAUTI.
//
// --emit-mps DIR writes the model of each seed as DIR/<family>_<seed>.mps instead of solving it
// (to regenerate an instance for a certificate, for example).
//
// Exit status: 0 when no mismatch was found, 1 otherwise, 2 for usage errors.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/mps.hpp"
#include "shodhan/rays.hpp"
#include "support/dense_ref_lp.hpp"
#include "support/lp_families.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

struct Tally {
  long long total = 0, agree = 0, inconclusive = 0, hypersensitive = 0, mismatches = 0;
  long long optimal = 0, infeasible = 0, unbounded = 0;
};

int usage() {
  std::cerr << "usage: shodhan_stress <family|all> <first_seed> <last_seed> [--emit-mps DIR] [--quiet]\n"
               "families: degenerate free ranged boxed wide infeasible unbounded\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) return usage();
  const std::string fam_arg = argv[1];
  const long long first = std::atoll(argv[2]);
  const long long last = std::atoll(argv[3]);
  std::string emit_dir;
  bool quiet = false;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--emit-mps" && i + 1 < argc) emit_dir = argv[++i];
    else if (a == "--quiet") quiet = true;
    else return usage();
  }
  if (first > last || first < 0) return usage();
  std::vector<int> families;
  if (fam_arg == "all") {
    for (int f = 0; f < kNumFamilies; ++f) families.push_back(f);
  } else {
    const int f = family_from_name(fam_arg.c_str());
    if (f < 0) return usage();
    families.push_back(f);
  }

  int exit_code = 0;
  for (const int f : families) {
    Tally t;
    for (long long seed = first; seed <= last; ++seed) {
      const LpModel model = make_family_instance(f, static_cast<std::uint64_t>(seed));
      if (!emit_dir.empty()) {
        const std::string path = emit_dir + "/" + fam_arg + "_" + std::to_string(seed) + ".mps";
        std::ofstream out(path, std::ios::binary);
        std::string error;
        if (!out || !write_mps(model, out, &error)) {
          std::cerr << "cannot write " << path << " " << error << "\n";
          return 2;
        }
        ++t.total;
        continue;
      }
      ++t.total;
      const RefLpResult ref = solve_dense_lp(model);
      if (ref.status == Status::NumericalError) {
        ++t.inconclusive;
        continue;
      }
      const LpResult r = LpSolver().solve(model);
      std::string why;
      bool sensitive = false;
      if (r.status != ref.status) {
        why = std::string("status ") + to_string(r.status) + ", oracle " + to_string(ref.status) + " [" + r.message + "]";
      } else if (r.status == Status::Optimal) {
        ++t.optimal;
        const double rel = std::fabs(r.solution.objective - ref.solution.objective) / (1.0 + std::fabs(ref.solution.objective));
        const KktReport kr = check_kkt(model, r.solution, 1e-6);
        double ysum = 0.0;
        for (const double yv : r.solution.y) ysum += std::fabs(yv);
        sensitive = ysum * 1e-9 > 1e-6 * (1.0 + std::fabs(ref.solution.objective));
        if (!kr.ok) why = "KKT on the original model failed: " + kr.summary();
        else if (!(rel <= 1e-6) && !sensitive) why = "objective " + std::to_string(r.solution.objective) + " vs oracle " + std::to_string(ref.solution.objective);
        else if (!(rel <= 1e-6)) {
          ++t.hypersensitive;
          std::cout << "HYPERSENSITIVE " << fam_arg << " seed " << seed << ": pipeline " << std::setprecision(15) << r.solution.objective << ", oracle "
                    << ref.solution.objective << ", sum|y| " << ysum << "\n";
        }
      } else if (r.status == Status::Infeasible) {
        ++t.infeasible;
        if (!check_farkas(model, r.farkas_ray, 1e-9).ok) why = "Farkas certificate does not check";
      } else if (r.status == Status::Unbounded) {
        ++t.unbounded;
        if (!check_unbounded_ray(model, r.unbounded_ray, 1e-7).ok) why = "unbounded ray does not check";
      }
      if (why.empty()) {
        ++t.agree;
      } else {
        ++t.mismatches;
        exit_code = 1;
        std::cout << "MISMATCH " << fam_arg << " seed " << seed << " (" << family_name(f) << ", m = " << model.n_rows << ", n = " << model.n_cols << "): " << why << "\n";
      }
    }
    if (emit_dir.empty() || !quiet) {
      std::cout << family_name(f) << ": " << t.total << " seeds, " << t.agree << " agree (" << t.optimal << " optimal, " << t.infeasible << " infeasible, " << t.unbounded
                << " unbounded), " << t.inconclusive << " oracle-inconclusive, " << t.hypersensitive << " hypersensitive, " << t.mismatches << " mismatches\n";
    }
  }
  return exit_code;
}
