// The synthetic refinery models written by bench/refinery/generate.py (SYNTHETIC: structure follows textbook formulations;
// not plant or MRPL data): the C++ reader parses and validates every file, a read-write-read round trip is stable, the
// tight and weak big-M variants of a model have the same optimum, and the tiny scheduling, blending and facility models
// (at most 14 binaries) agree with the brute-force oracle. The directory comes from SHODHAN_REFINERY_DIR (written by the
// ctest fixture `refinery_generate`); without it the tests print SKIPPED and pass.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "shodhan/mip/mip_solver.hpp"
#include "shodhan/mps.hpp"
#include "support/mip_oracle.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;
namespace fs = std::filesystem;

namespace {

std::vector<fs::path> refinery_files() {
  std::vector<fs::path> out;
  const char* dir = std::getenv("SHODHAN_REFINERY_DIR");
  if (dir == nullptr || *dir == 0) return out;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() == ".mps") out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string written(const LpModel& m) {
  std::ostringstream os;
  std::string err;
  const bool ok = write_mps(m, os, &err);
  if (!ok) std::cout << "  write_mps failed: " << err << "\n";
  return ok ? os.str() : std::string();
}

// Reads a file; on failure the message is printed and `ok` is false (REQUIRE cannot be used in a function with a value).
LpModel load(const fs::path& p, bool* ok) {
  const MpsReadResult r = read_mps_file(p.string());
  if (!r.ok) std::cout << "  " << p.string() << ": " << r.error << "\n";
  *ok = r.ok;
  return r.model;
}

}  // namespace

TEST_CASE(refinery_models_parse_validate_and_round_trip) {
  const std::vector<fs::path> files = refinery_files();
  if (files.empty()) {
    std::cout << "  SKIPPED: SHODHAN_REFINERY_DIR is not set or empty\n";
    return;
  }
  long long rows = 0, cols = 0, ints = 0;
  for (const fs::path& p : files) {
    bool read_ok = false;
    const LpModel m = load(p, &read_ok);
    REQUIRE(read_ok);
    CHECK(m.validate().empty());
    const std::string text = written(m);
    CHECK(!text.empty());
    const MpsReadResult again = read_mps_string(text, "round trip");
    CHECK(again.ok);
    if (again.ok) {
      CHECK(again.model == m);
      CHECK_EQ(written(again.model), text);
    }
    rows += m.n_rows;
    cols += m.n_cols;
    for (Index j = 0; j < m.n_cols; ++j) ints += m.is_integer(j) ? 1 : 0;
  }
  std::cout << "  refinery models: " << files.size() << " files parsed, validated and round-tripped (" << rows << " rows, " << cols << " columns, " << ints
            << " integer columns in total)\n";
}

TEST_CASE(refinery_tiny_models_agree_with_the_brute_force_oracle_and_loose_big_m_does_not_change_the_optimum) {
  const std::vector<fs::path> files = refinery_files();
  if (files.empty()) {
    std::cout << "  SKIPPED: SHODHAN_REFINERY_DIR is not set or empty\n";
    return;
  }
  int solved = 0, pairs = 0;
  for (const fs::path& p : files) {
    const std::string name = p.stem().string();
    const bool continuous_family = name.size() > 1 && name[0] == 'R' && (name[1] == '1' || name[1] == '2');
    if (name.find("_tiny_") == std::string::npos || continuous_family) continue;
    bool read_ok = false;
    const LpModel m = load(p, &read_ok);
    REQUIRE(read_ok);
    Index binaries = 0;
    for (Index j = 0; j < m.n_cols; ++j) binaries += m.is_integer(j) ? 1 : 0;
    REQUIRE(binaries <= 14);
    const MipRefResult truth = solve_mip_brute_force(m);
    REQUIRE(!truth.too_large);
    REQUIRE(truth.status == Status::Optimal);
    mip::MipOptions mo;
    mo.mip_gap = 1e-9;
    mo.mip_abs_gap = 1e-9;
    mo.params.verbosity = 0;
    const mip::MipResult r = mip::MipSolver(mo).solve(m);
    const bool ok = r.status == Status::Optimal && std::fabs(r.objective - truth.objective) <= 1e-6 * (1.0 + std::fabs(truth.objective));
    if (!ok) std::cout << "  " << name << ": SHODHAN " << to_string(r.status) << " " << r.objective << ", brute force " << truth.objective << "\n";
    CHECK(ok);
    ++solved;
    // the weak variant of a tight model has the same feasible set, hence the same optimum
    if (name.size() < 5 || name.substr(name.size() - 5) != "_weak") {
      const fs::path weak = p.parent_path() / (name + "_weak.mps");
      if (fs::exists(weak)) {
        bool weak_ok = false;
        const LpModel wm = load(weak, &weak_ok);
        REQUIRE(weak_ok);
        const MipRefResult wt = solve_mip_brute_force(wm);
        REQUIRE(wt.status == Status::Optimal);
        const bool same = std::fabs(wt.objective - truth.objective) <= 1e-7 * (1.0 + std::fabs(truth.objective));
        if (!same) std::cout << "  " << name << ": weak big-M optimum " << wt.objective << " vs " << truth.objective << "\n";
        CHECK(same);
        ++pairs;
      }
    }
  }
  std::cout << "  refinery tiny models: " << solved << " solved by SHODHAN and equal to the brute-force optimum, " << pairs << " tight/weak pairs with equal optimum\n";
}
