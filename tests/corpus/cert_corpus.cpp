// Generates a corpus of seeded LPs and MILPs, solves each with the full pipeline and writes, per model,
//   <dir>/<name>.mps          the model
//   <dir>/<name>.cert.json    its certificate
// plus <dir>/corpus.csv (name,family,status,attempts,configuration,expect). The Python side (KASAUTI) then
// verifies every certificate; `expect` is "pass" or "inconclusive" (a certificate that certifies nothing, such
// as an infeasibility proved by branching). Usage: shodhan_cert_corpus <dir> [lps_per_family] [mips_per_family]
// (defaults 60 and 15: 420 LPs, 6 special models and 150 MILPs).
//
// The model that is solved is the one READ BACK from the MPS file, so the certificate is about exactly the
// file whose SHA-256 it carries.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/certificate.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/mip/mip_solver.hpp"
#include "shodhan/mps.hpp"
#include "shodhan/sha256.hpp"
#include "shodhan/version.hpp"
#include "support/lp_families.hpp"
#include "support/mip_families.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

struct Entry {
  std::string name, family;
  LpModel model;
  bool mip = false;
  long long node_limit = 0;  // MILPs: 0 = no limit
};

LpModel special_model(int variant) {
  // Names with spaces (fixed-format MPS), a ranged row, a maximization and a free variable.
  LpModel m;
  m.name = "special " + std::to_string(variant);
  m.n_rows = 3;
  m.n_cols = 3;
  std::string err;
  SparseMatrix::from_triplets(3, 3, {{0, 0, 1.0}, {0, 1, 2.0}, {1, 1, 1.0}, {1, 2, -1.0}, {2, 0, 3.0}, {2, 2, 1.0}}, &m.A, &err);
  m.col_cost = {1.0 + variant, -1.0, 0.5};
  m.col_lower = {0.0, 0.0, -kInf};
  m.col_upper = {10.0, kInf, kInf};
  m.col_type.assign(3, ColType::Continuous);
  m.row_lower = {-kInf, 1.0, 2.0};
  m.row_upper = {14.0, 6.0, 20.0};
  m.col_names = {"COL A", "COL B", "C 3"};
  m.row_names = {"ROW 1", "ROW 2", "ROW 3"};
  if (variant % 2 == 1) m.sense = Sense::Maximize;
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: shodhan_cert_corpus <dir> [lps_per_family] [mips_per_family]\n";
    return 2;
  }
  const std::string dir = argv[1];
  const int per_family = argc > 2 ? std::atoi(argv[2]) : 60;
  const int mips_per_family = argc > 3 ? std::atoi(argv[3]) : 15;
  std::vector<Entry> entries;
  static const char* const keys[kNumFamilies] = {"degenerate", "free", "ranged", "boxed", "wide", "infeasible", "unbounded"};
  for (int f = 0; f < kNumFamilies; ++f) {
    for (int k = 1; k <= per_family; ++k) {
      const std::uint64_t seed = static_cast<std::uint64_t>(f) * 100000ULL + static_cast<std::uint64_t>(k);
      entries.push_back({std::string(keys[f]) + "_" + std::to_string(k), keys[f], make_family_instance(f, seed)});
    }
  }
  for (int v = 0; v < 6; ++v) entries.push_back({"special_" + std::to_string(v), "special", special_model(v)});
  for (int f = 0; f < kNumMipFamilies; ++f) {
    for (int k = 1; k <= mips_per_family; ++k) {
      const std::uint64_t seed = static_cast<std::uint64_t>(f) * 1000ULL + static_cast<std::uint64_t>(k);
      Entry e{std::string("mip_") + std::to_string(f) + "_" + std::to_string(k), std::string("mip ") + mip_family_name(f), make_mip_instance(f, seed), true, 0};
      if (k % 5 == 0) e.node_limit = 2;  // some runs stop early: a feasible certificate with a bound that is not tight
      entries.push_back(std::move(e));
    }
  }

  std::ofstream csv(dir + "/corpus.csv");
  csv << "name,family,status,attempts,configuration,expect\n";
  int failures = 0;
  for (Entry& e : entries) {
    const std::string mps_path = dir + "/" + e.name + ".mps";
    {
      std::ofstream out(mps_path, std::ios::binary);
      std::string error;
      if (!out || !write_mps(e.model, out, &error)) {
        std::cerr << "cannot write " << mps_path << ": " << error << "\n";
        return 2;
      }
    }
    const MpsReadResult read = read_mps_file(mps_path);
    if (!read.ok) {
      std::cerr << "cannot read back " << mps_path << ": " << read.error << "\n";
      return 2;
    }
    if (e.mip) {
      mip::MipOptions mo;
      mo.params.verbosity = 0;
      mo.params.time_limit = 30.0;
      if (e.node_limit > 0) mo.node_limit = e.node_limit;
      const mip::MipResult mr = mip::MipSolver(mo).solve(read.model);
      CertificateContext mctx;
      mctx.solver_version = kVersion;
      mctx.problem_name = read.model.name;
      mctx.options.params = mo.params;
      mctx.int_tol = mo.params.int_tol;
      mctx.mip_gap = mo.mip_gap;
      mctx.mip_abs_gap = mo.mip_abs_gap;
      if (!sha256_file_hex(mps_path, &mctx.file_sha256)) return 2;
      std::string merror;
      if (!write_mip_certificate_file(read.model, mctx, mr, dir + "/" + e.name + ".cert.json", &merror)) {
        std::cerr << merror << "\n";
        return 2;
      }
      const std::string mstatus = mip_certificate_status(mr);
      const bool pass = mstatus == "feasible" || (mstatus == "infeasible" && mr.lp_infeasible_certified);
      csv << e.name << "," << e.family << "," << mstatus << ",1,branch-and-bound," << (pass ? "pass" : "inconclusive") << "\n";
      continue;
    }
    const LpOptions options;
    const LpResult result = LpSolver(options).solve(read.model);
    CertificateContext ctx;
    ctx.solver_version = kVersion;
    ctx.problem_name = read.model.name;
    ctx.options = options;
    if (!sha256_file_hex(mps_path, &ctx.file_sha256)) return 2;
    std::string error;
    if (!write_certificate_file(read.model, ctx, result, dir + "/" + e.name + ".cert.json", &error)) {
      std::cerr << error << "\n";
      return 2;
    }
    csv << e.name << "," << e.family << "," << certificate_status(result) << "," << result.attempts << "," << result.configuration << ",pass\n";
    if (certificate_status(result) == "other") {
      ++failures;
      std::cerr << "no certificate for " << e.name << ": " << to_string(result.status) << " " << result.message << "\n";
    }
  }
  std::cout << "wrote " << entries.size() << " models and certificates to " << dir << " (" << failures << " LPs without evidence)\n";
  return 0;
}
