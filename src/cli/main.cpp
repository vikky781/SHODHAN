#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "factor_bench.hpp"
#include "info.hpp"
#include "presolve_report.hpp"
#include "shodhan/kkt.hpp"
#include "shodhan/mps.hpp"
#include "shodhan/presolve.hpp"
#include "shodhan/status.hpp"
#include "shodhan/version.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 1;
constexpr int kExitNotImplemented = 2;

void print_usage(std::ostream& out) {
  out << "SHODHAN " << shodhan::kVersion << " - a from-scratch LP/MILP/QP optimization solver core\n"
      << "\n"
      << "Usage:\n"
      << "  shodhan info <file>        print a summary of the model in an MPS file\n"
      << "  shodhan presolve <file> [--write-presolved out.mps] [--no-dual-needed] [--mip]\n"
      << "                             presolve the model and report what was reduced\n"
      << "  shodhan factor-bench <file> [--threshold u] [--max-updates k]\n"
      << "                             factorize a crash basis of the model and report LU\n"
      << "                             statistics, random solves and updates (developer\n"
      << "                             diagnostic, not a benchmark)\n"
      << "  shodhan solve <file>       presolve and scale the model, then solve it\n"
      << "                             (the solver itself is not implemented yet)\n"
      << "  shodhan --help             show this help\n"
      << "  shodhan --version          show the version\n"
      << "\n"
      << "Exit codes: 0 ok, 1 usage, read or write error, 2 not implemented.\n";
}

// Reads the model; on failure prints the error and returns false.
bool load(const std::string& path, shodhan::MpsReadResult* result) {
  *result = shodhan::read_mps_file(path);
  if (!result->ok) {
    std::cerr << "error: " << result->error << "\n";
    std::cerr << "status: " << shodhan::to_string(shodhan::Status::ReadError) << "\n";
    return false;
  }
  return true;
}

int usage_error(const std::string& message) {
  std::cerr << "error: " << message << "\n\n";
  print_usage(std::cerr);
  return kExitUsage;
}

int run_presolve(const std::vector<std::string>& args) {
  std::string path;
  std::string write_path;
  bool need_duals = true;
  bool force_mip = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--write-presolved") {
      if (i + 1 >= args.size()) return usage_error("--write-presolved needs a file name");
      write_path = args[++i];
    } else if (a == "--no-dual-needed") {
      need_duals = false;
    } else if (a == "--mip") {
      force_mip = true;
    } else if (!a.empty() && a[0] == '-') {
      return usage_error("unknown option '" + a + "'");
    } else if (path.empty()) {
      path = a;
    } else {
      return usage_error("'presolve' takes exactly one file argument");
    }
  }
  if (path.empty()) return usage_error("'presolve' needs a file argument");

  shodhan::MpsReadResult read;
  if (!load(path, &read)) return kExitUsage;

  shodhan::PresolveOptions opt = shodhan::cli::presolve_options_for(read.model, need_duals);
  if (force_mip) opt.is_mip = true;
  const shodhan::PresolveResult result = shodhan::presolve(read.model, opt);

  std::cout << "Model:          " << (read.model.name.empty() ? "(none)" : read.model.name) << "\n";
  std::cout << "Mode:           " << (opt.is_mip ? "MIP-safe reductions only" : "LP") << "\n";
  if (!opt.is_mip && !opt.need_duals) std::cout << "Duals:          not reconstructed (--no-dual-needed)\n";
  std::cout << "\n";
  shodhan::cli::print_presolve_report(result, std::cout);
  if (result.status == shodhan::PresolveStatus::Reduced) {
    std::cout << "\n";
    shodhan::cli::print_scaling_report(result.reduced, std::cout);
  }

  if (!write_path.empty()) {
    if (result.status != shodhan::PresolveStatus::Reduced &&
        result.status != shodhan::PresolveStatus::SolvedByPresolve) {
      std::cout << "\nNothing written: presolve concluded " << shodhan::to_string(result.status)
                << " and produced no reduced model.\n";
    } else {
      std::ofstream out(write_path, std::ios::binary);
      std::string error;
      if (!out || !shodhan::write_mps(result.reduced, out, &error)) {
        std::cerr << "error: cannot write '" << write_path << "'" << (error.empty() ? "" : ": " + error)
                  << "\n";
        return kExitUsage;
      }
      out.close();
      std::cout << "\nWrote the presolved model to " << write_path << "\n";
    }
  }
  return kExitOk;
}

int run_info_or_solve(const std::string& cmd, const std::vector<std::string>& args) {
  if (args.size() != 2) return usage_error("'" + cmd + "' takes exactly one file argument");
  shodhan::MpsReadResult read;
  if (!load(args[1], &read)) return kExitUsage;
  shodhan::cli::print_model_summary(read.model, read.warnings, std::cout);
  if (cmd == "info") return kExitOk;

  // ---- solve: presolve, scale, then (not yet) solve ----
  const shodhan::PresolveOptions opt = shodhan::cli::presolve_options_for(read.model, true);
  const shodhan::PresolveResult pre = shodhan::presolve(read.model, opt);
  std::cout << "\n";
  shodhan::cli::print_presolve_report(pre, std::cout);

  switch (pre.status) {
    case shodhan::PresolveStatus::Infeasible:
      std::cout << "\nStatus: Infeasible (proven by presolve)\n";
      return kExitOk;
    case shodhan::PresolveStatus::Unbounded:
      std::cout << "\nStatus: Unbounded (proven by presolve)\n";
      return kExitOk;
    case shodhan::PresolveStatus::InfeasibleOrUnbounded:
      std::cout << "\nStatus: InfeasibleOrUnbounded (presolve found an improving ray but could not "
                   "establish feasibility)\n";
      return kExitOk;
    case shodhan::PresolveStatus::SolvedByPresolve: {
      const shodhan::Solution sol = shodhan::postsolve(pre.stack, shodhan::Solution{});
      std::cout << "\nStatus: Optimal (solved by presolve)\n";
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.12g", sol.objective);
      std::cout << "Objective: " << buf << "\n";
      if (!opt.is_mip) {
        const shodhan::KktReport k = shodhan::check_kkt(read.model, sol, 1e-9);
        std::cout << "KKT check (tolerance 1e-9): " << (k.ok ? "passed" : "FAILED") << "\n";
      }
      return kExitOk;
    }
    case shodhan::PresolveStatus::Reduced:
      break;
  }
  std::cout << "\n";
  shodhan::cli::print_scaling_report(pre.reduced, std::cout);
  std::cout << "\nStatus: " << shodhan::to_string(shodhan::Status::NotImplemented)
            << " (the solver is not implemented yet; no solution is produced)\n";
  return kExitNotImplemented;
}

int run(const std::vector<std::string>& args) {
  if (args.empty()) {
    print_usage(std::cerr);
    return kExitUsage;
  }
  const std::string& cmd = args[0];
  if (cmd == "--help" || cmd == "-h" || cmd == "help") {
    print_usage(std::cout);
    return kExitOk;
  }
  if (cmd == "--version" || cmd == "-V") {
    std::cout << "shodhan " << shodhan::kVersion << "\n";
    return kExitOk;
  }
  if (cmd == "info" || cmd == "solve") return run_info_or_solve(cmd, args);
  if (cmd == "presolve") return run_presolve(args);
  if (cmd == "factor-bench") return shodhan::cli::run_factor_bench(args);
  return usage_error("unknown command '" + cmd + "'");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    return run(args);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return kExitUsage;
  }
}
