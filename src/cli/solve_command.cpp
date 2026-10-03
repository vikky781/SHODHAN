#include "solve_command.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "info.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/mps.hpp"
#include "shodhan/rays.hpp"
#include "shodhan/status.hpp"

namespace shodhan::cli {

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 1;
constexpr int kExitNotImplemented = 2;
constexpr int kExitStatus = 3;

int usage_error(const std::string& message) {
  std::cerr << "error: " << message << "\n";
  std::cerr << "usage: shodhan solve <file> [--no-presolve] [--no-scaling] [--no-perturb] [--time-limit seconds]\n"
               "                     [--iter-limit n] [--write-sol path] [--verbose]\n";
  return kExitUsage;
}

bool parse_number(const std::string& s, double* out) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0' || !std::isfinite(v)) return false;
  *out = v;
  return true;
}

std::string num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.12g", v);
  return buf;
}

std::string sci(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2e", v);
  return buf;
}

bool has_integer_columns(const LpModel& m) {
  for (const ColType t : m.col_type) {
    if (t != ColType::Continuous) return true;
  }
  return false;
}

}  // namespace

int run_solve(const std::vector<std::string>& args) {
  std::string path, sol_path;
  LpOptions opt;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--no-presolve") {
      opt.presolve = false;
    } else if (a == "--no-scaling") {
      opt.scaling = false;
    } else if (a == "--no-perturb") {
      opt.perturb = false;
    } else if (a == "--verbose") {
      opt.params.verbosity = 2;
    } else if (a == "--time-limit" || a == "--iter-limit") {
      double v = 0.0;
      if (i + 1 >= args.size() || !parse_number(args[i + 1], &v) || v < 0.0) {
        return usage_error(a + " needs a non-negative number");
      }
      if (a == "--time-limit") {
        opt.params.time_limit = v;
      } else {
        if (v != std::floor(v)) return usage_error("--iter-limit needs an integer");
        opt.iteration_limit = static_cast<long long>(v);
      }
      ++i;
    } else if (a == "--write-sol") {
      if (i + 1 >= args.size()) return usage_error("--write-sol needs a file name");
      sol_path = args[++i];
    } else if (!a.empty() && a[0] == '-') {
      return usage_error("unknown option '" + a + "'");
    } else if (path.empty()) {
      path = a;
    } else {
      return usage_error("'solve' takes exactly one file argument");
    }
  }
  if (path.empty()) return usage_error("'solve' needs a file argument");

  const MpsReadResult read = read_mps_file(path);
  if (!read.ok) {
    std::cerr << "error: " << read.error << "\n";
    std::cerr << "status: " << to_string(Status::ReadError) << "\n";
    return kExitUsage;
  }
  const LpModel& model = read.model;
  print_model_summary(model, read.warnings, std::cout);
  std::cout << "\n";

  if (has_integer_columns(model)) {
    std::cout << "Status: " << to_string(Status::NotImplemented)
              << " (the model has integer columns; branch and bound is not implemented yet, no solution is produced)\n";
    return kExitNotImplemented;
  }

  if (opt.params.verbosity >= 2) opt.log = &std::cerr;
  const LpResult r = LpSolver(opt).solve(model);

  std::cout << "Status:        " << to_string(r.status) << "\n";
  if (r.status == Status::Optimal) {
    std::cout << "Objective:     " << num(r.solution.objective) << "\n";
  }
  std::cout << "Iterations:    " << r.iterations << " (dual phase 1: " << r.phase1_iterations << ", primal cleanup: " << r.primal_iterations
            << "), refactorizations: " << r.refactors << "\n";
  std::cout << "Perturbation:  " << (r.perturbation_used ? "used" : "not used") << "\n";
  if (r.presolve_ran) {
    std::cout << "Presolve:      rows " << r.presolve_stats.rows_before << " -> " << r.presolve_stats.rows_after << ", columns "
              << r.presolve_stats.cols_before << " -> " << r.presolve_stats.cols_after;
    if (!r.presolve_status_note.empty()) std::cout << " (" << r.presolve_status_note << ")";
    std::cout << "\n";
  } else {
    std::cout << "Presolve:      off\n";
  }
  std::cout << "Scaling:       " << (opt.scaling ? "on" : "off") << "\n";
  {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "Time:          %.3f s (presolve %.3f, scaling %.3f, simplex %.3f)\n", r.total_seconds, r.presolve_seconds,
                  r.scaling_seconds, r.simplex_seconds);
    std::cout << buf;
  }
  if (r.attempts > 1) {
    std::cout << "Attempts:      " << r.attempts
              << (r.message.empty() ? " (the presolve result was confirmed on the original model)" : " (fallbacks were needed; see the message)") << "\n";
  }
  if (r.status == Status::Optimal) {
    std::cout << "KKT check on the original model (tolerance " << sci(opt.kkt_tol) << "): " << (r.kkt.ok ? "passed" : "FAILED") << "\n";
    std::cout << "  primal infeasibility " << sci(r.kkt.primal_infeasibility_abs) << " abs, " << sci(r.kkt.primal_infeasibility_rel) << " rel\n";
    std::cout << "  dual infeasibility   " << sci(r.kkt.dual_infeasibility_abs) << " abs, " << sci(r.kkt.dual_infeasibility_rel) << " rel\n";
    std::cout << "  complementarity      " << sci(r.kkt.complementarity_abs) << " abs, " << sci(r.kkt.complementarity_rel) << " rel\n";
    std::cout << "  duality gap          " << sci(r.kkt.gap_abs) << " abs, " << sci(r.kkt.gap_rel) << " rel\n";
  } else if (r.status == Status::Infeasible) {
    const RayCheck rc = check_farkas(model, r.farkas_ray, 1e-9);
    std::cout << "Certificate:   Farkas multipliers verified: " << rc.message << "\n";
  } else if (r.status == Status::Unbounded) {
    const RayCheck rc = check_unbounded_ray(model, r.unbounded_ray, 1e-7);
    std::cout << "Certificate:   improving ray verified: " << rc.message << "\n";
  }
  if (!r.message.empty()) std::cout << "Message:       " << r.message << "\n";

  if (!sol_path.empty()) {
    if (r.status != Status::Optimal) {
      std::cout << "Nothing written: no optimal solution.\n";
    } else {
      std::ofstream out(sol_path, std::ios::binary);
      if (!out) {
        std::cerr << "error: cannot write '" << sol_path << "'\n";
        return kExitUsage;
      }
      out << "objective " << num(r.solution.objective) << "\n";
      for (Index j = 0; j < model.n_cols; ++j) {
        const double x = r.solution.x[to_size(j)];
        if (std::fabs(x) <= 1e-12) continue;
        const std::string name = model.col_names.empty() ? "C" + std::to_string(j + 1) : model.col_names[to_size(j)];
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.15g", x);
        out << name << " " << buf << "\n";
      }
      std::cout << "Wrote the solution to " << sol_path << "\n";
    }
  }
  return r.status == Status::Optimal ? kExitOk : kExitStatus;
}

}  // namespace shodhan::cli
