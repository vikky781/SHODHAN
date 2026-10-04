#include "solve_mip.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "cert_output.hpp"
#include "shodhan/mip/mip_solver.hpp"
#include "shodhan/status.hpp"

namespace shodhan::cli {

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 1;
constexpr int kExitNotImplemented = 2;
constexpr int kExitStatus = 3;
constexpr int kExitLimit = 4;

std::string num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.12g", v);
  return buf;
}

}  // namespace

int run_solve_mip(const LpModel& model, const std::string& model_path, const mip::MipOptions& options, const std::string& sol_path,
                  const std::string& cert_path) {
  const mip::MipResult r = mip::MipSolver(options).solve(model);

  std::cout << "Status:        " << to_string(r.status) << "\n";
  if (r.has_solution) std::cout << "Objective:     " << num(r.objective) << "\n";
  if (r.has_bound) std::cout << "Best bound:    " << num(r.best_bound) << (model.sense == Sense::Maximize ? " (upper bound)" : " (lower bound)") << "\n";
  if (r.has_solution && r.has_bound) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "Gap:           %.3g absolute, %.4g%% relative\n", r.abs_gap, 100.0 * r.rel_gap);
    std::cout << buf;
  }
  std::cout << "Nodes:         " << r.nodes_processed << " processed, " << r.nodes_open << " open, maximum depth " << r.max_depth << "\n";
  std::cout << "LP iterations: " << r.lp_iterations << " (strong branching: " << r.strong_branching_iterations << " in " << r.strong_branching_calls << " child solves)\n";
  if (r.presolve_cols_before > 0) {
    std::cout << "Presolve:      rows " << r.presolve_rows_before << " -> " << r.presolve_rows_after << ", columns " << r.presolve_cols_before << " -> "
              << r.presolve_cols_after << "\n";
  }
  {
    char buf[200];
    std::snprintf(buf, sizeof(buf), "Time:          %.3f s (presolve %.3f, root LP %.3f, node LPs %.3f, branching %.3f, heuristics %.3f)\n", r.seconds_total,
                  r.seconds_presolve, r.seconds_root_lp, r.seconds_node_lps, r.seconds_strong_branching, r.seconds_heuristics);
    std::cout << buf;
  }
  if (!r.heuristics.empty()) {
    std::cout << "Heuristics:    ";
    bool first = true;
    for (const mip::HeuristicStats& h : r.heuristics) {
      if (h.calls == 0 && h.successes == 0) continue;
      std::cout << (first ? "" : ", ") << h.name << " " << h.successes << " found/" << h.calls << " runs";
      first = false;
    }
    if (first) std::cout << "none ran";
    std::cout << "\n";
  }
  if (r.has_solution) {
    std::cout << "Incumbent:     found by '" << r.incumbent_source << "'; " << r.incumbents_found << " improving solution(s), " << r.rejections.total()
              << " candidate(s) rejected (not better " << r.rejections.not_better << ", infeasible " << r.rejections.row_violated + r.rejections.column_violated
              << ", other " << r.rejections.not_integral + r.rejections.lp_resolve_failed + r.rejections.wrong_size << ")\n";
  }
  if (r.numerical_trouble_nodes > 0) std::cout << "Numerical trouble in " << r.numerical_trouble_nodes << " node(s): the search is incomplete.\n";
  if (!r.message.empty()) std::cout << "Message:       " << r.message << "\n";
  if (r.status == Status::TimeLimit || r.status == Status::NodeLimit) {
    std::cout << "Stopped at the " << (r.status == Status::TimeLimit ? "time" : "node") << " limit; " << (r.has_solution ? "an incumbent exists (see Objective and Best bound)" : "no incumbent was found")
              << ".\n";
  }
  if (r.has_solution) {
    std::cout << "The solution was verified against the original model (rows, bounds, integrality); optimality is not certified by the written certificate.\n";
  }

  if (!sol_path.empty()) {
    if (!r.has_solution) {
      std::cout << "Nothing written: no solution.\n";
    } else {
      std::ofstream out(sol_path, std::ios::binary);
      if (!out) {
        std::cerr << "error: cannot write '" << sol_path << "'\n";
        return kExitUsage;
      }
      out << "objective " << num(r.objective) << "\n";
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
  if (!cert_path.empty() && !write_mip_certificate_output(model, model_path, options, r, cert_path)) return kExitUsage;

  switch (r.status) {
    case Status::Optimal: return kExitOk;
    case Status::TimeLimit:
    case Status::NodeLimit: return kExitLimit;
    case Status::NotImplemented: return kExitNotImplemented;
    default: return kExitStatus;
  }
}

}  // namespace shodhan::cli
