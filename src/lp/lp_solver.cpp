#include "shodhan/lp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "shodhan/rays.hpp"
#include "shodhan/scaling.hpp"
#include "shodhan/simplex_engine.hpp"

namespace shodhan {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// Outcome of one pass through the pipeline.
struct Attempt {
  Status status = Status::NumericalError;
  bool verified = false;        // Optimal: KKT on the original passed; Infeasible/Unbounded: certificate passed
  bool needs_confirm = false;   // a status that cannot be certified in this configuration (presolve involved)
  Solution solution;
  KktReport kkt;
  std::vector<double> farkas, ray;
  long long iterations = 0, phase1 = 0, primal = 0;
  int refactors = 0;
  bool perturbed = false;
  double t_presolve = 0.0, t_scaling = 0.0, t_simplex = 0.0;
  PresolveStats pstats;
  bool presolve_ran = false;
  std::string note;             // what presolve concluded
  std::string message;
};

Status to_lp_status(EngineStatus s) {
  switch (s) {
    case EngineStatus::Optimal: return Status::Optimal;
    case EngineStatus::Infeasible: return Status::Infeasible;
    case EngineStatus::Unbounded: return Status::Unbounded;
    case EngineStatus::IterationLimit: return Status::IterationLimit;
    case EngineStatus::TimeLimit: return Status::TimeLimit;
    case EngineStatus::NumericalError: return Status::NumericalError;
  }
  return Status::NumericalError;
}

// `tight` multiplies the tolerances (1 = as given, 0.01 = a hundred times tighter).
Attempt run_attempt(const LpModel& model, const LpOptions& opt, bool use_presolve, bool use_scaling, double tight,
                    double time_left) {
  Attempt a;
  const LpModel* work = &model;
  PresolveResult pre;
  if (use_presolve) {
    const auto t0 = Clock::now();
    PresolveOptions po;
    po.need_duals = true;
    pre = presolve(model, po);
    a.t_presolve = seconds_since(t0);
    a.pstats = pre.stats;
    a.presolve_ran = true;
    switch (pre.status) {
      case PresolveStatus::Infeasible:
        a.status = Status::Infeasible;
        a.needs_confirm = true;
        a.note = "presolve: Infeasible" + (pre.note.empty() ? std::string() : " (" + pre.note + ")");
        return a;
      case PresolveStatus::Unbounded:
        a.status = Status::Unbounded;
        a.needs_confirm = true;
        a.note = "presolve: Unbounded";
        return a;
      case PresolveStatus::InfeasibleOrUnbounded:
        a.status = Status::InfeasibleOrUnbounded;
        a.needs_confirm = true;
        a.note = "presolve: InfeasibleOrUnbounded";
        return a;
      case PresolveStatus::SolvedByPresolve:
        a.solution = postsolve(pre.stack, Solution{});
        a.kkt = check_kkt(model, a.solution, opt.kkt_tol);
        a.status = a.kkt.ok ? Status::Optimal : Status::NumericalError;
        a.verified = a.kkt.ok;
        a.note = "solved by presolve";
        if (!a.kkt.ok) a.message = "presolve solved the model but its solution failed the KKT check: " + a.kkt.summary();
        return a;
      case PresolveStatus::Reduced:
        work = &pre.reduced;
        break;
    }
  }

  Scaling sc;
  LpModel scaled_storage;
  const LpModel* eng_model = work;
  if (use_scaling) {
    const auto t0 = Clock::now();
    sc = compute_scaling(*work);
    scaled_storage = apply_scaling(*work, sc);
    eng_model = &scaled_storage;
    a.t_scaling = seconds_since(t0);
  }

  SimplexOptions so;
  so.primal_tol = opt.params.primal_tol * tight;
  so.dual_tol = opt.params.dual_tol * tight;
  so.time_limit = is_inf(opt.params.time_limit) ? kInf : std::max(time_left, 0.0);
  so.iteration_limit = opt.iteration_limit;
  so.perturb = opt.perturb;
  so.seed = opt.params.seed;
  so.verbosity = opt.params.verbosity >= 2 ? 1 : 0;
  so.log = opt.log;
  so.final_tol = opt.kkt_tol;
  const auto t0 = Clock::now();
  SimplexEngine engine(*eng_model, so);
  const EngineStatus est = engine.solve();
  a.t_simplex = seconds_since(t0);
  const SimplexStats& st = engine.stats();
  a.iterations = st.iterations;
  a.phase1 = st.phase1_iterations;
  a.primal = st.primal_iterations;
  a.refactors = st.refactors;
  a.perturbed = st.perturbed;
  a.status = to_lp_status(est);

  switch (est) {
    case EngineStatus::Optimal: {
      Solution s = engine.solution();
      if (use_scaling) s = unscale_solution(sc, s);
      if (use_presolve) s = postsolve(pre.stack, s);
      a.solution = s;
      a.kkt = check_kkt(model, a.solution, opt.kkt_tol);
      a.verified = a.kkt.ok;
      if (!a.kkt.ok) {
        a.status = Status::NumericalError;
        a.message = "the solution failed the KKT check on the original model: " + a.kkt.summary();
      }
      break;
    }
    case EngineStatus::Infeasible: {
      if (use_presolve) {  // the certificate lives in the reduced model: confirm without presolve
        a.needs_confirm = true;
        break;
      }
      std::vector<double> y = engine.farkas_ray();
      if (use_scaling) {
        for (std::size_t i = 0; i < y.size(); ++i) y[i] *= sc.row_scale[i];
      }
      a.verified = check_farkas(model, y, 1e-9).ok;
      a.farkas = y;
      if (!a.verified) {
        a.status = Status::NumericalError;
        a.message = "the infeasibility certificate failed the check on the original model";
      }
      break;
    }
    case EngineStatus::Unbounded: {
      if (use_presolve) {
        a.needs_confirm = true;
        break;
      }
      std::vector<double> r = engine.unbounded_ray();
      if (use_scaling) {
        for (std::size_t j = 0; j < r.size(); ++j) r[j] *= sc.col_scale[j];
      }
      a.verified = check_unbounded_ray(model, r, 1e-7).ok;
      a.ray = r;
      if (!a.verified) {
        a.status = Status::NumericalError;
        a.message = "the unbounded ray failed the check on the original model";
      }
      break;
    }
    default:
      break;
  }
  return a;
}

}  // namespace

LpResult LpSolver::solve(const LpModel& model) const {
  const auto t_start = Clock::now();
  LpResult res;
  const std::vector<std::string> problems = model.validate();
  if (!problems.empty()) {
    res.status = Status::NumericalError;
    res.message = "invalid model: " + problems.front();
    return res;
  }

  struct Config {
    bool presolve, scaling;
    double tight;
  };
  std::vector<Config> ladder;
  ladder.push_back({options_.presolve, options_.scaling, 1.0});
  // Fallbacks, tried in order when the answer could not be certified.
  if (options_.presolve) ladder.push_back({false, options_.scaling, 1.0});
  ladder.push_back({false, options_.scaling, 0.01});
  if (options_.scaling) ladder.push_back({false, false, 0.01});

  std::string message;
  Attempt last;
  Attempt first;
  bool have_first = false;
  int tried = 0;
  for (std::size_t k = 0; k < ladder.size(); ++k) {
    const double elapsed = seconds_since(t_start);
    if (!is_inf(options_.params.time_limit) && elapsed >= options_.params.time_limit && k > 0) {
      res.status = Status::TimeLimit;
      break;
    }
    Attempt a = run_attempt(model, options_, ladder[k].presolve, ladder[k].scaling, ladder[k].tight,
                            options_.params.time_limit - elapsed);
    ++tried;
    res.iterations += a.iterations;
    res.phase1_iterations += a.phase1;
    res.primal_iterations += a.primal;
    res.refactors += a.refactors;
    res.perturbation_used = res.perturbation_used || a.perturbed;
    res.presolve_seconds += a.t_presolve;
    res.scaling_seconds += a.t_scaling;
    res.simplex_seconds += a.t_simplex;
    if (a.presolve_ran && !res.presolve_ran) {
      res.presolve_ran = true;
      res.presolve_stats = a.pstats;
    }
    if (!a.note.empty() && res.presolve_status_note.empty()) res.presolve_status_note = a.note;
    if (!a.message.empty()) message += (message.empty() ? "" : "; ") + a.message;
    if (!have_first) {
      first = a;
      have_first = true;
    }
    last = a;

    if (a.verified) {
      // A certified answer. (If an earlier attempt claimed something else without a certificate,
      // that claim is not trusted: say so.)
      if (k > 0 && have_first && !first.needs_confirm && first.status != a.status && first.status != Status::NumericalError) {
        message += (message.empty() ? "" : "; ") + std::string("the first attempt reported ") + to_string(first.status) +
                   " but it could not be certified; the certified result of a fallback is reported";
      }
      res.status = a.status;
      res.solution = a.solution;
      res.kkt = a.kkt;
      res.farkas_ray = a.farkas;
      res.unbounded_ray = a.ray;
      break;
    }
    if (a.status == Status::TimeLimit || a.status == Status::IterationLimit) {
      res.status = a.status;
      break;
    }
    if (options_.log != nullptr && k + 1 < ladder.size()) {
      *options_.log << "note: attempt " << (k + 1) << " (" << (ladder[k].presolve ? "presolve, " : "") << (ladder[k].scaling ? "scaling" : "no scaling")
                    << ") gave " << to_string(a.status) << (a.verified ? "" : " without a certificate") << "; trying a fallback\n";
    }
    res.status = Status::NumericalError;
  }
  res.attempts = tried;
  if (!res.farkas_ray.empty() || !res.unbounded_ray.empty() || res.status == Status::Optimal) {
    // certified: nothing to add
  } else if (res.status == Status::NumericalError) {
    if (message.empty()) message = "no attempt produced a verified result";
  }
  res.message = message;
  res.total_seconds = seconds_since(t_start);
  return res;
}

}  // namespace shodhan
