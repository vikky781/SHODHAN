#include "shodhan/lp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "shodhan/quadratic.hpp"
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
  DualBound bound;
  std::vector<double> farkas, ray, point;
  long long iterations = 0, phase1 = 0, primal = 0;
  int refactors = 0;
  bool perturbed = false;
  double t_presolve = 0.0, t_scaling = 0.0, t_simplex = 0.0;
  PresolveStats pstats;
  bool presolve_ran = false;
  std::string note;             // what presolve concluded
  std::string message;
};

// Final acceptance of an optimal answer on the ORIGINAL model: the KKT check, and, when the multipliers give a
// weak-duality bound (strictly, or tolerance-level after dropping multipliers below kDualZeroTol of their
// scale), that bound must support the objective. A strict bound of -infinity (float multipliers that are
// zero in theory but tiny in fact) does not reject the answer, and neither does the absence of any bound:
// KKT already limits the dual infeasibility, and the result is then recorded as not rigorous with no bound.
void accept_optimal(Attempt& a, const LpModel& model, double kkt_tol) {
  a.kkt = check_kkt(model, a.solution, kkt_tol);
  a.bound = compute_dual_bound(model, a.solution.x, a.solution.y, a.solution.objective, kkt_tol);
  const bool bound_vetoes = a.bound.finite && !a.bound.gap_ok;
  a.verified = a.kkt.ok && !bound_vetoes;
  a.status = a.verified ? Status::Optimal : Status::NumericalError;
  if (!a.kkt.ok) {
    a.message = "the solution failed the KKT check on the original model: " + a.kkt.summary();
  } else if (bound_vetoes) {
    a.message = "the dual bound of the multipliers misses the objective by a relative " + std::to_string(a.bound.gap_rel);
  }
}

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

// A feasible point of the model for an unbounded certificate. The point the primal simplex stops at is feasible
// only within the engine's scaled primal tolerance; unscaling can amplify that on rows with large scale factors
// beyond what the certificate check accepts. So the point is recomputed by a feasibility solve (zero costs) with a
// much tighter primal tolerance, warm-started from the basis at which the ray was found. Empty on failure.
std::vector<double> tight_feasible_point(const LpModel& model, SimplexOptions so, const BasisSnapshot* basis) {
  LpModel f = model;
  std::fill(f.col_cost.begin(), f.col_cost.end(), 0.0);
  f.objective_offset = 0.0;
  so.primal_tol = 1e-10;
  so.final_check = false;
  SimplexEngine e(f, so);
  if (basis != nullptr && !e.set_basis(*basis)) return {};
  if (e.solve() != EngineStatus::Optimal) return {};
  return std::vector<double>(e.primal_all().begin(), e.primal_all().begin() + model.n_cols);
}

// Cleans an unbounded ray: scales it to unit infinity norm and zeroes small entries. A ray from the simplex can
// have entries of size 1e10 next to rounding noise of size 1e-17 in components that are really zero, or small
// spurious components next to the real ones; on a row where such a component is the only term it breaks a
// recession-cone condition by a relative amount of 1. Thresholds from 1e-12 up to 1e-6 of the largest entry are
// tried; the first cleaned ray that passes the check at the stricter tolerance 1e-9 is used (it is then also an
// improving direction), otherwise the first that passes at 1e-7, otherwise the ray as it was.
std::vector<double> cleaned_ray(const LpModel& model, const std::vector<double>& ray) {
  double mx = 0.0;
  for (const double v : ray) mx = std::max(mx, std::fabs(v));
  if (!(mx > 0.0)) return ray;
  std::vector<double> loose;
  for (const double threshold : {1e-12, 1e-10, 1e-8, 1e-6}) {
    std::vector<double> c(ray.size());
    for (std::size_t j = 0; j < ray.size(); ++j) {
      const double v = ray[j] / mx;
      c[j] = std::fabs(v) <= threshold ? 0.0 : v;
    }
    if (check_unbounded_ray(model, c, 1e-9).ok) return c;
    if (loose.empty() && check_unbounded_ray(model, c, 1e-7).ok) loose = c;
  }
  return loose.empty() ? ray : loose;
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
        accept_optimal(a, model, opt.kkt_tol);
        a.note = "solved by presolve";
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
      accept_optimal(a, model, opt.kkt_tol);
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
      std::vector<double> pt = engine.unbounded_point();
      if (use_scaling) {
        for (std::size_t j = 0; j < r.size(); ++j) r[j] *= sc.col_scale[j];
        for (std::size_t j = 0; j < pt.size(); ++j) pt[j] *= sc.col_scale[j];
      }
      if (max_relative_violation(model, pt) > opt.kkt_tol) {
        // Candidates: a feasibility solve from the basis of the ray, and one from the slack basis (a different,
        // often better conditioned vertex). The most accurate one is kept.
        const BasisSnapshot snap = engine.get_basis_snapshot();
        for (const BasisSnapshot* start : {&snap, static_cast<const BasisSnapshot*>(nullptr)}) {
          std::vector<double> better = tight_feasible_point(*eng_model, so, start);
          if (better.empty()) continue;
          if (use_scaling) {
            for (std::size_t j = 0; j < better.size(); ++j) better[j] *= sc.col_scale[j];
          }
          if (max_relative_violation(model, better) < max_relative_violation(model, pt)) pt = better;
        }
      }
      r = cleaned_ray(model, r);
      a.verified = check_unbounded_ray(model, r, 1e-7).ok && max_relative_violation(model, pt) <= opt.kkt_tol;
      a.ray = r;
      a.point = pt;
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

// d = c + Q x - A^T y in the minimization form, from the original model: after a postsolve of a QP this is what the
// KKT check recomputes anyway, and the multipliers returned to the caller must agree with it.
void recompute_reduced_costs(const LpModel& model, Solution& s) {
  const double sgn = model.sense == Sense::Maximize ? -1.0 : 1.0;
  std::vector<double> aty(to_size(model.n_cols), 0.0), qx(to_size(model.n_cols), 0.0);
  model.A.multiply_transpose(s.y, aty);
  if (has_quadratic(model)) quad_multiply(model.quadratic, s.x, qx);
  s.d.assign(to_size(model.n_cols), 0.0);
  for (std::size_t j = 0; j < to_size(model.n_cols); ++j) s.d[j] = sgn * model.col_cost[j] + sgn * qx[j] - aty[j];
}

struct IpmAttempt {
  Attempt a;
  IpmResult ipm;
};

IpmAttempt run_ipm_attempt(const LpModel& model, const LpOptions& opt, bool use_presolve, bool use_scaling, double tight, double time_left) {
  IpmAttempt out;
  Attempt& a = out.a;
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
        return out;
      case PresolveStatus::Unbounded:
        a.status = Status::Unbounded;
        a.needs_confirm = true;
        a.note = "presolve: Unbounded";
        return out;
      case PresolveStatus::InfeasibleOrUnbounded:
        a.status = Status::InfeasibleOrUnbounded;
        a.needs_confirm = true;
        a.note = "presolve: InfeasibleOrUnbounded";
        return out;
      case PresolveStatus::SolvedByPresolve:
        a.solution = postsolve(pre.stack, Solution{});
        recompute_reduced_costs(model, a.solution);
        accept_optimal(a, model, opt.kkt_tol);
        a.note = "solved by presolve";
        return out;
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
  IpmOptions io;
  io.tol = opt.ipm_tol * tight;
  io.max_iterations = opt.ipm_max_iterations;
  io.time_limit = is_inf(opt.params.time_limit) ? kInf : std::max(time_left, 0.0);
  io.verbosity = opt.params.verbosity >= 2 ? 1 : 0;
  io.log = opt.log;
  const auto t0 = Clock::now();
  out.ipm = solve_ipm(*eng_model, io);
  a.t_simplex = seconds_since(t0);
  a.iterations = out.ipm.iterations;
  a.status = out.ipm.status;
  if (out.ipm.status == Status::Optimal) {
    Solution sol = out.ipm.solution;
    if (use_scaling) sol = unscale_solution(sc, sol);
    if (use_presolve) sol = postsolve(pre.stack, sol);
    recompute_reduced_costs(model, sol);
    sol.objective = model_objective(model, sol.x);
    a.solution = sol;
    accept_optimal(a, model, opt.kkt_tol);
  } else if (out.ipm.status == Status::NumericalError) {
    a.message = "interior point: " + out.ipm.message;
  }
  return out;
}

}  // namespace

const char* to_string(LpMethod method) noexcept {
  switch (method) {
    case LpMethod::Auto: return "auto";
    case LpMethod::Simplex: return "simplex";
    case LpMethod::Ipm: return "ipm";
    case LpMethod::IpmCrossover: return "ipm-crossover";
  }
  return "auto";
}

bool parse_method(const std::string& name, LpMethod* out) {
  for (const LpMethod m : {LpMethod::Auto, LpMethod::Simplex, LpMethod::Ipm, LpMethod::IpmCrossover}) {
    if (name == to_string(m)) {
      *out = m;
      return true;
    }
  }
  return false;
}

LpResult LpSolver::solve(const LpModel& model) const {
  const auto t_start = Clock::now();
  LpResult res;
  const std::vector<std::string> problems = model.validate();
  if (!problems.empty()) {
    res.status = Status::NumericalError;
    res.message = "invalid model: " + problems.front();
    return res;
  }

  const bool qp = has_quadratic(model);
  res.quadratic = qp;
  LpMethod method = options_.method;
  if (method == LpMethod::Auto) method = qp ? LpMethod::Ipm : LpMethod::Simplex;
  if (qp && method == LpMethod::Simplex) {
    res.status = Status::NotImplemented;
    res.message = "the simplex method cannot solve a quadratic program; use --method ipm (or auto)";
    return res;
  }
  if (qp) {
    const ConvexityReport cr = check_convexity(model);
    if (!cr.decided) {
      res.status = Status::NumericalError;
      res.message = "convexity of the quadratic term could not be decided: " + cr.note;
      return res;
    }
    if (!cr.convex) {
      res.status = Status::NonConvex;
      res.message = "the quadratic term is not positive semidefinite in the minimization form (" + cr.note + ")";
      return res;
    }
  }
  if (method == LpMethod::Ipm) return solve_with_ipm(model, t_start);

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
      res.dual_bound = a.bound;
      res.rigorous = a.bound.rigorous && a.status == Status::Optimal;
      res.farkas_ray = a.farkas;
      res.unbounded_ray = a.ray;
      res.unbounded_point = a.point;
      res.configuration = std::string(ladder[k].presolve ? "presolve" : "") + (ladder[k].presolve && ladder[k].scaling ? "+" : "") +
                          (ladder[k].scaling ? "scaling" : (ladder[k].presolve ? "" : "none")) + (ladder[k].tight < 1.0 ? "+tight" : "");
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
  res.method_used = "dual simplex";
  if (!res.farkas_ray.empty() || !res.unbounded_ray.empty() || res.status == Status::Optimal) {
    // certified: nothing to add
  } else if (res.status == Status::NumericalError) {
    if (message.empty()) message = "no attempt produced a verified result";
  }
  res.message = message;
  res.total_seconds = seconds_since(t_start);
  return res;
}

LpResult LpSolver::solve_with_ipm(const LpModel& model, std::chrono::steady_clock::time_point t_start) const {
  LpResult res;
  const bool qp = has_quadratic(model);
  res.quadratic = qp;
  res.method_used = "interior point";
  struct Config {
    bool presolve, scaling;
    double tight;
  };
  // The fallback ladder: the configuration as asked, then without presolve, then with a hundred times tighter tolerance,
  // then without scaling.
  std::vector<Config> ladder;
  ladder.push_back({options_.presolve, options_.scaling, 1.0});
  if (options_.presolve) ladder.push_back({false, options_.scaling, 1.0});
  ladder.push_back({false, options_.scaling, 0.01});
  if (options_.scaling) ladder.push_back({false, false, 0.01});
  std::string message;
  int tried = 0;
  res.status = Status::NumericalError;
  // The constraints of a QP alone are an LP feasibility problem: the dual simplex settles it with a Farkas certificate.
  auto check_constraints_with_simplex = [&](bool diverged) {
    LpModel feas = model;
    feas.quadratic = SparseMatrix();
    std::fill(feas.col_cost.begin(), feas.col_cost.end(), 0.0);
    feas.objective_offset = 0.0;
    feas.sense = Sense::Minimize;
    LpOptions so = options_;
    so.method = LpMethod::Simplex;
    const LpResult fr = LpSolver(so).solve(feas);
    if (fr.status == Status::Infeasible && !fr.farkas_ray.empty()) {
      res.status = Status::Infeasible;
      res.farkas_ray = fr.farkas_ray;
      message += (message.empty() ? "" : "; ") + std::string("the constraints are infeasible (Farkas multipliers verified)");
    } else if (diverged) {
      res.status = Status::InfeasibleOrUnbounded;
      message += (message.empty() ? "" : "; ") + std::string("the interior-point iterates diverged and the constraints are feasible: the QP is infeasible or unbounded (no certificate)");
    }
  };
  for (std::size_t k = 0; k < ladder.size(); ++k) {
    const double elapsed = seconds_since(t_start);
    if (!is_inf(options_.params.time_limit) && elapsed >= options_.params.time_limit && k > 0) {
      res.status = Status::TimeLimit;
      break;
    }
    IpmAttempt at = run_ipm_attempt(model, options_, ladder[k].presolve, ladder[k].scaling, ladder[k].tight, options_.params.time_limit - elapsed);
    const Attempt& a = at.a;
    ++tried;
    res.iterations += a.iterations;
    res.presolve_seconds += a.t_presolve;
    res.scaling_seconds += a.t_scaling;
    res.simplex_seconds += a.t_simplex;
    if (a.presolve_ran && !res.presolve_ran) {
      res.presolve_ran = true;
      res.presolve_stats = a.pstats;
    }
    if (!a.note.empty() && res.presolve_status_note.empty()) res.presolve_status_note = a.note;
    if (!a.message.empty()) message += (message.empty() ? "" : "; ") + a.message;
    res.ipm_nnz_l = at.ipm.nnz_l;
    res.ipm_regularizations += at.ipm.dynamic_regularizations;
    res.ipm_refinement_steps += at.ipm.refinement_steps;
    res.ipm_factorizations += at.ipm.factorizations;
    res.ipm_primal_residual = at.ipm.primal_residual;
    res.ipm_dual_residual = at.ipm.dual_residual;
    res.ipm_gap = at.ipm.gap;
    res.ipm_history = at.ipm.history;
    if (a.verified && a.status == Status::Optimal) {
      res.status = Status::Optimal;
      res.solution = a.solution;
      res.kkt = a.kkt;
      res.dual_bound = a.bound;
      res.rigorous = a.bound.rigorous;
      res.configuration = std::string(ladder[k].presolve ? "presolve" : "") + (ladder[k].presolve && ladder[k].scaling ? "+" : "") +
                          (ladder[k].scaling ? "scaling" : (ladder[k].presolve ? "" : "none")) + (ladder[k].tight < 1.0 ? "+tight" : "");
      break;
    }
    if (a.status == Status::TimeLimit || a.status == Status::IterationLimit) {
      res.status = a.status;
      break;
    }
    const bool divergence = a.status == Status::InfeasibleOrUnbounded && !a.needs_confirm;
    if (divergence) {
      // The interior-point method has no certificates. An LP is resolved by the dual simplex (which has them); a QP gets a
      // feasibility check of its constraints with the dual simplex.
      res.attempts = tried;
      if (!qp) {
        LpOptions so = options_;
        so.method = LpMethod::Simplex;
        LpResult sr = LpSolver(so).solve(model);
        sr.iterations += res.iterations;
        sr.attempts += tried;
        sr.method_used = "dual simplex (after the interior-point method diverged)";
        sr.message = "the interior-point iterates diverged; resolved by the dual simplex" + (sr.message.empty() ? std::string() : "; " + sr.message);
        return sr;
      }
      check_constraints_with_simplex(true);
      res.message = message;
      res.total_seconds = seconds_since(t_start);
      return res;
    }
    if (options_.log != nullptr && k + 1 < ladder.size()) {
      *options_.log << "note: interior-point attempt " << (k + 1) << " gave " << to_string(a.status) << "; trying a fallback\n";
    }
    res.status = Status::NumericalError;
  }
  res.attempts = tried;
  if (res.status == Status::NumericalError && qp) check_constraints_with_simplex(false);  // an infeasible QP may only make the IPM fail
  if (res.status == Status::NumericalError && message.empty()) message = "no attempt produced a verified result";
  res.message = message;
  res.total_seconds = seconds_since(t_start);
  return res;
}

}  // namespace shodhan
