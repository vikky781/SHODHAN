#include <algorithm>
#include <cmath>
#include <cstdint>

#include "shodhan/kkt.hpp"
#include "shodhan/rays.hpp"
#include "shodhan/simplex_engine.hpp"

namespace shodhan {

namespace {

std::uint64_t splitmix(std::uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

}  // namespace

// Cost perturbation against dual degeneracy: the cost of every nonbasic structural variable moves
// away from dual infeasibility by xi_j = scale * (1 + |c_j|) * (1 + u_j), u_j uniform in [0, 1)
// (deterministic, from the seed and the number of perturbations so far). At a lower bound the cost
// grows, at an upper bound it shrinks; free and fixed variables are left alone. d_j moves by exactly
// the change, so no recomputation is needed.
void SimplexEngine::perturb_costs(double multiplier) {
  std::uint64_t state = opt_.seed * 0x2545f4914f6cdd1dULL + static_cast<std::uint64_t>(stats_.perturbation_rounds) * 0x9e3779b9ULL + 12345ULL;
  const double base = opt_.perturb_scale * multiplier;
  for (Index j = 0; j < n_; ++j) {
    const VarStatus st = status_[to_size(j)];
    if (st != VarStatus::AtLower && st != VarStatus::AtUpper) continue;
    const double u = static_cast<double>(splitmix(state) >> 11) / 9007199254740992.0;
    const double xi = base * (1.0 + std::fabs(cost_orig_[to_size(j)])) * (1.0 + u);
    const double change = st == VarStatus::AtLower ? xi : -xi;
    cost_[to_size(j)] += change;
    d_[to_size(j)] += change;
  }
  costs_modified_ = true;
  stats_.perturbed = true;
  ++stats_.perturbation_rounds;
}

// After the dual simplex ended optimal for the working costs: restore the original costs and make
// sure the point is optimal for them. Dual infeasibilities that appear are removed by the primal
// simplex (the basis is primal feasible); the loop is bounded.
EngineStatus SimplexEngine::finish_after_dual() {
  bool polished = false;
  for (int round = 0; round <= opt_.max_cleanup_rounds; ++round) {
    if (costs_modified_) {
      cost_ = cost_orig_;
      costs_modified_ = false;
    }
    if (!refactor()) return EngineStatus::NumericalError;
    compute_primal();
    compute_dual();
    const InfeasibilitySummary s = infeasibility();
    if (s.dual_count == 0 && s.primal_count == 0) {
      if (!opt_.polish || polished) return EngineStatus::Optimal;
      // Polishing. The point satisfies the tolerances, but violations below them can still be
      // visible after unscaling and, when the multipliers are large, change the objective. Tighten
      // both tolerances, let the dual simplex (primal violations) or the primal simplex (dual
      // violations) finish the vertex, with an iteration cap so that noise cannot make it loop, and
      // let the next round re-establish and check the normal tolerances.
      polished = true;
      const double saved_p = opt_.primal_tol, saved_d = opt_.dual_tol;
      const long long saved_limit = opt_.iteration_limit;
      opt_.primal_tol = std::min(saved_p, opt_.polish_tol);
      opt_.dual_tol = std::min(saved_d, opt_.polish_tol);
      opt_.iteration_limit = std::min(saved_limit, stats_.iterations + 100 + 2 * static_cast<long long>(m_));
      const InfeasibilitySummary tight = infeasibility();
      if (tight.primal_count > 0 || tight.dual_count > 0) {
        if (tight.primal_count == 0) {
          run_primal_simplex();
        } else {
          fix_dual_infeasibilities(true);
          if (primal_stale_) compute_primal();
          run_dual_simplex();
        }
      }
      opt_.primal_tol = saved_p;
      opt_.dual_tol = saved_d;
      opt_.iteration_limit = saved_limit;
      continue;
    }
    if (round == opt_.max_cleanup_rounds) break;
    ++stats_.cleanup_rounds;
    EngineStatus st;
    if (s.dual_count > 0 && s.primal_count == 0) {
      st = run_primal_simplex();
    } else {
      // Primal infeasible (and dual feasible, or made so by shifting): the dual simplex again.
      fix_dual_infeasibilities(true);
      if (primal_stale_) compute_primal();
      st = run_dual_simplex();
    }
    if (st != EngineStatus::Optimal) return st;
  }
  return EngineStatus::NumericalError;
}

// Phase 2 from a dual feasible basis: perturb, dual simplex, then restore the original costs and
// clean up.
EngineStatus SimplexEngine::phase2() {
  if (opt_.perturb && opt_.perturb_at_start) perturb_costs(1.0);
  EngineStatus st = run_dual_simplex();
  if (st == EngineStatus::Optimal) st = finish_after_dual();
  return st;
}

// Dual phase 1 (Koberstein's artificial bounds subproblem): every variable gets a box that makes the
// slack basis dual feasible whatever the costs are: free [-B, B], lower bounded only [0, 1], upper
// bounded only [-1, 0], boxed or fixed [0, 0] (the logical variables, i.e. the rows, are treated the
// same way). The subproblem min c^T x is solved with the dual simplex; its optimal basis is dual
// feasible for the original problem exactly when no variable ends at an artificial bound that its
// original type does not allow. The real bounds are then restored.
EngineStatus SimplexEngine::run_dual_phase1(bool* dual_feasible) {
  const std::vector<double> lo0 = lo_, hi0 = hi_;
  const long long it0 = stats_.iterations;
  for (Index j = 0; j < N_; ++j) {
    const bool fl = !is_inf(lo0[to_size(j)]), fh = !is_inf(hi0[to_size(j)]);
    double lo, hi;
    if (!fl && !fh) {
      lo = -opt_.artificial_bound;
      hi = opt_.artificial_bound;
    } else if (fl && !fh) {
      lo = 0.0;
      hi = 1.0;
    } else if (!fl && fh) {
      lo = -1.0;
      hi = 0.0;
    } else {
      lo = 0.0;
      hi = 0.0;
    }
    lo_[to_size(j)] = lo;
    hi_[to_size(j)] = hi;
  }
  for (Index j = 0; j < N_; ++j) {
    if (status_[to_size(j)] != VarStatus::Basic) set_status_from_bounds(j, d_[to_size(j)] < 0.0);
  }
  sync_nonbasic_values();
  compute_primal();
  const bool saved_perturb = opt_.perturb, saved_harris = opt_.harris, saved_polish = opt_.polish;
  in_phase1_ = true;
  opt_.perturb = false;
  opt_.harris = false;
  opt_.polish = false;
  EngineStatus st = run_dual_simplex();
  if (st == EngineStatus::Optimal) st = finish_after_dual();
  in_phase1_ = false;
  opt_.perturb = saved_perturb;
  opt_.harris = saved_harris;
  opt_.polish = saved_polish;
  lo_ = lo0;
  hi_ = hi0;
  stats_.phase1_iterations += stats_.iterations - it0;
  if (st != EngineStatus::Optimal) {
    // The subproblem is feasible (x = 0) and bounded (all boxed): anything but a limit is a numerical failure.
    return st == EngineStatus::IterationLimit || st == EngineStatus::TimeLimit ? st : EngineStatus::NumericalError;
  }
  for (Index j = 0; j < N_; ++j) {
    if (status_[to_size(j)] != VarStatus::Basic) set_status_from_bounds(j, d_[to_size(j)] < 0.0);
  }
  sync_nonbasic_values();
  compute_primal();
  compute_dual();
  const Index remaining = fix_dual_infeasibilities(false);
  if (primal_stale_) compute_primal();
  *dual_feasible = remaining == 0;
  return EngineStatus::Optimal;
}

// The basis is not dual feasible and cannot be made so: the problem is infeasible or unbounded.
// First ignore the costs (every basis is dual feasible for zero costs) and look for a primal feasible
// basis with the dual simplex: if it proves infeasibility we are done. Otherwise run the primal
// simplex with the true costs from that basis: it either finds the optimum (the dual infeasibility
// was a tolerance artifact) or an unbounded ray.
EngineStatus SimplexEngine::resolve_dual_infeasible() {
  const std::vector<double> saved_costs = cost_orig_;
  std::fill(cost_orig_.begin(), cost_orig_.end(), 0.0);
  cost_ = cost_orig_;
  costs_modified_ = false;
  compute_dual();
  fix_dual_infeasibilities(false);
  if (primal_stale_) compute_primal();
  EngineStatus st = phase2();
  cost_orig_ = saved_costs;
  cost_ = cost_orig_;
  costs_modified_ = false;
  if (st != EngineStatus::Optimal) return st;  // Infeasible, or a limit, or a numerical failure
  if (!refactor()) return EngineStatus::NumericalError;
  compute_primal();
  compute_dual();
  st = run_primal_simplex();
  if (st != EngineStatus::Optimal) return st;  // Unbounded (ray kept), or a limit, or a numerical failure
  return finish_after_dual();
}

void SimplexEngine::refine_solution(int rounds) {
  for (int round = 0; round < rounds; ++round) {
    bool did = false;
    // Primal: B dx = -(A x - r) on the basic variables.
    rhs_.clear();
    double xmax = 0.0;
    for (Index j = 0; j < N_; ++j) {
      const double x = x_[to_size(j)];
      if (x == 0.0) continue;
      xmax = std::max(xmax, std::fabs(x));
      if (j < n_) {
        for (Index t = model_.A.col_start[to_size(j)]; t < model_.A.col_start[to_size(j) + 1]; ++t) {
          rhs_.add(model_.A.row_index[to_size(t)], model_.A.value[to_size(t)] * x);
        }
      } else {
        rhs_.add(j - n_, -x);
      }
    }
    if (rhs_.norm_inf() > 1e-13 * (1.0 + xmax)) {
      for (const Index i : rhs_.indices()) rhs_.set(i, -rhs_[i]);
      factor_.ftran(rhs_, false);
      for (const Index i : rhs_.indices()) x_[to_size(basis_[to_size(i)])] += rhs_[i];
      did = true;
    }
    // Dual: B^T dy = c_B - B^T y.
    rhs_.clear();
    double cmax = 0.0;
    for (Index p = 0; p < m_; ++p) {
      const Index v = basis_[to_size(p)];
      double bt;
      if (v < n_) {
        bt = 0.0;
        for (Index t = model_.A.col_start[to_size(v)]; t < model_.A.col_start[to_size(v) + 1]; ++t) {
          bt += model_.A.value[to_size(t)] * y_[to_size(model_.A.row_index[to_size(t)])];
        }
      } else {
        bt = -y_[to_size(v - n_)];
      }
      const double r = cost_[to_size(v)] - bt;
      cmax = std::max(cmax, std::fabs(cost_[to_size(v)]));
      if (r != 0.0) rhs_.set(p, r);
    }
    if (rhs_.norm_inf() > 1e-13 * (1.0 + cmax)) {
      factor_.btran(rhs_);
      for (const Index i : rhs_.indices()) y_[to_size(i)] += rhs_[i];
      update_duals_from_y();
      did = true;
    }
    if (!did) break;
  }
}

const LpModel& SimplexEngine::checked_model() {
  if (!bounds_modified_) return model_;
  check_model_ = model_;
  for (Index j = 0; j < n_; ++j) {
    check_model_.col_lower[to_size(j)] = lo_[to_size(j)];
    check_model_.col_upper[to_size(j)] = hi_[to_size(j)];
  }
  for (Index i = 0; i < m_; ++i) {
    check_model_.row_lower[to_size(i)] = lo_[to_size(n_ + i)];
    check_model_.row_upper[to_size(i)] = hi_[to_size(n_ + i)];
  }
  return check_model_;
}

// Never return a point or a certificate that was not checked.
EngineStatus SimplexEngine::accept(EngineStatus st) {
  if (!opt_.final_check) return st;
  if (st == EngineStatus::Optimal) {
    refine_solution(2);
    if (check_kkt(checked_model(), solution(), opt_.final_tol).ok) return st;
    // Tighten the tolerances once, redo the cleanup, refine and check again.
    const double saved_p = opt_.primal_tol, saved_d = opt_.dual_tol;
    opt_.primal_tol = std::min(saved_p, 1e-9);
    opt_.dual_tol = std::min(saved_d, 1e-9);
    const EngineStatus again = finish_after_dual();
    opt_.primal_tol = saved_p;
    opt_.dual_tol = saved_d;
    if (again != EngineStatus::Optimal) return EngineStatus::NumericalError;
    refine_solution(2);
    return check_kkt(checked_model(), solution(), opt_.final_tol).ok ? st : EngineStatus::NumericalError;
  }
  if (st == EngineStatus::Infeasible) {
    return check_farkas(checked_model(), farkas_, 1e-9).ok ? st : EngineStatus::NumericalError;
  }
  if (st == EngineStatus::Unbounded) {
    return check_unbounded_ray(checked_model(), ray_, 1e-7).ok ? st : EngineStatus::NumericalError;
  }
  return st;
}

EngineStatus SimplexEngine::solve() {
  t0_ = std::chrono::steady_clock::now();
  cost_ = cost_orig_;
  costs_modified_ = false;
  EngineStatus st;
  if (!refactor()) {
    st = EngineStatus::NumericalError;
  } else {
    compute_primal();
    compute_dual();
    const Index remaining = fix_dual_infeasibilities(false);
    if (primal_stale_) compute_primal();
    if (remaining > 0) {
      bool dual_feasible = false;
      st = run_dual_phase1(&dual_feasible);
      if (st == EngineStatus::Optimal) st = dual_feasible ? phase2() : resolve_dual_infeasible();
    } else {
      st = phase2();
    }
    st = accept(st);
  }
  stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  return st;
}

}  // namespace shodhan
