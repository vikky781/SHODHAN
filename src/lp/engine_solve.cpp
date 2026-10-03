#include <algorithm>
#include <cmath>
#include <cstdint>

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

EngineStatus SimplexEngine::solve() {
  t0_ = std::chrono::steady_clock::now();
  cost_ = cost_orig_;
  costs_modified_ = false;
  if (!refactor()) return EngineStatus::NumericalError;
  compute_primal();
  compute_dual();
  const Index remaining = fix_dual_infeasibilities(false);
  if (primal_stale_) compute_primal();
  EngineStatus st;
  if (remaining > 0) {
    st = EngineStatus::NumericalError;  // dual phase 1 comes with stage D
  } else {
    if (opt_.perturb && opt_.perturb_at_start) perturb_costs(1.0);
    st = run_dual_simplex();
    if (st == EngineStatus::Optimal) st = finish_after_dual();
  }
  stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  return st;
}

}  // namespace shodhan
