// Bounded dual simplex, phase 2 (docs/SIMPLEX.md).
//
// Notation. B x_B + N x_N = 0 (computational form A x - r = 0). For the basic
// variable p = basis[r] with an infeasibility delta = x_p - bound (bound the
// violated one), rho_r = B^-T e_r and alpha_j = rho_r^T a_j for nonbasic j. Row r
// of the tableau reads  x_p + sum_j alpha_j x_j = 0.
//
// The dual step. Let sigma = +1 if x_p leaves at its lower bound (delta < 0) and
// -1 at its upper bound. Moving y by -sigma*theta*rho_r (theta >= 0) changes
//   d_j -> d_j + sigma*theta*alpha_j   (nonbasic j),     d_p = sigma*theta.
// A nonbasic variable at its lower bound (d_j >= 0) blocks the step when
// sigma*alpha_j < 0, one at its upper bound (d_j <= 0) when sigma*alpha_j > 0,
// a free one whenever alpha_j != 0; fixed variables never block.

#include <algorithm>
#include <cmath>
#include <iostream>

#include "shodhan/simplex_engine.hpp"

namespace shodhan {

namespace {

// A reduced cost of the wrong sign by more than this is removed by shifting its cost.
constexpr double kShiftTol = 1e-11;

// Loads the column of variable j (structural or logical) as a row-space vector.
void load_column(const SparseMatrix& A, Index n, Index j, SparseWork& w) {
  w.clear();
  if (j < n) {
    for (Index t = A.col_start[to_size(j)]; t < A.col_start[to_size(j) + 1]; ++t) {
      if (A.value[to_size(t)] != 0.0) w.set(A.row_index[to_size(t)], A.value[to_size(t)]);
    }
  } else {
    w.set(j - n, -1.0);
  }
}

}  // namespace

Index SimplexEngine::choose_leaving_row(double* score_out) const {
  Index best = -1;
  double best_score = 0.0;
  for (Index p = 0; p < m_; ++p) {
    const Index v = basis_[to_size(p)];
    const double x = x_[to_size(v)];
    double viol = 0.0;
    if (x < lo_[to_size(v)] - ptol(lo_[to_size(v)])) viol = lo_[to_size(v)] - x;
    else if (x > hi_[to_size(v)] + ptol(hi_[to_size(v)])) viol = x - hi_[to_size(v)];
    else continue;
    const double score = viol * viol / weights_[to_size(p)];
    if (score > best_score) {
      best_score = score;
      best = p;
    }
  }
  if (score_out != nullptr) *score_out = best_score;
  return best;
}

// alpha_r over the nonbasic variables, row-wise from the CSR copy: only the
// nonzeros of rho_r are visited.
void SimplexEngine::compute_pivot_row() {
  row_alpha_.clear();
  for (const Index i : rho_.indices()) {
    const double v = rho_[i];
    if (v == 0.0) continue;
    for (Index t = csr_.row_start[to_size(i)]; t < csr_.row_start[to_size(i) + 1]; ++t) {
      const Index j = csr_.col_index[to_size(t)];
      if (status_[to_size(j)] != VarStatus::Basic) row_alpha_.add(j, v * csr_.value[to_size(t)]);
    }
    const Index jl = n_ + i;
    if (status_[to_size(jl)] != VarStatus::Basic) row_alpha_.add(jl, -v);
  }
}

// Ratio test. Candidates are the nonbasic variables that block the dual step
// (see the notation above); candidate j has the exact breakpoint
//   t_j = max(s_j, 0) / |alpha_j|,       s_j = d_j (at lower), -d_j (at upper), 0 (free),
// and, for Harris, the relaxed breakpoint  r_j = (s_j + tol) / |alpha_j|.
//
// Candidates are sorted by t_j. Harris pass 1: theta_max = min r_j over the
// remaining candidates; the group K holds those with t_j <= theta_max. With
// bound flipping, a group of boxed candidates whose bound flips keep the slope
// of the dual objective positive,
//   slope -= sum_{j in K} |alpha_j| (hi_j - lo_j) > 0,
// is passed (all its variables flip) and the search continues; otherwise the
// entering variable is the one of K with the largest |alpha_j| (Harris pass 2).
bool SimplexEngine::select_entering(double sigma, double delta, double margin, double* theta_dual, Index* entering,
                                    std::vector<Index>* flips) {
  flips->clear();
  double amax = 0.0;
  for (const Index j : row_alpha_.indices()) amax = std::max(amax, std::fabs(row_alpha_[j]));
  const double tol = opt_.harris ? 0.5 * opt_.dual_tol : 0.0;

  auto collect = [&](double threshold) {
    cand_.clear();
    for (const Index j : row_alpha_.indices()) {
      const double a = row_alpha_[j];
      const double aa = std::fabs(a);
      if (aa < threshold) continue;
      if (!banned_.empty() && std::find(banned_.begin(), banned_.end(), j) != banned_.end()) continue;
      const double abar = sigma * a;
      double s;
      switch (status_[to_size(j)]) {
        case VarStatus::AtLower:
          if (!(abar < 0.0)) continue;
          s = d_[to_size(j)];
          break;
        case VarStatus::AtUpper:
          if (!(abar > 0.0)) continue;
          s = -d_[to_size(j)];
          break;
        case VarStatus::FreeAtZero:
          s = 0.0;
          break;
        default:
          continue;
      }
      const double range = hi_[to_size(j)] - lo_[to_size(j)];
      const bool boxed = !is_inf(lo_[to_size(j)]) && !is_inf(hi_[to_size(j)]);
      cand_.push_back({std::max(s, 0.0) / aa, (std::max(s, 0.0) + tol) / aa, aa, boxed ? range : kInf, j});
    }
  };
  collect(std::max(opt_.min_pivot_abs, opt_.min_pivot_rel * amax));
  if (cand_.empty()) collect(opt_.min_pivot_abs);  // relative threshold removed everything
  if (cand_.empty()) return false;

  std::sort(cand_.begin(), cand_.end(), [](const Candidate& a, const Candidate& b) {
    return a.t < b.t || (a.t == b.t && (a.abs_alpha > b.abs_alpha || (a.abs_alpha == b.abs_alpha && a.var < b.var)));
  });
  const std::size_t nc = cand_.size();
  suffix_r_.assign(nc + 1, kInf);
  for (std::size_t k = nc; k-- > 0;) suffix_r_[k] = std::min(suffix_r_[k + 1], cand_[k].r);

  double slope = std::fabs(delta);
  std::size_t ptr = 0;
  std::size_t end = 0;
  for (;;) {
    const double theta_max = suffix_r_[ptr];
    end = ptr;
    double decrease = 0.0;
    while (end < nc && cand_[end].t <= theta_max) {
      decrease += cand_[end].abs_alpha * cand_[end].range;
      ++end;
    }
    if (end == ptr) end = ptr + 1;  // cannot happen (the minimizer of r has t <= r), kept for safety
    const bool can_pass = opt_.bound_flipping && std::isfinite(decrease) && decrease < kInf &&
                          slope - decrease > margin && end < nc;
    if (!can_pass) {
      // The last group is selected, unless it is the end and everything before could be flipped:
      break;
    }
    slope -= decrease;
    for (std::size_t k = ptr; k < end; ++k) flips->push_back(cand_[k].var);
    ptr = end;
  }
  // Entering: largest |alpha| in the group [ptr, end).
  std::size_t best = ptr;
  for (std::size_t k = ptr + 1; k < end; ++k) {
    if (cand_[k].abs_alpha > cand_[best].abs_alpha ||
        (cand_[k].abs_alpha == cand_[best].abs_alpha && cand_[k].var < cand_[best].var)) {
      best = k;
    }
  }
  // If the whole candidate set was passed with positive slope left, the row is infeasible (all
  // boxed variables sit at the bound that helps most and x_p still violates its bound): that is
  // only possible when `end == nc` and every member could be flipped.
  if (opt_.bound_flipping && end == nc && std::isfinite(cand_[best].range)) {
    double decrease = 0.0;
    for (std::size_t k = ptr; k < end; ++k) decrease += cand_[k].abs_alpha * cand_[k].range;
    if (std::isfinite(decrease) && slope - decrease > margin) {
      for (std::size_t k = ptr; k < end; ++k) flips->push_back(cand_[k].var);
      return false;  // dual unbounded: primal infeasible
    }
  }
  *entering = cand_[best].var;
  *theta_dual = cand_[best].t;
  return true;
}

void SimplexEngine::record_farkas(Index r) {
  (void)r;
  farkas_.assign(to_size(m_), 0.0);
  for (const Index i : rho_.indices()) farkas_[to_size(i)] = rho_[i];
}

bool SimplexEngine::handle_trouble(const char* what) {
  ++stats_.trouble_events;
  ++trouble_run_;
  if (opt_.verbosity > 0 && opt_.log != nullptr) *opt_.log << "numerical trouble: " << what << "\n";
  return trouble_run_ <= opt_.max_trouble;
}

// Refactorizes, recomputes x_B and d from scratch, restores dual feasibility by
// bound switches and cost shifts, and reports a large drift as a trouble event.
bool SimplexEngine::refactor_and_recompute() {
  std::vector<double> xb_old(to_size(m_));
  for (Index p = 0; p < m_; ++p) xb_old[to_size(p)] = x_[to_size(basis_[to_size(p)])];
  const std::vector<Index> basis_old = basis_;
  const int repairs_before = stats_.basis_repairs;
  if (!refactor()) return false;
  compute_primal();
  compute_dual();
  fix_dual_infeasibilities(true);
  if (primal_stale_) compute_primal();
  dual_objective_ = working_objective();
  if (stats_.basis_repairs == repairs_before && basis_old == basis_) {
    double err = 0.0;
    for (Index p = 0; p < m_; ++p) {
      const double xn = x_[to_size(basis_[to_size(p)])];
      err = std::max(err, std::fabs(xn - xb_old[to_size(p)]) / (1.0 + std::fabs(xb_old[to_size(p)])));
    }
    if (err > 1e-6) return handle_trouble("large primal drift after refactorization") || true;
  }
  return true;
}

void SimplexEngine::compute_exact_weights() {
  for (Index i = 0; i < m_; ++i) {
    rho_.clear();
    rho_.set(i, 1.0);
    factor_.btran(rho_);
    double w = 0.0;
    for (const Index k : rho_.indices()) w += rho_[k] * rho_[k];
    weights_[to_size(i)] = std::max(w, 1e-4);
  }
  weights_exact_ = true;
}

void SimplexEngine::update_weights(Index r, double alpha_r) {
  double wr = 0.0;
  for (const Index i : rho_.indices()) wr += rho_[i] * rho_[i];
  wr = std::max(wr, 1e-4);
  for (const Index i : col_.indices()) {
    if (i == r) continue;
    const double ratio = col_[i] / alpha_r;
    const double w = weights_[to_size(i)] - 2.0 * ratio * tau_[i] + ratio * ratio * wr;
    weights_[to_size(i)] = std::max(w, 1e-4);
  }
  weights_[to_size(r)] = std::max(wr / (alpha_r * alpha_r), 1e-4);
}

EngineStatus SimplexEngine::run_dual_simplex() {
  trouble_run_ = 0;
  banned_.clear();
  dual_objective_ = working_objective();
  best_dual_objective_ = dual_objective_;
  last_progress_iter_ = stats_.iterations;
  stall_rounds_ = 0;
  for (;;) {
    if (stats_.iterations >= opt_.iteration_limit) return EngineStatus::IterationLimit;
    if (time_exceeded()) return EngineStatus::TimeLimit;
    if (factor_.stats().growth > opt_.max_growth && updates_since_refactor_ > 0) {
      if (!refactor_and_recompute()) return EngineStatus::NumericalError;
    }

    Index r = choose_leaving_row(nullptr);
    if (r < 0) {
      // Before declaring optimality, make sure the point is not a product of
      // accumulated update error.
      if (updates_since_refactor_ > 0) {
        if (!refactor_and_recompute()) return EngineStatus::NumericalError;
        r = choose_leaving_row(nullptr);
      }
      if (r < 0) return EngineStatus::Optimal;
    }

    const Index p = basis_[to_size(r)];
    const double xp = x_[to_size(p)];
    double sigma, bound;
    if (xp < lo_[to_size(p)]) {
      sigma = 1.0;
      bound = lo_[to_size(p)];
    } else {
      sigma = -1.0;
      bound = hi_[to_size(p)];
    }
    double delta = xp - bound;

    rho_.clear();
    rho_.set(r, 1.0);
    factor_.btran(rho_);
    compute_pivot_row();

    double theta = 0.0;
    Index q = -1;
    const double margin = 0.5 * ptol(bound);
    if (!select_entering(sigma, delta, margin, &theta, &q, &flips_)) {
      if (updates_since_refactor_ > 0) {  // verify with a fresh factorization first
        if (!refactor_and_recompute()) return EngineStatus::NumericalError;
        continue;
      }
      if (!banned_.empty()) return EngineStatus::NumericalError;  // candidates were excluded as unreliable
      record_farkas(r);
      return EngineStatus::Infeasible;
    }

    // Entering column (spike saved for the update) and the pivot check.
    load_column(model_.A, n_, q, col_);
    factor_.ftran(col_, true);
    const double alpha_c = col_[r];
    const double alpha_r = row_alpha_[q];
    const double scale = std::max(std::fabs(alpha_c), std::fabs(alpha_r));
    if (!(std::fabs(alpha_c - alpha_r) <= opt_.pivot_agreement_tol * scale) || std::fabs(alpha_c) < opt_.min_pivot_abs * 0.5) {
      if (updates_since_refactor_ > 0) {
        if (!refactor_and_recompute()) return EngineStatus::NumericalError;
        continue;
      }
      banned_.push_back(q);
      if (!handle_trouble("row and column pivot values disagree")) return EngineStatus::NumericalError;
      continue;
    }

    if (opt_.dual_steepest_edge) {
      tau_.assign(rho_);
      factor_.ftran(tau_, false);
    }
    if (!flips_.empty()) apply_bound_flips(flips_, &delta, r);

    // Dual update. Flipped variables have just moved to their other bound; the step carries
    // their reduced costs to the sign that bound requires. A Harris step can leave a reduced
    // cost wrong-signed by at most the tolerance: its cost is shifted so that d_j = 0.
    const double step = sigma * theta;
    for (const Index j : row_alpha_.indices()) {
      if (j == q) continue;
      const double dj = d_[to_size(j)] + step * row_alpha_[j];
      d_[to_size(j)] = dj;
      bool wrong = false;
      switch (status_[to_size(j)]) {
        case VarStatus::AtLower: wrong = dj < -kShiftTol; break;
        case VarStatus::AtUpper: wrong = dj > kShiftTol; break;
        case VarStatus::FreeAtZero: wrong = std::fabs(dj) > kShiftTol; break;
        default: break;
      }
      if (wrong) {
        cost_[to_size(j)] -= dj;
        d_[to_size(j)] = 0.0;
        costs_modified_ = true;
        ++stats_.cost_shifts;
      }
    }
    {
      const double dq = d_[to_size(q)] + step * row_alpha_[q];
      if (std::fabs(dq) > kShiftTol) {  // only when theta was clamped at 0
        cost_[to_size(q)] -= dq;
        costs_modified_ = true;
        ++stats_.cost_shifts;
      }
      d_[to_size(q)] = 0.0;
    }
    d_[to_size(p)] = step;

    // Primal update (and the change of the dual objective c^T x it causes).
    const double t = delta / alpha_c;
    double dobj = cost_[to_size(q)] * t + cost_[to_size(p)] * (bound - x_[to_size(p)]);
    for (const Index i : col_.indices()) {
      if (i == r) continue;
      const Index bv = basis_[to_size(i)];
      x_[to_size(bv)] -= t * col_[i];
      dobj -= cost_[to_size(bv)] * t * col_[i];
    }
    x_[to_size(q)] += t;
    x_[to_size(p)] = bound;
    dual_objective_ += dobj + flip_objective_;
    flip_objective_ = 0.0;

    if (opt_.dual_steepest_edge) update_weights(r, alpha_c);

    // Basis change.
    status_[to_size(p)] = lo_[to_size(p)] == hi_[to_size(p)] ? VarStatus::Fixed
                                                             : (sigma > 0.0 ? VarStatus::AtLower : VarStatus::AtUpper);
    status_[to_size(q)] = VarStatus::Basic;
    basis_[to_size(r)] = q;
    pos_[to_size(q)] = r;
    pos_[to_size(p)] = -1;
    ++stats_.iterations;
    ++stats_.dual_iterations;
    banned_.clear();
    trouble_run_ = 0;

    if (factor_.update(r) == FactorStatus::Ok) {
      ++updates_since_refactor_;
    } else if (!refactor_and_recompute()) {
      return EngineStatus::NumericalError;
    }
    // Stall detection: no progress of the dual objective for a long time.
    if (dual_objective_ > best_dual_objective_ + 1e-9 * (1.0 + std::fabs(best_dual_objective_))) {
      best_dual_objective_ = dual_objective_;
      last_progress_iter_ = stats_.iterations;
    } else if (stats_.iterations - last_progress_iter_ >= opt_.stall_iterations) {
      if (opt_.perturb && stall_rounds_ < 5) {
        perturb_costs(std::pow(10.0, stall_rounds_ + 1));
        dual_objective_ = working_objective();
        best_dual_objective_ = dual_objective_;
      }
      ++stall_rounds_;
      last_progress_iter_ = stats_.iterations;
    }
    if (opt_.verbosity > 0 && stats_.iterations % std::max(1, opt_.log_interval) == 0) log_line("dual", objective());
  }
}

// Moves the flipped variables to their other bound and updates x_B with one ftran of the
// combined change: x_B -= B^-1 (sum_j a_j * change_j). Returns the new infeasibility of the
// leaving variable through delta.
void SimplexEngine::apply_bound_flips(const std::vector<Index>& flips, double* delta, Index r) {
  rhs_.clear();
  flip_objective_ = 0.0;
  for (const Index j : flips) {
    double change;
    if (status_[to_size(j)] == VarStatus::AtLower) {
      change = hi_[to_size(j)] - lo_[to_size(j)];
      status_[to_size(j)] = VarStatus::AtUpper;
      x_[to_size(j)] = hi_[to_size(j)];
    } else {
      change = lo_[to_size(j)] - hi_[to_size(j)];
      status_[to_size(j)] = VarStatus::AtLower;
      x_[to_size(j)] = lo_[to_size(j)];
    }
    flip_objective_ += cost_[to_size(j)] * change;
    if (j < n_) {
      for (Index t = model_.A.col_start[to_size(j)]; t < model_.A.col_start[to_size(j) + 1]; ++t) {
        rhs_.add(model_.A.row_index[to_size(t)], model_.A.value[to_size(t)] * change);
      }
    } else {
      rhs_.add(j - n_, -change);
    }
  }
  factor_.ftran(rhs_, false);
  for (const Index i : rhs_.indices()) {
    const Index bv = basis_[to_size(i)];
    x_[to_size(bv)] -= rhs_[i];
    flip_objective_ -= cost_[to_size(bv)] * rhs_[i];
  }
  stats_.bound_flips += static_cast<long long>(flips.size());
  const Index p = basis_[to_size(r)];
  const double xp = x_[to_size(p)];
  *delta = *delta < 0.0 ? xp - lo_[to_size(p)] : xp - hi_[to_size(p)];  // same side as before the flips
}

}  // namespace shodhan
