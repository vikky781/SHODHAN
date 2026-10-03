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
  const double tol = opt_.primal_tol;
  for (Index p = 0; p < m_; ++p) {
    const Index v = basis_[to_size(p)];
    const double x = x_[to_size(v)];
    double viol = 0.0;
    if (x < lo_[to_size(v)] - tol) viol = lo_[to_size(v)] - x;
    else if (x > hi_[to_size(v)] + tol) viol = x - hi_[to_size(v)];
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

bool SimplexEngine::select_entering(double sigma, double delta, double* theta_dual, Index* entering,
                                    std::vector<Index>* flips) {
  (void)delta;
  flips->clear();
  Index best = -1;
  double best_ratio = 0.0, best_abs = 0.0;
  for (const Index j : row_alpha_.indices()) {
    const double a = row_alpha_[j];
    const double abar = sigma * a;
    if (std::fabs(a) < opt_.min_pivot_abs) continue;
    if (std::find(banned_.begin(), banned_.end(), j) != banned_.end()) continue;
    double ratio;
    switch (status_[to_size(j)]) {
      case VarStatus::AtLower:
        if (!(abar < 0.0)) continue;
        ratio = std::max(d_[to_size(j)], 0.0) / -abar;
        break;
      case VarStatus::AtUpper:
        if (!(abar > 0.0)) continue;
        ratio = std::max(-d_[to_size(j)], 0.0) / abar;
        break;
      case VarStatus::FreeAtZero:
        ratio = 0.0;
        break;
      default:
        continue;
    }
    const double aa = std::fabs(a);
    const bool take = best < 0 || ratio < best_ratio || (ratio == best_ratio && (aa > best_abs || (aa == best_abs && j < best)));
    if (take) {
      best = j;
      best_ratio = ratio;
      best_abs = aa;
    }
  }
  if (best < 0) return false;
  *entering = best;
  *theta_dual = best_ratio;
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
    if (!select_entering(sigma, delta, &theta, &q, &flips_)) {
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

    // Dual update.
    const double step = sigma * theta;
    for (const Index j : row_alpha_.indices()) {
      if (j == q) continue;
      d_[to_size(j)] += step * row_alpha_[j];
    }
    d_[to_size(q)] = 0.0;
    d_[to_size(p)] = step;

    // Primal update.
    const double t = delta / alpha_c;
    for (const Index i : col_.indices()) {
      if (i == r) continue;
      x_[to_size(basis_[to_size(i)])] -= t * col_[i];
    }
    x_[to_size(q)] += t;
    x_[to_size(p)] = bound;

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
    if (opt_.verbosity > 0 && stats_.iterations % std::max(1, opt_.log_interval) == 0) log_line("dual", objective());
  }
}

void SimplexEngine::apply_bound_flips(const std::vector<Index>& flips, double* delta, Index r) {
  // Filled in with the bound flipping ratio test.
  (void)flips;
  (void)delta;
  (void)r;
}

}  // namespace shodhan
