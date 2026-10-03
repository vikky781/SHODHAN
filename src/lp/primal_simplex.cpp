// Bounded primal simplex, phase 2 (docs/SIMPLEX.md): a cleanup engine used after
// perturbation removal and to look for unbounded rays. Dantzig pricing, Harris
// ratio test, Bland's rule after a long run of degenerate steps. It starts from a
// basis whose basic variables are within bounds (up to the primal tolerance).
//
// Notation. x_q changes by dir * t (dir = +1 increases, -1 decreases) and the
// basic variables by  -dir * t * alpha  with alpha = B^-1 a_q. Reduced costs
// change after a pivot in row r as  d_j -= (d_q / alpha_r) * alpha_rj,
// d_p = -d_q / alpha_r.

#include <algorithm>
#include <cmath>

#include "shodhan/rays.hpp"
#include "shodhan/simplex_engine.hpp"

namespace shodhan {

namespace {

constexpr long long kBlandAfter = 100;  // degenerate steps in a row before Bland's rule

void load_col(const SparseMatrix& A, Index n, Index j, SparseWork& w) {
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

bool SimplexEngine::refresh_after_refactor_primal() {
  if (!refactor()) return false;
  compute_primal();
  compute_dual();
  return true;
}

EngineStatus SimplexEngine::run_primal_simplex() {
  const double dtol = opt_.dual_tol;
  const double hp = opt_.harris ? 0.5 * opt_.primal_tol : 0.0;
  long long degenerate_run = 0;
  bool refreshed_for_optimality = false;
  trouble_run_ = 0;

  for (;;) {
    if (stats_.iterations >= opt_.iteration_limit) return EngineStatus::IterationLimit;
    if (time_exceeded()) return EngineStatus::TimeLimit;
    if (factor_.stats().growth > opt_.max_growth && updates_since_refactor_ > 0) {
      if (!refresh_after_refactor_primal()) return EngineStatus::NumericalError;
    }
    const bool bland = degenerate_run > kBlandAfter;

    // ---- pricing: nonbasic variable whose reduced cost has the wrong sign ----
    Index q = -1;
    double best = 0.0, dir = 0.0;
    for (Index j = 0; j < N_; ++j) {
      const VarStatus st = status_[to_size(j)];
      if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;
      const double dj = d_[to_size(j)];
      double viol = 0.0, dr = 0.0;
      if (st == VarStatus::AtLower) {
        if (dj < -dtol) { viol = -dj; dr = 1.0; }
      } else if (st == VarStatus::AtUpper) {
        if (dj > dtol) { viol = dj; dr = -1.0; }
      } else if (std::fabs(dj) > dtol) {
        viol = std::fabs(dj);
        dr = dj < 0.0 ? 1.0 : -1.0;
      }
      if (viol == 0.0) continue;
      if (bland) {
        q = j;
        dir = dr;
        break;
      }
      if (viol > best) {
        best = viol;
        q = j;
        dir = dr;
      }
    }
    if (q < 0) {
      if (updates_since_refactor_ > 0 && !refreshed_for_optimality) {
        refreshed_for_optimality = true;
        if (!refresh_after_refactor_primal()) return EngineStatus::NumericalError;
        continue;
      }
      return EngineStatus::Optimal;
    }
    refreshed_for_optimality = false;

    // ---- entering column and ratio test ----
    load_col(model_.A, n_, q, col_);
    factor_.ftran(col_, true);
    double amax = 0.0;
    for (const Index i : col_.indices()) amax = std::max(amax, std::fabs(col_[i]));
    const double hp_now = bland ? 0.0 : hp;
    Index r = -1;
    double t_exact = kInf;

    // Harris two-pass ratio test over the basic variables whose |alpha| exceeds pivtol.
    auto ratio_test = [&](double pivtol) {
      r = -1;
      t_exact = kInf;
      double tmax = kInf;
      for (const Index i : col_.indices()) {
        const double a = col_[i];
        if (std::fabs(a) <= pivtol) continue;
        const double rate = -dir * a;
        const Index v = basis_[to_size(i)];
        const double x = x_[to_size(v)];
        double tl;
        if (rate < 0.0 && !is_inf(lo_[to_size(v)])) tl = (x - lo_[to_size(v)] + hp_now * (1.0 + std::fabs(lo_[to_size(v)]))) / -rate;
        else if (rate > 0.0 && !is_inf(hi_[to_size(v)])) tl = (hi_[to_size(v)] - x + hp_now * (1.0 + std::fabs(hi_[to_size(v)]))) / rate;
        else continue;
        tmax = std::min(tmax, tl);
      }
      if (!(tmax < kInf)) return;
      double best_abs = 0.0;
      for (const Index i : col_.indices()) {
        const double a = col_[i];
        if (std::fabs(a) <= pivtol) continue;
        const double rate = -dir * a;
        const Index v = basis_[to_size(i)];
        const double x = x_[to_size(v)];
        double te;
        if (rate < 0.0 && !is_inf(lo_[to_size(v)])) te = std::max(x - lo_[to_size(v)], 0.0) / -rate;
        else if (rate > 0.0 && !is_inf(hi_[to_size(v)])) te = std::max(hi_[to_size(v)] - x, 0.0) / rate;
        else continue;
        if (te > tmax) continue;
        const double aa = std::fabs(a);
        const bool take = r < 0 || (bland ? (te < t_exact || (te == t_exact && v < basis_[to_size(r)]))
                                          : (aa > best_abs || (aa == best_abs && v < basis_[to_size(r)])));
        if (take) {
          r = i;
          t_exact = te;
          best_abs = aa;
        }
      }
    };
    ratio_test(std::max(opt_.min_pivot_abs, opt_.min_pivot_rel * amax));
    const double range = hi_[to_size(q)] - lo_[to_size(q)];
    const bool boxed = !is_inf(lo_[to_size(q)]) && !is_inf(hi_[to_size(q)]);

    if (r < 0 && !boxed) {
      // No basic variable blocks and the entering variable has no bound in this direction. Only
      // claim it with a ray that checks; if not, basic variables with a tiny |alpha| that the
      // pivot threshold dropped may block after all, so look at them too.
      auto make_ray = [&]() {
        ray_.assign(to_size(n_), 0.0);
        if (q < n_) ray_[to_size(q)] = dir;
        for (const Index i : col_.indices()) {
          const Index v = basis_[to_size(i)];
          if (v < n_) ray_[to_size(v)] += -dir * col_[i];
        }
      };
      make_ray();
      if (!check_unbounded_ray(checked_model(), ray_, 1e-8).ok) {
        ratio_test(1e-11);
        if (r < 0) return EngineStatus::Unbounded;  // accept() will refuse an invalid ray
      } else {
        return EngineStatus::Unbounded;
      }
    }

    if (boxed && (r < 0 || range <= t_exact)) {
      // Bound flip of the entering variable: no basis change.
      const double t = range;
      for (const Index i : col_.indices()) x_[to_size(basis_[to_size(i)])] -= dir * t * col_[i];
      status_[to_size(q)] = dir > 0.0 ? VarStatus::AtUpper : VarStatus::AtLower;
      x_[to_size(q)] = dir > 0.0 ? hi_[to_size(q)] : lo_[to_size(q)];
      ++stats_.iterations;
      ++stats_.primal_iterations;
      ++stats_.bound_flips;
      degenerate_run = 0;
      continue;
    }

    // ---- pivot row (for the dual update and the pivot check) ----
    const Index p = basis_[to_size(r)];
    rho_.clear();
    rho_.set(r, 1.0);
    factor_.btran(rho_);
    compute_pivot_row();
    const double alpha_c = col_[r];
    const double alpha_r = row_alpha_[q];
    const double scale = std::max(std::fabs(alpha_c), std::fabs(alpha_r));
    if (!(std::fabs(alpha_c - alpha_r) <= opt_.pivot_agreement_tol * scale)) {
      if (updates_since_refactor_ > 0) {
        if (!refresh_after_refactor_primal()) return EngineStatus::NumericalError;
        continue;
      }
      if (!handle_trouble("primal: row and column pivot values disagree")) return EngineStatus::NumericalError;
      degenerate_run += 1;  // retried with the same pricing would loop: let Bland's rule break the tie
      continue;
    }

    // ---- update ----
    const double t = t_exact;
    const double rate_r = -dir * alpha_c;
    const double new_bound = rate_r < 0.0 ? lo_[to_size(p)] : hi_[to_size(p)];
    for (const Index i : col_.indices()) {
      if (i == r) continue;
      x_[to_size(basis_[to_size(i)])] -= dir * t * col_[i];
    }
    x_[to_size(q)] += dir * t;
    x_[to_size(p)] = new_bound;

    const double theta_d = d_[to_size(q)] / alpha_c;
    for (const Index j : row_alpha_.indices()) {
      if (j == q) continue;
      d_[to_size(j)] -= theta_d * row_alpha_[j];
    }
    d_[to_size(q)] = 0.0;
    d_[to_size(p)] = -theta_d;

    status_[to_size(p)] = lo_[to_size(p)] == hi_[to_size(p)] ? VarStatus::Fixed
                                                             : (rate_r < 0.0 ? VarStatus::AtLower : VarStatus::AtUpper);
    status_[to_size(q)] = VarStatus::Basic;
    basis_[to_size(r)] = q;
    pos_[to_size(q)] = r;
    pos_[to_size(p)] = -1;
    ++stats_.iterations;
    ++stats_.primal_iterations;
    degenerate_run = t < 1e-12 ? degenerate_run + 1 : 0;
    trouble_run_ = 0;

    if (factor_.update(r) == FactorStatus::Ok) {
      ++updates_since_refactor_;
    } else if (!refresh_after_refactor_primal()) {
      return EngineStatus::NumericalError;
    }
    if (opt_.verbosity > 0 && stats_.iterations % std::max(1, opt_.log_interval) == 0) log_line("primal", objective());
  }
}

}  // namespace shodhan
