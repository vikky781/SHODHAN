// Primal-dual interior-point method for LPs and convex QPs; the formulation and the derivation of the Newton system
// are in docs/IPM.md. Notation: w = (x, r) with r = A x the row activities, bounds lo <= w <= up (finite or not),
// equality  A x - r = 0, slacks s_l = w - lo, s_u = up - w, bound multipliers z_l, z_u >= 0, row multipliers y.
//
//   dual residual   rd_x = c + Q x - A^T y - z_l + z_u,   rd_r = y - z_l + z_u (rows with bounds)
//   primal residual rp   = A x - r
//   Newton:  (H + Theta^-1) dw - B^T dy = -rd - (rc_l/s_l - rc_u/s_u),   B dw = -rp,
//            Theta^-1 = z_l/s_l + z_u/s_u,  B = [A -I],  H = diag(Q, 0)
//   which, with dr eliminated, is the quasi-definite system
//     [ Q + Theta_x^-1   A^T            ] [ dx ]   [ rhs1_x                  ]
//     [ A               -(Dr^-1 + 0)    ] [ -dy ] = [ -rp + rhs1_r / Dr       ],  Dr = Theta_r^-1.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>

#include "shodhan/ipm.hpp"
#include "shodhan/quadratic.hpp"

namespace shodhan {

namespace {

std::size_t u(Index i) { return static_cast<std::size_t>(i); }

double norm_inf(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s = std::max(s, std::fabs(x));
  return s;
}

}  // namespace

namespace {

IpmResult solve_ipm_attempt(const LpModel& model, const IpmOptions& opt) {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point t0 = Clock::now();
  IpmResult res;
  auto finish = [&](Status st, const std::string& msg) {
    res.status = st;
    if (!msg.empty()) res.message = msg;
    res.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return res;
  };
  for (Index j = 0; j < model.n_cols; ++j) {
    if (model.is_integer(j)) return finish(Status::NotImplemented, "integer columns: the interior-point method solves continuous models");
  }
  const std::vector<std::string> problems = model.validate();
  if (!problems.empty()) return finish(Status::NumericalError, "invalid model: " + problems.front());

  const Index n = model.n_cols, m = model.n_rows, N = n + m;
  const double sgn = model.sense == Sense::Maximize ? -1.0 : 1.0;
  std::vector<double> c(u(n));
  for (Index j = 0; j < n; ++j) c[u(j)] = sgn * model.col_cost[u(j)];
  const double offset = sgn * model.objective_offset;
  SparseMatrix Q = model.quadratic;  // lower triangle, minimization form
  const bool has_q = Q.nnz() > 0;
  if (has_q) {
    for (double& v : Q.value) v *= sgn;
  }
  const SparseMatrix& A = model.A;

  // ---- variables, bounds, classes ----
  std::vector<double> lo(u(N)), up(u(N));
  for (Index j = 0; j < n; ++j) {
    lo[u(j)] = model.col_lower[u(j)];
    up[u(j)] = model.col_upper[u(j)];
  }
  for (Index i = 0; i < m; ++i) {
    lo[u(n + i)] = model.row_lower[u(i)];
    up[u(n + i)] = model.row_upper[u(i)];
  }
  std::vector<char> hl(u(N)), hu(u(N)), fixed(u(N)), excluded(u(N), 0);
  for (Index k = 0; k < N; ++k) {
    hl[u(k)] = !is_neg_inf(lo[u(k)]);
    hu[u(k)] = !is_pos_inf(up[u(k)]);
    fixed[u(k)] = hl[u(k)] && hu[u(k)] && lo[u(k)] == up[u(k)];
    if (k >= n && !hl[u(k)] && !hu[u(k)]) excluded[u(k)] = 1;  // a free row never binds
    if (hl[u(k)] && hu[u(k)] && lo[u(k)] > up[u(k)]) return finish(Status::Infeasible, "a variable or row has lower bound above upper bound");
  }
  // Indices in the augmented matrix: active columns first, then the rows that take part.
  std::vector<Index> kidx(u(N), -1);
  Index na = 0, ma = 0;
  for (Index j = 0; j < n; ++j) {
    if (!fixed[u(j)]) kidx[u(j)] = na++;
  }
  for (Index i = 0; i < m; ++i) {
    if (!excluded[u(n + i)]) kidx[u(n + i)] = na + ma++;
  }
  const Index nk = na + ma;

  // ---- augmented matrix pattern (lower triangle), diagonal entries always present ----
  std::vector<Triplet> trip;
  std::vector<double> qdiag(u(n), 0.0);
  for (Index j = 0; j < n; ++j) {
    if (kidx[u(j)] < 0) continue;
    trip.push_back({kidx[u(j)], kidx[u(j)], 0.0});
    if (has_q) {
      for (Index p = Q.col_start[u(j)]; p < Q.col_start[u(j) + 1]; ++p) {
        const Index i = Q.row_index[u(p)];
        if (i == j) qdiag[u(j)] = Q.value[u(p)];
        else if (kidx[u(i)] >= 0) trip.push_back({kidx[u(i)], kidx[u(j)], Q.value[u(p)]});
      }
    }
    for (Index p = A.col_start[u(j)]; p < A.col_start[u(j) + 1]; ++p) {
      const Index i = A.row_index[u(p)];
      if (kidx[u(n + i)] >= 0) trip.push_back({kidx[u(n + i)], kidx[u(j)], A.value[u(p)]});
    }
  }
  for (Index i = 0; i < m; ++i) {
    if (kidx[u(n + i)] >= 0) trip.push_back({kidx[u(n + i)], kidx[u(n + i)], 0.0});
  }
  SparseMatrix K;
  {
    std::string err;
    if (!SparseMatrix::from_triplets(nk, nk, std::move(trip), &K, &err)) return finish(Status::NumericalError, "cannot assemble the augmented system: " + err);
  }
  std::vector<Index> dpos(u(nk), -1);  // position of the diagonal entry of every K column
  for (Index k = 0; k < nk; ++k) {
    for (Index p = K.col_start[u(k)]; p < K.col_start[u(k) + 1]; ++p) {
      if (K.row_index[u(p)] == k) dpos[u(k)] = p;
    }
  }
  // Fill the constant entries: Q off-diagonals were given values already; the Q diagonal goes into the base diagonal.
  std::vector<signed char> sign(u(nk), 1);
  for (Index k = na; k < nk; ++k) sign[u(k)] = -1;
  LdlParams lp;
  lp.rho = opt.rho;
  lp.delta = opt.delta;
  SparseLdl ldl(lp);
  if (nk > 0 && !ldl.analyze(K, sign, SparseLdl::Ordering::Amd)) return finish(Status::NumericalError, "the augmented system could not be analyzed");
  res.nnz_l = ldl.stats().nnz_l;

  // ---- starting point ----
  // Primal: the point of least distance to a reference (x_ref = 0 projected on the bounds, r_ref likewise) that satisfies
  // A x - r = 0, from the regularized least-squares system [[Q + I, A^T], [A, -I]] [x; v] = [x_ref; r_ref - A x_fixed],
  // r = r_ref + v; then moved into the interior of the bounds. Dual: y = 0, bound multipliers cancel the dual residual
  // where their sign allows and are shifted to be positive and roughly centred (Mehrotra 1992).
  std::vector<double> w(u(N), 0.0), zl(u(N), 0.0), zu(u(N), 0.0), y(u(m), 0.0);
  std::vector<double> ax(u(m), 0.0), aty(u(n), 0.0), qx(u(n), 0.0);
  auto compute_ax = [&]() {
    std::fill(ax.begin(), ax.end(), 0.0);
    for (Index j = 0; j < n; ++j) {
      const double xj = w[u(j)];
      if (xj == 0.0) continue;
      for (Index p = A.col_start[u(j)]; p < A.col_start[u(j) + 1]; ++p) ax[u(A.row_index[u(p)])] += A.value[u(p)] * xj;
    }
  };
  auto clamp0 = [&](Index k) { return std::min(std::max(0.0, lo[u(k)]), up[u(k)]); };
  auto interior = [&](Index k, double t) {
    const double l = lo[u(k)], h = up[u(k)];
    if (fixed[u(k)]) return l;
    const double margin = std::max(0.1, 0.01 * std::fabs(t));
    if (hl[u(k)] && hu[u(k)]) {
      const double d = std::min(margin, 0.25 * (h - l));
      return std::min(std::max(t, l + d), h - d);
    }
    if (hl[u(k)]) return std::max(t, l + margin);
    if (hu[u(k)]) return std::min(t, h - margin);
    return t;
  };
  for (Index j = 0; j < n; ++j) w[u(j)] = fixed[u(j)] ? lo[u(j)] : clamp0(j);
  compute_ax();  // activity of the fixed columns at their values (the others are still at their reference)
  bool ls_ok = false;
  if (nk > 0) {
    std::vector<double> afx(u(m), 0.0), rhs0(u(nk), 0.0), sol0(u(nk), 0.0);
    for (Index j = 0; j < n; ++j) {
      if (!fixed[u(j)]) continue;
      for (Index p = A.col_start[u(j)]; p < A.col_start[u(j) + 1]; ++p) afx[u(A.row_index[u(p)])] += A.value[u(p)] * lo[u(j)];
    }
    for (Index j = 0; j < n; ++j) {
      if (kidx[u(j)] >= 0) {
        K.value[u(dpos[u(kidx[u(j)])])] = qdiag[u(j)] + 1.0;
        rhs0[u(kidx[u(j)])] = clamp0(j);
      }
    }
    for (Index i = 0; i < m; ++i) {
      const Index k = n + i;
      if (kidx[u(k)] < 0) continue;
      K.value[u(dpos[u(kidx[u(k)])])] = fixed[u(k)] ? 0.0 : -1.0;
      rhs0[u(kidx[u(k)])] = clamp0(k) - afx[u(i)];
    }
    if (ldl.factorize(K)) {
      ldl.solve(rhs0, sol0, opt.refinement_steps, 0.0);
      ls_ok = true;
      for (const double v : sol0) ls_ok = ls_ok && std::isfinite(v);
      if (ls_ok) {
        for (Index j = 0; j < n; ++j) {
          if (kidx[u(j)] >= 0) w[u(j)] = sol0[u(kidx[u(j)])];
        }
        for (Index i = 0; i < m; ++i) {
          const Index k = n + i;
          if (kidx[u(k)] >= 0) w[u(k)] = clamp0(k) + sol0[u(kidx[u(k)])];
        }
      }
    }
  }
  if (!ls_ok) {
    for (Index j = 0; j < n; ++j) w[u(j)] = fixed[u(j)] ? lo[u(j)] : clamp0(j);
  }
  for (Index j = 0; j < n; ++j) w[u(j)] = interior(j, w[u(j)]);
  compute_ax();
  for (Index i = 0; i < m; ++i) {
    const Index k = n + i;
    if (excluded[u(k)]) w[u(k)] = ax[u(i)];
    else w[u(k)] = interior(k, ls_ok ? w[u(k)] : ax[u(i)]);
  }
  if (has_q) quad_multiply(Q, std::span<const double>(w.data(), u(n)), qx);
  Index nbounds = 0;
  {
    double gmax = 0.0;
    for (Index j = 0; j < n; ++j) gmax = std::max(gmax, std::fabs(c[u(j)] + qx[u(j)]));
    const double zeta = std::max(1.0, 0.01 * gmax);
    for (Index k = 0; k < N; ++k) {
      if (fixed[u(k)] || excluded[u(k)]) continue;
      const double g = k < n ? c[u(k)] + qx[u(k)] : 0.0;
      if (hl[u(k)]) zl[u(k)] = std::max(g, 0.0) + zeta;
      if (hu[u(k)]) zu[u(k)] = std::max(-g, 0.0) + zeta;
    }
    // Centre: every product s * z at least a tenth of the average product.
    double sum = 0.0;
    Index cnt = 0;
    for (Index k = 0; k < N; ++k) {
      if (fixed[u(k)] || excluded[u(k)]) continue;
      if (hl[u(k)]) {
        sum += (w[u(k)] - lo[u(k)]) * zl[u(k)];
        ++cnt;
      }
      if (hu[u(k)]) {
        sum += (up[u(k)] - w[u(k)]) * zu[u(k)];
        ++cnt;
      }
    }
    const double avg = cnt > 0 ? sum / static_cast<double>(cnt) : 0.0;
    for (Index k = 0; k < N; ++k) {
      if (fixed[u(k)] || excluded[u(k)]) continue;
      if (hl[u(k)]) zl[u(k)] = std::max(zl[u(k)], 0.1 * avg / (w[u(k)] - lo[u(k)]));
      if (hu[u(k)]) zu[u(k)] = std::max(zu[u(k)], 0.1 * avg / (up[u(k)] - w[u(k)]));
    }
    nbounds = cnt;
  }

  // ---- work vectors ----
  std::vector<double> rp(u(m)), rd(u(N)), rcl(u(N)), rcu(u(N)), rhs(u(nk)), sol(u(nk));
  std::vector<double> dw(u(N)), dzl(u(N)), dzu(u(N)), dy(u(m));
  std::vector<double> dwa(u(N)), dzla(u(N)), dzua(u(N)), dya(u(m));
  std::vector<double> work_axabs(u(m)), work_atyabs(u(n));

  auto residuals = [&]() {
    std::fill(aty.begin(), aty.end(), 0.0);
    for (Index j = 0; j < n; ++j) {
      double s = 0.0;
      for (Index p = A.col_start[u(j)]; p < A.col_start[u(j) + 1]; ++p) s += A.value[u(p)] * y[u(A.row_index[u(p)])];
      aty[u(j)] = s;
    }
    compute_ax();
    if (has_q) quad_multiply(Q, std::span<const double>(w.data(), u(n)), qx);
    for (Index i = 0; i < m; ++i) rp[u(i)] = excluded[u(n + i)] ? 0.0 : ax[u(i)] - w[u(n + i)];
    for (Index j = 0; j < n; ++j) rd[u(j)] = fixed[u(j)] ? 0.0 : c[u(j)] + qx[u(j)] - aty[u(j)] - zl[u(j)] + zu[u(j)];
    for (Index i = 0; i < m; ++i) {
      const Index k = n + i;
      rd[u(k)] = (excluded[u(k)] || fixed[u(k)]) ? 0.0 : y[u(i)] - zl[u(k)] + zu[u(k)];
    }
  };
  auto complementarity = [&](double* mu_out) {
    double s = 0.0;
    for (Index k = 0; k < N; ++k) {
      if (fixed[u(k)] || excluded[u(k)]) continue;
      if (hl[u(k)]) s += (w[u(k)] - lo[u(k)]) * zl[u(k)];
      if (hu[u(k)]) s += (up[u(k)] - w[u(k)]) * zu[u(k)];
    }
    *mu_out = nbounds > 0 ? s / static_cast<double>(nbounds) : 0.0;
    return s;
  };

  // Theta^-1 of variable k: z_l / s_l + z_u / s_u over its finite bounds.
  auto theta_of = [&](Index k) {
    double th = 0.0;
    if (hl[u(k)]) th += zl[u(k)] / (w[u(k)] - lo[u(k)]);
    if (hu[u(k)]) th += zu[u(k)] / (up[u(k)] - w[u(k)]);
    return th > 0.0 ? std::min(std::max(th, opt.theta_floor), opt.theta_cap) : th;
  };

  std::vector<Index> row_from_dx;
  const double kSmallTheta = 1e-2;  // Theta^-1 of a row below which dr and dy are recomputed from the r-row equations
  std::vector<double> adx_work(u(m), 0.0);

  // One Newton solve. rcl/rcu are filled by the caller. Fills dw, dy, dzl, dzu. Returns false on a failed solve.
  const std::vector<double>* prp = &rp;  // residuals the Newton solve works with (zero for centrality correctors)
  const std::vector<double>* prd = &rd;
  auto newton = [&](std::vector<double>& out_dw, std::vector<double>& out_dy, std::vector<double>& out_dzl, std::vector<double>& out_dzu) {
    std::fill(rhs.begin(), rhs.end(), 0.0);
    // rhs1 = -rd - (rcl / s_l - rcu / s_u)
    for (Index k = 0; k < N; ++k) {
      if (fixed[u(k)] || excluded[u(k)]) continue;
      double r1 = -(*prd)[u(k)];
      if (hl[u(k)]) r1 -= rcl[u(k)] / (w[u(k)] - lo[u(k)]);
      if (hu[u(k)]) r1 += rcu[u(k)] / (up[u(k)] - w[u(k)]);
      if (k < n) {
        rhs[u(kidx[u(k)])] = r1;
      } else {
        const Index i = k - n;
        rhs[u(kidx[u(k)])] = -(*prp)[u(i)] + r1 / std::max(theta_of(k), 1e-300);
      }
    }
    for (Index i = 0; i < m; ++i) {
      const Index k = n + i;
      if (fixed[u(k)] && !excluded[u(k)]) rhs[u(kidx[u(k)])] = -(*prp)[u(i)];
    }
    ldl.solve(rhs, sol, opt.refinement_steps, 0.0);
    res.refinement_steps += ldl.stats().refinement_steps;
    if (opt.verbosity >= 2 && opt.log != nullptr) {
      *opt.log << "      solve: backward error " << ldl.stats().residual << " after " << ldl.stats().refinement_steps << " refinement steps, dynamic regularizations "
               << ldl.stats().dynamic_regularizations << ", min pivot " << ldl.stats().min_pivot << "\n";
    }
    for (const double v : sol) {
      if (!std::isfinite(v)) return false;
    }
    std::fill(out_dw.begin(), out_dw.end(), 0.0);
    std::fill(out_dy.begin(), out_dy.end(), 0.0);
    row_from_dx.clear();
    for (Index j = 0; j < n; ++j) {
      if (kidx[u(j)] >= 0) out_dw[u(j)] = sol[u(kidx[u(j)])];
    }
    for (Index i = 0; i < m; ++i) {
      const Index k = n + i;
      if (excluded[u(k)]) continue;
      const double v = sol[u(kidx[u(k)])];
      out_dy[u(i)] = -v;
      if (fixed[u(k)]) {
        out_dw[u(k)] = 0.0;
      } else {
        double r1 = -(*prd)[u(k)];
        if (hl[u(k)]) r1 -= rcl[u(k)] / (w[u(k)] - lo[u(k)]);
        if (hu[u(k)]) r1 += rcu[u(k)] / (up[u(k)] - w[u(k)]);
        // Rows near a bound (large Theta^-1): the multiplier is accurate in v and dr = (rhs1_r + v) / Theta^-1.
        // Inactive rows (small Theta^-1, a huge 1/Theta in the matrix): the matrix row cannot give dr accurately, so
        // dr = A dx + rp restores primal feasibility exactly and dy = rhs1_r - Theta^-1 dr (r-row stationarity).
        const double th = std::max(theta_of(k), 1e-300);
        if (th >= kSmallTheta) {
          out_dw[u(k)] = (r1 + v) / th;
        } else {
          row_from_dx.push_back(i);
        }
      }
    }
    // Rows of the second kind: dr = A dx + rp, dy = rhs1_r - Theta^-1 dr.
    if (!row_from_dx.empty()) {
      std::fill(adx_work.begin(), adx_work.end(), 0.0);
      for (Index j = 0; j < n; ++j) {
        const double dxj = out_dw[u(j)];
        if (dxj == 0.0) continue;
        for (Index p = A.col_start[u(j)]; p < A.col_start[u(j) + 1]; ++p) adx_work[u(A.row_index[u(p)])] += A.value[u(p)] * dxj;
      }
      for (const Index i : row_from_dx) {
        const Index k = n + i;
        double r1 = -(*prd)[u(k)];
        if (hl[u(k)]) r1 -= rcl[u(k)] / (w[u(k)] - lo[u(k)]);
        if (hu[u(k)]) r1 += rcu[u(k)] / (up[u(k)] - w[u(k)]);
        const double dr_new = adx_work[u(i)] + (*prp)[u(i)];
        out_dw[u(k)] = dr_new;
        out_dy[u(i)] = r1 - theta_of(k) * dr_new;
      }
    }
    if (opt.verbosity >= 2 && opt.log != nullptr) {
      // Diagnostic: error of the primal Newton equation A dx - dr + rp = 0 per row.
      std::fill(adx_work.begin(), adx_work.end(), 0.0);
      for (Index j = 0; j < n; ++j) {
        for (Index p = A.col_start[u(j)]; p < A.col_start[u(j) + 1]; ++p) adx_work[u(A.row_index[u(p)])] += A.value[u(p)] * out_dw[u(j)];
      }
      double worst = 0.0;
      Index wi = -1;
      for (Index i = 0; i < m; ++i) {
        if (excluded[u(n + i)]) continue;
        const double e = std::fabs(adx_work[u(i)] - out_dw[u(n + i)] + (*prp)[u(i)]);
        if (e > worst) {
          worst = e;
          wi = i;
        }
      }
      if (wi >= 0) *opt.log << "      newton: primal equation error " << worst << " in row " << wi << (fixed[u(n + wi)] ? " (equality)" : "") << " theta " << theta_of(n + wi) << " dr " << out_dw[u(n + wi)] << " dy " << out_dy[u(wi)] << "\n";
    }
    for (Index k = 0; k < N; ++k) {
      out_dzl[u(k)] = out_dzu[u(k)] = 0.0;
      if (fixed[u(k)] || excluded[u(k)]) continue;
      if (hl[u(k)]) out_dzl[u(k)] = (-rcl[u(k)] - zl[u(k)] * out_dw[u(k)]) / (w[u(k)] - lo[u(k)]);
      if (hu[u(k)]) out_dzu[u(k)] = (-rcu[u(k)] + zu[u(k)] * out_dw[u(k)]) / (up[u(k)] - w[u(k)]);
    }
    return true;
  };

  // Largest step in [0, 1] keeping slacks and multipliers positive (fraction to the boundary tau).
  auto max_steps = [&](const std::vector<double>& ddw, const std::vector<double>& ddzl, const std::vector<double>& ddzu, double tau,
                       double* ap, double* ad) {
    double a_p = 1.0, a_d = 1.0;
    for (Index k = 0; k < N; ++k) {
      if (fixed[u(k)] || excluded[u(k)]) continue;
      if (hl[u(k)]) {
        const double s = w[u(k)] - lo[u(k)];
        if (ddw[u(k)] < 0.0) a_p = std::min(a_p, -tau * s / ddw[u(k)]);
        if (ddzl[u(k)] < 0.0) a_d = std::min(a_d, -tau * zl[u(k)] / ddzl[u(k)]);
      }
      if (hu[u(k)]) {
        const double s = up[u(k)] - w[u(k)];
        if (ddw[u(k)] > 0.0) a_p = std::min(a_p, tau * s / ddw[u(k)]);
        if (ddzu[u(k)] < 0.0) a_d = std::min(a_d, -tau * zu[u(k)] / ddzu[u(k)]);
      }
    }
    *ap = a_p;
    *ad = a_d;
  };

  double best_merit = kInf;
  int best_it = 0, tiny_steps = 0;
  for (int it = 0; it <= opt.max_iterations; ++it) {
    residuals();
    double mu = 0.0;
    const double compl_sum = complementarity(&mu);
    double pobj = offset;
    for (Index j = 0; j < n; ++j) pobj += c[u(j)] * w[u(j)];
    if (has_q) {
      for (Index j = 0; j < n; ++j) pobj += 0.5 * qx[u(j)] * w[u(j)];
    }
    // Componentwise relative residuals, like check_kkt: every row and column is measured against the magnitude of the
    // terms it is computed from, so a residual that check_kkt would reject is not accepted here.
    std::vector<double>& axabs = work_axabs;
    std::vector<double>& atyabs = work_atyabs;
    std::fill(axabs.begin(), axabs.end(), 0.0);
    for (Index j = 0; j < n; ++j) {
      double acc = 0.0;
      for (Index p = A.col_start[u(j)]; p < A.col_start[u(j) + 1]; ++p) {
        const Index i = A.row_index[u(p)];
        axabs[u(i)] += std::fabs(A.value[u(p)] * w[u(j)]);
        acc += std::fabs(A.value[u(p)] * y[u(i)]);
      }
      atyabs[u(j)] = acc;
    }
    double pres = 0.0, dres = 0.0;
    for (Index i = 0; i < m; ++i) {
      if (excluded[u(n + i)]) continue;
      pres = std::max(pres, std::fabs(rp[u(i)]) / (1.0 + axabs[u(i)] + std::fabs(w[u(n + i)])));
      if (!fixed[u(n + i)]) dres = std::max(dres, std::fabs(rd[u(n + i)]) / (1.0 + std::fabs(y[u(i)]) + zl[u(n + i)] + zu[u(n + i)]));
    }
    for (Index j = 0; j < n; ++j) {
      if (fixed[u(j)]) continue;
      dres = std::max(dres, std::fabs(rd[u(j)]) / (1.0 + std::fabs(c[u(j)]) + std::fabs(qx[u(j)]) + atyabs[u(j)] + zl[u(j)] + zu[u(j)]));
    }
    const double gap = compl_sum / (1.0 + std::fabs(pobj));
    res.iterations = it;
    res.primal_residual = pres;
    res.dual_residual = dres;
    res.gap = gap;
    IpmIteration rec;
    rec.iteration = it;
    rec.primal_residual = pres;
    rec.dual_residual = dres;
    rec.gap = gap;
    rec.mu = mu;
    if (opt.verbosity >= 1 && opt.log != nullptr) {
      *opt.log << std::setw(4) << it << "  pobj " << std::setprecision(10) << pobj << "  pres " << std::setprecision(3) << std::scientific << pres << "  dres " << dres
               << "  gap " << gap << "  mu " << mu << std::defaultfloat << "\n";
    }
    if (!std::isfinite(pres) || !std::isfinite(dres) || !std::isfinite(gap)) {
      res.history.push_back(rec);
      return finish(Status::NumericalError, "non-finite residuals");
    }
    if (pres <= opt.tol && dres <= opt.tol && gap <= opt.tol) {
      res.history.push_back(rec);
      break;
    }
    if (it == opt.max_iterations) {
      res.history.push_back(rec);
      return finish(Status::IterationLimit, "iteration limit");
    }
    if (!is_inf(opt.time_limit) && std::chrono::duration<double>(Clock::now() - t0).count() >= opt.time_limit) {
      res.history.push_back(rec);
      return finish(Status::TimeLimit, "time limit");
    }
    // Divergence: iterates or multipliers far beyond any scale of the data.
    {
      double wmax = 0.0, zmax = norm_inf(y);
      for (Index k = 0; k < N; ++k) {
        if (excluded[u(k)]) continue;
        wmax = std::max(wmax, std::fabs(w[u(k)]));
        zmax = std::max(zmax, std::max(zl[u(k)], zu[u(k)]));
      }
      if (it >= 3 && (wmax > 1e13 || zmax > 1e13)) {
        res.history.push_back(rec);
        return finish(Status::InfeasibleOrUnbounded, "the iterates diverge (primal or dual infeasibility, not certified)");
      }
    }
    const double merit = std::max(pres, std::max(dres, gap));
    if (merit < 0.9 * best_merit) {
      best_merit = merit;
      best_it = it;
    } else if (it - best_it >= opt.stall_iterations) {
      res.history.push_back(rec);
      return finish(Status::NumericalError, "stagnation: the residuals stopped decreasing");
    }

    // ---- matrix values ----
    for (Index j = 0; j < n; ++j) {
      if (kidx[u(j)] < 0) continue;
      K.value[u(dpos[u(kidx[u(j)])])] = qdiag[u(j)] + theta_of(j);
    }
    for (Index i = 0; i < m; ++i) {
      const Index k = n + i;
      if (kidx[u(k)] < 0) continue;
      const double th = fixed[u(k)] ? 0.0 : theta_of(k);
      K.value[u(dpos[u(kidx[u(k)])])] = fixed[u(k)] ? 0.0 : -1.0 / std::max(th, 1e-13);
    }
    if (nk > 0) {
      if (!ldl.factorize(K)) {
        res.history.push_back(rec);
        return finish(Status::NumericalError, "factorization of the augmented system failed");
      }
      ++res.factorizations;
      res.dynamic_regularizations += ldl.stats().dynamic_regularizations;
    }

    // ---- predictor ----
    for (Index k = 0; k < N; ++k) {
      rcl[u(k)] = rcu[u(k)] = 0.0;
      if (fixed[u(k)] || excluded[u(k)]) continue;
      if (hl[u(k)]) rcl[u(k)] = (w[u(k)] - lo[u(k)]) * zl[u(k)];
      if (hu[u(k)]) rcu[u(k)] = (up[u(k)] - w[u(k)]) * zu[u(k)];
    }
    if (nk > 0 && !newton(dwa, dya, dzla, dzua)) {
      res.history.push_back(rec);
      return finish(Status::NumericalError, "non-finite Newton direction");
    }
    double ap = 1.0, ad = 1.0;
    max_steps(dwa, dzla, dzua, 1.0, &ap, &ad);
    if (has_q) ap = ad = std::min(ap, ad);
    double sigma = 0.0;
    if (nbounds > 0 && mu > 0.0) {
      double s = 0.0;
      for (Index k = 0; k < N; ++k) {
        if (fixed[u(k)] || excluded[u(k)]) continue;
        if (hl[u(k)]) s += (w[u(k)] - lo[u(k)] + ap * dwa[u(k)]) * (zl[u(k)] + ad * dzla[u(k)]);
        if (hu[u(k)]) s += (up[u(k)] - w[u(k)] - ap * dwa[u(k)]) * (zu[u(k)] + ad * dzua[u(k)]);
      }
      const double mu_aff = std::max(s, 0.0) / static_cast<double>(nbounds);
      sigma = std::pow(mu_aff / mu, 3.0);
      sigma = std::min(1.0, std::max(sigma, 0.0));
      // Do not let the complementarity collapse far below the infeasibilities: an infeasible-start method needs mu to
      // decrease no faster than the residuals, otherwise the Newton matrix becomes extremely ill conditioned first.
      if (gap < 0.01 * std::max(pres, dres)) sigma = std::max(sigma, 0.3);
    }
    if (opt.verbosity >= 1 && opt.log != nullptr) *opt.log << "      affine step " << ap << " " << ad << " sigma " << sigma << "\n";
    // ---- corrector ----
    const double second = std::min(1.0, std::min(ap, ad));  // the second-order term is only accurate for the affine step
    for (Index k = 0; k < N; ++k) {
      rcl[u(k)] = rcu[u(k)] = 0.0;
      if (fixed[u(k)] || excluded[u(k)]) continue;
      if (hl[u(k)]) rcl[u(k)] = (w[u(k)] - lo[u(k)]) * zl[u(k)] + second * dwa[u(k)] * dzla[u(k)] - sigma * mu;
      if (hu[u(k)]) rcu[u(k)] = (up[u(k)] - w[u(k)]) * zu[u(k)] - second * dwa[u(k)] * dzua[u(k)] - sigma * mu;
    }
    if (nk > 0 && !newton(dw, dy, dzl, dzu)) {
      res.history.push_back(rec);
      return finish(Status::NumericalError, "non-finite Newton direction");
    }
    const double tau = std::max(opt.step_fraction, 1.0 - mu);
    max_steps(dw, dzl, dzu, std::min(tau, 0.99999), &ap, &ad);
    if (has_q) ap = ad = std::min(ap, ad);
    // Multiple centrality correctors (Gondzio 1996): push the products s*z of the trial point into [0.1, 10] times the
    // target and take the extra direction if it lengthens the step.
    if (opt.centrality_correctors > 0 && nk > 0 && nbounds > 0) {
      const std::vector<double> zero_rp(u(m), 0.0), zero_rd(u(N), 0.0);
      std::vector<double> cw(u(N)), cy(u(m)), czl(u(N)), czu(u(N)), tw(u(N)), ty(u(m)), tzl(u(N)), tzu(u(N));
      const double mu_target = sigma * mu;
      for (int g = 0; g < opt.centrality_correctors; ++g) {
        const double atp = std::min(1.0, ap + 0.1), atd = std::min(1.0, ad + 0.1);
        bool any = false;
        for (Index k = 0; k < N; ++k) {
          rcl[u(k)] = rcu[u(k)] = 0.0;
          if (fixed[u(k)] || excluded[u(k)]) continue;
          if (hl[u(k)]) {
            const double v = (w[u(k)] - lo[u(k)] + atp * dw[u(k)]) * (zl[u(k)] + atd * dzl[u(k)]);
            double corr = std::min(std::max(v, 0.1 * mu_target), 10.0 * mu_target) - v;
            corr = std::max(corr, -10.0 * mu_target);
            rcl[u(k)] = -corr;
            any = any || corr != 0.0;
          }
          if (hu[u(k)]) {
            const double v = (up[u(k)] - w[u(k)] - atp * dw[u(k)]) * (zu[u(k)] + atd * dzu[u(k)]);
            double corr = std::min(std::max(v, 0.1 * mu_target), 10.0 * mu_target) - v;
            corr = std::max(corr, -10.0 * mu_target);
            rcu[u(k)] = -corr;
            any = any || corr != 0.0;
          }
        }
        if (!any) break;
        prp = &zero_rp;
        prd = &zero_rd;
        const bool ok = newton(cw, cy, czl, czu);
        prp = &rp;
        prd = &rd;
        if (!ok) break;
        for (Index k = 0; k < N; ++k) {
          tw[u(k)] = dw[u(k)] + cw[u(k)];
          tzl[u(k)] = dzl[u(k)] + czl[u(k)];
          tzu[u(k)] = dzu[u(k)] + czu[u(k)];
        }
        for (Index i = 0; i < m; ++i) ty[u(i)] = dy[u(i)] + cy[u(i)];
        double nap = 1.0, nad = 1.0;
        max_steps(tw, tzl, tzu, std::min(tau, 0.99999), &nap, &nad);
        if (has_q) nap = nad = std::min(nap, nad);
        if (std::min(nap, nad) >= std::min(ap, ad) + 0.01 * 0.1) {
          dw = tw;
          dzl = tzl;
          dzu = tzu;
          dy = ty;
          ap = nap;
          ad = nad;
        } else {
          break;
        }
      }
    }
    rec.step_primal = ap;
    rec.step_dual = ad;
    if (opt.verbosity >= 1 && opt.log != nullptr) *opt.log << "      step primal " << ap << " dual " << ad << " sigma " << sigma << "\n";
    res.history.push_back(rec);
    if (std::min(ap, ad) < 1e-9) {
      if (++tiny_steps >= 5) return finish(Status::NumericalError, "step lengths collapsed");
    } else {
      tiny_steps = 0;
    }
    for (Index k = 0; k < N; ++k) {
      if (fixed[u(k)] || excluded[u(k)]) continue;
      w[u(k)] += ap * dw[u(k)];
      zl[u(k)] += ad * dzl[u(k)];
      zu[u(k)] += ad * dzu[u(k)];
    }
    for (Index i = 0; i < m; ++i) y[u(i)] += ad * dy[u(i)];
  }

  // ---- solution in the original orientation ----
  res.solution.x.assign(w.begin(), w.begin() + n);
  res.solution.y = y;
  res.solution.d.assign(u(n), 0.0);
  for (Index j = 0; j < n; ++j) res.solution.d[u(j)] = c[u(j)] + qx[u(j)] - aty[u(j)];
  {
    double obj = offset;
    for (Index j = 0; j < n; ++j) obj += c[u(j)] * w[u(j)] + 0.5 * qx[u(j)] * w[u(j)];
    res.solution.objective = sgn * obj;
  }
  return finish(Status::Optimal, "");
}

}  // namespace

IpmResult solve_ipm(const LpModel& model, const IpmOptions& options) {
  // A numerical failure is retried with a larger regularization, a shorter step to the boundary and a floor on Theta,
  // which trade speed for robustness (docs/IPM.md). Divergence and limits are final.
  IpmResult res = solve_ipm_attempt(model, options);
  res.attempts = 1;
  if (res.status != Status::NumericalError) return res;
  IpmOptions o = options;
  const double reg[] = {1e-8, 1e-6};
  for (const double r : reg) {
    o.rho = std::max(options.rho, r);
    o.delta = std::max(options.delta, r);
    o.theta_floor = std::max(options.theta_floor, r * 1e-2);
    o.step_fraction = std::min(options.step_fraction, 0.9);
    IpmResult next = solve_ipm_attempt(model, o);
    next.attempts = res.attempts + 1;
    next.seconds += res.seconds;
    if (next.status != Status::NumericalError) return next;
    res = std::move(next);
  }
  return res;
}

}  // namespace shodhan
