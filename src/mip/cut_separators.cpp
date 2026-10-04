// Cut separators (docs/CUTS.md). Every separator returns inequalities  sum val x <= rhs  that hold for all
// integer-feasible points of the presolved model; the derivations are in the comments of each class. The
// references are the standard ones: Gomory (1960) and Cornuejols, Li, Vandenbussche (2003) for mixed-integer
// Gomory cuts; Nemhauser and Wolsey (1990) for MIR; Marchand and Wolsey (2001) for the aggregation heuristic;
// Balas (1975), Wolsey (1975) and Gu, Nemhauser, Savelsbergh (1999) for knapsack covers and their lifting;
// Atamturk, Nemhauser, Savelsbergh (2000) for clique cuts; Savelsbergh (1994) for implied bounds.

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>

#include "shodhan/mip/cuts.hpp"

namespace shodhan::mip {

namespace {

std::size_t u(Index i) { return static_cast<std::size_t>(i); }

bool is_binary_col(const LpModel& m, const std::vector<double>& lo, const std::vector<double>& hi, Index j) {
  return m.is_integer(j) && lo[u(j)] == 0.0 && hi[u(j)] == 1.0;
}

// Sparse accumulator over the columns.
class Acc {
 public:
  explicit Acc(Index n = 0) : v_(u(n), 0.0), mark_(u(n), 0) {}
  void add(Index j, double a) {
    if (!mark_[u(j)]) {
      mark_[u(j)] = 1;
      list_.push_back(j);
    }
    v_[u(j)] += a;
  }
  double operator[](Index j) const { return v_[u(j)]; }
  void set(Index j, double a) {
    if (!mark_[u(j)]) {
      mark_[u(j)] = 1;
      list_.push_back(j);
    }
    v_[u(j)] = a;
  }
  const std::vector<Index>& list() const { return list_; }
  void clear() {
    for (const Index j : list_) {
      v_[u(j)] = 0.0;
      mark_[u(j)] = 0;
    }
    list_.clear();
  }

 private:
  std::vector<double> v_;
  std::vector<char> mark_;
  std::vector<Index> list_;
};

Cut make_cut(const Acc& g, double rhs, int separator) {
  Cut c;
  c.rhs = rhs;
  c.separator = separator;
  for (const Index j : g.list()) {
    if (g[j] != 0.0) {
      c.idx.push_back(j);
      c.val.push_back(g[j]);
    }
  }
  return c;
}

double row_activity(const CsrMatrix& r, Index i, const std::vector<double>& x) {
  double s = 0.0;
  for (Index t = r.row_start[u(i)]; t < r.row_start[u(i) + 1]; ++t) s += r.value[u(t)] * x[u(r.col_index[u(t)])];
  return s;
}

// ======================================================================================================
// Gomory mixed-integer cuts from the optimal simplex tableau.
//
// For a basic integer column x_B with fractional value, the tableau row reads  x_B + sum_j a_j x_j = 0 over
// the nonbasic variables j of the computational form (structural columns and the logical variables r_i = row
// activities). Shifting every nonbasic variable to its bound (t_j = x_j - l_j at a lower bound, t_j = u_j - x_j at
// an upper bound, t_j >= 0) gives  x_B + sum a'_j t_j = beta  with beta = -sum a_j (bound value of j), and
// f0 = frac(beta) in (0,1). With t_j integer for integer columns (integral bounds) and continuous for logicals
// and continuous columns, the Gomory mixed-integer cut is
//     sum_{j integer} min(f_j / f0, (1 - f_j) / (1 - f0)) t_j
//   + sum_{j continuous} (a'_j >= 0 ? a'_j / f0 : -a'_j / (1 - f0)) t_j   >=  1,        f_j = frac(a'_j).
// The logical variables are substituted by their rows, so the cut lives on the structural columns; it is derived
// in the engine's scaled space (x' = x / col_scale, integer columns have scale 1) and returned unscaled.
class GomorySeparator final : public Separator {
 public:
  int id() const override { return kSepGomory; }
  void separate(const CutData& d, std::vector<Cut>& out) override {
    if (d.engine == nullptr || d.scaling == nullptr) return;
    SimplexEngine& e = *d.engine;
    const LpModel& M = *d.model;
    const Index n = e.n_structural();
    const Index m = e.n_rows();
    const std::vector<double>& xe = e.primal_all();
    const std::vector<Index>& basis = e.basis();
    struct Cand {
      Index pos;
      double key;
    };
    std::vector<Cand> cands;
    for (Index p = 0; p < m; ++p) {
      const Index v = basis[u(p)];
      if (v >= n || !M.is_integer(v)) continue;
      const double f = xe[u(v)] - std::floor(xe[u(v)]);
      if (f < 0.01 || f > 0.99) continue;
      cands.push_back({p, std::fabs(f - 0.5)});
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.key < b.key; });
    if (static_cast<int>(cands.size()) > d.options->cut_gomory_rows) cands.resize(u(d.options->cut_gomory_rows));
    const CsrMatrix& csr = e.row_csr();
    Acc g(n);
    std::vector<double> alpha;
    for (const Cand& c : cands) {
      e.tableau_row(c.pos, alpha);
      g.clear();
      // beta from the nonbasic values with the tableau row itself, so the cut is consistent with it.
      double beta = 0.0;
      bool ok = true;
      for (Index j = 0; j < e.n_structural() + m && ok; ++j) {
        const double a = alpha[u(j)];
        if (a == 0.0) continue;
        switch (e.status(j)) {
          case VarStatus::AtLower:
            beta -= a * (j < n ? e.col_lower(j) : e.row_lower(j - n));
            break;
          case VarStatus::AtUpper:
            beta -= a * (j < n ? e.col_upper(j) : e.row_upper(j - n));
            break;
          case VarStatus::Fixed:
            beta -= a * (j < n ? e.col_lower(j) : e.row_lower(j - n));
            break;
          case VarStatus::FreeAtZero:
            ok = false;
            break;
          case VarStatus::Basic:
            break;
        }
      }
      if (!ok) continue;
      const Index bv = basis[u(c.pos)];
      if (std::fabs(beta - xe[u(bv)]) > 1e-6 * (1.0 + std::fabs(beta))) continue;  // tableau row and values disagree
      const double f0 = beta - std::floor(beta);
      if (f0 < 0.01 || f0 > 0.99) continue;
      double rhs_ge = 1.0;
      for (Index j = 0; j < n + m; ++j) {
        const double a = alpha[u(j)];
        if (a == 0.0) continue;
        const VarStatus st = e.status(j);
        if (st == VarStatus::Fixed || st == VarStatus::Basic) continue;
        const bool lower = st == VarStatus::AtLower;
        const double ap = lower ? a : -a;  // coefficient of t_j
        double gamma;
        if (j < n && M.is_integer(j)) {
          double fj = ap - std::floor(ap);
          if (fj < 1e-9 || fj > 1.0 - 1e-9) fj = 0.0;
          gamma = fj <= f0 ? fj / f0 : (1.0 - fj) / (1.0 - f0);
        } else {
          gamma = ap >= 0.0 ? ap / f0 : -ap / (1.0 - f0);
        }
        if (gamma == 0.0) continue;
        // gamma * t_j with t_j = var - L (lower) or U - var (upper).
        const double bound = j < n ? (lower ? e.col_lower(j) : e.col_upper(j)) : (lower ? e.row_lower(j - n) : e.row_upper(j - n));
        const double sgn = lower ? 1.0 : -1.0;  // coefficient of var
        rhs_ge += sgn * gamma * bound;
        if (j < n) {
          g.add(j, sgn * gamma);
        } else {
          const Index i = j - n;
          for (Index t = csr.row_start[u(i)]; t < csr.row_start[u(i) + 1]; ++t) {
            g.add(csr.col_index[u(t)], sgn * gamma * csr.value[u(t)]);
          }
        }
      }
      // sum g'_k x'_k >= rhs_ge, x'_k = x_k / col_scale_k:  -sum (g'_k / s_k) x_k <= -rhs_ge.
      Cut cut;
      cut.separator = kSepGomory;
      cut.rhs = -rhs_ge;
      for (const Index k : g.list()) {
        if (g[k] == 0.0) continue;
        cut.idx.push_back(k);
        cut.val.push_back(-g[k] / d.scaling->col_scale[u(k)]);
      }
      if (!cut.idx.empty()) out.push_back(std::move(cut));
    }
  }
};

// ======================================================================================================
// Mixed-integer rounding cuts with row aggregation (complemented MIR, Marchand and Wolsey).
//
// Starting from a row that is tight at the LP point, up to `cut_mir_aggregation` rows are added (multipliers >= 0,
// so the sum is a valid inequality  sum a x <= b) to eliminate continuous columns that are strictly between their
// bounds. Continuous columns get their closest simple or variable bound (x_b = v implications from the
// presolve structure): y = bound + y' or bound - y' with y' >= 0; integer columns are shifted or complemented to
// [0, range]. With delta > 0 and f0 = frac(beta / delta) in [0.05, 0.95] the MIR inequality
//    sum_j (floor(a_j/delta) + max(0, f_j - f0) / (1 - f0)) x'_j + sum_{c_k < 0} c_k / (delta (1 - f0)) y'_k <= floor(beta/delta)
// is valid for integer x' >= 0 and y' >= 0; the best delta among the |a_j| of integer columns strictly between their
// bounds (and /2, /4, /8 of the best) is used.
struct VarBound {
  Index xb = -1;
  double c0 = 0.0;
  double c1 = 0.0;
};

class MirSeparator final : public Separator {
 public:
  int id() const override { return kSepMir; }

  void separate(const CutData& d, std::vector<Cut>& out) override {
    d_ = &d;
    M_ = d.model;
    const Index n = M_->n_cols, m = M_->n_rows;
    build_variable_bounds();
    const CsrMatrix& R = *d.rows;
    act_.assign(u(m), 0.0);
    for (Index i = 0; i < m; ++i) act_[u(i)] = row_activity(R, i, *d.x);
    struct Start {
      Index row;
      int side;
      double slack;
    };
    std::vector<Start> starts;
    for (Index i = 0; i < m; ++i) {
      const double ru = M_->row_upper[u(i)], rl = M_->row_lower[u(i)];
      if (!is_inf(ru)) {
        const double s = ru - act_[u(i)];
        if (s <= slack_tol(ru)) starts.push_back({i, 0, s});
      }
      if (!is_inf(rl)) {
        const double s = act_[u(i)] - rl;
        if (s <= slack_tol(rl)) starts.push_back({i, 1, s});
      }
    }
    std::stable_sort(starts.begin(), starts.end(), [](const Start& a, const Start& b) { return a.slack < b.slack; });
    if (starts.size() > 400) starts.resize(400);
    Acc agg(n);
    ia_ = Acc(n);
    const int max_agg = std::max(1, d.options->cut_mir_aggregation);
    for (const Start& s : starts) {
      agg.clear();
      std::vector<Index> used{s.row};
      double rhs = 0.0;
      add_side(agg, rhs, s.row, s.side == 0 ? 1.0 : -1.0, 1.0);
      for (int depth = 0; depth < max_agg; ++depth) {
        Cut cut;
        if (try_cmir(agg, rhs, cut)) out.push_back(std::move(cut));
        if (depth + 1 == max_agg) break;
        if (!eliminate_one(agg, rhs, used)) break;
      }
    }
  }

 private:
  static double slack_tol(double bound) { return 1e-6 * (1.0 + std::fabs(bound)); }

  // sign * row_i <= bound(sign), scaled by lambda, added to the aggregate.
  void add_side(Acc& agg, double& rhs, Index i, double sign, double lambda) const {
    const CsrMatrix& R = *d_->rows;
    for (Index t = R.row_start[u(i)]; t < R.row_start[u(i) + 1]; ++t) agg.add(R.col_index[u(t)], lambda * sign * R.value[u(t)]);
    rhs += lambda * (sign > 0 ? M_->row_upper[u(i)] : -M_->row_lower[u(i)]);
  }

  void build_variable_bounds() {
    const Index n = M_->n_cols;
    vub_.assign(u(n), {});
    vlb_.assign(u(n), {});
    if (d_->structure == nullptr) return;
    const std::vector<double>& lo = *d_->lo;
    const std::vector<double>& hi = *d_->hi;
    for (const Implication& im : d_->structure->implications) {
      const Index y = im.other;
      const Index b = im.var;
      if (M_->is_integer(y) || !is_binary_col(*M_, lo, hi, b)) continue;
      if (im.is_upper) {
        const double U = hi[u(y)];
        if (is_inf(U)) continue;
        // x_b = v => y <= B, and y <= U otherwise.
        VarBound vb;
        vb.xb = b;
        if (im.var_value == 0) {
          vb.c0 = im.bound;
          vb.c1 = U - im.bound;
        } else {
          vb.c0 = U;
          vb.c1 = im.bound - U;
        }
        vub_[u(y)].push_back(vb);
      } else {
        const double L = lo[u(y)];
        if (is_inf(L)) continue;
        VarBound vb;
        vb.xb = b;
        if (im.var_value == 0) {
          vb.c0 = im.bound;
          vb.c1 = L - im.bound;
        } else {
          vb.c0 = L;
          vb.c1 = im.bound - L;
        }
        vlb_[u(y)].push_back(vb);
      }
    }
  }

  // Adds one tight row to eliminate the continuous column strictly between its bounds that is farthest from them.
  bool eliminate_one(Acc& agg, double& rhs, std::vector<Index>& used) {
    const std::vector<double>& lo = *d_->lo;
    const std::vector<double>& hi = *d_->hi;
    const std::vector<double>& x = *d_->x;
    struct Pick {
      Index col;
      double dist;
    };
    std::vector<Pick> picks;
    for (const Index j : agg.list()) {
      if (agg[j] == 0.0 || M_->is_integer(j) || lo[u(j)] == hi[u(j)]) continue;
      double dist = kInf;
      if (!is_inf(lo[u(j)])) dist = std::min(dist, x[u(j)] - lo[u(j)]);
      if (!is_inf(hi[u(j)])) dist = std::min(dist, hi[u(j)] - x[u(j)]);
      if (dist > 1e-6) picks.push_back({j, std::min(dist, 1e12)});
    }
    std::sort(picks.begin(), picks.end(), [](const Pick& a, const Pick& b) { return a.dist > b.dist || (a.dist == b.dist && a.col < b.col); });
    const SparseMatrix& A = M_->A;
    for (std::size_t q = 0; q < picks.size() && q < 3; ++q) {
      const Index j = picks[q].col;
      Index best_row = -1;
      double best_sign = 0.0, best_slack = kInf;
      Index best_len = 0;
      for (Index t = A.col_start[u(j)]; t < A.col_start[u(j) + 1]; ++t) {
        const Index r = A.row_index[u(t)];
        if (std::find(used.begin(), used.end(), r) != used.end()) continue;
        for (int side = 0; side < 2; ++side) {
          const double sign = side == 0 ? 1.0 : -1.0;
          const double bound = side == 0 ? M_->row_upper[u(r)] : M_->row_lower[u(r)];
          if (is_inf(bound)) continue;
          const double slack = side == 0 ? bound - act_[u(r)] : act_[u(r)] - bound;
          if (slack > slack_tol(bound)) continue;
          const double e = sign * A.value[u(t)];
          if (e * agg[j] >= 0.0) continue;
          const Index len = d_->rows->row_start[u(r) + 1] - d_->rows->row_start[u(r)];
          if (slack < best_slack - 1e-12 || (std::fabs(slack - best_slack) <= 1e-12 && len < best_len)) {
            best_row = r;
            best_sign = sign;
            best_slack = slack;
            best_len = len;
          }
        }
      }
      if (best_row < 0) continue;
      // Coefficient of column j in the chosen side.
      double e = 0.0;
      for (Index t = A.col_start[u(j)]; t < A.col_start[u(j) + 1]; ++t) {
        if (A.row_index[u(t)] == best_row) e = best_sign * A.value[u(t)];
      }
      const double lambda = -agg[j] / e;
      add_side(agg, rhs, best_row, best_sign, lambda);
      agg.set(j, 0.0);
      used.push_back(best_row);
      return true;
    }
    return false;
  }

  struct IntTerm {
    Index j;
    double c;       // coefficient of x' (x' = x - l, or u - x)
    bool upper;     // complemented: x' = u - x
    double range;   // u - l (infinite when the other bound is infinite)
    double xs;      // x' at the LP point
  };
  struct ContTerm {
    Index j;
    double c;       // coefficient of y' in the row
    int mode;       // 0: y = L + y'  1: y = U - y'  2: y = (c0 + c1 xb) + y'  3: y = (c0 + c1 xb) - y'
    VarBound vb;
    double ys;      // y' at the LP point
  };

  bool try_cmir(const Acc& agg, double rhs, Cut& out) {
    const std::vector<double>& lo = *d_->lo;
    const std::vector<double>& hi = *d_->hi;
    const std::vector<double>& x = *d_->x;
    Acc& ia = ia_;
    ia.clear();
    std::vector<ContTerm> cont;
    double beta = rhs;
    bool has_int = false;
    for (const Index j : agg.list()) {
      const double a = agg[j];
      if (a == 0.0) continue;
      if (lo[u(j)] == hi[u(j)]) {
        beta -= a * lo[u(j)];
        continue;
      }
      if (M_->is_integer(j)) {
        ia.add(j, a);
        continue;
      }
      // Continuous: closest bound, simple or variable.
      ContTerm ct;
      ct.j = j;
      double best = kInf;
      int mode = -1;
      VarBound vb;
      if (!is_inf(lo[u(j)]) && x[u(j)] - lo[u(j)] < best) {
        best = x[u(j)] - lo[u(j)];
        mode = 0;
      }
      if (!is_inf(hi[u(j)]) && hi[u(j)] - x[u(j)] < best - 1e-12) {
        best = hi[u(j)] - x[u(j)];
        mode = 1;
      }
      for (const VarBound& v : vlb_[u(j)]) {
        const double dist = x[u(j)] - (v.c0 + v.c1 * x[u(v.xb)]);
        if (dist < best - 1e-12) {
          best = dist;
          mode = 2;
          vb = v;
        }
      }
      for (const VarBound& v : vub_[u(j)]) {
        const double dist = (v.c0 + v.c1 * x[u(v.xb)]) - x[u(j)];
        if (dist < best - 1e-12) {
          best = dist;
          mode = 3;
          vb = v;
        }
      }
      if (mode < 0) return false;  // free continuous column: no valid transformation
      ct.mode = mode;
      ct.vb = vb;
      ct.ys = std::max(0.0, best);
      switch (mode) {
        case 0:
          beta -= a * lo[u(j)];
          ct.c = a;
          break;
        case 1:
          beta -= a * hi[u(j)];
          ct.c = -a;
          break;
        case 2:
          beta -= a * vb.c0;
          ia.add(vb.xb, a * vb.c1);
          ct.c = a;
          break;
        default:
          beta -= a * vb.c0;
          ia.add(vb.xb, a * vb.c1);
          ct.c = -a;
          break;
      }
      cont.push_back(ct);
    }
    std::vector<IntTerm> ints;
    for (const Index j : ia.list()) {
      const double a = ia[j];
      if (a == 0.0) continue;
      if (lo[u(j)] == hi[u(j)]) {
        beta -= a * lo[u(j)];
        continue;
      }
      IntTerm t;
      t.j = j;
      const bool flo = !is_inf(lo[u(j)]), fhi = !is_inf(hi[u(j)]);
      if (!flo && !fhi) return false;
      bool upper = !flo;
      if (flo && fhi) upper = hi[u(j)] - x[u(j)] < x[u(j)] - lo[u(j)];
      t.upper = upper;
      t.range = flo && fhi ? hi[u(j)] - lo[u(j)] : kInf;
      if (upper) {
        t.c = -a;
        beta -= a * hi[u(j)];
        t.xs = hi[u(j)] - x[u(j)];
      } else {
        t.c = a;
        beta -= a * lo[u(j)];
        t.xs = x[u(j)] - lo[u(j)];
      }
      t.xs = std::max(0.0, t.xs);
      ints.push_back(t);
      has_int = true;
    }
    if (!has_int) return false;
    // delta candidates: |c| of integer terms strictly between their bounds.
    std::vector<double> deltas;
    for (const IntTerm& t : ints) {
      if (t.xs > 1e-6 && t.xs < t.range - 1e-6) deltas.push_back(std::fabs(t.c));
    }
    std::sort(deltas.begin(), deltas.end());
    deltas.erase(std::unique(deltas.begin(), deltas.end(), [](double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(a, b); }),
                 deltas.end());
    if (deltas.empty()) return false;
    if (deltas.size() > 8) deltas.resize(8);
    auto efficacy = [&](double delta, bool* valid) {
      *valid = false;
      if (!(delta > 1e-9)) return 0.0;
      const double bd = beta / delta;
      const double f0 = bd - std::floor(bd);
      if (f0 < 0.05 || f0 > 0.95) return 0.0;
      double lhs = 0.0, nrm = 0.0;
      for (const IntTerm& t : ints) {
        const double v = t.c / delta;
        const double fl = std::floor(v);
        const double F = fl + std::max(0.0, v - fl - f0) / (1.0 - f0);
        lhs += F * t.xs;
        nrm += F * F;
      }
      for (const ContTerm& t : cont) {
        if (t.c >= 0.0) continue;
        const double G = t.c / (delta * (1.0 - f0));
        lhs += G * t.ys;
        nrm += G * G;
      }
      if (nrm <= 0.0) return 0.0;
      *valid = true;
      return (lhs - std::floor(bd)) / std::sqrt(nrm);
    };
    double best_eff = 0.0, best_delta = 0.0;
    auto consider = [&](double delta) {
      bool ok;
      const double eff = efficacy(delta, &ok);
      if (ok && eff > best_eff) {
        best_eff = eff;
        best_delta = delta;
      }
    };
    for (const double dl : deltas) consider(dl);
    if (!(best_eff > 0.0)) return false;
    const double base = best_delta;
    for (const double div : {2.0, 4.0, 8.0}) consider(base / div);
    // Build the cut for best_delta and map it back to the original columns.
    const double delta = best_delta;
    const double bd = beta / delta;
    const double f0 = bd - std::floor(bd);
    double r = std::floor(bd);
    Acc g(M_->n_cols);
    for (const IntTerm& t : ints) {
      const double v = t.c / delta;
      const double fl = std::floor(v);
      const double F = fl + std::max(0.0, v - fl - f0) / (1.0 - f0);
      if (F == 0.0) continue;
      if (t.upper) {  // x' = u - x
        g.add(t.j, -F);
        r -= F * hi[u(t.j)];
      } else {  // x' = x - l
        g.add(t.j, F);
        r += F * lo[u(t.j)];
      }
    }
    for (const ContTerm& t : cont) {
      if (t.c >= 0.0) continue;
      const double G = t.c / (delta * (1.0 - f0));
      switch (t.mode) {
        case 0:  // y' = y - L
          g.add(t.j, G);
          r += G * lo[u(t.j)];
          break;
        case 1:  // y' = U - y
          g.add(t.j, -G);
          r -= G * hi[u(t.j)];
          break;
        case 2:  // y' = y - c0 - c1 xb
          g.add(t.j, G);
          g.add(t.vb.xb, -G * t.vb.c1);
          r += G * t.vb.c0;
          break;
        default:  // y' = c0 + c1 xb - y
          g.add(t.j, -G);
          g.add(t.vb.xb, G * t.vb.c1);
          r -= G * t.vb.c0;
          break;
      }
    }
    out = make_cut(g, r, kSepMir);
    return !out.idx.empty();
  }

  const CutData* d_ = nullptr;
  const LpModel* M_ = nullptr;
  Acc ia_;
  std::vector<double> act_;
  std::vector<std::vector<VarBound>> vub_, vlb_;
};

// ======================================================================================================
// Lifted knapsack cover cuts.
//
// A row side  sum a_j x_j <= b  is relaxed to a 0-1 knapsack  sum w_j z_j <= b'  over literals z (x or 1 - x of binary
// columns, weights w > 0): fixed columns move to b', every other column is replaced by its minimum activity.
// A cover C (sum_C w > b') gives  sum_C z <= |C| - 1. Items outside C are lifted one by one (sequential up-lifting,
// Padberg): alpha_j = |C| - 1 - max{ sum alpha_i z_i : sum w_i z_i <= b' - w_j } over the items already in the cut,
// computed exactly by a dynamic program over the profit. Each lifting step keeps the inequality valid.
class CoverSeparator final : public Separator {
 public:
  int id() const override { return kSepCover; }
  void separate(const CutData& d, std::vector<Cut>& out) override {
    const LpModel& M = *d.model;
    const CsrMatrix& R = *d.rows;
    const std::vector<double>& lo = *d.lo;
    const std::vector<double>& hi = *d.hi;
    const std::vector<double>& x = *d.x;
    for (Index i = 0; i < M.n_rows; ++i) {
      const Index len = R.row_start[u(i) + 1] - R.row_start[u(i)];
      if (len < 2 || len > 2000) continue;
      for (int side = 0; side < 2; ++side) {
        const double bound = side == 0 ? M.row_upper[u(i)] : -M.row_lower[u(i)];
        if (is_inf(bound)) continue;
        const double sg = side == 0 ? 1.0 : -1.0;
        double b = bound;
        struct Item {
          Index j;
          double w;
          bool comp;  // literal is 1 - x
          double zs;  // literal value at the LP point
        };
        std::vector<Item> items;
        bool ok = true;
        for (Index t = R.row_start[u(i)]; t < R.row_start[u(i) + 1] && ok; ++t) {
          const Index j = R.col_index[u(t)];
          const double a = sg * R.value[u(t)];
          if (a == 0.0) continue;
          if (lo[u(j)] == hi[u(j)]) {
            b -= a * lo[u(j)];
          } else if (is_binary_col(M, lo, hi, j)) {
            if (a > 0.0) {
              items.push_back({j, a, false, x[u(j)]});
            } else {
              b -= a;
              items.push_back({j, -a, true, 1.0 - x[u(j)]});
            }
          } else {
            if ((a > 0.0 && is_inf(lo[u(j)])) || (a < 0.0 && is_inf(hi[u(j)]))) ok = false;
            else b -= a > 0.0 ? a * lo[u(j)] : a * hi[u(j)];
          }
        }
        if (!ok || items.size() < 2) continue;
        double total = 0.0;
        for (const Item& it : items) total += it.w;
        if (!(total > b + 1e-9) || b < 0.0) continue;
        for (int order = 0; order < 2; ++order) {
          std::vector<std::size_t> idx(items.size());
          for (std::size_t k = 0; k < idx.size(); ++k) idx[k] = k;
          std::stable_sort(idx.begin(), idx.end(), [&](std::size_t p, std::size_t q) {
            const double kp = order == 0 ? (1.0 - items[p].zs) / items[p].w : (1.0 - items[p].zs);
            const double kq = order == 0 ? (1.0 - items[q].zs) / items[q].w : (1.0 - items[q].zs);
            if (kp != kq) return kp < kq;
            return items[p].w > items[q].w;
          });
          // Greedy cover.
          std::vector<std::size_t> cover;
          double wsum = 0.0;
          for (const std::size_t k : idx) {
            cover.push_back(k);
            wsum += items[k].w;
            if (wsum > b + 1e-9) break;
          }
          if (!(wsum > b + 1e-9)) continue;
          // Make it minimal: drop items (largest slack in the LP sense first) while it stays a cover.
          std::vector<std::size_t> by_zs = cover;
          std::stable_sort(by_zs.begin(), by_zs.end(), [&](std::size_t p, std::size_t q) { return items[p].zs < items[q].zs; });
          std::vector<char> in_cover(items.size(), 0);
          for (const std::size_t k : cover) in_cover[k] = 1;
          for (const std::size_t k : by_zs) {
            if (wsum - items[k].w > b + 1e-9) {
              in_cover[k] = 0;
              wsum -= items[k].w;
            }
          }
          std::vector<std::size_t> c;
          for (std::size_t k = 0; k < items.size(); ++k) {
            if (in_cover[k]) c.push_back(k);
          }
          if (c.size() < 2) continue;
          const int cap = static_cast<int>(c.size()) - 1;
          // Lifting by decreasing LP value.
          std::vector<double> alpha(items.size(), 0.0);
          std::vector<std::size_t> lifted = c;
          for (const std::size_t k : c) alpha[k] = 1.0;
          std::vector<std::size_t> rest;
          for (std::size_t k = 0; k < items.size(); ++k) {
            if (!in_cover[k]) rest.push_back(k);
          }
          std::stable_sort(rest.begin(), rest.end(), [&](std::size_t p, std::size_t q) { return items[p].zs > items[q].zs; });
          if (rest.size() + c.size() > 400) rest.resize(400 - std::min<std::size_t>(c.size(), 400));
          for (const std::size_t j : rest) {
            double a_j;
            if (items[j].w > b) {
              a_j = cap;  // z_j = 0 in every feasible point
            } else {
              // min weight per profit 0..cap over the lifted items
              std::vector<double> minw(u(cap) + 1, kInf);
              minw[0] = 0.0;
              for (const std::size_t q : lifted) {
                const int p = static_cast<int>(alpha[q]);
                if (p <= 0) continue;
                for (int s = cap; s >= p; --s) {
                  if (minw[u(s - p)] + items[q].w < minw[u(s)]) minw[u(s)] = minw[u(s - p)] + items[q].w;
                }
              }
              int best = 0;
              for (int s = 0; s <= cap; ++s) {
                if (minw[u(s)] <= b - items[j].w + 1e-12) best = s;
              }
              a_j = static_cast<double>(cap - best);
            }
            if (a_j >= 1.0) {
              alpha[j] = a_j;
              lifted.push_back(j);
            }
          }
          // sum alpha z <= cap, z = x or 1 - x.
          Acc g(M.n_cols);
          double rhs = static_cast<double>(cap);
          for (const std::size_t k : lifted) {
            if (items[k].comp) {
              g.add(items[k].j, -alpha[k]);
              rhs -= alpha[k];
            } else {
              g.add(items[k].j, alpha[k]);
            }
          }
          out.push_back(make_cut(g, rhs, kSepCover));
        }
      }
    }
  }
};

// ======================================================================================================
// Clique cuts. A clique is a set of binary literals (x or 1 - x) of which at most one can be 1 in an
// integer-feasible point. The stored cliques (from rows and implications) are extended greedily with literals that
// conflict with every member (two literals conflict when they share a stored clique); a clique whose literal values
// at the LP point sum to more than 1 gives the cut  sum_{x in K} x - sum_{1-x in K} x <= 1 - |{1-x in K}|.
class CliqueSeparator final : public Separator {
 public:
  int id() const override { return kSepClique; }
  void separate(const CutData& d, std::vector<Cut>& out) override {
    if (d.structure == nullptr || d.structure->cliques.cliques.empty()) return;
    const std::vector<std::vector<int>>& cl = d.structure->cliques.cliques;
    const std::vector<double>& x = *d.x;
    const Index n = d.model->n_cols;
    std::unordered_map<int, std::vector<int>> member;  // literal -> clique ids
    for (std::size_t k = 0; k < cl.size(); ++k) {
      for (const int l : cl[k]) member[l].push_back(static_cast<int>(k));
    }
    auto lv = [&](int l) {
      const double v = x[u(l >> 1)];
      return (l & 1) ? 1.0 - v : v;
    };
    auto conflict = [&](int a, int b) {
      const auto ia = member.find(a);
      const auto ib = member.find(b);
      if (ia == member.end() || ib == member.end()) return false;
      std::size_t p = 0, q = 0;
      const std::vector<int>& va = ia->second;
      const std::vector<int>& vb = ib->second;
      while (p < va.size() && q < vb.size()) {
        if (va[p] == vb[q]) return true;
        va[p] < vb[q] ? ++p : ++q;
      }
      return false;
    };
    std::set<std::vector<int>> seen;
    auto emit = [&](std::vector<int> lits) {
      std::sort(lits.begin(), lits.end());
      if (!seen.insert(lits).second) return;
      double sum = 0.0;
      for (const int l : lits) sum += lv(l);
      if (!(sum > 1.0 + 1e-6)) return;
      Acc g(n);
      double rhs = 1.0;
      for (const int l : lits) {
        if (l & 1) {
          g.add(l >> 1, -1.0);
          rhs -= 1.0;
        } else {
          g.add(l >> 1, 1.0);
        }
      }
      out.push_back(make_cut(g, rhs, kSepClique));
    };
    // Candidate literals with a positive LP value, best first.
    std::vector<int> pos;
    for (const auto& kv : member) {
      if (lv(kv.first) > 1e-6) pos.push_back(kv.first);
    }
    std::sort(pos.begin(), pos.end(), [&](int a, int b) { return lv(a) > lv(b) || (lv(a) == lv(b) && a < b); });
    auto extend = [&](std::vector<int> k) {
      for (const int c : pos) {
        if (std::find(k.begin(), k.end(), c) != k.end()) continue;
        bool all = true;
        for (const int m : k) {
          if (!conflict(c, m)) {
            all = false;
            break;
          }
        }
        if (all) k.push_back(c);
      }
      return k;
    };
    long long budget = 200000;
    for (const std::vector<int>& k : cl) {
      double sum = 0.0;
      for (const int l : k) sum += lv(l);
      if (sum <= 0.5) continue;  // far from violated even after extension
      if (budget-- <= 0) break;
      emit(extend(k));
    }
    // Greedy cliques grown from the fractional literals.
    std::size_t seeds = 0;
    for (const int s : pos) {
      if (lv(s) >= 1.0 - 1e-6 || ++seeds > 100) continue;
      if (budget-- <= 0) break;
      emit(extend({s}));
    }
  }
};

// ======================================================================================================
// Implied bound cuts. An implication  x_b = v  =>  y <= B  (or >= B) with x_b binary and finite global bounds of y
// gives the valid inequality  y <= B + (U - B) x_b  (v = 0),  y <= U + (B - U) x_b  (v = 1)  and the mirrored
// ones for lower bounds. Only violated ones are returned.
class ImpliedBoundSeparator final : public Separator {
 public:
  int id() const override { return kSepImpliedBound; }
  void separate(const CutData& d, std::vector<Cut>& out) override {
    if (d.structure == nullptr) return;
    const LpModel& M = *d.model;
    const std::vector<double>& lo = *d.lo;
    const std::vector<double>& hi = *d.hi;
    const std::vector<double>& x = *d.x;
    for (const Implication& im : d.structure->implications) {
      const Index b = im.var, y = im.other;
      if (b == y || !is_binary_col(M, lo, hi, b)) continue;
      Cut c;
      c.separator = kSepImpliedBound;
      double slope, icpt;  // y <= icpt + slope x_b   (upper)   or   y >= icpt + slope x_b   (lower)
      if (im.is_upper) {
        const double U = hi[u(y)];
        if (is_inf(U)) continue;
        if (im.var_value == 0) {
          icpt = im.bound;
          slope = U - im.bound;
        } else {
          icpt = U;
          slope = im.bound - U;
        }
        if (std::fabs(slope) < 1e-9) continue;
        // y - slope x_b <= icpt
        c.idx = {y, b};
        c.val = {1.0, -slope};
        c.rhs = icpt;
        if (!(x[u(y)] - slope * x[u(b)] > icpt + 1e-6)) continue;
      } else {
        const double L = lo[u(y)];
        if (is_inf(L)) continue;
        if (im.var_value == 0) {
          icpt = im.bound;
          slope = L - im.bound;
        } else {
          icpt = L;
          slope = im.bound - L;
        }
        if (std::fabs(slope) < 1e-9) continue;
        // -y + slope x_b <= -icpt
        c.idx = {y, b};
        c.val = {-1.0, slope};
        c.rhs = -icpt;
        if (!(-x[u(y)] + slope * x[u(b)] > -icpt + 1e-6)) continue;
      }
      out.push_back(std::move(c));
    }
  }
};

}  // namespace

std::vector<std::unique_ptr<Separator>> make_separators(const MipOptions& o) {
  std::vector<std::unique_ptr<Separator>> v;
  if (o.cut_gomory) v.push_back(std::make_unique<GomorySeparator>());
  if (o.cut_mir) v.push_back(std::make_unique<MirSeparator>());
  if (o.cut_cover) v.push_back(std::make_unique<CoverSeparator>());
  if (o.cut_clique) v.push_back(std::make_unique<CliqueSeparator>());
  if (o.cut_implied_bound) v.push_back(std::make_unique<ImpliedBoundSeparator>());
  return v;
}

}  // namespace shodhan::mip
