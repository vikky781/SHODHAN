// Feasibility Jump, a Lagrangian-style local search for MIP feasibility that needs no LP.
//
// Luteberget, Sartor: "Feasibility Jump: an LP-free Lagrangian MIP heuristic", Mathematical Programming
// Computation (2023). Implemented here from the paper's description, in this project's own code.
//
// The search minimizes the weighted constraint violation  sum_i w_i * viol_i(x)  over the box of the columns,
// where viol_i is the distance of the row activity to the row's range. For one column this is a convex
// piecewise-linear function of its value, so the best value for the column (its "jump value") is found from
// the breakpoints of the rows it appears in; the "score" of a column is the decrease of the weighted
// violation that moving it to its jump value gives. Each step moves one of the best-scoring columns of a
// small random sample of the columns with a positive score. At a local minimum (no positive score) the weights
// of the violated rows are increased, which changes the landscape. Integer columns only take integer values.
//
// Simplification against the paper: after a move, the jump values and scores of all columns that share a row
// with the moved column are recomputed from scratch (the paper updates them incrementally). Tie-breaking is
// seeded, so runs are deterministic. A work limit (matrix entries touched) bounds the effort. The objective
// is not optimized: the heuristic looks for feasible points; the incumbent manager decides whether one is
// better than the incumbent.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "shodhan/mip/plugins.hpp"

namespace shodhan::mip {

namespace {

class FeasibilityJump final : public PrimalHeuristic {
 public:
  const char* name() const override { return "feasibility-jump"; }

  bool wants(const MipOptions& o, HeuristicWhen when, long long nodes, Index) const override {
    if (!o.feasibility_jump.enabled) return false;
    if (when == HeuristicWhen::BeforeRootLp) return true;
    if (when == HeuristicWhen::AfterRootLp) return false;
    return o.feasibility_jump.frequency > 0 && nodes % o.feasibility_jump.frequency == 0;
  }

  void run(SearchState& s, HeuristicWhen) override {
    const LpModel& m = *s.model;
    const MipOptions& opt = *s.options;
    n_ = m.n_cols;
    rows_ = m.n_rows;
    if (n_ == 0 || rows_ == 0) return;
    const CsrMatrix csr = m.A.to_csr();
    csr_ = &csr;
    model_ = &m;
    // ---- start point: the rounded LP solution when there is one, else zero clamped into the bounds ----
    x_.assign(to_size(n_), 0.0);
    for (Index j = 0; j < n_; ++j) {
      double v = s.lp.x.empty() ? 0.0 : s.lp.x[to_size(j)];
      v = clamp(v, m.col_lower[to_size(j)], m.col_upper[to_size(j)]);
      if (m.is_integer(j)) v = clamp(std::round(v), m.col_lower[to_size(j)], m.col_upper[to_size(j)]);
      x_[to_size(j)] = v;
    }
    act_.assign(to_size(rows_), 0.0);
    for (Index j = 0; j < n_; ++j) {
      for (Index p = m.A.col_start[to_size(j)]; p < m.A.col_start[to_size(j) + 1]; ++p) act_[to_size(m.A.row_index[to_size(p)])] += m.A.value[to_size(p)] * x_[to_size(j)];
    }
    w_.assign(to_size(rows_), 1.0);
    jump_.assign(to_size(n_), 0.0);
    score_.assign(to_size(n_), 0.0);
    in_good_.assign(to_size(n_), 0);
    good_pos_.assign(to_size(n_), -1);
    good_.clear();
    stamp_.assign(to_size(n_), 0);
    cur_stamp_ = 0;
    violated_ = 0;
    for (Index i = 0; i < rows_; ++i) {
      if (violation(i, act_[to_size(i)]) > 0.0) ++violated_;
    }
    work_ = 0;
    for (Index j = 0; j < n_; ++j) refresh(j);

    const long long work_limit = opt.fj_work_limit;
    const long long max_steps = 50000 + 200LL * n_;
    for (long long step = 0; step < max_steps && work_ < work_limit; ++step) {
      if ((step & 1023) == 0 && s.time_up()) break;
      if (violated_ == 0) {
        s.submit(x_, name());
        return;
      }
      if (good_.empty()) {
        // Local minimum: raise the weights of the violated rows and look at their columns again.
        ++cur_stamp_;
        std::vector<Index> touched;
        for (Index i = 0; i < rows_; ++i) {
          if (violation(i, act_[to_size(i)]) <= 0.0) continue;
          w_[to_size(i)] += 1.0;
          for (Index p = csr.row_start[to_size(i)]; p < csr.row_start[to_size(i) + 1]; ++p) {
            const Index k = csr.col_index[to_size(p)];
            if (stamp_[to_size(k)] != cur_stamp_) {
              stamp_[to_size(k)] = cur_stamp_;
              touched.push_back(k);
            }
          }
          work_ += csr.row_start[to_size(i) + 1] - csr.row_start[to_size(i)];
        }
        for (const Index k : touched) refresh(k);
        if (good_.empty()) {
          // Even the new weights give no improving move: perturb by moving a random column of a violated row.
          if (!kick(s)) return;
        }
        continue;
      }
      // Sample a few good columns and take the best score (seeded tie-breaking).
      const std::size_t sample = std::min<std::size_t>(good_.size(), 25);
      Index best = -1;
      double best_score = -1.0;
      for (std::size_t t = 0; t < sample; ++t) {
        const Index j = good_.size() <= 25 ? good_[t] : good_[static_cast<std::size_t>(s.rng->below(good_.size()))];
        const double sc = score_[to_size(j)];
        if (sc > best_score + 1e-12 || (std::fabs(sc - best_score) <= 1e-12 && (s.rng->next() & 1ULL))) {
          best_score = sc;
          best = j;
        }
      }
      move(best, jump_[to_size(best)]);
    }
    if (violated_ == 0) s.submit(x_, name());
  }

 private:
  static double clamp(double v, double lo, double hi) {
    if (!is_inf(lo) && v < lo) v = lo;
    if (!is_inf(hi) && v > hi) v = hi;
    return v;
  }

  double tol(double bound) const { return 1e-9 * (1.0 + std::fabs(bound)); }

  // Distance of a row activity to the row's range, ignoring distances within the tolerance.
  double violation(Index i, double a) const {
    const double lo = model_->row_lower[to_size(i)], hi = model_->row_upper[to_size(i)];
    if (!is_inf(lo) && a < lo - tol(lo)) return lo - a;
    if (!is_inf(hi) && a > hi + tol(hi)) return a - hi;
    return 0.0;
  }
  double violation_raw(Index i, double a) const {
    const double lo = model_->row_lower[to_size(i)], hi = model_->row_upper[to_size(i)];
    if (!is_inf(lo) && a < lo) return lo - a;
    if (!is_inf(hi) && a > hi) return a - hi;
    return 0.0;
  }

  // Weighted violation of column j's rows if x_j had value t.
  double weighted_violation(Index j, double t) const {
    double f = 0.0;
    const double xj = x_[to_size(j)];
    for (Index p = model_->A.col_start[to_size(j)]; p < model_->A.col_start[to_size(j) + 1]; ++p) {
      const Index i = model_->A.row_index[to_size(p)];
      f += w_[to_size(i)] * violation_raw(i, act_[to_size(i)] + model_->A.value[to_size(p)] * (t - xj));
    }
    return f;
  }

  // Recomputes the jump value and score of column j and updates the set of good columns.
  void refresh(Index j) {
    const LpModel& m = *model_;
    events_.clear();
    double slope = 0.0;
    const double xj = x_[to_size(j)];
    const Index begin = m.A.col_start[to_size(j)], end = m.A.col_start[to_size(j) + 1];
    for (Index p = begin; p < end; ++p) {
      const Index i = m.A.row_index[to_size(p)];
      const double a = m.A.value[to_size(p)];
      const double rest = act_[to_size(i)] - a * xj;
      const double lo = m.row_lower[to_size(i)], hi = m.row_upper[to_size(i)];
      const double wa = w_[to_size(i)] * std::fabs(a);
      double lb, rb;
      if (a > 0.0) {
        lb = is_inf(lo) ? -kInf : (lo - rest) / a;
        rb = is_inf(hi) ? kInf : (hi - rest) / a;
      } else {
        lb = is_inf(hi) ? -kInf : (hi - rest) / a;
        rb = is_inf(lo) ? kInf : (lo - rest) / a;
      }
      if (!is_inf(lb)) {
        slope -= wa;
        events_.push_back({lb, wa});
      }
      if (!is_inf(rb)) events_.push_back({rb, wa});
    }
    work_ += end - begin + 1;
    double target = xj;
    if (!events_.empty()) {
      std::sort(events_.begin(), events_.end(), [](const Event& a, const Event& b) { return a.pos < b.pos; });
      double ta = -kInf, tb = kInf;
      double cum = slope;
      std::size_t k = 0;
      if (cum >= -1e-12) {
        // no row pushes the value up from the left: flat until the first event, then increasing
        ta = -kInf;
        tb = events_[0].pos;
      } else {
        while (k < events_.size()) {
          const double pos = events_[k].pos;
          while (k < events_.size() && events_[k].pos == pos) cum += events_[k++].delta;
          if (cum >= -1e-12) {
            ta = pos;
            tb = cum > 1e-12 ? pos : (k < events_.size() ? events_[k].pos : kInf);
            break;
          }
        }
      }
      const double lo_j = m.col_lower[to_size(j)], hi_j = m.col_upper[to_size(j)];
      double a_lo = std::max(ta, is_inf(lo_j) ? -kInf : lo_j);
      double b_hi = std::min(tb, is_inf(hi_j) ? kInf : hi_j);
      if (a_lo > b_hi) {
        target = tb < (is_inf(lo_j) ? -kInf : lo_j) ? lo_j : hi_j;
      } else {
        target = std::min(std::max(xj, a_lo), b_hi);
      }
      if (m.is_integer(j) && target != std::round(target)) {
        const double low_i = std::ceil(a_lo - 1e-9), high_i = std::floor(b_hi + 1e-9);
        if (a_lo <= b_hi && low_i <= high_i) {
          target = std::min(std::max(std::round(xj), low_i), high_i);
        } else {
          double c1 = clamp(std::floor(target), lo_j, hi_j), c2 = clamp(std::ceil(target), lo_j, hi_j);
          const double f1 = weighted_violation(j, c1), f2 = weighted_violation(j, c2);
          target = f1 < f2 - 1e-12 ? c1 : (f2 < f1 - 1e-12 ? c2 : (std::fabs(c1 - xj) <= std::fabs(c2 - xj) ? c1 : c2));
        }
      } else if (m.is_integer(j)) {
        target = clamp(target, lo_j, hi_j);
      }
    }
    double sc = 0.0;
    if (target != xj && !is_inf(target)) {
      sc = weighted_violation(j, xj) - weighted_violation(j, target);
      if (!(sc > 1e-9)) sc = 0.0;
    }
    jump_[to_size(j)] = target;
    score_[to_size(j)] = sc;
    const bool good = sc > 0.0;
    if (good && !in_good_[to_size(j)]) {
      in_good_[to_size(j)] = 1;
      good_pos_[to_size(j)] = static_cast<Index>(good_.size());
      good_.push_back(j);
    } else if (!good && in_good_[to_size(j)]) {
      const Index pos = good_pos_[to_size(j)];
      const Index last = good_.back();
      good_[to_size(pos)] = last;
      good_pos_[to_size(last)] = pos;
      good_.pop_back();
      in_good_[to_size(j)] = 0;
      good_pos_[to_size(j)] = -1;
    }
  }

  void move(Index j, double value) {
    const LpModel& m = *model_;
    const double delta = value - x_[to_size(j)];
    x_[to_size(j)] = value;
    ++cur_stamp_;
    std::vector<Index>& touched = touched_;
    touched.clear();
    touched.push_back(j);
    stamp_[to_size(j)] = cur_stamp_;
    for (Index p = m.A.col_start[to_size(j)]; p < m.A.col_start[to_size(j) + 1]; ++p) {
      const Index i = m.A.row_index[to_size(p)];
      const bool was = violation(i, act_[to_size(i)]) > 0.0;
      act_[to_size(i)] += m.A.value[to_size(p)] * delta;
      const bool is = violation(i, act_[to_size(i)]) > 0.0;
      violated_ += static_cast<int>(is) - static_cast<int>(was);
      for (Index q = csr_->row_start[to_size(i)]; q < csr_->row_start[to_size(i) + 1]; ++q) {
        const Index k = csr_->col_index[to_size(q)];
        if (stamp_[to_size(k)] != cur_stamp_) {
          stamp_[to_size(k)] = cur_stamp_;
          touched.push_back(k);
        }
      }
      work_ += csr_->row_start[to_size(i) + 1] - csr_->row_start[to_size(i)];
    }
    for (const Index k : touched) refresh(k);
  }

  // Moves a random column of a random violated row to a random nearby value when no weight change helped.
  bool kick(SearchState& s) {
    std::vector<Index> vio;
    for (Index i = 0; i < rows_; ++i) {
      if (violation(i, act_[to_size(i)]) > 0.0) vio.push_back(i);
    }
    if (vio.empty()) return true;
    const Index i = vio[static_cast<std::size_t>(s.rng->below(vio.size()))];
    const Index begin = csr_->row_start[to_size(i)], end = csr_->row_start[to_size(i) + 1];
    if (begin == end) return false;  // an empty violated row cannot be repaired
    const Index j = csr_->col_index[to_size(begin + static_cast<Index>(s.rng->below(static_cast<std::uint64_t>(end - begin))))];
    const LpModel& m = *model_;
    double v = x_[to_size(j)] + (s.rng->next() & 1ULL ? 1.0 : -1.0);
    v = clamp(v, m.col_lower[to_size(j)], m.col_upper[to_size(j)]);
    if (m.is_integer(j)) v = std::round(v);
    if (v != x_[to_size(j)]) move(j, v);
    return true;
  }

  struct Event {
    double pos;
    double delta;
  };

  const LpModel* model_ = nullptr;
  const CsrMatrix* csr_ = nullptr;
  Index n_ = 0, rows_ = 0;
  std::vector<double> x_, act_, w_, jump_, score_;
  std::vector<char> in_good_;
  std::vector<Index> good_pos_, good_, touched_;
  std::vector<unsigned> stamp_;
  unsigned cur_stamp_ = 0;
  int violated_ = 0;
  long long work_ = 0;
  std::vector<Event> events_;
};

}  // namespace

void register_heuristics_fj(PluginRegistry& reg) {
  reg.heuristics.add("feasibility-jump", [] { return std::make_unique<FeasibilityJump>(); });
}

}  // namespace shodhan::mip
