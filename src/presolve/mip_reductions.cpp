#include "mip_reductions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace shodhan::presolve_detail {

namespace {

std::size_t u(int i) { return static_cast<std::size_t>(i); }

constexpr double kMinImprovement = 1e-3;  // continuous bounds: relative improvement worth recording
constexpr double kBoundCap = 1e12;        // derived bounds beyond this are ignored

// Activity-based bound propagation (Achterberg et al. 2020, section on domain propagation) over a private copy of
// the column bounds, with a trail so that probing can undo its tentative changes. Derived bounds of integer
// columns are rounded outwards by the same tolerance Context::btol the standard passes use, so rounding noise can
// only weaken a bound, never cut off a feasible point.
class Propagator {
 public:
  struct Change {
    int col;
    double lo, up;  // values before the change
  };

  Propagator(Context& c, long long* work, long long limit)
      : lo(c.w.cl), up(c.w.cu), c_(c), w_(c.w), work_(work), limit_(limit), inq_(u(c.w.m), 0) {}

  std::vector<double> lo, up;
  std::vector<Change> trail;
  bool conflict = false;

  void queue_row(int i) {
    if (!w_.row_alive[u(i)] || inq_[u(i)]) return;
    inq_[u(i)] = 1;
    queue_.push_back(i);
  }
  void queue_all() {
    for (int i = 0; i < w_.m; ++i) {
      if (w_.row_alive[u(i)] && w_.row_cnt[u(i)] > 0) queue_row(i);
    }
  }
  void clear_queue() {
    for (std::size_t k = head_; k < queue_.size(); ++k) inq_[u(queue_[k])] = 0;
    queue_.clear();
    head_ = 0;
  }
  bool out_of_budget() const { return *work_ > limit_; }

  /// Sets the bounds of column j (never loosening is the caller's duty). False on a conflict.
  bool set(int j, double nl, double nu) {
    if (nl > nu) {
      const bool ok = !w_.is_int[u(j)] && nl - nu <= c_.tol(nu);
      if (!ok) {
        conflict = true;
        return false;
      }
      nl = nu;
    }
    trail.push_back({j, lo[u(j)], up[u(j)]});
    lo[u(j)] = nl;
    up[u(j)] = nu;
    for (const Entry& e : w_.cols[u(j)]) queue_row(e.idx);
    return true;
  }

  std::size_t mark() const { return trail.size(); }
  void undo(std::size_t m) {
    while (trail.size() > m) {
      const Change& ch = trail.back();
      lo[u(ch.col)] = ch.lo;
      up[u(ch.col)] = ch.up;
      trail.pop_back();
    }
    clear_queue();
    conflict = false;
  }

  /// Processes the queue until empty, a conflict, or the work budget runs out. False on a conflict.
  bool propagate() {
    while (head_ < queue_.size()) {
      if (out_of_budget()) {
        clear_queue();
        return !conflict;
      }
      const int i = queue_[head_++];
      inq_[u(i)] = 0;
      if (head_ > 4096 && head_ * 2 > queue_.size()) {
        queue_.erase(queue_.begin(), queue_.begin() + static_cast<std::ptrdiff_t>(head_));
        head_ = 0;
      }
      process_row(i);
      if (conflict) {
        clear_queue();
        return false;
      }
    }
    queue_.clear();
    head_ = 0;
    return !conflict;
  }

 private:
  void tighten_up(int i, int j, double a, double v) {
    if (conflict) return;
    const std::size_t jj = u(j);
    if (w_.is_int[jj]) {
      const double nv = std::floor(v + c_.btol(i, a, v));
      if (std::fabs(nv) < kBoundCap && nv < up[jj]) set(j, lo[jj], nv);
    } else {
      const double nv = v + c_.btol(i, a, v);
      if (std::fabs(nv) >= kBoundCap) return;
      if (is_pos_inf(up[jj]) || nv < up[jj] - kMinImprovement * std::max(1.0, std::fabs(up[jj]))) {
        set(j, lo[jj], nv);
      }
    }
  }
  void tighten_lo(int i, int j, double a, double v) {
    if (conflict) return;
    const std::size_t jj = u(j);
    if (w_.is_int[jj]) {
      const double nv = std::ceil(v - c_.btol(i, a, v));
      if (std::fabs(nv) < kBoundCap && nv > lo[jj]) set(j, nv, up[jj]);
    } else {
      const double nv = v - c_.btol(i, a, v);
      if (std::fabs(nv) >= kBoundCap) return;
      if (is_neg_inf(lo[jj]) || nv > lo[jj] + kMinImprovement * std::max(1.0, std::fabs(lo[jj]))) {
        set(j, nv, up[jj]);
      }
    }
  }

  void process_row(int i) {
    const std::vector<Entry>& row = w_.rows[u(i)];
    double minf = 0.0, maxf = 0.0;
    int mini = 0, maxi = 0;
    for (const Entry& e : row) {
      if (!w_.col_alive[u(e.idx)]) continue;
      const double l = lo[u(e.idx)], h = up[u(e.idx)];
      if (e.val > 0.0) {
        if (is_neg_inf(l)) ++mini; else minf += e.val * l;
        if (is_pos_inf(h)) ++maxi; else maxf += e.val * h;
      } else {
        if (is_pos_inf(h)) ++mini; else minf += e.val * h;
        if (is_neg_inf(l)) ++maxi; else maxf += e.val * l;
      }
    }
    *work_ += static_cast<long long>(row.size());
    const double rl = w_.rl[u(i)], ru = w_.ru[u(i)];
    if (!is_inf(ru) && mini == 0 && minf > ru + c_.rtol(i, ru)) {
      conflict = true;
      return;
    }
    if (!is_inf(rl) && maxi == 0 && maxf < rl - c_.rtol(i, rl)) {
      conflict = true;
      return;
    }
    // A side that the current bounds cannot violate yields no deduction.
    const bool upper_side = !is_inf(ru) && mini <= 1 && !(maxi == 0 && maxf <= ru);
    const bool lower_side = !is_inf(rl) && maxi <= 1 && !(mini == 0 && minf >= rl);
    if (!upper_side && !lower_side) return;
    for (const Entry& e : row) {
      if (!w_.col_alive[u(e.idx)] || conflict) continue;
      const int j = e.idx;
      const double a = e.val, l = lo[u(j)], h = up[u(j)];
      if (upper_side) {  // sum a x <= ru
        const double cmin = a > 0.0 ? (is_neg_inf(l) ? kInf : a * l) : (is_pos_inf(h) ? kInf : a * h);
        const bool cinf = a > 0.0 ? is_neg_inf(l) : is_pos_inf(h);
        double rest;
        bool ok = true;
        if (mini == 0) rest = minf - cmin;
        else if (cinf) rest = minf;
        else { rest = 0.0; ok = false; }
        if (ok) {
          const double v = (ru - rest) / a;
          if (a > 0.0) tighten_up(i, j, a, v); else tighten_lo(i, j, a, v);
        }
      }
      if (lower_side && !conflict) {  // sum a x >= rl
        const double l2 = lo[u(j)], h2 = up[u(j)];
        const bool cinf = a > 0.0 ? is_pos_inf(h2) : is_neg_inf(l2);
        const double cmax = cinf ? 0.0 : (a > 0.0 ? a * h2 : a * l2);
        double rest;
        bool ok = true;
        if (maxi == 0) rest = maxf - cmax;
        else if (cinf) rest = maxf;
        else { rest = 0.0; ok = false; }
        if (ok) {
          const double v = (rl - rest) / a;
          if (a > 0.0) tighten_lo(i, j, a, v); else tighten_up(i, j, a, v);
        }
      }
    }
  }

  Context& c_;
  WorkModel& w_;
  long long* work_;
  long long limit_;
  std::vector<char> inq_;
  std::vector<int> queue_;
  std::size_t head_ = 0;
};

// Writes the integer bounds of the propagator into the model; returns true when any changed.
bool sync_bounds(Context& c, const Propagator& p) {
  WorkModel& w = c.w;
  bool changed = false;
  for (int j = 0; j < w.n; ++j) {
    if (!w.col_alive[u(j)] || !w.is_int[u(j)]) continue;
    bool touched = false;
    if (p.lo[u(j)] > w.cl[u(j)]) {
      w.cl[u(j)] = p.lo[u(j)];
      ++c.stats.propagated_bounds;
      touched = true;
    }
    if (p.up[u(j)] < w.cu[u(j)]) {
      w.cu[u(j)] = p.up[u(j)];
      ++c.stats.propagated_bounds;
      touched = true;
    }
    if (touched) {
      w.touch_col_bounds(j);
      changed = true;
    }
  }
  return changed;
}

bool is_binary(const WorkModel& w, int j) {
  return w.col_alive[u(j)] && w.is_int[u(j)] && w.cl[u(j)] == 0.0 && w.cu[u(j)] == 1.0;
}

// Coefficient tightening (Savelsbergh 1994) on rows with exactly one finite side.
bool tighten_coefficients(Context& c) {
  WorkModel& w = c.w;
  bool changed = false;
  for (int i = 0; i < w.m; ++i) {
    if (!w.row_alive[u(i)] || w.row_cnt[u(i)] < 2) continue;
    const bool has_u = !is_inf(w.ru[u(i)]), has_l = !is_inf(w.rl[u(i)]);
    if (has_u == has_l) continue;
    const double s = has_u ? 1.0 : -1.0;
    double b = has_u ? w.ru[u(i)] : -w.rl[u(i)];
    double maxact = 0.0;
    bool finite = true, any_bin = false;
    for (const Entry& e : w.rows[u(i)]) {
      if (!w.col_alive[u(e.idx)]) continue;
      const double a = s * e.val;
      const double ext = a > 0.0 ? w.cu[u(e.idx)] : w.cl[u(e.idx)];
      if (is_inf(ext)) {
        finite = false;
        break;
      }
      maxact += a * ext;
      any_bin = any_bin || is_binary(w, e.idx);
    }
    c.stats.mip_work += static_cast<long long>(w.rows[u(i)].size());
    if (!finite || !any_bin) continue;
    const double tol = 1e-9 * (1.0 + std::fabs(b) + std::fabs(maxact));
    if (b >= maxact - tol) continue;  // redundant, left to the standard passes
    const Entries snap = w.row_snapshot(i);
    bool row_changed = false;
    for (const auto& pr : snap) {
      const int j = pr.first;
      if (!is_binary(w, j)) continue;
      const double a = s * pr.second;
      if (a == 0.0) continue;
      double d, a_new;
      if (a > 0.0) {
        d = b - (maxact - a);
        a_new = a - d;
      } else {
        d = b - maxact - a;
        a_new = a + d;
      }
      if (!(d > 1e-6 * std::max(1.0, std::fabs(a)))) continue;
      if (std::fabs(a_new) < 1e-9 * std::max(1.0, std::fabs(a)) && a_new != 0.0) continue;
      if ((a > 0.0 && a_new < 0.0) || (a < 0.0 && a_new > 0.0)) continue;
      w.set_coef(i, j, s * a_new);
      w.col_dirty[u(j)] = 1;
      if (a > 0.0) {
        b -= d;
        maxact -= d;
      }
      ++c.stats.coefficients_tightened;
      row_changed = true;
    }
    if (row_changed) {
      if (has_u) w.ru[u(i)] = b; else w.rl[u(i)] = -b;
      w.row_mag[u(i)] = std::max(w.row_mag[u(i)], std::fabs(b));
      w.touch_row_bounds(i);
      changed = true;
    }
  }
  return changed;
}

std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

std::vector<std::pair<int, double>> sorted_row(const WorkModel& w, int i) {
  std::vector<std::pair<int, double>> v;
  v.reserve(w.rows[u(i)].size());
  for (const Entry& e : w.rows[u(i)]) {
    if (w.col_alive[u(e.idx)]) v.emplace_back(e.idx, e.val);
  }
  std::sort(v.begin(), v.end());
  return v;
}

// Rows with the same column set and proportional coefficients: the intersection of their ranges is kept in one.
bool merge_parallel_rows(Context& c) {
  WorkModel& w = c.w;
  std::unordered_map<std::uint64_t, std::vector<int>> groups;
  std::vector<std::vector<std::pair<int, double>>> sorted(u(w.m));
  for (int i = 0; i < w.m; ++i) {
    if (!w.row_alive[u(i)] || w.row_cnt[u(i)] < 2) continue;
    sorted[u(i)] = sorted_row(w, i);
    std::uint64_t h = static_cast<std::uint64_t>(sorted[u(i)].size());
    for (const auto& pr : sorted[u(i)]) h = mix(h, static_cast<std::uint64_t>(pr.first));
    groups[h].push_back(i);
    c.stats.mip_work += static_cast<long long>(sorted[u(i)].size());
  }
  bool changed = false;
  for (auto& g : groups) {
    std::vector<int>& rows = g.second;
    if (rows.size() < 2) continue;
    std::sort(rows.begin(), rows.end());
    for (std::size_t a = 0; a < rows.size() && a < 64; ++a) {
      const int i = rows[a];
      if (!w.row_alive[u(i)]) continue;
      for (std::size_t b = a + 1; b < rows.size() && b < a + 64; ++b) {
        const int k = rows[b];
        if (!w.row_alive[u(k)] || !w.row_alive[u(i)]) continue;
        const auto& ei = sorted[u(i)];
        const auto& ek = sorted[u(k)];
        if (ei.size() != ek.size()) continue;
        const double lam = ek[0].second / ei[0].second;
        bool same = std::isfinite(lam) && lam != 0.0;
        for (std::size_t t = 0; same && t < ei.size(); ++t) {
          same = ei[t].first == ek[t].first &&
                 std::fabs(ek[t].second - lam * ei[t].second) <= 1e-13 * std::fabs(ek[t].second);
        }
        if (!same) continue;
        // Row k in the scale of row i.
        double kl = w.rl[u(k)], ku = w.ru[u(k)];
        double nl, nu;
        if (lam > 0.0) {
          nl = is_inf(kl) ? -kInf : kl / lam;
          nu = is_inf(ku) ? kInf : ku / lam;
        } else {
          nl = is_inf(ku) ? -kInf : ku / lam;
          nu = is_inf(kl) ? kInf : kl / lam;
        }
        const double rl = std::max(w.rl[u(i)], nl);
        const double ru = std::min(w.ru[u(i)], nu);
        if (rl > ru) continue;  // empty intersection: left to the infeasibility checks of the standard passes
        w.rl[u(i)] = rl;
        w.ru[u(i)] = ru;
        w.row_mag[u(i)] = std::max(w.row_mag[u(i)], w.row_mag[u(k)] / std::fabs(lam));
        w.remove_row(k);
        w.touch_row_bounds(i);
        ++c.stats.parallel_rows;
        changed = true;
      }
    }
  }
  return changed;
}

// Identical columns: merge (same cost) or fix the dominated one (cheaper copy can absorb it).
bool merge_columns(Context& c, MipWork& mw) {
  WorkModel& w = c.w;
  std::unordered_map<std::uint64_t, std::vector<int>> groups;
  std::vector<Entries> sorted(u(w.n));
  for (int j = 0; j < w.n; ++j) {
    if (!w.col_alive[u(j)] || w.col_cnt[u(j)] < 1 || w.cl[u(j)] == w.cu[u(j)]) continue;
    Entries v = w.col_snapshot(j);
    std::sort(v.begin(), v.end());
    std::uint64_t h = static_cast<std::uint64_t>(v.size());
    for (const auto& pr : v) {
      std::uint64_t bits;
      std::memcpy(&bits, &pr.second, sizeof bits);
      h = mix(mix(h, static_cast<std::uint64_t>(pr.first)), bits);
    }
    groups[h].push_back(j);
    c.stats.mip_work += static_cast<long long>(v.size());
    sorted[u(j)] = std::move(v);
  }
  bool changed = false;
  for (auto& g : groups) {
    std::vector<int>& cols = g.second;
    if (cols.size() < 2) continue;
    std::sort(cols.begin(), cols.end());
    for (std::size_t a = 0; a < cols.size() && a < 64; ++a) {
      const int j = cols[a];
      for (std::size_t b = a + 1; b < cols.size() && b < a + 64; ++b) {
        const int k = cols[b];
        if (!w.col_alive[u(j)] || !w.col_alive[u(k)]) continue;
        if (sorted[u(j)] != sorted[u(k)] || w.is_int[u(j)] != w.is_int[u(k)]) continue;
        const bool integer = w.is_int[u(j)] != 0;
        const double lj = w.cl[u(j)], uj = w.cu[u(j)], lk = w.cl[u(k)], uk = w.cu[u(k)];
        if (integer && !(std::floor(lj) == lj && std::floor(lk) == lk &&
                         (is_inf(uj) || std::floor(uj) == uj) && (is_inf(uk) || std::floor(uk) == uk))) {
          continue;
        }
        const double cj = w.cost[u(j)], ck = w.cost[u(k)];
        if (cj == ck && !is_inf(lj) && !is_inf(lk)) {
          c.record(std::make_shared<DuplicateColumnRecord>(j, k, lj, uj, lk, uk, integer));
          w.cl[u(j)] = lj + lk;
          w.cu[u(j)] = (is_inf(uj) || is_inf(uk)) ? kInf : uj + uk;
          mw.tainted[u(j)] = mw.tainted[u(k)] = 1;
          w.remove_col(k);
          w.touch_col_bounds(j);
          ++c.stats.duplicate_columns;
          changed = true;
          continue;
        }
        // Dominated: x_k (more expensive) can move its excess over l_k to x_j, which has no upper bound.
        for (int pass = 0; pass < 2; ++pass) {
          const int cheap = pass == 0 ? j : k, dear = pass == 0 ? k : j;
          if (!w.col_alive[u(cheap)] || !w.col_alive[u(dear)]) continue;
          if (w.cost[u(cheap)] < w.cost[u(dear)] && is_pos_inf(w.cu[u(cheap)]) && !is_inf(w.cl[u(dear)])) {
            FixedColumnReduction::fix(c, dear, w.cl[u(dear)], false);
            ++c.stats.dominated_columns;
            changed = true;
            break;
          }
        }
      }
    }
  }
  return changed;
}

struct Snap {
  int col;
  double lo, up;
};

// Probing on binary columns (Savelsbergh 1994): fix to 0 and to 1, propagate, and learn from the two outcomes.
bool probe(Context& c, MipWork& mw, bool& any_change) {
  WorkModel& w = c.w;
  const PresolveOptions& o = c.opt;
  Propagator p(c, &mw.probe_work, o.probing_work_limit);
  std::vector<int> cand;
  for (int j = 0; j < w.n; ++j) {
    if (is_binary(w, j) && w.col_cnt[u(j)] > 0) cand.push_back(j);
  }
  std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return w.col_cnt[u(a)] > w.col_cnt[u(b)]; });
  if (static_cast<int>(cand.size()) > o.probing_column_limit) cand.resize(u(o.probing_column_limit));
  std::vector<int> stamp(u(w.n), -1);
  int stamp_id = 0;
  for (const int j : cand) {
    if (p.out_of_budget()) break;
    if (!w.col_alive[u(j)] || p.lo[u(j)] == p.up[u(j)]) continue;
    std::vector<Snap> snaps[2];
    bool feasible[2] = {false, false};
    for (int v = 0; v < 2; ++v) {
      const std::size_t m = p.mark();
      bool ok = p.set(j, v, v);
      if (ok) {
        for (const Entry& e : w.cols[u(j)]) p.queue_row(e.idx);
        ok = p.propagate();
      }
      feasible[v] = ok;
      if (ok) {
        ++stamp_id;
        for (std::size_t t = m; t < p.trail.size(); ++t) {
          const int k = p.trail[t].col;
          if (k == j || stamp[u(k)] == stamp_id) continue;
          stamp[u(k)] = stamp_id;
          snaps[v].push_back({k, p.lo[u(k)], p.up[u(k)]});
        }
      }
      p.undo(m);
    }
    if (!feasible[0] && !feasible[1]) {
      c.mark_infeasible("probing: both values of binary column " + std::to_string(j) + " are infeasible");
      return true;
    }
    if (feasible[0] != feasible[1]) {
      const int v = feasible[0] ? 0 : 1;
      p.set(j, v, v);
      for (const Entry& e : w.cols[u(j)]) p.queue_row(e.idx);
      if (!p.propagate()) {
        c.mark_infeasible("probing: conflict after fixing column " + std::to_string(j));
        return true;
      }
      p.trail.clear();
      ++c.stats.probing_fixings;
      any_change = true;
      continue;
    }
    // Both branches feasible: bounds common to both hold globally.
    std::unordered_map<int, Snap> second;
    for (const Snap& s : snaps[1]) second[s.col] = s;
    bool tightened = false;
    for (const Snap& s : snaps[0]) {
      const auto it = second.find(s.col);
      if (it == second.end()) continue;
      const double nl = std::min(s.lo, it->second.lo), nu = std::max(s.up, it->second.up);
      const int k = s.col;
      if (!w.is_int[u(k)]) continue;
      if (nl > p.lo[u(k)] || nu < p.up[u(k)]) {
        if (!p.set(k, std::max(nl, p.lo[u(k)]), std::min(nu, p.up[u(k)]))) {
          c.mark_infeasible("probing: conflict tightening column " + std::to_string(k));
          return true;
        }
        ++c.stats.probing_bounds;
        tightened = true;
      }
    }
    if (tightened) {
      if (!p.propagate()) {
        c.mark_infeasible("probing: conflict after tightening bounds of column " + std::to_string(j));
        return true;
      }
      p.trail.clear();
      any_change = true;
    }
    // Implications of x_j = v on columns whose bound is still loose.
    for (int v = 0; v < 2; ++v) {
      for (const Snap& s : snaps[v]) {
        if (static_cast<int>(mw.implications.size()) >= o.max_implications) break;
        const std::size_t k = u(s.col);
        if (!w.col_alive[k]) continue;
        const bool bin = w.is_int[k] && w.cl[k] == 0.0 && w.cu[k] == 1.0;
        if (s.up < p.up[k] - (bin ? 0.5 : kMinImprovement * std::max(1.0, std::fabs(s.up))) && !is_inf(s.up)) {
          mw.implications.push_back({j, v, s.col, true, s.up});
        }
        if (s.lo > p.lo[k] + (bin ? 0.5 : kMinImprovement * std::max(1.0, std::fabs(s.lo))) && !is_inf(s.lo)) {
          mw.implications.push_back({j, v, s.col, false, s.lo});
        }
      }
    }
  }
  if (any_change) sync_bounds(c, p);
  return false;
}

}  // namespace

bool run_mip_round(Context& c, MipWork& mw) {
  WorkModel& w = c.w;
  const PresolveOptions& o = c.opt;
  if (mw.tainted.empty()) mw.tainted.assign(u(w.n), 0);
  bool changed = false;
  const long long work_before = mw.prop_work + mw.probe_work;
  w.compact();
  if (o.mip_propagation) {
    Propagator p(c, &mw.prop_work, o.propagation_work_limit);
    p.queue_all();
    if (!p.propagate()) {
      c.mark_infeasible("bound propagation found a conflict");
      return true;
    }
    changed = sync_bounds(c, p) || changed;
  }
  if (o.coefficient_tightening) changed = tighten_coefficients(c) || changed;
  if (o.parallel_rows) changed = merge_parallel_rows(c) || changed;
  if (o.duplicate_columns) changed = merge_columns(c, mw) || changed;
  if (o.probing) {
    w.compact();
    bool any = false;
    if (probe(c, mw, any)) return true;
    changed = any || changed;
  }
  c.stats.mip_work += mw.prop_work + mw.probe_work - work_before;
  return changed;
}

bool run_probing(Context& c, MipWork& mw) {
  if (mw.tainted.empty()) mw.tainted.assign(u(c.w.n), 0);
  c.w.compact();
  bool any = false;
  if (probe(c, mw, any)) return true;
  return any;
}

MipPresolveInfo collect_mip_structure(Context& c, const MipWork& mw) {
  WorkModel& w = c.w;
  const PresolveOptions& o = c.opt;
  MipPresolveInfo info;
  auto usable = [&](int j) {
    return is_binary(w, j) && !(u(j) < mw.tainted.size() && mw.tainted[u(j)]);
  };
  for (const Implication& im : mw.implications) {
    if (!usable(im.var) || !w.col_alive[u(im.other)]) continue;
    if (u(im.other) < mw.tainted.size() && mw.tainted[u(im.other)]) continue;
    info.implications.push_back(im);
  }
  if (!o.clique_table) return info;

  std::unordered_set<std::string> seen;
  auto add_clique = [&](std::vector<int> lits) {
    if (lits.size() < 2 || static_cast<int>(info.cliques.cliques.size()) >= o.max_cliques) return;
    std::sort(lits.begin(), lits.end());
    lits.erase(std::unique(lits.begin(), lits.end()), lits.end());
    for (std::size_t t = 0; t + 1 < lits.size(); ++t) {
      if ((lits[t] >> 1) == (lits[t + 1] >> 1)) return;  // x and 1-x together: handled by the model itself
    }
    std::string key;
    for (const int l : lits) key += std::to_string(l) + ",";
    if (!seen.insert(key).second) return;
    info.cliques.cliques.push_back(std::move(lits));
  };
  // From rows: sum of literals weighted by positive weights <= b.
  for (int i = 0; i < w.m; ++i) {
    if (!w.row_alive[u(i)] || w.row_cnt[u(i)] < 2) continue;
    for (int side = 0; side < 2; ++side) {
      const double s = side == 0 ? 1.0 : -1.0;
      const double b0 = side == 0 ? w.ru[u(i)] : -w.rl[u(i)];
      if (is_inf(b0)) continue;
      double b = b0;
      bool ok = true;
      std::vector<std::pair<double, int>> lits;  // weight, literal
      for (const Entry& e : w.rows[u(i)]) {
        const int j = e.idx;
        if (!w.col_alive[u(j)]) continue;
        const double a = s * e.val;
        if (usable(j)) {
          if (a > 0.0) {
            lits.emplace_back(a, 2 * j);
          } else {
            b -= a;  // a x = a + |a| (1 - x)
            lits.emplace_back(-a, 2 * j + 1);
          }
        } else {
          const double ext = a > 0.0 ? w.cl[u(j)] : w.cu[u(j)];
          if (is_inf(ext)) {
            ok = false;
            break;
          }
          b -= a * ext;
        }
      }
      if (!ok || lits.size() < 2) continue;
      std::sort(lits.begin(), lits.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
      const double margin = 1e-9 * (1.0 + std::fabs(b));
      if (lits[0].first + lits[1].first <= b + margin) continue;
      std::size_t t = 2;
      while (t < lits.size() && lits[t - 1].first + lits[t].first > b + margin) ++t;
      std::vector<int> cl;
      for (std::size_t q = 0; q < t; ++q) cl.push_back(lits[q].second);
      add_clique(std::move(cl));
    }
  }
  // From implications: literal a true implies literal b true gives the clique {a, not b}.
  for (const Implication& im : info.implications) {
    if (!is_binary(w, im.other)) continue;
    const int a = 2 * im.var + (im.var_value == 1 ? 0 : 1);
    const int b = 2 * im.other + (im.is_upper ? 1 : 0);  // x_other <= 0 means literal not-x true
    if (im.is_upper && im.bound != 0.0) continue;
    if (!im.is_upper && im.bound != 1.0) continue;
    add_clique({a, b ^ 1});
  }
  return info;
}

}  // namespace shodhan::presolve_detail
