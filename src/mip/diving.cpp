// Diving heuristics: repeatedly tighten one fractional integer column in a rounding direction and resolve the
// LP with the warm-started dual simplex until the LP solution is integral (offered to the incumbent manager)
// or the dive fails. The dive runs on COPIES of the engine, so the search state is untouched afterwards.
//
//   fractional diving   the column closest to an integer, rounded to the nearest integer;
//   coefficient diving  the column with the fewest locks in its rounding direction (a direction with no
//                       locks cannot make any row infeasible), ties by fractionality.
//
// At most one backtrack per dive (the opposite bound of the last decision when it made the LP infeasible),
// an iteration budget relative to the LP iterations of the whole search, a depth limit and a stall limit
// (no reduction of the number of fractional columns for a number of steps). All limits are targets.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "locks.hpp"
#include "shodhan/mip/plugins.hpp"

namespace shodhan::mip {

namespace {

enum class DiveKind { Fractional, Coefficient };

class DivingHeuristic final : public PrimalHeuristic {
 public:
  explicit DivingHeuristic(DiveKind kind) : kind_(kind) {}
  const char* name() const override { return kind_ == DiveKind::Fractional ? "diving-fractional" : "diving-coefficient"; }

  bool wants(const MipOptions& o, HeuristicWhen when, long long nodes, Index) const override {
    const HeuristicSetting& set = kind_ == DiveKind::Fractional ? o.diving_fractional : o.diving_coefficient;
    if (!set.enabled || when == HeuristicWhen::BeforeRootLp) return false;
    if (when == HeuristicWhen::AfterRootLp) return true;
    return set.frequency > 0 && nodes % set.frequency == 0;
  }

  void run(SearchState& s, HeuristicWhen) override {
    if (s.lp.x.empty() || s.lp.fractional.empty() || s.engine == nullptr) return;
    const MipOptions& opt = *s.options;
    const long long budget_total = opt.dive_iteration_floor + static_cast<long long>(opt.dive_iteration_fraction * static_cast<double>(s.lp_iterations()));
    long long budget = budget_total - used_;
    if (budget <= 0) return;
    const LpModel& m = *s.model;
    if (!locks_ready_) {
      locks_ = compute_locks(m);
      locks_ready_ = true;
    }
    auto cur = std::make_unique<SimplexEngine>(*s.engine);
    cur->options().final_check = false;
    std::unique_ptr<SimplexEngine> backup;  // the state before the last decision
    struct Decision {
      Index col;
      bool went_down;
      double value;
    };
    bool have_last = false;
    Decision last{0, true, 0.0};
    bool backtracked = false;
    const long long start_iters = cur->stats().iterations;
    long long wasted = 0;  // iterations of a failed branch that a backtrack discarded
    auto spent_so_far = [&]() { return cur->stats().iterations - start_iters + wasted; };
    std::vector<double> x(to_size(m.n_cols));
    std::vector<Index> fractional = s.lp.fractional;
    std::vector<double> value(s.lp.x);
    const std::size_t max_steps = s.int_cols->size() * 2 + 10;
    std::size_t best_fractional = fractional.size();
    int stall = 0;

    for (std::size_t step = 0; step < max_steps; ++step) {
      if (s.time_up()) break;
      // choose a column and a direction
      Index pick = -1;
      bool down = true;
      double best_key1 = kInf, best_key2 = kInf;
      for (const Index j : fractional) {
        const double v = value[to_size(j)];
        const double f = v - std::floor(v);
        const bool nearest_down = f < 0.5;
        double key1, key2;
        bool dir_down;
        if (kind_ == DiveKind::Fractional) {
          dir_down = nearest_down;
          key1 = std::min(f, 1.0 - f);
          key2 = 0.0;
        } else {
          const int dl = locks_.down[to_size(j)], ul = locks_.up[to_size(j)];
          if (dl < ul) dir_down = true;
          else if (ul < dl) dir_down = false;
          else dir_down = nearest_down;
          key1 = static_cast<double>(dir_down ? dl : ul);
          key2 = dir_down ? f : 1.0 - f;
        }
        if (key1 < best_key1 - 1e-12 || (std::fabs(key1 - best_key1) <= 1e-12 && key2 < best_key2 - 1e-12)) {
          best_key1 = key1;
          best_key2 = key2;
          pick = j;
          down = dir_down;
        }
      }
      if (pick < 0) break;
      const double v = value[to_size(pick)];
      const double lo = cur->col_lower(pick), hi = cur->col_upper(pick);
      const double nlo = down ? lo : std::ceil(v);
      const double nhi = down ? std::floor(v) : hi;
      if (nlo > nhi) break;
      backup = std::make_unique<SimplexEngine>(*cur);
      cur->change_col_bounds(pick, nlo, nhi);
      have_last = true;
      last = {pick, down, v};
      long long remaining = budget - spent_so_far();
      if (remaining <= 0) break;
      EngineStatus st = cur->solve_limited(remaining);
      if (st == EngineStatus::Infeasible && !backtracked && backup) {
        // one backtrack: the other side of the last decision, from the state before it
        backtracked = true;
        const long long failed = cur->stats().iterations - backup->stats().iterations;
        wasted += failed;
        cur = std::move(backup);
        const double lo2 = cur->col_lower(last.col), hi2 = cur->col_upper(last.col);
        const double nlo2 = last.went_down ? std::ceil(last.value) : lo2;
        const double nhi2 = last.went_down ? hi2 : std::floor(last.value);
        if (nlo2 > nhi2) break;
        cur->change_col_bounds(last.col, nlo2, nhi2);
        remaining = budget - spent_so_far();
        if (remaining <= 0) break;
        st = cur->solve_limited(remaining);
      }
      if (st != EngineStatus::Optimal) break;
      const double z = cur->objective() / s.obj_scale;
      if (s.cutoff() < kInf && z >= s.cutoff() - 1e-9 * std::max(1.0, std::fabs(s.cutoff()))) break;
      // read the new LP solution
      fractional.clear();
      const std::vector<double>& xs = cur->primal_all();
      for (Index j = 0; j < m.n_cols; ++j) value[to_size(j)] = xs[to_size(j)] * s.scaling->col_scale[to_size(j)];
      for (const Index j : *s.int_cols) {
        const double vj = value[to_size(j)];
        if (std::fabs(vj - std::round(vj)) > s.options->params.int_tol) fractional.push_back(j);
      }
      if (fractional.empty()) {
        for (Index j = 0; j < m.n_cols; ++j) x[to_size(j)] = value[to_size(j)];
        s.submit(x, name());
        break;
      }
      if (fractional.size() < best_fractional) {
        best_fractional = fractional.size();
        stall = 0;
      } else if (++stall >= 12) {
        break;  // stall limit
      }
    }
    (void)have_last;
    const long long spent = cur ? spent_so_far() : 0;
    used_ += spent;
    s.add_iterations(spent, false);
  }

 private:
  DiveKind kind_;
  Locks locks_;
  bool locks_ready_ = false;
  long long used_ = 0;  // iterations this heuristic has used in this search
};

}  // namespace

void register_heuristics_diving(PluginRegistry& reg) {
  reg.heuristics.add("diving-fractional", [] { return std::make_unique<DivingHeuristic>(DiveKind::Fractional); });
  reg.heuristics.add("diving-coefficient", [] { return std::make_unique<DivingHeuristic>(DiveKind::Coefficient); });
}

}  // namespace shodhan::mip
