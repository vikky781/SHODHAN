// Trivial heuristic and simple rounding with row locks. Both only propose points; every proposal goes through
// the incumbent manager, which verifies it on the original model.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "locks.hpp"
#include "shodhan/mip/plugins.hpp"

namespace shodhan::mip {

namespace {

double clamp_to_bounds(double v, double lo, double hi) {
  if (!is_inf(lo) && v < lo) v = lo;
  if (!is_inf(hi) && v > hi) v = hi;
  return v;
}

// Tries every integer column at its lower bound, at its upper bound, and at zero (clamped into the bounds).
// Continuous columns are set to the value nearest to zero inside their bounds; when the original model has
// continuous columns the incumbent manager re-solves them, so only the integer part has to be right.
class TrivialHeuristic final : public PrimalHeuristic {
 public:
  const char* name() const override { return "trivial"; }
  bool wants(const MipOptions& o, HeuristicWhen when, long long, Index) const override { return o.trivial.enabled && when == HeuristicWhen::BeforeRootLp; }
  void run(SearchState& s, HeuristicWhen) override {
    const LpModel& m = *s.model;
    for (int variant = 0; variant < 3; ++variant) {
      std::vector<double> x(to_size(m.n_cols));
      for (Index j = 0; j < m.n_cols; ++j) {
        const double lo = m.col_lower[to_size(j)], hi = m.col_upper[to_size(j)];
        double v = 0.0;
        if (m.is_integer(j)) {
          if (variant == 0) v = is_inf(lo) ? clamp_to_bounds(0.0, lo, hi) : lo;
          else if (variant == 1) v = is_inf(hi) ? clamp_to_bounds(0.0, lo, hi) : hi;
        }
        x[to_size(j)] = clamp_to_bounds(v, lo, hi);
        if (m.is_integer(j)) x[to_size(j)] = std::round(x[to_size(j)]);
      }
      s.submit(x, name());
    }
  }
};

// Simple rounding: every fractional integer column is rounded in a direction that cannot violate any row
// (no locks in that direction). Fails (no proposal) as soon as one fractional column has locks both ways.
class RoundingHeuristic final : public PrimalHeuristic {
 public:
  const char* name() const override { return "rounding"; }
  bool wants(const MipOptions& o, HeuristicWhen when, long long nodes, Index) const override {
    if (!o.rounding.enabled || when == HeuristicWhen::BeforeRootLp) return false;
    if (when == HeuristicWhen::AfterRootLp) return true;
    return o.rounding.frequency > 0 && nodes % o.rounding.frequency == 0;
  }
  void run(SearchState& s, HeuristicWhen) override {
    if (s.lp.x.empty() || s.lp.fractional.empty()) return;
    const LpModel& m = *s.model;
    if (!locks_ready_) {
      locks_ = compute_locks(m);
      locks_ready_ = true;
    }
    std::vector<double> x = s.lp.x;
    for (const Index j : s.lp.fractional) {
      const double v = x[to_size(j)];
      const bool can_down = locks_.down[to_size(j)] == 0, can_up = locks_.up[to_size(j)] == 0;
      if (can_down && can_up) {
        x[to_size(j)] = m.col_cost[to_size(j)] >= 0.0 ? std::floor(v) : std::ceil(v);  // either is safe: take the cheaper
      } else if (can_down) {
        x[to_size(j)] = std::floor(v);
      } else if (can_up) {
        x[to_size(j)] = std::ceil(v);
      } else {
        return;  // locked both ways: simple rounding cannot work here
      }
    }
    s.submit(x, name());
  }

 private:
  Locks locks_;
  bool locks_ready_ = false;
};

}  // namespace

void register_heuristics_simple(PluginRegistry& reg) {
  reg.heuristics.add("trivial", [] { return std::make_unique<TrivialHeuristic>(); });
  reg.heuristics.add("rounding", [] { return std::make_unique<RoundingHeuristic>(); });
}

}  // namespace shodhan::mip
