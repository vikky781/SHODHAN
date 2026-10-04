// Pseudocost branching and reliability branching.
//
// T. Achterberg, T. Koch, A. Martin, "Branching rules revisited", Operations Research Letters 33 (2005):
// pseudocosts estimate the objective gain per unit change of a column from past branchings; a column is
// "reliable" once it has been observed often enough in both directions; the gains of unreliable candidates
// are measured by strong branching (a short dual simplex solve of both children) instead of estimated. The
// candidate score is the product max(gain_down, eps) * max(gain_up, eps).
//
// Strong branching works on COPIES of the engine: the search state is read-only here, so it cannot be
// disturbed, and a node LP that the probe changes is simply discarded with the copy.

#include <algorithm>
#include <cmath>
#include <vector>

#include "shodhan/mip/plugins.hpp"

namespace shodhan::mip {

namespace {

constexpr double kEps = 1e-6;

struct Candidate {
  Index col;
  double frac;     // value - floor(value)
  double score;    // pseudocost product score
};

std::vector<Candidate> rank_candidates(const SearchState& s) {
  std::vector<Candidate> c;
  c.reserve(s.lp.fractional.size());
  for (const Index j : s.lp.fractional) {
    const double v = s.lp.x[to_size(j)];
    const double f = v - std::floor(v);
    c.push_back({j, f, s.pseudo->score(j, f)});
  }
  std::stable_sort(c.begin(), c.end(), [](const Candidate& a, const Candidate& b) {
    if (std::fabs(a.score - b.score) > 1e-12 * std::max(a.score, b.score)) return a.score > b.score;
    return a.col < b.col;  // deterministic tie-break: lowest index
  });
  return c;
}

class PseudocostRule final : public BranchingRule {
 public:
  const char* name() const override { return "pseudocost"; }
  BranchDecision select(SearchState& s) override {
    const std::vector<Candidate> ranked = rank_candidates(s);
    BranchDecision d;
    d.col = ranked.front().col;
    return d;
  }
};

// Outcome of strong branching on one candidate.
struct StrongResult {
  bool down_cut = false, up_cut = false;  // child infeasible or not better than the incumbent
  double down_gain = 0.0, up_gain = 0.0;
  bool down_exact = false, up_exact = false;  // child LP solved to optimality: its objective is a valid bound
  double down_obj = -kInf, up_obj = -kInf;
};

class ReliabilityRule final : public BranchingRule {
 public:
  const char* name() const override { return "reliability"; }

  BranchDecision select(SearchState& s) override {
    const MipOptions& opt = *s.options;
    const std::vector<Candidate> ranked = rank_candidates(s);
    BranchDecision best;
    best.col = ranked.front().col;
    double best_score = -1.0;
    int evaluated = 0, since_improvement = 0;
    bool strong_allowed = !s.time_up();
    for (const Candidate& c : ranked) {
      double score = c.score;
      const bool unreliable = !s.pseudo->reliable(c.col, opt.reliability_threshold);
      double down_gain = s.pseudo->value(c.col, -1) * c.frac;
      double up_gain = s.pseudo->value(c.col, +1) * (1.0 - c.frac);
      double down_obj = -kInf, up_obj = -kInf;
      if (unreliable && strong_allowed && evaluated < opt.strong_candidate_limit) {
        ++evaluated;
        const StrongResult r = strong_branch(s, c.col, c.frac);
        const double v = s.lp.x[to_size(c.col)];
        const double lo = (*s.lo)[to_size(c.col)], hi = (*s.hi)[to_size(c.col)];
        if (r.down_cut && r.up_cut) {
          BranchDecision d;
          d.kind = BranchDecision::Kind::Prune;
          return d;
        }
        if (r.down_cut || r.up_cut) {
          BranchDecision d;
          d.kind = BranchDecision::Kind::Tighten;
          // down child cut: the column must be at least ceil(v); up child cut: at most floor(v).
          d.tightenings.push_back(r.down_cut ? BoundChange{c.col, std::ceil(v), hi} : BoundChange{c.col, lo, std::floor(v)});
          return d;
        }
        down_gain = r.down_gain;
        up_gain = r.up_gain;
        down_obj = r.down_exact ? r.down_obj : -kInf;
        up_obj = r.up_exact ? r.up_obj : -kInf;
        s.pseudo->update(c.col, -1, r.down_gain, c.frac);
        s.pseudo->update(c.col, +1, r.up_gain, 1.0 - c.frac);
        score = std::max(down_gain, kEps) * std::max(up_gain, kEps);
        if (score > best_score) since_improvement = 0;
        else ++since_improvement;
        if (since_improvement >= opt.strong_lookahead) strong_allowed = false;
      } else {
        score = std::max(down_gain, kEps) * std::max(up_gain, kEps);
      }
      if (score > best_score + 1e-12 * std::max(1.0, best_score)) {
        best_score = score;
        best.col = c.col;
        best.down_gain = down_gain;
        best.up_gain = up_gain;
        best.down_bound = down_obj;
        best.up_bound = up_obj;
      }
    }
    return best;
  }

 private:
  // Solves the two children of `col` with a short dual simplex run each, on copies of the engine.
  static StrongResult strong_branch(SearchState& s, Index col, double frac) {
    StrongResult r;
    const MipOptions& opt = *s.options;
    const double v = s.lp.x[to_size(col)];
    const double z = s.lp.objective;
    const double cutoff = s.cutoff();
    const double lo = s.engine->col_lower(col), hi = s.engine->col_upper(col);
    (void)frac;
    for (int dir = -1; dir <= 1; dir += 2) {
      const double nlo = dir < 0 ? lo : std::ceil(v);
      const double nhi = dir < 0 ? std::floor(v) : hi;
      bool cut = false, exact = false;
      double gain = 0.0, obj = -kInf;
      if (nlo > nhi) {
        cut = true;
      } else {
        SimplexEngine probe = *s.engine;  // a copy: the search's engine is untouched
        probe.options().final_check = false;
        const long long before = probe.stats().iterations;
        probe.change_col_bounds(col, nlo, nhi);
        const EngineStatus st = probe.solve_limited(opt.strong_iteration_limit);
        s.add_iterations(probe.stats().iterations - before, true);
        s.count_strong_solve();
        if (st == EngineStatus::Infeasible) {
          cut = true;
        } else if (st == EngineStatus::Optimal) {
          obj = probe.objective() / s.obj_scale;
          exact = true;
          gain = std::max(0.0, obj - z);
          if (cutoff < kInf && obj >= cutoff - 1e-9 * std::max(1.0, std::fabs(cutoff))) cut = true;
        } else {
          // Iteration limit or trouble: the dual objective reached so far is an estimate of the gain.
          gain = std::max(0.0, probe.current_dual_objective() / s.obj_scale - z);
        }
      }
      if (dir < 0) {
        r.down_cut = cut;
        r.down_gain = gain;
        r.down_exact = exact;
        r.down_obj = obj;
      } else {
        r.up_cut = cut;
        r.up_gain = gain;
        r.up_exact = exact;
        r.up_obj = obj;
      }
    }
    return r;
  }
};

}  // namespace

void register_branching_reliability(PluginRegistry& reg) {
  reg.branching.add("pseudocost", [] { return std::make_unique<PseudocostRule>(); });
  reg.branching.add("reliability", [] { return std::make_unique<ReliabilityRule>(); });
}

}  // namespace shodhan::mip
