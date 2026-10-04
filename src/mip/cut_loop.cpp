// The root cut loop (docs/CUTS.md).

#include <algorithm>
#include <chrono>
#include <cmath>

#include "shodhan/mip/cuts.hpp"

namespace shodhan::mip {

namespace {

std::size_t u(Index i) { return static_cast<std::size_t>(i); }

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

CutLoopResult run_root_cut_loop(const CutLoopInput& in) {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point t_start = Clock::now();
  CutLoopResult res;
  CutStats& st = res.stats;
  SimplexEngine& e = *in.engine;
  const LpModel& M = *in.model;
  const MipOptions& o = *in.options;
  const Scaling& sc = *in.scaling;
  const Index n = M.n_cols;
  const Index m0 = e.n_rows();
  const CsrMatrix rows = M.A.to_csr();
  std::vector<std::unique_ptr<Separator>> seps = make_separators(o);
  std::vector<Index> int_cols;
  for (Index j = 0; j < n; ++j) {
    if (M.is_integer(j)) int_cols.push_back(j);
  }
  auto bound = [&]() { return e.objective() / sc.obj_scale; };
  auto unscaled_x = [&]() {
    std::vector<double> x(u(n));
    const std::vector<double>& xs = e.primal_all();
    for (Index j = 0; j < n; ++j) x[u(j)] = xs[u(j)] * sc.col_scale[u(j)];
    return x;
  };
  st.root_bound_before = bound();
  double prev = st.root_bound_before;
  const long long it0 = e.stats().iterations;
  std::vector<Cut> active;   // the cuts that are rows m0, m0+1, ... of the engine
  std::vector<int> age;
  int stall = 0;
  st.stopped_because = "round limit";
  for (int round = 0; round < o.cut_rounds; ++round) {
    if (in.time_up && in.time_up()) {
      st.stopped_because = "time limit";
      break;
    }
    const std::vector<double> x = unscaled_x();
    bool fractional = false;
    for (const Index j : int_cols) {
      if (std::fabs(x[u(j)] - std::round(x[u(j)])) > o.params.int_tol) {
        fractional = true;
        break;
      }
    }
    if (!fractional) {
      st.stopped_because = "LP solution integral";
      break;
    }
    CutData data;
    data.model = &M;
    data.rows = &rows;
    data.lo = in.lo;
    data.hi = in.hi;
    data.x = &x;
    data.structure = in.structure;
    data.engine = &e;
    data.scaling = &sc;
    data.options = &o;
    std::vector<Cut> candidates;
    for (const std::unique_ptr<Separator>& s : seps) {
      const Clock::time_point t0 = Clock::now();
      SeparatorStats& ss = st.sep[s->id()];
      ++ss.calls;
      std::vector<Cut> raw;
      s->separate(data, raw);
      ss.generated += static_cast<long long>(raw.size());
      for (Cut& c : raw) {
        c.separator = s->id();
        if (clean_cut(c, *in.lo, *in.hi, x, o, ss)) {
          ++ss.candidates;
          candidates.push_back(std::move(c));
        }
      }
      ss.seconds += seconds_since(t0);
    }
    if (in.keep_candidates) res.candidates.insert(res.candidates.end(), candidates.begin(), candidates.end());
    if (candidates.empty()) {
      st.stopped_because = "no violated cut found";
      break;
    }
    const std::vector<std::size_t> chosen = select_cuts(candidates, n, static_cast<std::size_t>(o.cut_max_per_round), o, st);
    if (chosen.empty()) {
      st.stopped_because = "no cut survived the filters";
      break;
    }
    std::vector<RowSpec> specs;
    for (const std::size_t k : chosen) {
      specs.push_back(cut_to_row(candidates[k], sc));
      ++st.sep[candidates[k].separator].added;
      active.push_back(candidates[k]);
      age.push_back(0);
    }
    st.cuts_added += static_cast<long long>(chosen.size());
    e.add_rows(specs);
    ++st.rounds;
    const CutLpOutcome out = in.resolve();
    if (out == CutLpOutcome::Infeasible) {
      st.infeasible = true;
      st.stopped_because = "the LP with cuts is infeasible";
      break;
    }
    if (out == CutLpOutcome::Failed) {
      st.lp_failed = true;
      st.stopped_because = "LP failure after adding cuts";
      break;
    }
    const double now = bound();
    const double progress = (now - prev) / std::max(1.0, std::fabs(prev));
    stall = progress < o.cut_min_progress ? stall + 1 : 0;
    prev = now;
    // Age the cuts and drop those that stayed slack for too long.
    std::vector<Index> drop;
    for (std::size_t k = 0; k < active.size(); ++k) {
      const Index logical = e.n_structural() + m0 + static_cast<Index>(k);
      if (e.status(logical) == VarStatus::Basic) ++age[k];
      else age[k] = 0;
      if (age[k] >= o.cut_age_limit) drop.push_back(m0 + static_cast<Index>(k));
    }
    if (!drop.empty() && e.remove_rows(drop)) {
      std::vector<Cut> keep;
      std::vector<int> keep_age;
      std::size_t d = 0;
      for (std::size_t k = 0; k < active.size(); ++k) {
        if (d < drop.size() && drop[d] == m0 + static_cast<Index>(k)) {
          ++d;
          continue;
        }
        keep.push_back(std::move(active[k]));
        keep_age.push_back(age[k]);
      }
      st.cuts_removed += static_cast<long long>(drop.size());
      active = std::move(keep);
      age = std::move(keep_age);
    }
    if (stall >= o.cut_stall_rounds) {
      st.stopped_because = "bound stalled";
      break;
    }
  }
  if (!st.lp_failed && !st.infeasible) {
    // Keep only the cuts that are tight at the final LP optimum: the slack ones do not change it.
    std::vector<Index> drop;
    for (std::size_t k = 0; k < active.size(); ++k) {
      if (e.status(e.n_structural() + m0 + static_cast<Index>(k)) == VarStatus::Basic) drop.push_back(m0 + static_cast<Index>(k));
    }
    if (!drop.empty() && e.remove_rows(drop)) {
      std::vector<Cut> keep;
      std::size_t d = 0;
      for (std::size_t k = 0; k < active.size(); ++k) {
        if (d < drop.size() && drop[d] == m0 + static_cast<Index>(k)) {
          ++d;
          continue;
        }
        keep.push_back(std::move(active[k]));
      }
      st.cuts_removed += static_cast<long long>(drop.size());
      active = std::move(keep);
    }
    st.root_bound_after = bound();
  } else {
    st.root_bound_after = st.root_bound_before;
  }
  st.cuts_kept = static_cast<long long>(active.size());
  res.kept = std::move(active);
  st.lp_iterations = e.stats().iterations - it0;
  st.seconds = seconds_since(t_start);
  return res;
}

}  // namespace shodhan::mip
