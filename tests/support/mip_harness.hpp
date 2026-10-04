#pragma once

// Test-only harness: a SearchState around an engine that has just solved the LP relaxation of a model (no
// presolve, no scaling), with a real IncumbentManager behind submit(). Used to run one plugin in isolation.

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "shodhan/mip/incumbent.hpp"
#include "shodhan/mip/plugins.hpp"

namespace shodhan::testing {

/// The minimization form of a model (costs and offset negated for a max model).
inline LpModel min_form(const LpModel& m) {
  LpModel out = m;
  if (m.sense == Sense::Maximize) {
    for (double& c : out.col_cost) c = -c;
    out.objective_offset = -out.objective_offset;
    out.sense = Sense::Minimize;
  }
  return out;
}

struct MipHarness {
  LpModel original;
  LpModel model;  // minimization form: the "presolved" model the plugins see
  Scaling scaling;
  std::unique_ptr<SimplexEngine> engine;
  mip::MipOptions options;
  mip::Pseudocosts pseudo;
  mip::Random rng{1};
  std::vector<Index> int_cols;
  std::unique_ptr<mip::IncumbentManager> incumbent;
  std::vector<double> lo, hi;
  long long strong_iterations = 0, other_iterations = 0, strong_solves = 0;
  bool lp_ok = false;
  mip::SearchState state;

  explicit MipHarness(const LpModel& m, bool solve_lp = true) : original(m), model(min_form(m)), pseudo(m.n_cols) {
    options.params.verbosity = 0;
    scaling.row_scale.assign(to_size(m.n_rows), 1.0);
    scaling.col_scale.assign(to_size(m.n_cols), 1.0);
    engine = std::make_unique<SimplexEngine>(model);
    incumbent = std::make_unique<mip::IncumbentManager>(original, nullptr, options);
    for (Index j = 0; j < m.n_cols; ++j) {
      if (m.is_integer(j)) int_cols.push_back(j);
    }
    lo = model.col_lower;
    hi = model.col_upper;
    state.model = &model;
    state.scaling = &scaling;
    state.engine = engine.get();
    state.options = &options;
    state.pseudo = &pseudo;
    state.rng = &rng;
    state.int_cols = &int_cols;
    state.lo = &lo;
    state.hi = &hi;
    state.submit = [this](const std::vector<double>& x, const std::string& src) { return incumbent->submit(x, src) == mip::SubmitOutcome::Accepted; };
    state.cutoff = [this]() { return incumbent->has_incumbent() ? incumbent->objective_min() : kInf; };
    state.time_up = []() { return false; };
    state.lp_iterations = []() { return 1000LL; };
    state.add_iterations = [this](long long n, bool strong) { (strong ? strong_iterations : other_iterations) += n; };
    state.count_strong_solve = [this]() { ++strong_solves; };
    if (solve_lp) solve();
  }

  /// Solves the LP relaxation and fills state.lp.
  void solve() {
    lp_ok = engine->solve() == EngineStatus::Optimal;
    if (!lp_ok) return;
    state.lp.objective = engine->objective();
    state.lp.x.assign(engine->primal_all().begin(), engine->primal_all().begin() + model.n_cols);
    state.lp.fractional.clear();
    for (const Index j : int_cols) {
      const double v = state.lp.x[to_size(j)];
      if (std::fabs(v - std::round(v)) > 1e-5) state.lp.fractional.push_back(j);
    }
  }
};

}  // namespace shodhan::testing
