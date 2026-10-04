#include "shodhan/mip/incumbent.hpp"

#include <algorithm>
#include <cmath>

#include "shodhan/lp_solver.hpp"

namespace shodhan::mip {

PointCheck check_point(const LpModel& model, const std::vector<double>& x) {
  PointCheck pc;
  std::vector<double> act(to_size(model.n_rows), 0.0);
  for (Index j = 0; j < model.n_cols; ++j) {
    const double xj = x[to_size(j)];
    if (xj == 0.0) continue;
    for (Index p = model.A.col_start[to_size(j)]; p < model.A.col_start[to_size(j) + 1]; ++p) {
      act[to_size(model.A.row_index[to_size(p)])] += model.A.value[to_size(p)] * xj;
    }
  }
  for (Index i = 0; i < model.n_rows; ++i) {
    const double lo = model.row_lower[to_size(i)], hi = model.row_upper[to_size(i)], a = act[to_size(i)];
    if (!is_inf(lo) && a < lo) pc.max_row_violation = std::max(pc.max_row_violation, (lo - a) / (1.0 + std::fabs(lo)));
    if (!is_inf(hi) && a > hi) pc.max_row_violation = std::max(pc.max_row_violation, (a - hi) / (1.0 + std::fabs(hi)));
  }
  for (Index j = 0; j < model.n_cols; ++j) {
    const double lo = model.col_lower[to_size(j)], hi = model.col_upper[to_size(j)], v = x[to_size(j)];
    if (!is_inf(lo) && v < lo) pc.max_column_violation = std::max(pc.max_column_violation, (lo - v) / (1.0 + std::fabs(lo)));
    if (!is_inf(hi) && v > hi) pc.max_column_violation = std::max(pc.max_column_violation, (v - hi) / (1.0 + std::fabs(hi)));
    if (model.is_integer(j)) pc.max_integrality_violation = std::max(pc.max_integrality_violation, std::fabs(v - std::round(v)));
  }
  return pc;
}

IncumbentManager::IncumbentManager(const LpModel& original, const PostsolveStack* stack, const MipOptions& options)
    : original_(original), stack_(stack), opt_(options), sense_(original.sense == Sense::Maximize ? -1.0 : 1.0) {
  for (Index j = 0; j < original.n_cols; ++j) {
    if (!original.is_integer(j)) has_continuous_ = true;
  }
  opt_.log = nullptr;
}

SubmitOutcome IncumbentManager::submit(const std::vector<double>& working_x, const std::string& source) {
  ++submitted_;
  if (stack_ == nullptr) return finish(working_x, source, true);
  if (working_x.size() != stack_->col_map.size()) {
    ++rej_.wrong_size;
    return SubmitOutcome::Rejected;
  }
  Solution reduced;
  reduced.x = working_x;
  const Solution full = postsolve(*stack_, reduced);
  return finish(full.x, source, true);
}

SubmitOutcome IncumbentManager::submit_original(const std::vector<double>& original_x, const std::string& source) {
  ++submitted_;
  return finish(original_x, source, true);
}

SubmitOutcome IncumbentManager::finish(std::vector<double> x, const std::string& source, bool try_completion) {
  const LpModel& m = original_;
  if (x.size() != to_size(m.n_cols)) {
    ++rej_.wrong_size;
    return SubmitOutcome::Rejected;
  }
  // 3. Snap the integer columns; they must stay inside their bounds.
  for (Index j = 0; j < m.n_cols; ++j) {
    if (!m.is_integer(j)) continue;
    const double r = std::round(x[to_size(j)]);
    const double lo = m.col_lower[to_size(j)], hi = m.col_upper[to_size(j)];
    if ((!is_inf(lo) && r < lo - 1e-9 * (1.0 + std::fabs(lo))) || (!is_inf(hi) && r > hi + 1e-9 * (1.0 + std::fabs(hi)))) {
      ++rej_.not_integral;
      return SubmitOutcome::Rejected;
    }
    x[to_size(j)] = r;
  }
  // 4. Complete the continuous part: LP with the integer columns fixed.
  if (has_continuous_ && try_completion) {
    LpModel fixed = m;
    for (Index j = 0; j < m.n_cols; ++j) {
      if (!m.is_integer(j)) continue;
      fixed.col_lower[to_size(j)] = fixed.col_upper[to_size(j)] = x[to_size(j)];
      fixed.col_type[to_size(j)] = ColType::Continuous;
    }
    LpOptions lo;
    lo.params.primal_tol = opt_.params.primal_tol;
    lo.params.dual_tol = opt_.params.dual_tol;
    lo.params.seed = opt_.params.seed;
    lo.params.verbosity = 0;
    const LpResult lr = LpSolver(lo).solve(fixed);
    if (lr.status == Status::Optimal) {
      for (Index j = 0; j < m.n_cols; ++j) {
        if (!m.is_integer(j)) x[to_size(j)] = lr.solution.x[to_size(j)];
      }
    } else if (lr.status == Status::Infeasible) {
      ++rej_.lp_resolve_failed;
      return SubmitOutcome::Rejected;
    }
    // Any other LP outcome (a numerical failure): fall through and verify the candidate's own continuous values.
  }
  // 5. Verify on the original model.
  const PointCheck pc = check_point(m, x);
  if (pc.max_row_violation > opt_.params.primal_tol) {
    ++rej_.row_violated;
    return SubmitOutcome::Rejected;
  }
  if (pc.max_column_violation > opt_.params.primal_tol) {
    ++rej_.column_violated;
    return SubmitOutcome::Rejected;
  }
  if (pc.max_integrality_violation != 0.0) {  // exact after snapping
    ++rej_.not_integral;
    return SubmitOutcome::Rejected;
  }
  // 6. Strictly better?
  double obj = m.objective_offset;
  for (Index j = 0; j < m.n_cols; ++j) obj += m.col_cost[to_size(j)] * x[to_size(j)];
  const double obj_min = sense_ * obj;
  if (has_ && !(obj_min < objective_min_ - 1e-9 * std::max(1.0, std::fabs(objective_min_)))) {
    ++rej_.not_better;
    return SubmitOutcome::NotBetter;
  }
  has_ = true;
  objective_min_ = obj_min;
  x_ = std::move(x);
  source_ = source;
  ++found_;
  auto it = std::find_if(by_source_.begin(), by_source_.end(), [&](const auto& p) { return p.first == source; });
  if (it == by_source_.end()) by_source_.emplace_back(source, 1);
  else ++it->second;
  return SubmitOutcome::Accepted;
}

}  // namespace shodhan::mip
