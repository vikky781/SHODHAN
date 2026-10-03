#include "shodhan/simplex_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace shodhan {

const char* to_string(EngineStatus status) noexcept {
  switch (status) {
    case EngineStatus::Optimal: return "Optimal";
    case EngineStatus::Infeasible: return "Infeasible";
    case EngineStatus::Unbounded: return "Unbounded";
    case EngineStatus::IterationLimit: return "IterationLimit";
    case EngineStatus::TimeLimit: return "TimeLimit";
    case EngineStatus::NumericalError: return "NumericalError";
  }
  return "Unknown";
}

SimplexEngine::SimplexEngine(const LpModel& model, const SimplexOptions& options)
    : opt_(options), model_(model), n_(model.n_cols), m_(model.n_rows), N_(model.n_cols + model.n_rows) {
  sgn_ = model.sense == Sense::Maximize ? -1.0 : 1.0;
  csr_ = model.A.to_csr();
  build_state();
  initialize_slack_basis();
}

void SimplexEngine::build_state() {
  const std::size_t sn = to_size(N_);
  lo_.assign(sn, 0.0);
  hi_.assign(sn, 0.0);
  cost_orig_.assign(sn, 0.0);
  for (Index j = 0; j < n_; ++j) {
    lo_[to_size(j)] = model_.col_lower[to_size(j)];
    hi_[to_size(j)] = model_.col_upper[to_size(j)];
    cost_orig_[to_size(j)] = sgn_ * model_.col_cost[to_size(j)];
  }
  for (Index i = 0; i < m_; ++i) {
    lo_[to_size(n_ + i)] = model_.row_lower[to_size(i)];
    hi_[to_size(n_ + i)] = model_.row_upper[to_size(i)];
  }
  cost_ = cost_orig_;
  costs_modified_ = false;
  basis_.assign(to_size(m_), 0);
  pos_.assign(sn, -1);
  status_.assign(sn, VarStatus::AtLower);
  x_.assign(sn, 0.0);
  d_.assign(sn, 0.0);
  y_.assign(to_size(m_), 0.0);
  weights_.assign(to_size(m_), 1.0);
  rho_.resize(m_);
  col_.resize(m_);
  tau_.resize(m_);
  rhs_.resize(m_);
  row_alpha_.resize(N_);
  FactorParams fp = opt_.factor;
  fp.max_updates = opt_.refactor_interval;
  factor_.set_params(fp);
}

bool SimplexEngine::time_exceeded() const {
  if (is_inf(opt_.time_limit)) return false;
  const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  return t >= opt_.time_limit;
}

double SimplexEngine::nonbasic_value(Index j) const {
  switch (status_[to_size(j)]) {
    case VarStatus::AtLower:
    case VarStatus::Fixed: return lo_[to_size(j)];
    case VarStatus::AtUpper: return hi_[to_size(j)];
    case VarStatus::FreeAtZero: return 0.0;
    case VarStatus::Basic: return x_[to_size(j)];
  }
  return 0.0;
}

void SimplexEngine::set_status_from_bounds(Index j, bool prefer_upper) {
  const double lo = lo_[to_size(j)], hi = hi_[to_size(j)];
  const bool flo = !is_inf(lo), fhi = !is_inf(hi);
  VarStatus st;
  if (flo && fhi) {
    st = lo == hi ? VarStatus::Fixed : (prefer_upper ? VarStatus::AtUpper : VarStatus::AtLower);
  } else if (flo) {
    st = VarStatus::AtLower;
  } else if (fhi) {
    st = VarStatus::AtUpper;
  } else {
    st = VarStatus::FreeAtZero;
  }
  status_[to_size(j)] = st;
}

void SimplexEngine::sync_nonbasic_values() {
  for (Index j = 0; j < N_; ++j) {
    if (status_[to_size(j)] != VarStatus::Basic) x_[to_size(j)] = nonbasic_value(j);
  }
}

void SimplexEngine::initialize_slack_basis() {
  std::fill(pos_.begin(), pos_.end(), -1);
  for (Index i = 0; i < m_; ++i) {
    basis_[to_size(i)] = n_ + i;
    pos_[to_size(n_ + i)] = i;
    status_[to_size(n_ + i)] = VarStatus::Basic;
  }
  cost_ = cost_orig_;
  costs_modified_ = false;
  for (Index j = 0; j < n_; ++j) set_status_from_bounds(j, cost_[to_size(j)] < 0.0);
  std::fill(weights_.begin(), weights_.end(), 1.0);
  weights_exact_ = true;
  sync_nonbasic_values();
  refactor();
  compute_primal();
  compute_dual();
}

bool SimplexEngine::set_basis(const std::vector<Index>& basis) {
  if (basis.size() != to_size(m_)) throw std::invalid_argument("SimplexEngine::set_basis: wrong basis size");
  std::vector<char> seen(to_size(N_), 0);
  for (const Index v : basis) {
    if (v < 0 || v >= N_ || seen[to_size(v)]) throw std::invalid_argument("SimplexEngine::set_basis: invalid basis");
    seen[to_size(v)] = 1;
  }
  basis_ = basis;
  std::fill(pos_.begin(), pos_.end(), -1);
  for (Index p = 0; p < m_; ++p) pos_[to_size(basis_[to_size(p)])] = p;
  for (Index j = 0; j < N_; ++j) {
    if (pos_[to_size(j)] >= 0) {
      status_[to_size(j)] = VarStatus::Basic;
    } else {
      set_status_from_bounds(j, cost_[to_size(j)] < 0.0);
    }
  }
  sync_nonbasic_values();
  const int repairs_before = stats_.basis_repairs;
  const bool ok = refactor(true);
  if (ok) {
    compute_primal();
    compute_dual();
    if (stats_.basis_repairs == repairs_before) compute_exact_weights();
  }
  return ok;
}

bool SimplexEngine::refactor(bool reset_weights) {
  FactorStatus st = factor_.factorize(model_.A, basis_);
  bool repaired = false;
  if (st == FactorStatus::RankDeficient) {
    const std::vector<Index> before = basis_;
    const std::vector<BasisSubstitution> changes = factor_.repair(model_.A, basis_);
    for (const BasisSubstitution& c : changes) {
      const Index old_var = c.old_var;
      const Index new_var = c.new_var;
      pos_[to_size(old_var)] = -1;
      set_status_from_bounds(old_var, d_[to_size(old_var)] < 0.0);
      x_[to_size(old_var)] = nonbasic_value(old_var);
      status_[to_size(new_var)] = VarStatus::Basic;
    }
    for (Index p = 0; p < m_; ++p) pos_[to_size(basis_[to_size(p)])] = p;
    ++stats_.basis_repairs;
    repaired = true;
    primal_stale_ = true;
    st = factor_.status();
  }
  if (st != FactorStatus::Ok) return false;
  updates_since_refactor_ = 0;
  ++stats_.refactors;
  if (reset_weights || repaired) {
    std::fill(weights_.begin(), weights_.end(), 1.0);
    weights_exact_ = false;
  }
  return true;
}

void SimplexEngine::compute_primal() {
  rhs_.clear();
  for (Index j = 0; j < N_; ++j) {
    if (status_[to_size(j)] == VarStatus::Basic) continue;
    const double v = x_[to_size(j)];
    if (v == 0.0) continue;
    if (j < n_) {
      for (Index t = model_.A.col_start[to_size(j)]; t < model_.A.col_start[to_size(j) + 1]; ++t) {
        rhs_.add(model_.A.row_index[to_size(t)], -model_.A.value[to_size(t)] * v);
      }
    } else {
      rhs_.add(j - n_, v);
    }
  }
  factor_.ftran(rhs_, false);
  for (Index p = 0; p < m_; ++p) x_[to_size(basis_[to_size(p)])] = rhs_[p];
  primal_stale_ = false;
}

void SimplexEngine::compute_dual() {
  rhs_.clear();
  for (Index p = 0; p < m_; ++p) {
    const double c = cost_[to_size(basis_[to_size(p)])];
    if (c != 0.0) rhs_.set(p, c);
  }
  factor_.btran(rhs_);
  for (Index i = 0; i < m_; ++i) y_[to_size(i)] = rhs_[i];
  update_duals_from_y();
}

void SimplexEngine::update_duals_from_y() {
  for (Index j = 0; j < n_; ++j) {
    if (status_[to_size(j)] == VarStatus::Basic) {
      d_[to_size(j)] = 0.0;
      continue;
    }
    double s = cost_[to_size(j)];
    for (Index t = model_.A.col_start[to_size(j)]; t < model_.A.col_start[to_size(j) + 1]; ++t) {
      s -= model_.A.value[to_size(t)] * y_[to_size(model_.A.row_index[to_size(t)])];
    }
    d_[to_size(j)] = s;
  }
  for (Index i = 0; i < m_; ++i) {
    const Index j = n_ + i;
    d_[to_size(j)] = status_[to_size(j)] == VarStatus::Basic ? 0.0 : cost_[to_size(j)] + y_[to_size(i)];
  }
}

Index SimplexEngine::fix_dual_infeasibilities(bool allow_shift) {
  Index remaining = 0;
  const double tol = opt_.dual_tol;
  for (Index j = 0; j < N_; ++j) {
    const VarStatus st = status_[to_size(j)];
    if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;
    const double dj = d_[to_size(j)];
    const bool boxed = !is_inf(lo_[to_size(j)]) && !is_inf(hi_[to_size(j)]);
    bool bad = false;
    if (boxed) {
      if (st == VarStatus::AtLower && dj < -tol) {
        status_[to_size(j)] = VarStatus::AtUpper;
        x_[to_size(j)] = hi_[to_size(j)];
        primal_stale_ = true;
      } else if (st == VarStatus::AtUpper && dj > tol) {
        status_[to_size(j)] = VarStatus::AtLower;
        x_[to_size(j)] = lo_[to_size(j)];
        primal_stale_ = true;
      }
      continue;
    }
    if (st == VarStatus::AtLower) bad = dj < -tol;
    else if (st == VarStatus::AtUpper) bad = dj > tol;
    else bad = std::fabs(dj) > tol;
    if (!bad) continue;
    if (allow_shift) {
      cost_[to_size(j)] -= dj;
      d_[to_size(j)] = 0.0;
      costs_modified_ = true;
      ++stats_.cost_shifts;
    } else {
      ++remaining;
    }
  }
  return remaining;
}

InfeasibilitySummary SimplexEngine::infeasibility() const {
  InfeasibilitySummary s;
  for (Index p = 0; p < m_; ++p) {
    const Index v = basis_[to_size(p)];
    const double x = x_[to_size(v)];
    double viol = 0.0, bound = 0.0;
    if (x < lo_[to_size(v)]) {
      viol = lo_[to_size(v)] - x;
      bound = lo_[to_size(v)];
    } else if (x > hi_[to_size(v)]) {
      viol = x - hi_[to_size(v)];
      bound = hi_[to_size(v)];
    }
    if (viol > ptol(bound)) {
      s.primal_sum += viol;
      s.primal_max = std::max(s.primal_max, viol);
      ++s.primal_count;
    }
  }
  for (Index j = 0; j < N_; ++j) {
    const VarStatus st = status_[to_size(j)];
    if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;
    const double dj = d_[to_size(j)];
    double viol = 0.0;
    if (st == VarStatus::AtLower) viol = -dj;
    else if (st == VarStatus::AtUpper) viol = dj;
    else viol = std::fabs(dj);
    if (viol > opt_.dual_tol) {
      s.dual_sum += viol;
      s.dual_max = std::max(s.dual_max, viol);
      ++s.dual_count;
    }
  }
  return s;
}

double SimplexEngine::objective() const {
  double s = 0.0;
  for (Index j = 0; j < N_; ++j) s += cost_orig_[to_size(j)] * x_[to_size(j)];
  return s + sgn_ * model_.objective_offset;
}

double SimplexEngine::working_objective() const {
  double s = 0.0;
  for (Index j = 0; j < N_; ++j) s += cost_[to_size(j)] * x_[to_size(j)];
  return s + sgn_ * model_.objective_offset;
}

Solution SimplexEngine::solution() const {
  Solution s;
  s.x.assign(x_.begin(), x_.begin() + n_);
  s.y = y_;
  s.d.assign(d_.begin(), d_.begin() + n_);
  s.objective = sgn_ * objective();
  return s;
}

void SimplexEngine::change_col_bounds(Index j, double lo, double hi) {
  if (j < 0 || j >= n_) throw std::invalid_argument("SimplexEngine::change_col_bounds: column out of range");
  if (lo > hi) throw std::invalid_argument("SimplexEngine::change_col_bounds: lo > hi");
  lo_[to_size(j)] = lo;
  hi_[to_size(j)] = hi;
  bounds_modified_ = true;
  if (status_[to_size(j)] != VarStatus::Basic) {
    set_status_from_bounds(j, d_[to_size(j)] < 0.0);
    x_[to_size(j)] = nonbasic_value(j);
    primal_stale_ = true;
  }
}

void SimplexEngine::change_row_bounds(Index i, double lo, double hi) {
  if (i < 0 || i >= m_) throw std::invalid_argument("SimplexEngine::change_row_bounds: row out of range");
  if (lo > hi) throw std::invalid_argument("SimplexEngine::change_row_bounds: lo > hi");
  const Index j = n_ + i;
  lo_[to_size(j)] = lo;
  hi_[to_size(j)] = hi;
  bounds_modified_ = true;
  if (status_[to_size(j)] != VarStatus::Basic) {
    set_status_from_bounds(j, d_[to_size(j)] < 0.0);
    x_[to_size(j)] = nonbasic_value(j);
    primal_stale_ = true;
  }
}

void SimplexEngine::log_line(const char* phase, double objective) {
  if (opt_.verbosity <= 0 || opt_.log == nullptr) return;
  const InfeasibilitySummary s = infeasibility();
  char buf[200];
  std::snprintf(buf, sizeof(buf), "%-8s iter %8lld  obj %.10e  sum pinf %.3e (%d)  refactors %d\n", phase,
                stats_.iterations, objective, s.primal_sum, static_cast<int>(s.primal_count), stats_.refactors);
  *opt_.log << buf;
}

}  // namespace shodhan
