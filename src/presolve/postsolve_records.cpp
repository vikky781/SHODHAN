#include "postsolve_records.hpp"

#include <algorithm>
#include <cmath>

namespace shodhan::presolve_detail {

namespace {

std::size_t u(int i) { return static_cast<std::size_t>(i); }

// c - sum over entries (r != skip_row) of a * y_r.
double reduced_cost_without(double cost, const Entries& entries, int skip_row,
                            const std::vector<double>& y) {
  double v = cost;
  for (const auto& e : entries) {
    if (e.first == skip_row) continue;
    v -= e.second * y[u(e.first)];
  }
  return v;
}

}  // namespace

void EmptyColumnRecord::undo(PostsolveState& s) const {
  s.x[u(col_)] = value_;
  if (s.duals) s.d[u(col_)] = cost_;
}

void FixedColumnRecord::undo(PostsolveState& s) const {
  s.x[u(col_)] = value_;
  if (s.duals) s.d[u(col_)] = reduced_cost_without(cost_, entries_, -1, s.y);
}

// M_{k+1} dual d' = c_j - A_rest^T y for column j. Row i adds a * y_i, so
// d_j = d' - a * y_i. If the bound that came from the row is the one the
// column rests on (d' pushes against it), choose y_i = d'/a so that d_j = 0;
// the sign of y_i then matches the row side that produced that bound.
// Otherwise the row is inactive: y_i = 0.
void SingletonRowRecord::undo(PostsolveState& s) const {
  if (!s.duals) return;
  const double dprime = s.d[u(col_)];
  if ((lb_from_row_ && dprime > 0.0) || (ub_from_row_ && dprime < 0.0)) {
    s.y[u(row_)] = dprime / a_;
    s.d[u(col_)] = 0.0;
  } else {
    s.y[u(row_)] = 0.0;
  }
}

// Every column of the row sits at its activity-extreme bound. With
// e_j = c_j - sum_{r != i} a_rj y_r and d_j = e_j - a_ij y_i the sign rules on
// the forced columns are  y_i <= e_j / a_ij  for all j  (min-forcing: row at its
// upper bound) or  y_i >= e_j / a_ij  (max-forcing). Taking the extreme ratio,
// capped at 0 / floored at 0 unless the row is an equality, satisfies them all.
void ForcingRowRecord::undo(PostsolveState& s) const {
  for (const Forced& f : cols_) s.x[u(f.col)] = f.value;
  if (!s.duals) return;
  std::vector<double> e(cols_.size());
  double yi = 0.0;
  bool first = true;
  for (std::size_t k = 0; k < cols_.size(); ++k) {
    e[k] = reduced_cost_without(cols_[k].cost, cols_[k].entries, row_, s.y);
    const double ratio = e[k] / cols_[k].a;
    if (first) {
      yi = ratio;
      first = false;
    } else {
      yi = min_forcing_ ? std::min(yi, ratio) : std::max(yi, ratio);
    }
  }
  if (!equality_) yi = min_forcing_ ? std::min(yi, 0.0) : std::max(yi, 0.0);
  s.y[u(row_)] = yi;
  for (std::size_t k = 0; k < cols_.size(); ++k) s.d[u(cols_[k].col)] = e[k] - cols_[k].a * yi;
}

// x_j = beta - alpha x_k. With e_j, e_k the reduced costs without the removed
// row, M_{k+1} has d'_k = e_k - alpha e_j. Choosing y_i = e_j / a_j gives
// d_j = 0 and d_k = d'_k. If x_k rests on a bound that was derived from x_j's
// bounds, that choice has the wrong sign for x_k, so choose y_i = e_k / a_k
// instead: then d_k = 0 and d_j = -d'_k / alpha, which has the sign x_j's
// active bound needs.
void DoubletonRecord::undo(PostsolveState& s) const {
  const Data& d = d_;
  s.x[u(d.elim)] = d.beta - d.alpha * s.x[u(d.kept)];
  if (!s.duals) return;
  const double ej = reduced_cost_without(d.cost_elim, d.entries_elim, d.row, s.y);
  const double ek = reduced_cost_without(d.cost_kept, d.entries_kept, d.row, s.y);
  const double dk_prime = s.d[u(d.kept)];
  double yi = 0.0;
  if ((dk_prime > 0.0 && d.lb_from_elim) || (dk_prime < 0.0 && d.ub_from_elim)) {
    yi = ek / d.a_kept;
  } else {
    yi = ej / d.a_elim;
  }
  s.y[u(d.row)] = yi;
  s.d[u(d.elim)] = ej - d.a_elim * yi;
  s.d[u(d.kept)] = ek - d.a_kept * yi;
}

void DuplicateColumnRecord::undo(PostsolveState& s) const {
  double z = s.x[u(keep_)];
  if (integer_) z = std::round(z);
  double xk = lo_keep_;
  if (!is_inf(up_gone_)) xk = std::max(xk, z - up_gone_);
  if (!is_inf(up_keep_)) xk = std::min(xk, up_keep_);
  s.x[u(keep_)] = xk;
  s.x[u(gone_)] = z - xk;
}

}  // namespace shodhan::presolve_detail
