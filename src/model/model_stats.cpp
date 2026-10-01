#include "shodhan/model_stats.hpp"

#include <algorithm>
#include <cmath>

namespace shodhan {

namespace {

void accumulate(AbsRange* r, double v) {
  const double a = std::fabs(v);
  if (a == 0.0) return;
  if (!r->valid) {
    r->min = r->max = a;
    r->valid = true;
  } else {
    r->min = std::min(r->min, a);
    r->max = std::max(r->max, a);
  }
}

}  // namespace

ModelStats compute_stats(const LpModel& model) {
  ModelStats s;
  s.rows = model.n_rows;
  s.cols = model.n_cols;
  s.nnz = model.A.nnz();
  if (model.n_rows > 0 && model.n_cols > 0) {
    s.density = static_cast<double>(s.nnz) /
                (static_cast<double>(model.n_rows) * static_cast<double>(model.n_cols));
  }

  for (const double v : model.A.value) accumulate(&s.coefficient, v);
  if (s.coefficient.valid) s.coefficient_ratio = s.coefficient.max / s.coefficient.min;

  for (std::size_t j = 0; j < to_size(model.n_cols); ++j) {
    switch (model.col_type[j]) {
      case ColType::Continuous:
        ++s.continuous_cols;
        break;
      case ColType::Integer:
        ++s.integer_cols;
        break;
      case ColType::Binary:
        ++s.binary_cols;
        break;
    }
    accumulate(&s.cost, model.col_cost[j]);
    const double lo = model.col_lower[j];
    const double up = model.col_upper[j];
    const bool lo_inf = is_inf(lo);
    const bool up_inf = is_inf(up);
    if (!lo_inf) accumulate(&s.bound, lo);
    if (!up_inf) accumulate(&s.bound, up);
    if (lo_inf && up_inf) {
      ++s.cols_free;
    } else if (lo_inf || up_inf) {
      ++s.cols_one_sided;
    } else if (lo == up) {
      ++s.cols_fixed;
    } else {
      ++s.cols_boxed;
    }
  }

  for (std::size_t i = 0; i < to_size(model.n_rows); ++i) {
    const double lo = model.row_lower[i];
    const double up = model.row_upper[i];
    const bool lo_inf = is_inf(lo);
    const bool up_inf = is_inf(up);
    if (lo_inf && up_inf) {
      ++s.rows_free;
    } else if (lo_inf) {
      ++s.rows_le;
    } else if (up_inf) {
      ++s.rows_ge;
    } else if (lo == up) {
      ++s.rows_eq;
    } else {
      ++s.rows_ranged;
    }
  }
  return s;
}

}  // namespace shodhan
