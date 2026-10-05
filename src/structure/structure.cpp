#include "shodhan/structure.hpp"

#include <cmath>

namespace shodhan {

namespace {

bool is_binary_col(const LpModel& m, Index j) {
  return m.is_integer(j) && m.col_lower[to_size(j)] == 0.0 && m.col_upper[to_size(j)] == 1.0;
}

}  // namespace

std::size_t StructureInfo::n_set(SetRowKind kind) const {
  std::size_t k = 0;
  for (const SetRow& s : set_rows) k += s.kind == kind ? 1 : 0;
  return k;
}

bool match_vub_shape(double ax, double ay, double b, double* u, bool* complemented) {
  if (!(ax > 0.0) || ay == 0.0 || !std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(b)) return false;
  const double tol = 1e-12 * std::fabs(ay);
  if (ay < 0.0 && std::fabs(b) <= tol) {
    *u = -ay / ax;
    *complemented = false;
    return true;
  }
  if (ay > 0.0 && std::fabs(b - ay) <= tol) {
    *u = ay / ax;
    *complemented = true;
    return true;
  }
  return false;
}

StructureInfo detect_structure(const LpModel& model) {
  StructureInfo info;
  info.is_balance_row.assign(to_size(model.n_rows), 0);
  const CsrMatrix R = model.A.to_csr();
  for (Index i = 0; i < model.n_rows; ++i) {
    const std::size_t b0 = to_size(R.row_start[to_size(i)]), b1 = to_size(R.row_start[to_size(i) + 1]);
    const std::size_t len = b1 - b0;
    if (len < 2) continue;
    const double rl = model.row_lower[to_size(i)], ru = model.row_upper[to_size(i)];

    // Variable upper bound: exactly two columns, one finite side.
    if (len == 2 && (is_inf(rl) != is_inf(ru))) {
      const double s = is_inf(rl) ? 1.0 : -1.0;
      const double b = is_inf(rl) ? ru : -rl;
      for (int first = 0; first < 2; ++first) {
        const std::size_t kx = b0 + static_cast<std::size_t>(first), ky = b0 + static_cast<std::size_t>(1 - first);
        const Index x = R.col_index[kx], y = R.col_index[ky];
        if (model.col_lower[to_size(x)] != 0.0 || is_binary_col(model, x) || !is_binary_col(model, y)) continue;
        double u = 0.0;
        bool comp = false;
        if (match_vub_shape(s * R.value[kx], s * R.value[ky], b, &u, &comp)) {
          info.vubs.push_back({i, x, y, u, comp});
          break;
        }
      }
    }

    // Set rows over binary columns with unit coefficients and right-hand side 1.
    bool all_binary_unit = true;
    for (std::size_t k = b0; k < b1 && all_binary_unit; ++k) {
      all_binary_unit = is_binary_col(model, R.col_index[k]) && R.value[k] == 1.0;
    }
    if (all_binary_unit) {
      if (rl == 1.0 && ru == 1.0) info.set_rows.push_back({i, SetRowKind::Partition});
      else if (is_inf(rl) && ru == 1.0) info.set_rows.push_back({i, SetRowKind::Packing});
      else if (rl == 1.0 && is_inf(ru)) info.set_rows.push_back({i, SetRowKind::Covering});
    }

    // Flow balance: equality, all coefficients +-1, both signs, at least one column that is not binary.
    if (rl == ru) {
      bool pm1 = true, plus = false, minus = false, non_binary = false;
      for (std::size_t k = b0; k < b1 && pm1; ++k) {
        const double a = R.value[k];
        if (a == 1.0) plus = true;
        else if (a == -1.0) minus = true;
        else pm1 = false;
        if (!is_binary_col(model, R.col_index[k])) non_binary = true;
      }
      if (pm1 && plus && minus && non_binary) {
        info.balance_rows.push_back(i);
        info.is_balance_row[to_size(i)] = 1;
      }
    }
  }
  return info;
}

}  // namespace shodhan
