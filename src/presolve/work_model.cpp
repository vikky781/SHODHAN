#include "work_model.hpp"

#include <algorithm>

namespace shodhan::presolve_detail {

namespace {
std::size_t u(int i) { return static_cast<std::size_t>(i); }
}  // namespace

WorkModel::WorkModel(const LpModel& model) {
  m = model.n_rows;
  n = model.n_cols;
  const double sgn = model.sense == Sense::Maximize ? -1.0 : 1.0;
  rows.assign(u(m), {});
  cols.assign(u(n), {});
  for (int j = 0; j < n; ++j) {
    for (Index p = model.A.col_start[u(j)]; p < model.A.col_start[u(j) + 1]; ++p) {
      const double v = model.A.value[to_size(p)];
      if (v == 0.0) continue;
      const int i = model.A.row_index[to_size(p)];
      cols[u(j)].push_back({i, v});
      rows[u(i)].push_back({j, v});
    }
  }
  rl = model.row_lower;
  ru = model.row_upper;
  cl = model.col_lower;
  cu = model.col_upper;
  cost.resize(u(n));
  for (int j = 0; j < n; ++j) cost[u(j)] = sgn * model.col_cost[u(j)];
  offset = sgn * model.objective_offset;
  is_int.resize(u(n));
  for (int j = 0; j < n; ++j) is_int[u(j)] = model.col_type[u(j)] != ColType::Continuous ? 1 : 0;
  row_alive.assign(u(m), 1);
  col_alive.assign(u(n), 1);
  row_dirty.assign(u(m), 1);
  col_dirty.assign(u(n), 1);
  row_cnt.resize(u(m));
  col_cnt.resize(u(n));
  for (int i = 0; i < m; ++i) row_cnt[u(i)] = static_cast<int>(rows[u(i)].size());
  for (int j = 0; j < n; ++j) col_cnt[u(j)] = static_cast<int>(cols[u(j)].size());
  row_mag.assign(u(m), 0.0);
  for (int i = 0; i < m; ++i) {
    double mag = 0.0;
    if (!is_inf(rl[u(i)])) mag = std::max(mag, std::fabs(rl[u(i)]));
    if (!is_inf(ru[u(i)])) mag = std::max(mag, std::fabs(ru[u(i)]));
    double sum = 0.0;
    for (const Entry& e : rows[u(i)]) {
      double bmag = 0.0;
      if (!is_inf(cl[u(e.idx)])) bmag = std::max(bmag, std::fabs(cl[u(e.idx)]));
      if (!is_inf(cu[u(e.idx)])) bmag = std::max(bmag, std::fabs(cu[u(e.idx)]));
      sum += std::fabs(e.val) * bmag;
    }
    row_mag[u(i)] = std::max(mag, sum);
  }
}

int WorkModel::alive_rows() const {
  int c = 0;
  for (const char a : row_alive) c += a ? 1 : 0;
  return c;
}

int WorkModel::alive_cols() const {
  int c = 0;
  for (const char a : col_alive) c += a ? 1 : 0;
  return c;
}

Entries WorkModel::col_snapshot(int j) const {
  Entries out;
  for_col(j, [&](int i, double a) { out.emplace_back(i, a); });
  return out;
}

Entries WorkModel::row_snapshot(int i) const {
  Entries out;
  for_row(i, [&](int j, double a) { out.emplace_back(j, a); });
  return out;
}

void WorkModel::remove_row(int i) {
  row_alive[u(i)] = 0;
  for (const Entry& e : rows[u(i)]) {
    if (!col_alive[u(e.idx)]) continue;
    --col_cnt[u(e.idx)];
    col_dirty[u(e.idx)] = 1;
  }
}

void WorkModel::remove_col(int j) {
  col_alive[u(j)] = 0;
  for (const Entry& e : cols[u(j)]) {
    if (!row_alive[u(e.idx)]) continue;
    --row_cnt[u(e.idx)];
    row_dirty[u(e.idx)] = 1;
  }
}

void WorkModel::substitute_fixed(int j, double v) {
  for_col(j, [&](int i, double a) {
    if (!is_inf(rl[u(i)])) rl[u(i)] -= a * v;
    if (!is_inf(ru[u(i)])) ru[u(i)] -= a * v;
    row_mag[u(i)] = std::max(row_mag[u(i)], std::fabs(a * v));
  });
  offset += cost[u(j)] * v;
  remove_col(j);
}

double WorkModel::coef(int r, int c) const {
  for (const Entry& e : rows[u(r)]) {
    if (e.idx == c) return e.val;
  }
  return 0.0;
}

void WorkModel::erase_entry(std::vector<Entry>& v, int idx) {
  for (std::size_t k = 0; k < v.size(); ++k) {
    if (v[k].idx == idx) {
      v[k] = v.back();
      v.pop_back();
      return;
    }
  }
}

void WorkModel::set_coef(int r, int c, double v) {
  // Entries pointing at dead rows/columns are never matched: a dead column
  // index cannot be passed in here.
  std::vector<Entry>& row = rows[u(r)];
  std::vector<Entry>& col = cols[u(c)];
  bool found = false;
  for (std::size_t k = 0; k < row.size(); ++k) {
    if (row[k].idx == c) {
      found = true;
      if (v == 0.0) {
        row[k] = row.back();
        row.pop_back();
      } else {
        row[k].val = v;
      }
      break;
    }
  }
  if (found) {
    if (v == 0.0) {
      erase_entry(col, r);
      --row_cnt[u(r)];
      --col_cnt[u(c)];
    } else {
      for (Entry& e : col) {
        if (e.idx == r) {
          e.val = v;
          break;
        }
      }
    }
  } else if (v != 0.0) {
    row.push_back({c, v});
    col.push_back({r, v});
    ++row_cnt[u(r)];
    ++col_cnt[u(c)];
  }
  row_dirty[u(r)] = 1;
  col_dirty[u(c)] = 1;
}

void WorkModel::touch_col_bounds(int j) {
  col_dirty[u(j)] = 1;
  double bmag = 0.0;
  if (!is_inf(cl[u(j)])) bmag = std::max(bmag, std::fabs(cl[u(j)]));
  if (!is_inf(cu[u(j)])) bmag = std::max(bmag, std::fabs(cu[u(j)]));
  for_col(j, [&](int i, double a) {
    row_dirty[u(i)] = 1;
    row_mag[u(i)] = std::max(row_mag[u(i)], std::fabs(a) * bmag);
  });
}

void WorkModel::touch_row_bounds(int i) {
  row_dirty[u(i)] = 1;
  for_row(i, [&](int j, double) { col_dirty[u(j)] = 1; });
}

Activity WorkModel::activity(int i) const {
  Activity act;
  for_row(i, [&](int j, double a) {
    const double lo = cl[u(j)];
    const double up = cu[u(j)];
    if (a > 0.0) {
      if (is_neg_inf(lo)) {
        ++act.min_inf;
      } else {
        act.min_fin += a * lo;
      }
      if (is_pos_inf(up)) {
        ++act.max_inf;
      } else {
        act.max_fin += a * up;
      }
    } else {
      if (is_pos_inf(up)) {
        ++act.min_inf;
      } else {
        act.min_fin += a * up;
      }
      if (is_neg_inf(lo)) {
        ++act.max_inf;
      } else {
        act.max_fin += a * lo;
      }
    }
  });
  return act;
}

void WorkModel::compact() {
  for (int i = 0; i < m; ++i) {
    std::vector<Entry>& v = rows[u(i)];
    if (!row_alive[u(i)]) {
      std::vector<Entry>().swap(v);
      continue;
    }
    v.erase(std::remove_if(v.begin(), v.end(), [&](const Entry& e) { return !col_alive[u(e.idx)]; }),
            v.end());
  }
  for (int j = 0; j < n; ++j) {
    std::vector<Entry>& v = cols[u(j)];
    if (!col_alive[u(j)]) {
      std::vector<Entry>().swap(v);
      continue;
    }
    v.erase(std::remove_if(v.begin(), v.end(), [&](const Entry& e) { return !row_alive[u(e.idx)]; }),
            v.end());
  }
}

bool WorkModel::any_dirty() const {
  for (int i = 0; i < m; ++i) {
    if (row_alive[u(i)] && row_dirty[u(i)]) return true;
  }
  for (int j = 0; j < n; ++j) {
    if (col_alive[u(j)] && col_dirty[u(j)]) return true;
  }
  return false;
}

}  // namespace shodhan::presolve_detail
