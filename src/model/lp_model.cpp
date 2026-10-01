#include "shodhan/lp_model.hpp"

#include <cmath>
#include <unordered_set>
#include <utility>

namespace shodhan {

namespace {

constexpr std::size_t kMaxReportedProblems = 50;

void check_names(const std::vector<std::string>& names, std::size_t expected, const char* what,
                 std::vector<std::string>* problems) {
  if (names.empty()) return;
  if (names.size() != expected) {
    problems->push_back(std::string(what) + " names: expected " + std::to_string(expected) +
                        " entries, found " + std::to_string(names.size()));
    return;
  }
  std::unordered_set<std::string> seen;
  for (const std::string& n : names) {
    if (n.empty()) {
      problems->push_back(std::string("empty ") + what + " name");
    } else if (!seen.insert(n).second) {
      problems->push_back(std::string("duplicate ") + what + " name '" + n + "'");
    }
  }
}

}  // namespace

std::vector<std::string> LpModel::validate() const {
  std::vector<std::string> problems;
  auto add = [&](std::string message) {
    if (problems.size() < kMaxReportedProblems) problems.push_back(std::move(message));
  };

  if (n_rows < 0 || n_cols < 0) {
    add("negative dimensions");
    return problems;
  }
  const std::size_t m = to_size(n_rows);
  const std::size_t n = to_size(n_cols);

  if (A.n_rows != n_rows || A.n_cols != n_cols) {
    add("matrix is " + std::to_string(A.n_rows) + "x" + std::to_string(A.n_cols) +
        " but model is " + std::to_string(n_rows) + "x" + std::to_string(n_cols));
  }
  for (const std::string& p : A.validate()) add("A: " + p);

  auto size_check = [&](std::size_t actual, std::size_t expected, const char* what) {
    if (actual != expected) {
      add(std::string(what) + " has " + std::to_string(actual) + " entries, expected " +
          std::to_string(expected));
      return false;
    }
    return true;
  };
  const bool cost_ok = size_check(col_cost.size(), n, "col_cost");
  const bool lo_ok = size_check(col_lower.size(), n, "col_lower");
  const bool up_ok = size_check(col_upper.size(), n, "col_upper");
  const bool type_ok = size_check(col_type.size(), n, "col_type");
  const bool rlo_ok = size_check(row_lower.size(), m, "row_lower");
  const bool rup_ok = size_check(row_upper.size(), m, "row_upper");

  if (!std::isfinite(objective_offset)) add("objective offset is not finite");

  if (cost_ok) {
    for (std::size_t j = 0; j < n; ++j) {
      if (!std::isfinite(col_cost[j])) add("column " + std::to_string(j) + ": cost is not finite");
    }
  }
  if (lo_ok && up_ok) {
    for (std::size_t j = 0; j < n; ++j) {
      const double lo = col_lower[j];
      const double up = col_upper[j];
      if (std::isnan(lo) || std::isnan(up)) {
        add("column " + std::to_string(j) + ": NaN bound");
        continue;
      }
      if (is_pos_inf(lo)) add("column " + std::to_string(j) + ": lower bound is +infinity");
      if (is_neg_inf(up)) add("column " + std::to_string(j) + ": upper bound is -infinity");
      if (lo > up) {
        add("column " + std::to_string(j) + ": lower bound " + std::to_string(lo) +
            " exceeds upper bound " + std::to_string(up));
      }
      if (type_ok && col_type[j] == ColType::Binary && (lo != 0.0 || up != 1.0)) {
        add("column " + std::to_string(j) + ": binary column must have bounds [0, 1]");
      }
    }
  }
  if (rlo_ok && rup_ok) {
    for (std::size_t i = 0; i < m; ++i) {
      const double lo = row_lower[i];
      const double up = row_upper[i];
      if (std::isnan(lo) || std::isnan(up)) {
        add("row " + std::to_string(i) + ": NaN bound");
        continue;
      }
      if (is_pos_inf(lo)) add("row " + std::to_string(i) + ": lower bound is +infinity");
      if (is_neg_inf(up)) add("row " + std::to_string(i) + ": upper bound is -infinity");
      if (lo > up) {
        add("row " + std::to_string(i) + ": lower bound " + std::to_string(lo) +
            " exceeds upper bound " + std::to_string(up));
      }
    }
  }

  check_names(row_names, m, "row", &problems);
  check_names(col_names, n, "column", &problems);

  const bool has_quadratic = quadratic.n_rows != 0 || quadratic.n_cols != 0 || quadratic.nnz() != 0;
  if (has_quadratic) {
    if (quadratic.n_rows != n_cols || quadratic.n_cols != n_cols) {
      add("quadratic term must be n_cols x n_cols");
    }
    for (const std::string& p : quadratic.validate()) add("quadratic: " + p);
    if (quadratic.validate().empty()) {
      for (std::size_t j = 0; j < to_size(quadratic.n_cols); ++j) {
        for (Index p = quadratic.col_start[j]; p < quadratic.col_start[j + 1]; ++p) {
          if (to_size(quadratic.row_index[to_size(p)]) < j) {
            add("quadratic: entry above the diagonal (only the lower triangle is stored)");
            return problems;
          }
        }
      }
    }
  }
  return problems;
}

}  // namespace shodhan
