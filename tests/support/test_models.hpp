#pragma once

// Helpers for building small LpModels in tests (test-only code).

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan::testing {

/// Builds a model from triplets and per-column / per-row data. Throws on
/// inconsistent input so a broken test fixture fails loudly.
inline LpModel make_model(Index rows, Index cols, std::vector<Triplet> triplets,
                          std::vector<double> cost, std::vector<double> col_lower,
                          std::vector<double> col_upper, std::vector<double> row_lower,
                          std::vector<double> row_upper, Sense sense = Sense::Minimize,
                          double offset = 0.0) {
  LpModel m;
  m.n_rows = rows;
  m.n_cols = cols;
  m.sense = sense;
  m.objective_offset = offset;
  std::string err;
  if (!SparseMatrix::from_triplets(rows, cols, std::move(triplets), &m.A, &err)) {
    throw std::runtime_error("make_model: " + err);
  }
  m.col_cost = std::move(cost);
  m.col_lower = std::move(col_lower);
  m.col_upper = std::move(col_upper);
  m.row_lower = std::move(row_lower);
  m.row_upper = std::move(row_upper);
  m.col_type.assign(to_size(cols), ColType::Continuous);
  const auto problems = m.validate();
  if (!problems.empty()) throw std::runtime_error("make_model: " + problems.front());
  return m;
}

}  // namespace shodhan::testing
