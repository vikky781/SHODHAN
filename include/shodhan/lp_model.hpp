#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shodhan/constants.hpp"
#include "shodhan/sparse_matrix.hpp"

namespace shodhan {

enum class Sense { Minimize, Maximize };

enum class ColType : std::uint8_t { Continuous, Integer, Binary };

/// Linear (and, later, convex quadratic) model in the form
///
///   min/max  offset + c^T x   (+ 1/2 x^T Q x, reserved)
///   s.t.     row_lower <= A x <= row_upper
///            col_lower <=  x  <= col_upper
///
/// Rows are normalized to ranges: "<= b" is [-kInf, b], ">= b" is [b, kInf],
/// "= b" is [b, b]. Infinite bounds are stored as +/-kInf (see is_inf()).
struct LpModel {
  std::string name;
  /// Name of the objective row (used when writing MPS).
  std::string objective_name = "OBJ";
  Sense sense = Sense::Minimize;
  double objective_offset = 0.0;

  Index n_rows = 0;
  Index n_cols = 0;
  SparseMatrix A;  // n_rows x n_cols, CSC

  std::vector<double> col_cost;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<ColType> col_type;

  std::vector<double> row_lower;
  std::vector<double> row_upper;

  /// Either empty (no names) or exactly one name per row/column.
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;

  /// Reserved for step 8: lower triangle of Q in CSC. Left empty (0x0) until
  /// quadratic models are supported.
  SparseMatrix quadratic;

  /// Returns a description of every structural problem found (empty means
  /// valid): size mismatches, invalid matrix, NaN, lower > upper, bounds on
  /// the wrong infinity, bad binary bounds, bad or duplicate names.
  std::vector<std::string> validate() const;

  bool is_integer(Index col) const noexcept {
    return col_type[to_size(col)] != ColType::Continuous;
  }

  friend bool operator==(const LpModel&, const LpModel&) = default;
};

}  // namespace shodhan
