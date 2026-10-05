#pragma once

// Structure of mixed-integer models that the cut code and the big-M tightening use (docs/STRUCTURE.md).
//
// Three kinds of rows are recognised on a model (in practice the presolved MIP model):
//   * variable-upper-bound rows (VUB):  x <= u * y  or  x <= u * (1 - y)  with y binary and x >= 0 continuous or
//     integer, written as a row with exactly two columns and one finite side;
//   * flow-balance rows: equality rows whose coefficients are all +1 or -1, with both signs present and at least one
//     column that is not binary (inflow - outflow = constant, inventory balances);
//   * set rows over binary columns with all coefficients +1 and right-hand side 1: partitioning (=), packing (<=),
//     covering (>=).

#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan {

/// One VUB row: x <= u * y (complemented = false) or x <= u * (1 - y) (complemented = true).
struct VubRow {
  Index row = -1;
  Index x = -1;
  Index y = -1;
  double u = 0.0;
  bool complemented = false;
};

enum class SetRowKind { Partition, Packing, Covering };

struct SetRow {
  Index row = -1;
  SetRowKind kind = SetRowKind::Packing;
};

struct StructureInfo {
  std::vector<VubRow> vubs;
  std::vector<Index> balance_rows;
  std::vector<SetRow> set_rows;
  /// Per row of the model: 1 when it is a balance row (so that a separator can ask in O(1)).
  std::vector<char> is_balance_row;

  std::size_t n_vubs() const { return vubs.size(); }
  std::size_t n_balance() const { return balance_rows.size(); }
  std::size_t n_set(SetRowKind kind) const;
};

/// The shape of a two-column row after normalization to  ax * x + ay * y <= b  with ax > 0:
/// ay < 0 and b = 0 gives x <= u y with u = -ay / ax; ay > 0 and b = ay gives x <= u (1 - y) with u = ay / ax.
/// (The comparison with zero and with ay uses a relative 1e-12.) Returns false for any other row.
bool match_vub_shape(double ax, double ay, double b, double* u, bool* complemented);

/// Scans every row of `model` (cost of one pass over the matrix). Binary means an integer column with bounds [0, 1];
/// the x column of a VUB must have lower bound 0 and must not itself be binary (a row over two binaries is an
/// implication, which the clique table handles), and differs from y.
StructureInfo detect_structure(const LpModel& model);

}  // namespace shodhan
