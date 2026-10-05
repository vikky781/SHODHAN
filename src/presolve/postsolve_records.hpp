#pragma once

// Internal to the presolve module: one postsolve record per reduction.
//
// Postsolve replays records in reverse order. Each record maps a point of the
// model AFTER its reduction (M_{k+1}) to a point of the model BEFORE it (M_k),
// keeping the invariant d = c_k - A_k^T y for the columns alive in M_k, so the
// KKT conditions of M_k hold whenever those of M_{k+1} do. Indices are original
// row/column indices. See docs/PRESOLVE.md for the derivations.

#include <vector>

#include "shodhan/presolve.hpp"
#include "work_model.hpp"

namespace shodhan {

namespace presolve_detail {

struct PostsolveState {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> d;
  bool duals = false;
};

}  // namespace presolve_detail

class PostsolveRecord {
 public:
  virtual ~PostsolveRecord() = default;
  virtual void undo(presolve_detail::PostsolveState& state) const = 0;
};

namespace presolve_detail {

/// An empty column fixed at the bound its cost favours: d = c.
class EmptyColumnRecord final : public PostsolveRecord {
 public:
  EmptyColumnRecord(int col, double value, double cost) : col_(col), value_(value), cost_(cost) {}
  void undo(PostsolveState& s) const override;

 private:
  int col_;
  double value_;
  double cost_;
};

/// A column substituted out at `value`: d_j = c_j - A_j^T y.
class FixedColumnRecord : public PostsolveRecord {
 public:
  FixedColumnRecord(int col, double value, double cost, Entries entries, Entries qentries = {}, double qdiag = 0.0)
      : col_(col), value_(value), cost_(cost), entries_(std::move(entries)), qentries_(std::move(qentries)), qdiag_(qdiag) {}
  void undo(PostsolveState& s) const override;

 private:
  int col_;
  double value_;
  double cost_;
  Entries entries_;  // (row, coefficient) of the alive entries when it was removed
  Entries qentries_;  // (column, q_jk) of the alive quadratic neighbours when it was removed (QP only)
  double qdiag_;      // q_jj
};

/// A column fixed by the dual-fixing argument (same recovery as a fixed column).
class DualFixingRecord final : public FixedColumnRecord {
 public:
  using FixedColumnRecord::FixedColumnRecord;
};

/// A singleton row a * x_j in [rl, ru] turned into bounds on x_j.
class SingletonRowRecord final : public PostsolveRecord {
 public:
  SingletonRowRecord(int row, int col, double a, bool lb_from_row, bool ub_from_row)
      : row_(row), col_(col), a_(a), lb_from_row_(lb_from_row), ub_from_row_(ub_from_row) {}
  void undo(PostsolveState& s) const override;

 private:
  int row_;
  int col_;
  double a_;
  bool lb_from_row_;  // the row's lower-derived bound is at least as tight as the old one
  bool ub_from_row_;
};

/// A forcing row: every column in it was fixed at its activity-extreme bound.
class ForcingRowRecord final : public PostsolveRecord {
 public:
  struct Forced {
    int col;
    double value;
    double cost;
    double a;  // coefficient in the forcing row
    Entries entries;  // all alive entries of the column when it was removed (incl. this row)
  };
  ForcingRowRecord(int row, bool min_forcing, bool equality, std::vector<Forced> cols)
      : row_(row), min_forcing_(min_forcing), equality_(equality), cols_(std::move(cols)) {}
  void undo(PostsolveState& s) const override;

 private:
  int row_;
  bool min_forcing_;  // activity minimum equals the row upper bound (else maximum equals the lower)
  bool equality_;     // rl == ru: the dual may have either sign
  std::vector<Forced> cols_;
};

/// Doubleton equation a_j x_j + a_k x_k = b: x_j = beta - alpha x_k.
class DoubletonRecord final : public PostsolveRecord {
 public:
  struct Data {
    int row = 0;
    int elim = 0;  // column j, removed
    int kept = 0;  // column k, stays
    double a_elim = 0.0;
    double a_kept = 0.0;
    double alpha = 0.0;  // a_kept / a_elim
    double beta = 0.0;   // b / a_elim
    double cost_elim = 0.0;
    double cost_kept = 0.0;  // before the update
    Entries entries_elim;    // alive entries of column j (incl. the doubleton row)
    Entries entries_kept;    // alive entries of column k before modification
    bool lb_from_elim = false;  // x_k's new lower bound came from x_j's bounds
    bool ub_from_elim = false;
  };
  explicit DoubletonRecord(Data d) : d_(std::move(d)) {}
  void undo(PostsolveState& s) const override;

 private:
  Data d_;
};

/// Two identical columns j and k (same entries, same cost, same type, finite lower bounds) merged into one column
/// that stands for z = x_j + x_k. Primal recovery only: x_j = max(l_j, z - u_k), x_k = z - x_j.
class DuplicateColumnRecord final : public PostsolveRecord {
 public:
  DuplicateColumnRecord(int keep, int gone, double lo_keep, double up_keep, double lo_gone, double up_gone,
                        bool integer)
      : keep_(keep), gone_(gone), lo_keep_(lo_keep), up_keep_(up_keep), lo_gone_(lo_gone), up_gone_(up_gone),
        integer_(integer) {}
  void undo(PostsolveState& s) const override;

 private:
  int keep_;
  int gone_;
  double lo_keep_, up_keep_, lo_gone_, up_gone_;
  bool integer_;
};

}  // namespace presolve_detail

}  // namespace shodhan
