#include "info.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ostream>
#include <string>

#include "shodhan/model_stats.hpp"
#include "shodhan/quadratic.hpp"
#include "shodhan/scaling.hpp"

namespace shodhan::cli {

namespace {

std::string sci(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}

std::string range_text(const AbsRange& r) {
  if (!r.valid) return "n/a";
  return "min " + sci(r.min) + ", max " + sci(r.max);
}

}  // namespace

void print_model_summary(const LpModel& model, const std::vector<std::string>& warnings,
                         std::ostream& out) {
  const ModelStats s = compute_stats(model);
  out << "Name:           " << (model.name.empty() ? "(none)" : model.name) << "\n";
  out << "Sense:          " << (model.sense == Sense::Maximize ? "maximize" : "minimize") << "\n";
  out << "Rows:           " << s.rows << "\n";
  out << "Columns:        " << s.cols << "\n";
  out << "Nonzeros:       " << s.nnz << "\n";
  out << "Density:        " << sci(s.density * 100.0) << " %\n";
  out << "Column types:   " << s.continuous_cols << " continuous, " << s.integer_cols
      << " integer, " << s.binary_cols << " binary\n";
  out << "Row types:      " << s.rows_le << " <=, " << s.rows_ge << " >=, " << s.rows_eq << " =, "
      << s.rows_ranged << " ranged, " << s.rows_free << " free\n";
  out << "Column bounds:  " << s.cols_fixed << " fixed, " << s.cols_free << " free, "
      << s.cols_boxed << " boxed, " << s.cols_one_sided << " one-sided\n";
  out << "Coefficients:   " << range_text(s.coefficient);
  if (s.coefficient.valid) out << ", ratio " << sci(s.coefficient_ratio);
  out << "\n";
  {
    // One-line scaling-quality indicator: the coefficient ratio before and after scaling.
    ScalingReport rep;
    compute_scaling(model, {}, &rep);
    out << "Scaling:        ";
    if (rep.before.valid) {
      out << "coefficient ratio " << sci(rep.before.ratio) << " -> " << sci(rep.after.ratio)
          << " after scaling\n";
    } else {
      out << "n/a (no nonzeros)\n";
    }
  }
  out << "Costs:          " << range_text(s.cost) << "\n";
  out << "Bounds:         " << range_text(s.bound) << " (finite, nonzero)\n";
  if (model.objective_offset != 0.0) {
    out << "Objective offset: " << sci(model.objective_offset) << "\n";
  }
  if (has_quadratic(model)) {
    Index diag = 0;
    double qmax = 0.0;
    for (Index j = 0; j < model.n_cols; ++j) {
      for (Index p = model.quadratic.col_start[to_size(j)]; p < model.quadratic.col_start[to_size(j) + 1]; ++p) {
        if (model.quadratic.row_index[to_size(p)] == j) ++diag;
        qmax = std::max(qmax, std::fabs(model.quadratic.value[to_size(p)]));
      }
    }
    out << "Quadratic term: " << model.quadratic.nnz() << " nonzeros in the lower triangle of Q (" << diag << " diagonal, "
        << model.quadratic.nnz() - static_cast<std::size_t>(diag) << " off-diagonal), largest |q| " << sci(qmax) << "\n";
    const ConvexityReport cr = check_convexity(model);
    out << "Convexity:      " << (!cr.decided ? "not decided (" + cr.note + ")" : cr.convex ? "positive semidefinite (rank " + std::to_string(cr.rank) + ")" : "NOT convex: " + cr.note) << "\n";
  }
  if (!warnings.empty()) {
    out << "Warnings (" << warnings.size() << "):\n";
    for (const std::string& w : warnings) out << "  " << w << "\n";
  }
}

}  // namespace shodhan::cli
