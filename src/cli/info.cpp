#include "info.hpp"

#include <cstdio>
#include <ostream>
#include <string>

#include "shodhan/model_stats.hpp"

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
  out << "Costs:          " << range_text(s.cost) << "\n";
  out << "Bounds:         " << range_text(s.bound) << " (finite, nonzero)\n";
  if (model.objective_offset != 0.0) {
    out << "Objective offset: " << sci(model.objective_offset) << "\n";
  }
  if (!warnings.empty()) {
    out << "Warnings (" << warnings.size() << "):\n";
    for (const std::string& w : warnings) out << "  " << w << "\n";
  }
}

}  // namespace shodhan::cli
