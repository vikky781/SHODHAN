#include "presolve_report.hpp"

#include <cstdio>
#include <ostream>
#include <string>

namespace shodhan::cli {

namespace {

std::string fixed(double v, int digits) {
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%.*f", digits, v);
  return buf;
}

std::string sci(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}

}  // namespace

PresolveOptions presolve_options_for(const LpModel& model, bool need_duals) {
  PresolveOptions o;
  o.need_duals = need_duals;
  for (const ColType t : model.col_type) {
    if (t != ColType::Continuous) o.is_mip = true;
  }
  return o;
}

void print_presolve_report(const PresolveResult& r, std::ostream& out) {
  const PresolveStats& s = r.stats;
  out << "Presolve\n";
  out << "  status:       " << to_string(r.status) << "\n";
  if (!r.note.empty()) out << "  reason:       " << r.note << "\n";
  if (r.status == PresolveStatus::Reduced || r.status == PresolveStatus::SolvedByPresolve) {
    out << "  rows:         " << s.rows_before << " -> " << s.rows_after << "\n";
    out << "  columns:      " << s.cols_before << " -> " << s.cols_after << "\n";
    out << "  nonzeros:     " << s.nnz_before << " -> " << s.nnz_after << "\n";
  } else {
    out << "  rows:         " << s.rows_before << " (not reduced)\n";
    out << "  columns:      " << s.cols_before << " (not reduced)\n";
    out << "  nonzeros:     " << s.nnz_before << " (not reduced)\n";
  }
  out << "  passes:       " << s.passes << "\n";
  out << "  time:         " << fixed(s.seconds * 1e3, 3) << " ms\n";
  out << "  reductions:\n";
  for (const auto& c : s.reduction_counts()) {
    char line[96];
    std::snprintf(line, sizeof(line), "    %-26s %d\n", c.first.c_str(), c.second);
    out << line;
  }
}

void print_scaling_report(const LpModel& model, std::ostream& out) {
  ScalingReport rep;
  compute_scaling(model, {}, &rep);
  out << "Scaling\n";
  if (!rep.before.valid) {
    out << "  coefficient ratio: n/a (no nonzeros)\n";
    return;
  }
  out << "  coefficients:      min " << sci(rep.before.min) << ", max " << sci(rep.before.max) << "\n";
  out << "  coefficient ratio: " << sci(rep.before.ratio) << " -> " << sci(rep.after.ratio)
      << " after scaling\n";
  out << "  geometric passes:  " << rep.geometric_passes << "\n";
}

}  // namespace shodhan::cli
