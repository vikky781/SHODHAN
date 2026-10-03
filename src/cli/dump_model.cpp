#include "dump_model.hpp"

#include <charconv>
#include <ostream>
#include <string>

namespace shodhan::cli {

namespace {

std::string num(double v) {
  if (is_pos_inf(v)) return "inf";
  if (is_neg_inf(v)) return "-inf";
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, res.ptr);
}

std::string row_name(const LpModel& m, Index i) { return m.row_names.empty() ? "R" + std::to_string(i + 1) : m.row_names[to_size(i)]; }
std::string col_name(const LpModel& m, Index j) { return m.col_names.empty() ? "C" + std::to_string(j + 1) : m.col_names[to_size(j)]; }

}  // namespace

void print_model_dump(const LpModel& m, std::ostream& out) {
  out << "name\t" << m.name << "\n";
  out << "sense\t" << (m.sense == Sense::Maximize ? "max" : "min") << "\n";
  out << "offset\t" << num(m.objective_offset) << "\n";
  out << "rows\t" << m.n_rows << "\n";
  out << "cols\t" << m.n_cols << "\n";
  for (Index i = 0; i < m.n_rows; ++i) {
    out << "row\t" << row_name(m, i) << "\t" << num(m.row_lower[to_size(i)]) << "\t" << num(m.row_upper[to_size(i)]) << "\n";
  }
  for (Index j = 0; j < m.n_cols; ++j) {
    out << "col\t" << col_name(m, j) << "\t" << num(m.col_cost[to_size(j)]) << "\t" << num(m.col_lower[to_size(j)]) << "\t"
        << num(m.col_upper[to_size(j)]) << "\t" << (m.col_type[to_size(j)] == ColType::Continuous ? "C" : "I") << "\n";
  }
  for (Index j = 0; j < m.n_cols; ++j) {
    for (Index t = m.A.col_start[to_size(j)]; t < m.A.col_start[to_size(j) + 1]; ++t) {
      out << "entry\t" << row_name(m, m.A.row_index[to_size(t)]) << "\t" << col_name(m, j) << "\t" << num(m.A.value[to_size(t)]) << "\n";
    }
  }
}

}  // namespace shodhan::cli
