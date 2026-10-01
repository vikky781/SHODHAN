#include <charconv>
#include <cmath>
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "shodhan/mps.hpp"

namespace shodhan {

namespace {

// Fixed-format field limits (see docs/MPS_FORMAT.md).
constexpr std::size_t kFixedNameWidth = 8;
constexpr std::size_t kFixedNumberWidth = 12;

std::string fmt(double v) {
  char buf[40];
  const auto res = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, res.ptr);
}

bool has_control_whitespace(const std::string& s) {
  for (const char c : s) {
    if (c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v') return true;
  }
  return false;
}

bool has_space(const std::string& s) { return s.find(' ') != std::string::npos; }

void put(std::string& line, std::size_t pos, const std::string& s) {
  if (s.empty()) return;
  if (line.size() < pos + s.size()) line.resize(pos + s.size(), ' ');
  line.replace(pos, s.size(), s);
}

// One fixed-format data line: fields at 0-based columns 1, 4, 14, 24-36
// (right-aligned number), 39 and 49-61 (right-aligned number).
std::string fixed_line(const std::string& f1, const std::string& f2, const std::string& f3,
                       const std::string& f4 = "") {
  std::string line;
  put(line, 1, f1);
  put(line, 4, f2);
  put(line, 14, f3);
  if (!f4.empty()) put(line, 36 - f4.size(), f4);
  return line + "\n";
}

struct RangeEncoding {
  char type;  // 'G': [rhs, rhs + R]   'L': [rhs - R, rhs]
  double rhs;
  double range;
};

// Picks an (type, rhs, R) triple that the reader maps back to [lo, hi].
// hi - lo is tried first, then a few neighbouring doubles, and the "L" form
// is tried when the "G" form does not reproduce the bounds exactly. A range is
// stored as a width, so some (lo, hi) pairs have no exact encoding; the result
// then differs from the input by about one unit in the last place.
RangeEncoding encode_range(double lo, double hi) {
  const double base = hi - lo;
  constexpr int kSteps = 4;
  for (int pass = 0; pass < 2; ++pass) {
    double up = base;
    double down = base;
    for (int k = 0; k <= kSteps; ++k) {
      for (const double r : {up, down}) {
        if (!(r > 0.0)) continue;
        if (pass == 0 && lo + r == hi) return {'G', lo, r};
        if (pass == 1 && hi - r == lo) return {'L', hi, r};
      }
      up = std::nextafter(up, HUGE_VAL);
      down = std::nextafter(down, 0.0);
    }
  }
  return {'G', lo, base};
}

}  // namespace

bool write_mps(const LpModel& model, std::ostream& out, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  const std::vector<std::string> problems = model.validate();
  if (!problems.empty()) return fail("model is invalid: " + problems.front());
  if (has_control_whitespace(model.name)) return fail("model name contains a control character");

  const std::size_t m = to_size(model.n_rows);
  const std::size_t n = to_size(model.n_cols);

  std::vector<std::string> row_names = model.row_names;
  std::vector<std::string> col_names = model.col_names;
  if (row_names.empty()) {
    for (std::size_t i = 0; i < m; ++i) row_names.push_back("R" + std::to_string(i + 1));
  }
  if (col_names.empty()) {
    for (std::size_t j = 0; j < n; ++j) col_names.push_back("C" + std::to_string(j + 1));
  }
  const std::string& obj_name = model.objective_name;

  // Free format cannot hold names with spaces; fall back to the fixed layout
  // (which also restricts name and number widths).
  bool fixed = has_space(obj_name);
  for (const std::string& r : row_names) fixed = fixed || has_space(r);
  for (const std::string& c : col_names) fixed = fixed || has_space(c);

  auto check_name = [&](const std::string& name, const char* what) -> std::string {
    if (name.empty() || has_control_whitespace(name)) {
      return std::string(what) + " name '" + name + "' is empty or contains a control character";
    }
    if (!fixed) return "";
    if (name.front() == ' ' || name.back() == ' ') {
      return std::string(what) + " name '" + name + "' starts or ends with a space";
    }
    if (name.size() > kFixedNameWidth) {
      return std::string(what) + " name '" + name +
             "' is longer than 8 characters, which fixed format (needed because another name "
             "contains a space) cannot hold";
    }
    return "";
  };
  {
    std::unordered_set<std::string> seen;
    std::string msg = check_name(obj_name, "objective row");
    if (!msg.empty()) return fail(msg);
    seen.insert(obj_name);
    for (const std::string& r : row_names) {
      msg = check_name(r, "row");
      if (!msg.empty()) return fail(msg);
      if (!seen.insert(r).second) return fail("duplicate row name '" + r + "'");
    }
    std::unordered_set<std::string> cols;
    for (const std::string& c : col_names) {
      msg = check_name(c, "column");
      if (!msg.empty()) return fail(msg);
      if (!cols.insert(c).second) return fail("duplicate column name '" + c + "'");
    }
  }

  bool number_too_wide = false;
  auto num = [&](double v) {
    std::string s = fmt(v);
    if (fixed && s.size() > kFixedNumberWidth) number_too_wide = true;
    return s;
  };
  auto entry = [&](const std::string& col, const std::string& row, double v) {
    const std::string s = num(v);
    if (fixed) return fixed_line("", col, row, s);
    return "    " + col + "  " + row + "  " + s + "\n";
  };
  auto set_entry = [&](const char* set, const std::string& row, double v) {
    const std::string s = num(v);
    if (fixed) return fixed_line("", set, row, s);
    return "    " + std::string(set) + "       " + row + "  " + s + "\n";
  };

  std::ostringstream os;
  os << "NAME";
  if (!model.name.empty()) os << "          " << model.name;
  os << "\n";
  if (model.sense == Sense::Maximize) os << "OBJSENSE\n    MAX\n";

  os << "ROWS\n N  " << obj_name << "\n";
  struct RowEncoding {
    char type;
    double rhs;
    bool has_range;
    double range;
  };
  std::vector<RowEncoding> rows(m);
  for (std::size_t i = 0; i < m; ++i) {
    const double lo = model.row_lower[i];
    const double up = model.row_upper[i];
    const bool lo_inf = is_inf(lo);
    const bool up_inf = is_inf(up);
    if (lo_inf && up_inf) {
      rows[i] = {'L', kInf, false, 0.0};  // free row: "<= +inf"
    } else if (lo_inf) {
      rows[i] = {'L', up, false, 0.0};
    } else if (up_inf) {
      rows[i] = {'G', lo, false, 0.0};
    } else if (lo == up) {
      rows[i] = {'E', lo, false, 0.0};
    } else {
      const RangeEncoding e = encode_range(lo, up);
      rows[i] = {e.type, e.rhs, true, e.range};
    }
    os << ' ' << rows[i].type << "  " << row_names[i] << "\n";
  }

  const char* intorg = "    MARKER                 'MARKER'                 'INTORG'\n";
  const char* intend = "    MARKER                 'MARKER'                 'INTEND'\n";
  os << "COLUMNS\n";
  bool in_int = false;
  for (std::size_t j = 0; j < n; ++j) {
    const bool is_int = model.col_type[j] != ColType::Continuous;
    if (is_int && !in_int) {
      os << intorg;
      in_int = true;
    } else if (!is_int && in_int) {
      os << intend;
      in_int = false;
    }
    const Index begin = model.A.col_start[j];
    const Index end = model.A.col_start[j + 1];
    // A column with no entries still needs one line to exist in the file.
    if (model.col_cost[j] != 0.0 || begin == end) {
      os << entry(col_names[j], obj_name, model.col_cost[j]);
    }
    for (Index p = begin; p < end; ++p) {
      os << entry(col_names[j], row_names[to_size(model.A.row_index[to_size(p)])],
                  model.A.value[to_size(p)]);
    }
  }
  if (in_int) os << intend;

  os << "RHS\n";
  if (model.objective_offset != 0.0) {
    os << set_entry("RHS", obj_name, -model.objective_offset);
  }
  for (std::size_t i = 0; i < m; ++i) {
    if (rows[i].rhs != 0.0) os << set_entry("RHS", row_names[i], rows[i].rhs);
  }

  bool any_range = false;
  for (std::size_t i = 0; i < m; ++i) any_range = any_range || rows[i].has_range;
  if (any_range) {
    os << "RANGES\n";
    for (std::size_t i = 0; i < m; ++i) {
      if (rows[i].has_range) os << set_entry("RNG", row_names[i], rows[i].range);
    }
  }

  std::ostringstream bounds;
  auto bound = [&](const char* type, const std::string& col, const double* v) {
    const std::string value = v != nullptr ? num(*v) : std::string();
    if (fixed) {
      bounds << fixed_line(type, "BND", col, value);
    } else {
      bounds << ' ' << type << " BND       " << col;
      if (v != nullptr) bounds << "  " << value;
      bounds << "\n";
    }
  };
  for (std::size_t j = 0; j < n; ++j) {
    const double lo = model.col_lower[j];
    const double up = model.col_upper[j];
    const std::string& c = col_names[j];
    const bool lo_inf = is_inf(lo);
    const bool up_inf = is_inf(up);
    if (model.col_type[j] == ColType::Binary) {
      bound("BV", c, nullptr);
    } else if (!lo_inf && !up_inf && lo == up) {
      bound("FX", c, &lo);
    } else if (lo_inf && up_inf) {
      bound("FR", c, nullptr);
    } else if (lo_inf) {
      bound("MI", c, nullptr);
      bound("UP", c, &up);
    } else if (up_inf) {
      if (lo != 0.0) bound("LO", c, &lo);
    } else {
      if (lo != 0.0) bound("LO", c, &lo);
      bound("UP", c, &up);
    }
  }
  const std::string bounds_text = bounds.str();
  if (!bounds_text.empty()) os << "BOUNDS\n" << bounds_text;
  os << "ENDATA\n";

  if (number_too_wide) {
    return fail(
        "a number does not fit the 12-character fixed-format field (fixed format is needed "
        "because a name contains a space)");
  }
  out << os.str();
  return true;
}

}  // namespace shodhan
