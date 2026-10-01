#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <istream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shodhan/mps.hpp"

#ifdef SHODHAN_HAVE_ZLIB
#include <zlib.h>
#endif

namespace shodhan {

namespace {

// ---------------------------------------------------------------------------
// Small text helpers
// ---------------------------------------------------------------------------

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }

std::string_view trim(std::string_view s) {
  while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
  return s;
}

std::vector<std::string_view> split_ws(std::string_view s) {
  std::vector<std::string_view> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && is_space(s[i])) ++i;
    const std::size_t start = i;
    while (i < s.size() && !is_space(s[i])) ++i;
    if (i > start) out.push_back(s.substr(start, i - start));
  }
  return out;
}

std::string to_upper(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  }
  return out;
}

bool ends_with_ci(std::string_view s, std::string_view suffix) {
  if (s.size() < suffix.size()) return false;
  return to_upper(s.substr(s.size() - suffix.size())) == to_upper(suffix);
}

// Parses an MPS number: decimal or exponent form, 'D'/'d' exponent letters,
// "Inf"/"Infinity" (signed). Overflow yields +/-infinity. Rejects NaN, hex
// floats and trailing junk. Negative zero is normalized to +0.
bool parse_number(std::string_view tok, double* out) {
  if (tok.empty()) return false;
  {
    std::string_view t = tok;
    double sign = 1.0;
    if (t.front() == '+' || t.front() == '-') {
      if (t.front() == '-') sign = -1.0;
      t.remove_prefix(1);
    }
    const std::string u = to_upper(t);
    if (u == "INF" || u == "INFINITY") {
      *out = sign * HUGE_VAL;
      return true;
    }
  }
  std::string s(tok);
  for (char& c : s) {
    if (c == 'D' || c == 'd') {
      c = 'E';
    } else if (c == 'x' || c == 'X' || c == 'n' || c == 'N' || c == 'p' || c == 'P') {
      return false;
    }
  }
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end != s.c_str() + s.size()) return false;
  if (std::isnan(v)) return false;
  *out = (v == 0.0) ? 0.0 : v;
  return true;
}

double clamp_inf(double v) {
  if (v >= kInf) return kInf;
  if (v <= -kInf) return -kInf;
  return v;
}

std::string snippet(std::string_view line) {
  std::string_view t = trim(line);
  constexpr std::size_t kMax = 100;
  if (t.size() > kMax) return std::string(t.substr(0, kMax)) + "...";
  return std::string(t);
}

// ---------------------------------------------------------------------------
// Sections and headers
// ---------------------------------------------------------------------------

enum class Section { None, Name, ObjSense, Rows, Columns, Rhs, Ranges, Bounds, End };

struct HeaderInfo {
  bool is_header = false;
  std::string keyword;  // upper-cased first token
  std::vector<std::string_view> tokens;
};

bool is_sense_word(const std::string& upper_word) {
  return upper_word == "MAX" || upper_word == "MAXIMIZE" || upper_word == "MIN" ||
         upper_word == "MINIMIZE";
}

// A line is a section header when it starts in column 1 and either has a
// single token or begins with NAME/OBJSENSE. Unindented multi-token lines are
// data (free format without indentation). In an OBJSENSE section an
// unindented MAX/MIN word is data.
HeaderInfo classify_line(std::string_view line, Section current) {
  HeaderInfo h;
  if (line.empty() || is_space(line.front())) return h;
  h.tokens = split_ws(line);
  if (h.tokens.empty()) return h;
  h.keyword = to_upper(h.tokens[0]);
  if (current == Section::ObjSense && h.tokens.size() == 1 && is_sense_word(h.keyword)) return h;
  if (h.tokens.size() == 1 || h.keyword == "NAME" || h.keyword == "OBJSENSE") {
    h.is_header = true;
  }
  return h;
}

Section section_from_keyword(const std::string& kw) {
  if (kw == "NAME") return Section::Name;
  if (kw == "OBJSENSE") return Section::ObjSense;
  if (kw == "ROWS") return Section::Rows;
  if (kw == "COLUMNS") return Section::Columns;
  if (kw == "RHS") return Section::Rhs;
  if (kw == "RANGES") return Section::Ranges;
  if (kw == "BOUNDS") return Section::Bounds;
  if (kw == "ENDATA") return Section::End;
  return Section::None;
}

// ---------------------------------------------------------------------------
// Fixed-format field extraction
// ---------------------------------------------------------------------------
// 0-based character ranges: field1 [1,3) field2 [4,12) field3 [14,22)
// field4 [24,36) field5 [39,47) field6 [49,61).

std::string_view field(std::string_view line, std::size_t begin, std::size_t end) {
  if (begin >= line.size()) return {};
  return trim(line.substr(begin, std::min(end, line.size()) - begin));
}

// MARKER lines are usually not aligned to the fixed layout; they are read by
// whitespace in both formats.
bool is_marker_line(const std::vector<std::string_view>& ws) {
  return ws.size() == 3 && ws[1] == "'MARKER'";
}

bool fixed_line_conforms(Section sec, std::string_view line, bool* has_space) {
  if (sec == Section::Columns && is_marker_line(split_ws(line))) return true;
  static constexpr std::size_t kSeparators[] = {0, 3, 12, 13, 22, 23, 36, 37, 38, 47, 48};
  for (const std::size_t p : kSeparators) {
    if (p < line.size() && line[p] != ' ') return false;
  }
  auto spaced = [](std::string_view s) { return s.find(' ') != std::string_view::npos; };
  auto numeric = [](std::string_view s) {
    double v = 0.0;
    return parse_number(s, &v);
  };
  if (sec == Section::Rows) {
    const std::string t = to_upper(field(line, 1, 3));
    if (t.size() != 1 || std::string("NLGE").find(t[0]) == std::string::npos) return false;
    const std::string_view name = line.size() > 4 ? trim(line.substr(4)) : std::string_view();
    if (name.empty()) return false;
    if (spaced(name)) *has_space = true;
    return true;
  }
  if (line.size() > 61 && !trim(line.substr(61)).empty()) return false;
  const std::string_view f2 = field(line, 4, 12);
  const std::string_view f3 = field(line, 14, 22);
  const std::string_view f4 = field(line, 24, 36);
  const std::string_view f5 = field(line, 39, 47);
  const std::string_view f6 = field(line, 49, 61);
  if (sec == Section::Columns) {
    if (f2.empty() || f3.empty()) return false;
    if (f3 != "'MARKER'") {
      if (!numeric(f4)) return false;
      if (f5.empty() ? !f6.empty() : !numeric(f6)) return false;
    }
  } else if (sec == Section::Rhs || sec == Section::Ranges) {
    if (f3.empty() || !numeric(f4)) return false;
    if (f5.empty() ? !f6.empty() : !numeric(f6)) return false;
  } else if (sec == Section::Bounds) {
    const std::string_view f1 = field(line, 1, 3);
    if (f1.size() != 2 || f3.empty()) return false;
    if (!f4.empty() && !numeric(f4)) return false;
    if (!f5.empty() || !f6.empty()) return false;
  } else {
    return false;
  }
  if (spaced(f2) || spaced(f3) || spaced(f5)) *has_space = true;
  return true;
}

// Fixed format is only distinguishable from free format when a name contains
// a space. Choose fixed iff every data line in ROWS..BOUNDS conforms to the
// column layout and at least one name contains a space.
bool detect_fixed(const std::vector<std::string_view>& lines) {
  Section sec = Section::None;
  bool any_space = false;
  for (const std::string_view line : lines) {
    if (trim(line).empty() || line.front() == '*') continue;
    const HeaderInfo h = classify_line(line, sec);
    if (h.is_header) {
      sec = section_from_keyword(h.keyword);
      if (sec == Section::End) break;
      continue;
    }
    if (sec == Section::Rows || sec == Section::Columns || sec == Section::Rhs ||
        sec == Section::Ranges || sec == Section::Bounds) {
      if (!fixed_line_conforms(sec, line, &any_space)) return false;
    }
  }
  return any_space;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct ParseError {
  std::string message;
};

struct StrHash {
  using is_transparent = void;
  std::size_t operator()(std::string_view s) const noexcept {
    return std::hash<std::string_view>{}(s);
  }
};
using NameMap = std::unordered_map<std::string, int, StrHash, std::equal_to<>>;

constexpr int kObjectiveRow = -1;
constexpr int kIgnoredRow = -2;

struct SetTracker {
  bool chosen = false;
  std::string name;
  bool warned = false;
};

class Parser {
 public:
  Parser(std::string_view text, std::string source, MpsFormat format)
      : source_(std::move(source)) {
    std::size_t pos = 0;
    while (pos <= text.size()) {
      std::size_t nl = text.find('\n', pos);
      if (nl == std::string_view::npos) nl = text.size();
      std::string_view line = text.substr(pos, nl - pos);
      while (!line.empty() && line.back() == '\r') line.remove_suffix(1);
      lines_.push_back(line);
      if (nl == text.size()) break;
      pos = nl + 1;
    }
    switch (format) {
      case MpsFormat::Fixed:
        fixed_ = true;
        break;
      case MpsFormat::Free:
        fixed_ = false;
        break;
      case MpsFormat::Auto:
        fixed_ = detect_fixed(lines_);
        break;
    }
  }

  MpsReadResult run() {
    MpsReadResult result;
    result.detected_format = fixed_ ? MpsFormat::Fixed : MpsFormat::Free;
    try {
      parse();
      finish();
      result.ok = true;
      result.model = std::move(model_);
    } catch (const ParseError& e) {
      result.error = e.message;
    } catch (const std::exception& e) {
      result.error = source_ + ": internal error while reading: " + e.what();
    }
    result.warnings = std::move(warnings_);
    return result;
  }

 private:
  // ---- diagnostics ----
  [[noreturn]] void fail(const std::string& message) const {
    throw ParseError{source_ + ":" + std::to_string(line_no_) + ": " + message + ": '" +
                     snippet(cur_) + "'"};
  }
  void warn(const std::string& message) {
    warnings_.push_back(source_ + ":" + std::to_string(line_no_) + ": " + message);
  }

  // ---- numbers ----
  double number(std::string_view tok) const {
    double v = 0.0;
    if (!parse_number(tok, &v)) fail("invalid number '" + std::string(tok) + "'");
    return v;
  }
  double coefficient(std::string_view tok) const {
    const double v = number(tok);
    if (!std::isfinite(v)) fail("coefficient '" + std::string(tok) + "' is not finite");
    return v;
  }

  // ---- main loop ----
  void parse() {
    for (std::size_t i = 0; i < lines_.size(); ++i) {
      line_no_ = i + 1;
      cur_ = lines_[i];
      if (trim(cur_).empty() || cur_.front() == '*') continue;
      last_line_no_ = line_no_;
      last_text_ = cur_;
      const HeaderInfo h = classify_line(cur_, section_);
      if (h.is_header) {
        handle_header(h);
        if (section_ == Section::End) return;
        continue;
      }
      handle_data();
    }
    line_no_ = last_line_no_ == 0 ? 1 : last_line_no_;
    cur_ = last_text_;
    fail("missing ENDATA (end of file reached)");
  }

  void leave_section() {
    if (section_ == Section::Columns && in_integer_block_) {
      warn("INTORG marker block is not closed by INTEND before the end of COLUMNS");
      in_integer_block_ = false;
    }
  }

  void handle_header(const HeaderInfo& h) {
    leave_section();
    const std::string& kw = h.keyword;
    if (kw == "QUADOBJ" || kw == "QMATRIX" || kw == "QSECTION" || kw == "QCMATRIX") {
      fail("QPS not yet supported: quadratic section '" + std::string(h.tokens[0]) + "'");
    }
    if (kw == "NAME") {
      section_ = Section::Name;
      std::string_view rest = trim(cur_);
      rest.remove_prefix(h.tokens[0].size());
      model_.name = std::string(trim(rest));
      return;
    }
    if (kw == "OBJSENSE") {
      section_ = Section::ObjSense;
      if (h.tokens.size() > 2) fail("OBJSENSE takes a single MAX or MIN value");
      if (h.tokens.size() == 2) set_sense(h.tokens[1]);
      return;
    }
    const Section next = section_from_keyword(kw);
    switch (next) {
      case Section::Rows:
        seen_rows_ = true;
        break;
      case Section::Columns:
      case Section::Rhs:
      case Section::Ranges:
      case Section::Bounds:
        if (!seen_rows_) fail("section " + kw + " appears before ROWS");
        break;
      case Section::End:
        break;
      default:
        if (kw == "SOS" || kw == "INDICATORS" || kw == "SOS1" || kw == "SOS2" ||
            kw == "OBJNAME") {
          fail("unsupported section '" + std::string(h.tokens[0]) + "'");
        }
        fail("unknown section '" + std::string(h.tokens[0]) + "'");
    }
    section_ = next;
  }

  void set_sense(std::string_view word) {
    const std::string u = to_upper(word);
    if (u == "MAX" || u == "MAXIMIZE") {
      model_.sense = Sense::Maximize;
    } else if (u == "MIN" || u == "MINIMIZE") {
      model_.sense = Sense::Minimize;
    } else {
      fail("unknown objective sense '" + std::string(word) + "' (expected MAX or MIN)");
    }
  }

  // ---- data lines ----
  std::vector<std::string_view> tokenize() const {
    if (!fixed_ || section_ == Section::ObjSense) return split_ws(cur_);
    if (section_ == Section::Columns) {
      std::vector<std::string_view> ws = split_ws(cur_);
      if (is_marker_line(ws)) return ws;
    }
    std::vector<std::string_view> tok;
    auto add = [&](std::string_view f) {
      if (!f.empty()) tok.push_back(f);
    };
    switch (section_) {
      case Section::Rows:
        add(field(cur_, 1, 3));
        add(cur_.size() > 4 ? trim(cur_.substr(4)) : std::string_view());
        break;
      case Section::Columns:
      case Section::Rhs:
      case Section::Ranges:
        add(field(cur_, 4, 12));
        add(field(cur_, 14, 22));
        add(field(cur_, 24, 36));
        add(field(cur_, 39, 47));
        add(field(cur_, 49, 61));
        break;
      case Section::Bounds:
        add(field(cur_, 1, 3));
        add(field(cur_, 4, 12));
        add(field(cur_, 14, 22));
        add(field(cur_, 24, 36));
        break;
      default:
        return split_ws(cur_);
    }
    return tok;
  }

  void handle_data() {
    const std::vector<std::string_view> tok = tokenize();
    switch (section_) {
      case Section::None:
        fail("data line outside of any section");
      case Section::Name:
        fail("unexpected data line after NAME");
      case Section::ObjSense:
        if (tok.size() != 1) fail("expected MAX or MIN");
        set_sense(tok[0]);
        return;
      case Section::Rows:
        handle_row(tok);
        return;
      case Section::Columns:
        handle_column(tok);
        return;
      case Section::Rhs:
        handle_rhs(tok);
        return;
      case Section::Ranges:
        handle_range(tok);
        return;
      case Section::Bounds:
        handle_bound(tok);
        return;
      case Section::End:
        return;
    }
  }

  void handle_row(const std::vector<std::string_view>& tok) {
    if (tok.size() != 2) fail("expected a row type (N/L/G/E) and a row name");
    const std::string t = to_upper(tok[0]);
    if (t.size() != 1 || std::string("NLGE").find(t[0]) == std::string::npos) {
      fail("unknown row type '" + std::string(tok[0]) + "'");
    }
    const std::string_view name = tok[1];
    if (row_map_.find(name) != row_map_.end()) fail("duplicate row name '" + std::string(name) + "'");
    if (t[0] == 'N') {
      if (!have_objective_) {
        have_objective_ = true;
        objective_name_ = std::string(name);
        row_map_.emplace(std::string(name), kObjectiveRow);
      } else {
        warn("additional N row '" + std::string(name) +
             "' ignored (the first N row is the objective)");
        row_map_.emplace(std::string(name), kIgnoredRow);
      }
      return;
    }
    row_map_.emplace(std::string(name), static_cast<int>(row_type_.size()));
    row_type_.push_back(t[0]);
    row_names_.emplace_back(name);
    row_stamp_.push_back(-1);
    rhs_.push_back(0.0);
    range_.push_back(0.0);
    has_range_.push_back(false);
  }

  int lookup_row(std::string_view name) const {
    const auto it = row_map_.find(name);
    if (it == row_map_.end()) fail("unknown row name '" + std::string(name) + "'");
    return it->second;
  }

  int lookup_col(std::string_view name) const {
    const auto it = col_map_.find(name);
    if (it == col_map_.end()) fail("unknown column name '" + std::string(name) + "'");
    return it->second;
  }

  void handle_column(const std::vector<std::string_view>& tok) {
    if (tok.size() == 3 && tok[1] == "'MARKER'") {
      if (tok[2] == "'INTORG'") {
        if (in_integer_block_) fail("INTORG inside an open integer block");
        in_integer_block_ = true;
      } else if (tok[2] == "'INTEND'") {
        if (!in_integer_block_) fail("INTEND without a matching INTORG");
        in_integer_block_ = false;
      } else {
        fail("unknown MARKER type '" + std::string(tok[2]) + "'");
      }
      return;
    }
    if (tok.size() != 3 && tok.size() != 5) {
      fail("expected 3 or 5 fields in COLUMNS, found " + std::to_string(tok.size()));
    }
    const std::string_view name = tok[0];
    if (current_col_ < 0 || col_names_[to_size(current_col_)] != name) {
      if (col_map_.find(name) != col_map_.end()) {
        fail("column '" + std::string(name) + "' appears again non-contiguously");
      }
      current_col_ = static_cast<int>(col_names_.size());
      col_map_.emplace(std::string(name), current_col_);
      col_names_.emplace_back(name);
      col_cost_.push_back(0.0);
      col_lower_.push_back(0.0);
      col_upper_.push_back(kInf);
      col_type_.push_back(in_integer_block_ ? ColType::Integer : ColType::Continuous);
    }
    for (std::size_t k = 1; k + 1 < tok.size(); k += 2) {
      const int row = lookup_row(tok[k]);
      const double v = coefficient(tok[k + 1]);
      if (row == kObjectiveRow) {
        if (obj_stamp_ == current_col_) {
          fail("duplicate entry for column '" + std::string(name) + "' in objective row '" +
               objective_name_ + "'");
        }
        obj_stamp_ = current_col_;
        col_cost_[to_size(current_col_)] = v;
      } else if (row == kIgnoredRow) {
        continue;
      } else {
        if (row_stamp_[to_size(row)] == current_col_) {
          fail("duplicate entry for column '" + std::string(name) + "' in row '" +
               std::string(tok[k]) + "'");
        }
        row_stamp_[to_size(row)] = current_col_;
        triplets_.push_back({row, current_col_, v});
      }
    }
  }

  // Returns true when the line belongs to the selected set.
  bool accept_set(SetTracker* t, std::string_view set, const char* what) {
    if (!t->chosen) {
      t->chosen = true;
      t->name = std::string(set);
      return true;
    }
    if (t->name == set) return true;
    if (!t->warned) {
      warn(std::string("entries of ") + what + " set '" + std::string(set) +
           "' ignored (only the first set '" + t->name + "' is used)");
      t->warned = true;
    }
    return false;
  }

  void handle_rhs(const std::vector<std::string_view>& tok) {
    if (tok.size() < 2) fail("expected at least one row/value pair in RHS");
    const bool has_set = tok.size() % 2 == 1;
    const std::string_view set = has_set ? tok[0] : std::string_view();
    if (!accept_set(&rhs_set_, set, "RHS")) return;
    for (std::size_t k = has_set ? 1 : 0; k + 1 < tok.size(); k += 2) {
      const int row = lookup_row(tok[k]);
      const double v = number(tok[k + 1]);
      if (row == kObjectiveRow) {
        if (!std::isfinite(v)) fail("objective offset is not finite");
        model_.objective_offset = (v == 0.0) ? 0.0 : -v;
      } else if (row != kIgnoredRow) {
        rhs_[to_size(row)] = v;
      }
    }
  }

  void handle_range(const std::vector<std::string_view>& tok) {
    if (tok.size() < 2) fail("expected at least one row/value pair in RANGES");
    const bool has_set = tok.size() % 2 == 1;
    const std::string_view set = has_set ? tok[0] : std::string_view();
    if (!accept_set(&ranges_set_, set, "RANGES")) return;
    for (std::size_t k = has_set ? 1 : 0; k + 1 < tok.size(); k += 2) {
      const int row = lookup_row(tok[k]);
      const double v = number(tok[k + 1]);
      if (row < 0) {
        warn("RANGES entry on N row '" + std::string(tok[k]) + "' ignored");
        continue;
      }
      range_[to_size(row)] = v;
      has_range_[to_size(row)] = true;
    }
  }

  void handle_bound(const std::vector<std::string_view>& tok) {
    if (tok.size() < 2) fail("expected a bound type and a column name");
    const std::string type = to_upper(tok[0]);
    if (type == "SC") fail("semi-continuous (SC) bounds are not supported");
    const bool with_value = type == "UP" || type == "LO" || type == "FX" || type == "LI" ||
                            type == "UI";
    const bool no_value = type == "FR" || type == "MI" || type == "PL" || type == "BV";
    if (!with_value && !no_value) fail("unknown bound type '" + std::string(tok[0]) + "'");

    std::string_view set;
    std::string_view col;
    std::string_view value;
    if (with_value) {
      if (tok.size() == 4) {
        set = tok[1];
        col = tok[2];
        value = tok[3];
      } else if (tok.size() == 3) {
        col = tok[1];
        value = tok[2];
      } else {
        fail("bound type " + type + " needs a column name and a value");
      }
    } else {
      if (tok.size() == 3 || tok.size() == 4) {
        set = tok[1];
        col = tok[2];
      } else if (tok.size() == 2) {
        col = tok[1];
      } else {
        fail("bound type " + type + " needs a column name");
      }
    }
    if (!accept_set(&bounds_set_, set, "BOUNDS")) return;

    const int j = lookup_col(col);
    const std::size_t c = to_size(j);
    double v = 0.0;
    if (with_value) v = clamp_inf(number(value));

    auto apply_upper = [&](double ub) {
      if (ub < 0.0 && col_lower_[c] == 0.0) {
        warn("UP bound " + std::string(value) + " < 0 on column '" + col_names_[c] +
             "' with lower bound 0: setting lower bound to -infinity");
        col_lower_[c] = -kInf;
      }
      col_upper_[c] = ub;
    };

    if (type == "UP") {
      apply_upper(v);
    } else if (type == "LO") {
      col_lower_[c] = v;
    } else if (type == "FX") {
      col_lower_[c] = v;
      col_upper_[c] = v;
    } else if (type == "FR") {
      col_lower_[c] = -kInf;
      col_upper_[c] = kInf;
    } else if (type == "MI") {
      col_lower_[c] = -kInf;
    } else if (type == "PL") {
      col_upper_[c] = kInf;
    } else if (type == "BV") {
      col_lower_[c] = 0.0;
      col_upper_[c] = 1.0;
      col_type_[c] = ColType::Binary;
    } else if (type == "LI") {
      col_lower_[c] = v;
      col_type_[c] = ColType::Integer;
    } else {  // UI
      apply_upper(v);
      col_type_[c] = ColType::Integer;
    }
  }

  // ---- model assembly ----
  void finish() {
    LpModel& m = model_;
    m.objective_name = have_objective_ ? objective_name_ : "OBJ";
    m.n_rows = static_cast<Index>(row_type_.size());
    m.n_cols = static_cast<Index>(col_names_.size());

    m.row_lower.assign(row_type_.size(), 0.0);
    m.row_upper.assign(row_type_.size(), 0.0);
    for (std::size_t i = 0; i < row_type_.size(); ++i) {
      const double rhs = clamp_inf(rhs_[i]);
      const double r = std::fabs(range_[i]);
      const bool ranged = has_range_[i] && !is_inf(rhs);
      double lo = 0.0;
      double up = 0.0;
      switch (row_type_[i]) {
        case 'L':
          up = rhs;
          lo = ranged ? rhs - r : -kInf;
          break;
        case 'G':
          lo = rhs;
          up = ranged ? rhs + r : kInf;
          break;
        default:  // 'E'
          lo = rhs;
          up = rhs;
          if (ranged) {
            if (range_[i] > 0.0) {
              up = rhs + r;
            } else if (range_[i] < 0.0) {
              lo = rhs - r;
            }
          }
          break;
      }
      m.row_lower[i] = clamp_inf(lo);
      m.row_upper[i] = clamp_inf(up);
    }

    for (std::size_t j = 0; j < col_names_.size(); ++j) {
      if (col_type_[j] == ColType::Binary && (col_lower_[j] != 0.0 || col_upper_[j] != 1.0)) {
        col_type_[j] = ColType::Integer;
      }
      if (col_lower_[j] > col_upper_[j]) {
        warnings_.push_back(source_ + ": column '" + col_names_[j] +
                            "' has lower bound greater than upper bound (infeasible bounds)");
      }
    }

    std::string error;
    if (!SparseMatrix::from_triplets(m.n_rows, m.n_cols, std::move(triplets_), &m.A, &error)) {
      throw ParseError{source_ + ": internal error assembling the matrix: " + error};
    }
    m.col_cost = std::move(col_cost_);
    m.col_lower = std::move(col_lower_);
    m.col_upper = std::move(col_upper_);
    m.col_type = std::move(col_type_);
    m.row_names = std::move(row_names_);
    m.col_names = std::move(col_names_);
  }

  // ---- state ----
  std::string source_;
  std::vector<std::string_view> lines_;
  bool fixed_ = false;

  std::size_t line_no_ = 0;
  std::string_view cur_;
  std::size_t last_line_no_ = 0;
  std::string_view last_text_;

  Section section_ = Section::None;
  bool seen_rows_ = false;
  bool have_objective_ = false;
  std::string objective_name_;
  bool in_integer_block_ = false;

  NameMap row_map_;
  NameMap col_map_;
  std::vector<char> row_type_;
  std::vector<std::string> row_names_;
  std::vector<std::string> col_names_;
  std::vector<double> col_cost_;
  std::vector<double> col_lower_;
  std::vector<double> col_upper_;
  std::vector<ColType> col_type_;
  std::vector<Triplet> triplets_;
  int current_col_ = -1;
  std::vector<int> row_stamp_;
  int obj_stamp_ = -1;

  std::vector<double> rhs_;
  std::vector<double> range_;
  std::vector<bool> has_range_;
  SetTracker rhs_set_;
  SetTracker ranges_set_;
  SetTracker bounds_set_;

  LpModel model_;
  std::vector<std::string> warnings_;
};

bool load_file(const std::string& path, std::string* text, std::string* error) {
  if (ends_with_ci(path, ".gz")) {
#ifdef SHODHAN_HAVE_ZLIB
    gzFile f = gzopen(path.c_str(), "rb");
    if (f == nullptr) {
      *error = "cannot open file";
      return false;
    }
    char buf[1 << 16];
    for (;;) {
      const int n = gzread(f, buf, static_cast<unsigned>(sizeof(buf)));
      if (n < 0) {
        int code = 0;
        const char* msg = gzerror(f, &code);
        *error = std::string("gzip read error: ") + (msg != nullptr ? msg : "unknown");
        gzclose(f);
        return false;
      }
      if (n == 0) break;
      text->append(buf, static_cast<std::size_t>(n));
    }
    gzclose(f);
    return true;
#else
    *error = "gzip input requires building with SHODHAN_ENABLE_ZLIB=ON";
    return false;
#endif
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open file";
    return false;
  }
  text->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  if (in.bad()) {
    *error = "read error";
    return false;
  }
  return true;
}

}  // namespace

MpsReadResult read_mps_string(std::string_view text, const std::string& source_name,
                              const MpsReadOptions& options) {
  try {
    Parser parser(text, source_name, options.format);
    return parser.run();
  } catch (const std::exception& e) {
    MpsReadResult r;
    r.error = source_name + ": internal error while reading: " + e.what();
    return r;
  }
}

MpsReadResult read_mps_stream(std::istream& in, const std::string& source_name,
                              const MpsReadOptions& options) {
  std::string text;
  try {
    text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  } catch (const std::exception& e) {
    MpsReadResult r;
    r.error = source_name + ": read error: " + e.what();
    return r;
  }
  return read_mps_string(text, source_name, options);
}

MpsReadResult read_mps_file(const std::string& path, const MpsReadOptions& options) {
  std::string text;
  std::string error;
  try {
    if (!load_file(path, &text, &error)) {
      MpsReadResult r;
      r.error = path + ": " + error;
      return r;
    }
  } catch (const std::exception& e) {
    MpsReadResult r;
    r.error = path + ": read error: " + e.what();
    return r;
  }
  return read_mps_string(text, path, options);
}

}  // namespace shodhan
