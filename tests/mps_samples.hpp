#pragma once

// Small hand-written MPS models used by several test files. These are toy
// models written for these tests; no benchmark instance is reproduced here.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace samples {

// Places `s` at 0-based column `pos`, padding with spaces as needed.
inline void put(std::string& line, std::size_t pos, const std::string& s) {
  if (s.empty()) return;
  if (line.size() < pos + s.size()) line.resize(pos + s.size(), ' ');
  line.replace(pos, s.size(), s);
}

// Builds one fixed-format line: fields 1..6 at columns 2-3, 5-12, 15-22,
// 25-36 (right-aligned number), 40-47 and 50-61 (right-aligned number).
inline std::string fx(const std::string& f1, const std::string& f2, const std::string& f3,
                      const std::string& f4 = "", const std::string& f5 = "",
                      const std::string& f6 = "") {
  std::string line;
  put(line, 1, f1);
  put(line, 4, f2);
  put(line, 14, f3);
  if (!f4.empty()) put(line, 36 - f4.size(), f4);
  put(line, 39, f5);
  if (!f6.empty()) put(line, 61 - f6.size(), f6);
  return line;
}

inline std::string join(const std::vector<std::string>& lines) {
  std::string out;
  for (const std::string& l : lines) {
    out += l;
    out += '\n';
  }
  return out;
}

// Fixed-format model whose names contain spaces.
inline std::string fixed_with_spaces() {
  return join({
      "NAME          FIXED SPC",
      "ROWS",
      fx("N", "TOT COST", ""),
      fx("L", "ROW ONE", ""),
      fx("G", "ROW TWO", ""),
      "COLUMNS",
      fx("", "COL A", "TOT COST", "1.0", "ROW ONE", "2.0"),
      fx("", "COL A", "ROW TWO", "1.5"),
      fx("", "COL B", "TOT COST", "-1.0", "ROW TWO", "1.0"),
      "RHS",
      fx("", "RHS", "ROW ONE", "8.0", "ROW TWO", "1.0"),
      "BOUNDS",
      fx("UP", "BND", "COL A", "4.0"),
      "ENDATA",
  });
}

// Fixed-format layout, but no spaces inside names (so it is also valid free
// format).
inline std::string fixed_without_spaces() {
  return join({
      "NAME          FIXEDOK",
      "ROWS",
      fx("N", "TOTCOST", ""),
      fx("L", "ROW1", ""),
      fx("G", "ROW2", ""),
      "COLUMNS",
      fx("", "COLA", "TOTCOST", "1.0", "ROW1", "2.0"),
      fx("", "COLA", "ROW2", "1.5"),
      fx("", "COLB", "TOTCOST", "-1.0", "ROW2", "1.0"),
      "RHS",
      fx("", "RHS", "ROW1", "8.0", "ROW2", "1.0"),
      "BOUNDS",
      fx("UP", "BND", "COLA", "4.0"),
      "ENDATA",
  });
}

inline const char* kBasicFree = R"(NAME          TINYFREE
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    X         COST         1.0   LIM1         1.0
    X         LIM2         1.0
    Y         COST         2.0   LIM1         1.0
    Y         MYEQN       -1.0
    Z         COST        -1.0   MYEQN        1.0
RHS
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0
BOUNDS
 UP BND       X            4.0
 LO BND       Y           -1.0
 UP BND       Y            1.0
ENDATA
)";

inline const char* kAllBounds = R"(NAME          BOUNDS
ROWS
 N  OBJ
 L  R1
COLUMNS
    CUP       R1        1
    CLO       R1        1
    CFX       R1        1
    CFR       R1        1
    CMI       R1        1
    CPL       R1        1
    CBV       R1        1
    CLI       R1        1
    CUI       R1        1
    CNEG      R1        1
    CNEG2     R1        1
    CINFU     R1        1
    CINFL     R1        1
RHS
    RHS       R1        100
BOUNDS
 UP BND       CUP       5
 LO BND       CLO       2
 FX BND       CFX       3
 FR BND       CFR
 MI BND       CMI
 UP BND       CPL       7
 PL BND       CPL
 BV BND       CBV
 LI BND       CLI       4
 UI BND       CUI       9
 UP BND       CNEG      -2
 LO BND       CNEG2     -5
 UP BND       CNEG2     -2
 UP BND       CINFU     Inf
 LO BND       CINFL     -Inf
ENDATA
)";

inline const char* kAllRanges = R"(NAME          RANGES
ROWS
 N  OBJ
 L  LR
 G  GR
 E  EP
 E  EN
 E  EZ
 L  LN
 G  GN
COLUMNS
    X         LR        1
    X         GR        1
    X         EP        1
    X         EN        1
    X         EZ        1
    X         LN        1
    X         GN        1
RHS
    RHS       LR        10
    RHS       GR        2
    RHS       EP        5
    RHS       EN        5
    RHS       EZ        5
    RHS       LN        10
    RHS       GN        2
RANGES
    RNG       LR        4
    RNG       GR        3
    RNG       EP        2
    RNG       EN        -2
    RNG       EZ        0
    RNG       LN        -4
    RNG       GN        -3
ENDATA
)";

inline const char* kMarkers = R"(NAME          MARKERS
ROWS
 N  OBJ
 L  R1
COLUMNS
    XC        R1        1
    MARKER                 'MARKER'                 'INTORG'
    XI        R1        1
    XJ        R1        1
    MARKER                 'MARKER'                 'INTEND'
    XD        R1        1
RHS
    RHS       R1        9
ENDATA
)";

inline const char* kRhsOnObjective = R"(NAME          OBJRHS
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST      1
    X         R1        1
RHS
    RHS       COST      -5.5
    RHS       R1        3
ENDATA
)";

inline const char* kObjsenseSection = R"(NAME          MAXSEC
OBJSENSE
    MAX
ROWS
 N  OBJ
 L  R1
COLUMNS
    X         OBJ       3
    X         R1        1
RHS
    RHS       R1        4
ENDATA
)";

inline const char* kObjsenseInline = R"(NAME          MAXINL
OBJSENSE MAXIMIZE
ROWS
 N  OBJ
 L  R1
COLUMNS
    X         OBJ       3
    X         R1        1
RHS
    RHS       R1        4
ENDATA
)";

inline const char* kMultiN = R"(NAME          MULTIN
ROWS
 N  COST
 N  OTHER
 L  R1
COLUMNS
    X         COST      1   OTHER     5
    X         R1        1
    Y         OTHER     7
RHS
    RHS       OTHER     9   R1        4
ENDATA
)";

inline const char* kDExponents = R"(NAME          DEXP
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST      1.0D+2   R1        -0
    Y         COST      1.5d-1   R1        2.5E0
RHS
    RHS       R1        1.0D+1
BOUNDS
 UP BND       X         2.5D0
 LO BND       Y         -0
ENDATA
)";

inline const char* kCommentsAndBlanks = R"(* leading comment
NAME          COMMENTS

* another comment
ROWS
 N  OBJ

 L  R1
COLUMNS
* comment inside a section
    X         OBJ       1
    X         R1        1

RHS
    RHS       R1        2
ENDATA
)";

inline const char* kInfiniteRhs = R"(NAME          INFRHS
ROWS
 N  OBJ
 L  FREE1
 G  FREE2
 L  NORMAL
COLUMNS
    X         FREE1     1
    X         FREE2     1
    X         NORMAL    1
RHS
    RHS       FREE1     1e30
    RHS       FREE2     -1e30
    RHS       NORMAL    5
ENDATA
)";

inline const char* kNoSetNames = R"(NAME          NOSETS
ROWS
 N  OBJ
 L  R1
COLUMNS
    X         OBJ       1
    X         R1        1
    Y         R1        2
RHS
    R1        5
BOUNDS
 UP X         4
 FR Y
ENDATA
)";

inline const char* kEmptyColumnAndNoRows = R"(NAME          EMPTYCOL
ROWS
 N  OBJ
 N  IGNORED
COLUMNS
    X         IGNORED   5
    Y         OBJ       2
BOUNDS
 UP BND       Y         3
ENDATA
)";

inline const char* kFreeNoIndent = R"(NAME FREEFORM
ROWS
N OBJ
L R1
G R2
COLUMNS
x1 OBJ 1 R1 1
x1 R2 1
x2 OBJ 2 R1 1
RHS
RHS R1 4 R2 1
BOUNDS
UP BND x1 3
ENDATA
)";

/// (name, text) of every embedded model that must read cleanly.
inline const std::vector<std::pair<std::string, std::string>>& all_valid() {
  static const std::vector<std::pair<std::string, std::string>> v = {
      {"basic_free", kBasicFree},
      {"fixed_with_spaces", fixed_with_spaces()},
      {"fixed_without_spaces", fixed_without_spaces()},
      {"all_bounds", kAllBounds},
      {"all_ranges", kAllRanges},
      {"markers", kMarkers},
      {"rhs_on_objective", kRhsOnObjective},
      {"objsense_section", kObjsenseSection},
      {"objsense_inline", kObjsenseInline},
      {"multi_n", kMultiN},
      {"d_exponents", kDExponents},
      {"comments_and_blanks", kCommentsAndBlanks},
      {"infinite_rhs", kInfiniteRhs},
      {"no_set_names", kNoSetNames},
      {"empty_column_and_no_rows", kEmptyColumnAndNoRows},
      {"free_no_indent", kFreeNoIndent},
  };
  return v;
}

}  // namespace samples
