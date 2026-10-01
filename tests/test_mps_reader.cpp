#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mps_samples.hpp"
#include "shodhan/mps.hpp"
#include "test_harness.hpp"

using namespace shodhan;

namespace {

MpsReadResult read(const std::string& text, MpsFormat format = MpsFormat::Auto) {
  MpsReadOptions opt;
  opt.format = format;
  return read_mps_string(text, "t.mps", opt);
}

std::vector<std::vector<double>> dense(const SparseMatrix& a) {
  std::vector<std::vector<double>> d(to_size(a.n_rows), std::vector<double>(to_size(a.n_cols), 0.0));
  for (Index j = 0; j < a.n_cols; ++j) {
    for (Index p = a.col_start[to_size(j)]; p < a.col_start[to_size(j) + 1]; ++p) {
      d[to_size(a.row_index[to_size(p)])][to_size(j)] = a.value[to_size(p)];
    }
  }
  return d;
}

bool has_warning(const MpsReadResult& r, const std::string& needle) {
  for (const std::string& w : r.warnings) {
    if (w.find(needle) != std::string::npos) return true;
  }
  return false;
}

// Reads text expected to fail and checks the error mentions every needle.
void expect_error(const std::string& text, const std::vector<std::string>& needles) {
  const MpsReadResult r = read(text);
  CHECK(!r.ok);
  for (const std::string& n : needles) {
    if (r.error.find(n) == std::string::npos) {
      std::cerr << "  expected '" << n << "' in error: " << r.error << "\n";
      CHECK_CONTAINS(r.error, n);
    }
  }
  CHECK_CONTAINS(r.error, "t.mps");
}

}  // namespace

TEST_CASE(mps_reads_basic_free_format_model) {
  const MpsReadResult r = read(samples::kBasicFree);
  REQUIRE(r.ok);
  const LpModel& m = r.model;
  CHECK(r.detected_format == MpsFormat::Free);
  CHECK_EQ(m.name, std::string("TINYFREE"));
  CHECK_EQ(m.objective_name, std::string("COST"));
  CHECK(m.sense == Sense::Minimize);
  CHECK_EQ(m.n_rows, 3);
  CHECK_EQ(m.n_cols, 3);
  CHECK((m.row_names == std::vector<std::string>{"LIM1", "LIM2", "MYEQN"}));
  CHECK((m.col_names == std::vector<std::string>{"X", "Y", "Z"}));
  CHECK((m.col_cost == std::vector<double>{1.0, 2.0, -1.0}));
  CHECK((dense(m.A) == std::vector<std::vector<double>>{{1, 1, 0}, {1, 0, 0}, {0, -1, 1}}));
  CHECK(m.row_lower[0] == -kInf && m.row_upper[0] == 4.0);
  CHECK(m.row_lower[1] == 1.0 && m.row_upper[1] == kInf);
  CHECK(m.row_lower[2] == 7.0 && m.row_upper[2] == 7.0);
  CHECK(m.col_lower[0] == 0.0 && m.col_upper[0] == 4.0);
  CHECK(m.col_lower[1] == -1.0 && m.col_upper[1] == 1.0);
  CHECK(m.col_lower[2] == 0.0 && m.col_upper[2] == kInf);
  CHECK(m.validate().empty());
  CHECK(r.warnings.empty());
}

TEST_CASE(mps_fixed_format_allows_spaces_in_names) {
  const std::string text = samples::fixed_with_spaces();
  const MpsReadResult r = read(text);  // auto-detect
  REQUIRE(r.ok);
  CHECK(r.detected_format == MpsFormat::Fixed);
  const LpModel& m = r.model;
  CHECK_EQ(m.name, std::string("FIXED SPC"));
  CHECK_EQ(m.objective_name, std::string("TOT COST"));
  CHECK((m.row_names == std::vector<std::string>{"ROW ONE", "ROW TWO"}));
  CHECK((m.col_names == std::vector<std::string>{"COL A", "COL B"}));
  CHECK((m.col_cost == std::vector<double>{1.0, -1.0}));
  CHECK((dense(m.A) == std::vector<std::vector<double>>{{2.0, 0.0}, {1.5, 1.0}}));
  CHECK(m.row_upper[0] == 8.0 && m.row_lower[1] == 1.0);
  CHECK(m.col_upper[0] == 4.0);

  const MpsReadResult forced = read(text, MpsFormat::Fixed);
  REQUIRE(forced.ok);
  CHECK(forced.model == m);

  // Whitespace tokenization cannot represent these names, so free format fails.
  CHECK(!read(text, MpsFormat::Free).ok);
}

TEST_CASE(mps_fixed_and_free_agree_when_names_have_no_spaces) {
  const std::string text = samples::fixed_without_spaces();
  const MpsReadResult a = read(text, MpsFormat::Auto);
  const MpsReadResult f = read(text, MpsFormat::Fixed);
  const MpsReadResult g = read(text, MpsFormat::Free);
  REQUIRE(a.ok);
  REQUIRE(f.ok);
  REQUIRE(g.ok);
  CHECK(a.model == f.model);
  CHECK(a.model == g.model);
  CHECK(a.detected_format == MpsFormat::Free);
  CHECK_EQ(a.model.n_rows, 2);
}

TEST_CASE(mps_free_format_without_indentation) {
  const MpsReadResult r = read(samples::kFreeNoIndent);
  REQUIRE(r.ok);
  CHECK_EQ(r.model.name, std::string("FREEFORM"));
  CHECK_EQ(r.model.n_rows, 2);
  CHECK_EQ(r.model.n_cols, 2);
  CHECK(r.model.row_upper[0] == 4.0);
  CHECK(r.model.row_lower[1] == 1.0);
  CHECK(r.model.col_upper[0] == 3.0);
}

TEST_CASE(mps_every_bound_type) {
  const MpsReadResult r = read(samples::kAllBounds);
  REQUIRE(r.ok);
  const LpModel& m = r.model;
  auto col = [&](const std::string& name) -> std::size_t {
    for (std::size_t j = 0; j < m.col_names.size(); ++j) {
      if (m.col_names[j] == name) return j;
    }
    return m.col_names.size();
  };
  auto bounds_are = [&](const std::string& name, double lo, double up) {
    const std::size_t j = col(name);
    REQUIRE(j < m.col_names.size());
    CHECK_EQ(m.col_lower[j], lo);
    CHECK_EQ(m.col_upper[j], up);
  };
  bounds_are("CUP", 0.0, 5.0);
  bounds_are("CLO", 2.0, kInf);
  bounds_are("CFX", 3.0, 3.0);
  bounds_are("CFR", -kInf, kInf);
  bounds_are("CMI", -kInf, kInf);  // MI leaves the upper bound untouched (+inf by default)
  bounds_are("CPL", 0.0, kInf);    // PL resets the upper bound set by UP
  bounds_are("CBV", 0.0, 1.0);
  bounds_are("CLI", 4.0, kInf);
  bounds_are("CUI", 0.0, 9.0);
  bounds_are("CNEG", -kInf, -2.0);
  bounds_are("CNEG2", -5.0, -2.0);
  bounds_are("CINFU", 0.0, kInf);
  bounds_are("CINFL", -kInf, kInf);

  CHECK(m.col_type[col("CBV")] == ColType::Binary);
  CHECK(m.col_type[col("CLI")] == ColType::Integer);
  CHECK(m.col_type[col("CUI")] == ColType::Integer);
  CHECK(m.col_type[col("CUP")] == ColType::Continuous);

  // Exactly one warning: the UP < 0 on CNEG (CNEG2 had a nonzero lower bound).
  CHECK_EQ(r.warnings.size(), std::size_t{1});
  CHECK(has_warning(r, "CNEG"));
  CHECK(has_warning(r, "-infinity"));
  CHECK(m.validate().empty());
}

TEST_CASE(mps_bounds_without_a_set_name) {
  const MpsReadResult r = read(samples::kNoSetNames);
  REQUIRE(r.ok);
  CHECK_EQ(r.model.col_upper[0], 4.0);
  CHECK_EQ(r.model.col_lower[1], -kInf);
  CHECK_EQ(r.model.col_upper[1], kInf);
  CHECK_EQ(r.model.row_upper[0], 5.0);
  CHECK(r.warnings.empty());
}

TEST_CASE(mps_ranges_for_each_row_type_and_sign) {
  const MpsReadResult r = read(samples::kAllRanges);
  REQUIRE(r.ok);
  const LpModel& m = r.model;
  // rows: LR GR EP EN EZ LN GN
  const double lo[] = {6, 2, 5, 3, 5, 6, 2};
  const double up[] = {10, 5, 7, 5, 5, 10, 5};
  REQUIRE(m.n_rows == 7);
  for (std::size_t i = 0; i < 7; ++i) {
    CHECK_EQ(m.row_lower[i], lo[i]);
    CHECK_EQ(m.row_upper[i], up[i]);
  }
}

TEST_CASE(mps_integer_marker_blocks) {
  const MpsReadResult r = read(samples::kMarkers);
  REQUIRE(r.ok);
  const LpModel& m = r.model;
  REQUIRE(m.n_cols == 4);
  CHECK((m.col_names == std::vector<std::string>{"XC", "XI", "XJ", "XD"}));
  CHECK(m.col_type[0] == ColType::Continuous);
  CHECK(m.col_type[1] == ColType::Integer);
  CHECK(m.col_type[2] == ColType::Integer);
  CHECK(m.col_type[3] == ColType::Continuous);
  // Integer columns without bound entries are [0, +inf), not binary.
  CHECK_EQ(m.col_lower[1], 0.0);
  CHECK_EQ(m.col_upper[1], kInf);
  CHECK_EQ(m.col_lower[2], 0.0);
  CHECK_EQ(m.col_upper[2], kInf);
  CHECK(r.warnings.empty());
}

TEST_CASE(mps_rhs_on_objective_row_sets_negated_offset) {
  const MpsReadResult r = read(samples::kRhsOnObjective);
  REQUIRE(r.ok);
  CHECK_EQ(r.model.objective_offset, 5.5);
  CHECK_EQ(r.model.row_upper[0], 3.0);
  CHECK_EQ(r.model.n_rows, 1);

  const MpsReadResult positive = read(
      "NAME t\nROWS\n N COST\nCOLUMNS\n X COST 1\nRHS\n RHS COST 4\nENDATA\n");
  REQUIRE(positive.ok);
  CHECK_EQ(positive.model.objective_offset, -4.0);
}

TEST_CASE(mps_objsense_both_styles) {
  const MpsReadResult section = read(samples::kObjsenseSection);
  REQUIRE(section.ok);
  CHECK(section.model.sense == Sense::Maximize);

  const MpsReadResult inline_style = read(samples::kObjsenseInline);
  REQUIRE(inline_style.ok);
  CHECK(inline_style.model.sense == Sense::Maximize);

  const std::string min_section =
      "NAME t\nOBJSENSE\n    MIN\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n";
  const MpsReadResult mn = read(min_section);
  REQUIRE(mn.ok);
  CHECK(mn.model.sense == Sense::Minimize);

  const std::string max_inline = "NAME t\nOBJSENSE MAX\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n";
  const MpsReadResult mx = read(max_inline);
  REQUIRE(mx.ok);
  CHECK(mx.model.sense == Sense::Maximize);

  // Default is minimize.
  const MpsReadResult def = read(samples::kBasicFree);
  REQUIRE(def.ok);
  CHECK(def.model.sense == Sense::Minimize);
}

TEST_CASE(mps_multiple_n_rows_keep_first_and_warn) {
  const MpsReadResult r = read(samples::kMultiN);
  REQUIRE(r.ok);
  const LpModel& m = r.model;
  CHECK_EQ(m.objective_name, std::string("COST"));
  CHECK_EQ(m.n_rows, 1);
  CHECK_EQ(m.row_names[0], std::string("R1"));
  CHECK((m.col_cost == std::vector<double>{1.0, 0.0}));
  CHECK_EQ(m.row_upper[0], 4.0);
  CHECK_EQ(m.A.nnz(), std::size_t{1});
  CHECK_EQ(r.warnings.size(), std::size_t{1});
  CHECK(has_warning(r, "OTHER"));
  CHECK(has_warning(r, "t.mps:"));
}

TEST_CASE(mps_accepts_d_exponents_and_negative_zero) {
  const MpsReadResult r = read(samples::kDExponents);
  REQUIRE(r.ok);
  const LpModel& m = r.model;
  CHECK_EQ(m.col_cost[0], 100.0);
  CHECK_EQ(m.col_cost[1], 0.15);
  CHECK_EQ(m.row_upper[0], 10.0);
  CHECK_EQ(m.col_upper[0], 2.5);
  CHECK_EQ(m.col_lower[1], 0.0);
  CHECK(!std::signbit(m.col_lower[1]));  // -0 is normalized to +0
  CHECK_EQ(m.A.nnz(), std::size_t{2});   // the explicit "-0" coefficient is kept
  CHECK_EQ(m.A.value[0], 0.0);
  CHECK(!std::signbit(m.A.value[0]));
}

TEST_CASE(mps_skips_comments_and_blank_lines) {
  const MpsReadResult r = read(samples::kCommentsAndBlanks);
  REQUIRE(r.ok);
  CHECK_EQ(r.model.name, std::string("COMMENTS"));
  CHECK_EQ(r.model.n_rows, 1);
  CHECK_EQ(r.model.n_cols, 1);
  CHECK_EQ(r.model.row_upper[0], 2.0);
}

TEST_CASE(mps_infinite_rhs_values_make_free_rows) {
  const MpsReadResult r = read(samples::kInfiniteRhs);
  REQUIRE(r.ok);
  CHECK_EQ(r.model.row_lower[0], -kInf);
  CHECK_EQ(r.model.row_upper[0], kInf);
  CHECK_EQ(r.model.row_lower[1], -kInf);
  CHECK_EQ(r.model.row_upper[1], kInf);
  CHECK_EQ(r.model.row_upper[2], 5.0);
}

TEST_CASE(mps_column_only_in_ignored_row_is_kept_empty) {
  const MpsReadResult r = read(samples::kEmptyColumnAndNoRows);
  REQUIRE(r.ok);
  CHECK_EQ(r.model.n_rows, 0);
  CHECK_EQ(r.model.n_cols, 2);
  CHECK((r.model.col_cost == std::vector<double>{0.0, 2.0}));
  CHECK(r.model.validate().empty());
}

TEST_CASE(mps_error_unknown_section) {
  expect_error("NAME t\nFOOBAR\nROWS\n N OBJ\nENDATA\n", {"t.mps:2:", "unknown section", "FOOBAR"});
}

TEST_CASE(mps_error_duplicate_column_entry_names_the_line) {
  const std::string text =
      "NAME t\n"       // 1
      "ROWS\n"         // 2
      " N  OBJ\n"      // 3
      " L  R1\n"       // 4
      "COLUMNS\n"      // 5
      "    X  R1  1\n" // 6
      "    X  R1  2\n" // 7
      "ENDATA\n";
  expect_error(text, {"t.mps:7:", "duplicate entry", "R1", "X  R1  2"});
}

TEST_CASE(mps_error_duplicate_objective_entry) {
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\n X OBJ 2\nENDATA\n",
               {"t.mps:6:", "duplicate entry", "OBJ"});
}

TEST_CASE(mps_error_unknown_row_name) {
  expect_error("NAME t\nROWS\n N OBJ\n L R1\nCOLUMNS\n X NOPE 1\nENDATA\n",
               {"t.mps:6:", "unknown row name", "NOPE"});
  expect_error("NAME t\nROWS\n N OBJ\n L R1\nCOLUMNS\n X R1 1\nRHS\n RHS NOPE 1\nENDATA\n",
               {"t.mps:8:", "unknown row name", "NOPE"});
}

TEST_CASE(mps_error_semi_continuous_bound) {
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nBOUNDS\n SC BND X 5\nENDATA\n",
               {"t.mps:7:", "semi-continuous", "not supported"});
}

TEST_CASE(mps_error_missing_endata) {
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\n", {"t.mps:5:", "missing ENDATA"});
  expect_error("", {"missing ENDATA"});
}

TEST_CASE(mps_error_qps_sections_are_rejected_clearly) {
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nQUADOBJ\n X X 2\nENDATA\n",
               {"t.mps:6:", "QPS not yet supported", "QUADOBJ"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nQMATRIX\nENDATA\n",
               {"QPS not yet supported"});
}

TEST_CASE(mps_error_other_malformed_input) {
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ abc\nENDATA\n",
               {"t.mps:5:", "invalid number", "abc"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ nan\nENDATA\n", {"invalid number"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 0x10\nENDATA\n", {"invalid number"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1 R1\nENDATA\n", {"3 or 5 fields"});
  expect_error("NAME t\nROWS\n Q OBJ\nENDATA\n", {"t.mps:3:", "unknown row type"});
  expect_error("NAME t\nROWS\n N OBJ\n L R1\n L R1\nENDATA\n",
               {"t.mps:5:", "duplicate row name", "R1"});
  expect_error("NAME t\nCOLUMNS\n X OBJ 1\nENDATA\n", {"t.mps:2:", "before ROWS"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nBOUNDS\n UP BND NOPE 1\nENDATA\n",
               {"t.mps:7:", "unknown column name", "NOPE"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nBOUNDS\n ZZ BND X 1\nENDATA\n",
               {"unknown bound type"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nBOUNDS\n UP X\nENDATA\n",
               {"needs a column name and a value"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\n Y OBJ 1\n X OBJ 2\nENDATA\n",
               {"t.mps:7:", "non-contiguously"});
  expect_error("NAME t\nOBJSENSE\n    SIDEWAYS\nROWS\nENDATA\n", {"t.mps:3:", "objective sense"});
  expect_error("  X OBJ 1\nENDATA\n", {"t.mps:1:", "outside of any section"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n    MARKER   'MARKER'   'INTEND'\nENDATA\n",
               {"INTEND without"});
  expect_error(
      "NAME t\nROWS\n N OBJ\nCOLUMNS\n    MARKER   'MARKER'   'INTORG'\n"
      "    MARKER   'MARKER'   'INTORG'\nENDATA\n",
      {"INTORG inside"});
  expect_error("NAME t\nROWS\n N OBJ\nCOLUMNS\n    MARKER   'MARKER'   'BOGUS'\nENDATA\n",
               {"unknown MARKER type"});
  expect_error("NAME t\nOBJSENSE MAX MIN\nROWS\nENDATA\n", {"single MAX or MIN"});
  expect_error("NAME t\nSOS\nENDATA\n", {"unsupported section", "SOS"});
}

TEST_CASE(mps_unclosed_integer_block_is_a_warning) {
  const MpsReadResult r = read(
      "NAME t\nROWS\n N OBJ\nCOLUMNS\n    MARKER   'MARKER'   'INTORG'\n X OBJ 1\nENDATA\n");
  REQUIRE(r.ok);
  CHECK(r.model.col_type[0] == ColType::Integer);
  CHECK(has_warning(r, "not closed"));
}

TEST_CASE(mps_infeasible_bounds_warn_but_read) {
  const MpsReadResult r = read(
      "NAME t\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nBOUNDS\n LO BND X 5\n UP BND X 2\nENDATA\n");
  REQUIRE(r.ok);
  CHECK(has_warning(r, "lower bound greater than upper bound"));
}

TEST_CASE(mps_reading_a_missing_file_reports_an_error) {
  const MpsReadResult r = read_mps_file("definitely_not_here/nothing.mps");
  CHECK(!r.ok);
  CHECK_CONTAINS(r.error, "nothing.mps");
  CHECK_CONTAINS(r.error, "cannot open");
}

TEST_CASE(mps_gz_without_zlib_is_rejected_clearly) {
#ifndef SHODHAN_HAVE_ZLIB
  const MpsReadResult r = read_mps_file("whatever.mps.gz");
  CHECK(!r.ok);
  CHECK_CONTAINS(r.error, "SHODHAN_ENABLE_ZLIB");
#endif
}

TEST_CASE(mps_reader_survives_truncated_and_corrupted_input) {
  // Not an assertion about the result, only that nothing crashes or throws.
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  const std::string alphabet = " \t\n'*-+.0123456789eEdDRXY NAMEROWSCOLUMNSRHSBOUNDSENDATA";
  int cases = 0;
  for (const auto& sample : samples::all_valid()) {
    const std::string& text = sample.second;
    for (std::size_t len = 0; len <= text.size(); len += 3) {
      (void)read(text.substr(0, len));
      ++cases;
    }
    for (int k = 0; k < 300; ++k) {
      std::string mutated = text;
      const int edits = 1 + static_cast<int>(next() % 4);
      for (int e = 0; e < edits; ++e) {
        const std::size_t pos = static_cast<std::size_t>(next() % mutated.size());
        mutated[pos] = alphabet[static_cast<std::size_t>(next() % alphabet.size())];
      }
      (void)read(mutated);
      (void)read(mutated, MpsFormat::Fixed);
      ++cases;
    }
  }
  CHECK(cases > 1000);
}

TEST_CASE(mps_every_embedded_model_reads_and_validates) {
  for (const auto& sample : samples::all_valid()) {
    const MpsReadResult r = read(sample.second);
    if (!r.ok) {
      CHECK_EQ(sample.first + ": " + r.error, std::string());
      continue;
    }
    CHECK_EQ(sample.first, sample.first);
    const std::vector<std::string> problems = r.model.validate();
    if (!problems.empty()) CHECK_EQ(sample.first + ": " + problems.front(), std::string());
  }
}
