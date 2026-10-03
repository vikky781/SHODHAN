import gzip
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kasauti.mps import MpsError, Unsupported, parse_mps, parse_number  # noqa: E402


def parse(text, exact=True):
    return parse_mps(text.encode("utf-8"), exact)


FREE = """NAME          TOY
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
    RHS       COST        -3.5
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0
BOUNDS
 UP BND       X            4.0
 LO BND       Y           -1.0
 UP BND       Y            1.0
ENDATA
"""


class NumberTests(unittest.TestCase):
    def test_decimal_text_is_exact(self):
        self.assertEqual(parse_number("0.1"), Fraction(1, 10))
        self.assertEqual(parse_number("1.5e3"), 1500)
        self.assertEqual(parse_number("-2.5E-1"), Fraction(-1, 4))
        self.assertEqual(parse_number("1.0D+2"), 100)
        self.assertEqual(parse_number("3d-1"), Fraction(3, 10))
        self.assertEqual(parse_number(".5"), Fraction(1, 2))
        self.assertEqual(parse_number("5."), 5)
        self.assertEqual(parse_number("-0"), 0)

    def test_infinity_and_rejections(self):
        self.assertEqual(parse_number("Inf"), float("inf"))
        self.assertEqual(parse_number("-Infinity"), float("-inf"))
        for bad in ("nan", "0x1p3", "1e", "abc", "1.2.3", "--1", ""):
            with self.assertRaises(MpsError, msg=bad):
                parse_number(bad)
        with self.assertRaises(Unsupported):
            parse_number("1e99999")

    def test_float_mode(self):
        self.assertEqual(parse_number("0.1", exact=False), 0.1)


class ParserTests(unittest.TestCase):
    def test_free_format_basics(self):
        m = parse(FREE)
        self.assertEqual((m.name, m.sense, m.n_rows, m.n_cols, m.nnz), ("TOY", "min", 3, 3, 5))
        self.assertEqual(m.row_names, ["LIM1", "LIM2", "MYEQN"])
        self.assertEqual(m.col_names, ["X", "Y", "Z"])
        self.assertEqual(m.col_cost, [1, 2, -1])
        self.assertEqual(m.offset, Fraction(7, 2))  # RHS on the objective row: offset = -value
        self.assertEqual((m.row_lo, m.row_hi), ([None, 1, 7], [4, None, 7]))
        self.assertEqual((m.col_lo, m.col_hi), ([0, -1, 0], [4, 1, None]))
        self.assertTrue(m.exact)

    def test_negative_up_bound_with_zero_lower_makes_lower_minus_infinity(self):
        m = parse(FREE.replace("UP BND       X            4.0", "UP BND       X           -4.0"))
        self.assertEqual((m.col_lo[0], m.col_hi[0]), (None, -4))
        # An explicit nonzero lower bound first: it stays.
        m2 = parse(FREE.replace(" UP BND       X            4.0", " LO BND       X           -9.0\n UP BND       X           -4.0"))
        self.assertEqual((m2.col_lo[0], m2.col_hi[0]), (-9, -4))

    def test_ranges_follow_the_table(self):
        text = """NAME R
ROWS
 N  OBJ
 L  RL
 G  RG
 E  EP
 E  EN
 E  EZ
 L  RINF
COLUMNS
    X  OBJ 1  RL 1
    X  RG 1  EP 1
    X  EN 1  EZ 1
    X  RINF 1
RHS
    RHS  RL 10  RG 20
    RHS  EP 30  EN 40
    RHS  EZ 50  RINF 1e30
RANGES
    RNG  RL 4  RG 5
    RNG  EP 6  EN -7
    RNG  EZ 0  RINF 3
ENDATA
"""
        m = parse(text)
        self.assertEqual(m.row_lo, [6, 20, 30, 33, 50, None])
        self.assertEqual(m.row_hi, [10, 25, 36, 40, 50, None])  # RINF: infinite rhs, range ignored, free row

    def test_ranges_before_rhs_and_other_sets_ignored(self):
        text = FREE.replace("RHS\n", "RANGES\n    RNG  LIM1 2\nRHS\n").replace(
            "    RHS       MYEQN        7.0", "    RHS       MYEQN        7.0\n    OTHER     LIM1       100.0")
        m = parse(text)
        self.assertEqual((m.row_lo[0], m.row_hi[0]), (2, 4))
        self.assertTrue(any("ignored" in w for w in m.warnings))

    def test_objsense_forms(self):
        sect = FREE.replace("ROWS", "OBJSENSE\n    MAX\nROWS", 1)
        self.assertEqual(parse(sect).sense, "max")
        inline = FREE.replace("ROWS", "OBJSENSE MAXIMIZE\nROWS", 1)
        self.assertEqual(parse(inline).sense, "max")
        self.assertEqual(parse(FREE.replace("ROWS", "OBJSENSE\n    MIN\nROWS", 1)).sense, "min")
        with self.assertRaises(MpsError):
            parse(FREE.replace("ROWS", "OBJSENSE\n    SIDEWAYS\nROWS", 1))

    def test_integer_markers_and_bound_types(self):
        text = """NAME M
ROWS
 N  OBJ
 L  R1
COLUMNS
    A  OBJ 1  R1 1
    MARKER                 'MARKER'                 'INTORG'
    B  OBJ 2  R1 1
    C  OBJ 3  R1 1
    MARKER                 'MARKER'                 'INTEND'
    D  OBJ 4  R1 1
    E  OBJ 5  R1 1
    F  OBJ 6  R1 1
    G  OBJ 7  R1 1
    H  OBJ 8  R1 1
RHS
    RHS  R1 100
BOUNDS
 BV BND  A
 UI BND  B 7
 FX BND  D 2.5
 FR BND  E
 MI BND  F
 PL BND  F
 LI BND  G 3
 UP BND  H 9
ENDATA
"""
        m = parse(text)
        self.assertEqual(m.col_integer, [True, True, True, False, False, False, True, False])
        self.assertEqual([(m.col_lo[j], m.col_hi[j]) for j in range(8)],
                         [(0, 1), (0, 7), (0, None), (Fraction(5, 2), Fraction(5, 2)), (None, None), (None, None), (3, None), (0, 9)])
        self.assertEqual(m.n_integer, 4)

    def test_bounds_of_1e30_are_infinite(self):
        m = parse(FREE.replace("UP BND       X            4.0", "UP BND       X            1e30"))
        self.assertIsNone(m.col_hi[0])

    def test_extra_objective_rows_are_ignored(self):
        text = FREE.replace(" N  COST", " N  COST@ N  OTHER").replace(
            "    Z         COST        -1.0   MYEQN        1.0", "    Z         COST        -1.0   MYEQN        1.0@    Z         OTHER        5.0").replace("@", chr(10))
        m = parse(text)
        self.assertEqual(m.n_rows, 3)
        self.assertEqual(m.nnz, 5)
        self.assertTrue(any("OTHER" in w for w in m.warnings))

    def test_fixed_format_with_spaces_in_names(self):
        def fx(f1="", f2="", f3="", f4="", f5="", f6=""):
            return " " + f1.ljust(2) + " " + f2.ljust(8) + "  " + f3.ljust(8) + "  " + f4.rjust(12) + "   " + f5.ljust(8) + "  " + f6.rjust(12)
        text = chr(10).join([
            "NAME          SPACES", "ROWS", " N  COST", " L  ROW ONE", " G  ROW TWO", "COLUMNS",
            fx("", "COL A", "COST", "1.", "ROW ONE", "2."),
            fx("", "COL B", "ROW TWO", "4."),
            "RHS",
            fx("", "RHS", "ROW ONE", "5."),
            "BOUNDS",
            fx("UP", "BND", "COL A", "3."),
            "ENDATA", "",
        ])
        m = parse(text)
        self.assertEqual(m.row_names, ["ROW ONE", "ROW TWO"])
        self.assertEqual(m.col_names, ["COL A", "COL B"])
        self.assertEqual((m.col_cost, m.row_hi, m.col_hi), ([1, 0], [5, None], [3, None]))
        self.assertEqual(m.col_entries, [[(0, 2)], [(1, 4)]])

    def test_a_number_that_does_not_fit_a_fixed_field_is_never_silently_truncated(self):
        from tests.mpsgen import fixed_line
        long_number = "3.14159265358979"  # 16 characters in a 12-character field
        text = chr(10).join(["NAME X", "ROWS", " N  COST", " L  ROW 1", "COLUMNS",
                             fixed_line("", "COL 1", "COST", "1.", "ROW 1", long_number), "RHS", fixed_line("", "RHS", "ROW 1", "5."),
                             "ENDATA", ""])
        with self.assertRaises(MpsError):
            parse(text)

    def test_free_format_names_are_case_sensitive_and_sections_are_not(self):
        text = FREE.replace("ROWS", "rows").replace("COLUMNS", "Columns").replace("ENDATA", "endata")
        m = parse(text)
        self.assertEqual(m.n_cols, 3)
        text2 = FREE.replace("    Y         COST", "    y         COST", 1)
        self.assertIn("y", parse(text2).col_names)

    def test_gzip_is_detected(self):
        m = parse_mps(gzip.compress(FREE.encode()), True)
        self.assertEqual(m.n_cols, 3)

    def test_float_mode_matches_exact_mode_values(self):
        e, f = parse(FREE, True), parse(FREE, False)
        self.assertEqual([float(v) for v in e.col_cost], f.col_cost)
        self.assertFalse(f.exact)

    def test_errors_are_reported_not_guessed(self):
        bad = {
            "missing ENDATA": FREE.replace("ENDATA\n", ""),
            "duplicate entry": FREE.replace("    X         LIM2         1.0", "    X         LIM1         1.0"),
            "column returns": FREE.replace("    Z         COST", "    X         COST", 1).replace("    X         COST         1.0", "    X         COST         1.0", 1),
            "unknown row": FREE.replace("MYEQN       -1.0", "NOPE        -1.0"),
            "unknown column bound": FREE.replace("UP BND       X ", "UP BND       Q "),
            "bad number": FREE.replace("LIM1         4.0", "LIM1         4.x"),
            "unknown section": FREE.replace("BOUNDS", "FROBNICATE"),
            "bad row type": FREE.replace(" E  MYEQN", " Q  MYEQN"),
        }
        for what, text in bad.items():
            with self.assertRaises(MpsError, msg=what):
                parse(text)

    def test_unsupported_features(self):
        with self.assertRaises(Unsupported):
            parse(FREE.replace(" UP BND       X            4.0", " SC BND       X            4.0"))
        with self.assertRaises(Unsupported):
            parse(FREE.replace("ENDATA", "QUADOBJ\n    X  X  2.0\nENDATA"))
        with self.assertRaises(Unsupported):
            parse(FREE.replace("ROWS", "SOS\nROWS", 1))

    def test_comments_and_blank_lines_and_crlf(self):
        text = "* a comment\r\n\r\n" + FREE.replace("\n", "\r\n").replace("ROWS", "* another\r\nROWS")
        self.assertEqual(parse(text).n_cols, 3)

    def test_optional_set_names_in_free_format(self):
        text = (FREE.replace("    RHS       COST        -3.5", "    COST        -3.5")
                .replace("    RHS       LIM1         4.0   LIM2         1.0", "    LIM1         4.0   LIM2         1.0")
                .replace("    RHS       MYEQN        7.0", "    MYEQN        7.0")
                .replace(" UP BND       X            4.0", " UP X            4.0")
                .replace(" LO BND       Y           -1.0", " LO Y           -1.0")
                .replace(" UP BND       Y            1.0", " UP Y            1.0"))
        m = parse(text)
        self.assertEqual((m.row_hi[0], m.row_lo[1], m.col_hi[0]), (4, 1, 4))


if __name__ == "__main__":
    unittest.main()
