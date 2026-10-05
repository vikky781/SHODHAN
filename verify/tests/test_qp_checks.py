"""Hand-made QP certificates: the exact objective with Q, the rigorous dual bound with the first-order underestimate,
the exact PSD test and its cap, and the verdict when convexity is not verified."""

import unittest

from tests.helpers import make_cert, verify

# min 1/2 (x1^2 + x2^2) - x1 - 2 x2 + 3  s.t.  x1 + x2 <= 1, x >= 0.   Optimum x = (0, 1), y = -1, objective 3/2.
# Dual bound at x~ = (0, 1), y = -1: rows: y*ru = -1; d = c + Qx~ - A^T y = (-1 + 0 + 1, -2 + 1 + 1) = (0, 0);
# constant -1/2 x~^T Q x~ = -1/2; offset 3. LB = 3 - 1/2 - 1 = 3/2 = the primal objective.
QP = """NAME QP
ROWS
 N OBJ
 L R1
COLUMNS
 X1 OBJ -1 R1 1
 X2 OBJ -2 R1 1
RHS
 RHS OBJ -3 R1 1
QUADOBJ
 X1 X1 1
 X2 X2 1
ENDATA
"""
PROB = dict(rows=1, cols=2, nnz=2)


def cert(x, y, claimed=1.5, text=QP, **kw):
    c = make_cert(text, "optimal", {"x": x, "y": y, "claimed_objective": claimed}, **PROB)
    c["problem"]["quadratic"] = True
    c["problem"]["q_nnz"] = kw.get("q_nnz", 2)
    return c


class QpCheckTests(unittest.TestCase):
    def test_the_known_optimum_passes_rigorously(self):
        code, rep = verify(QP, cert({"X2": 1}, {"R1": -1}))
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_OPTIMAL", True))
        self.assertEqual(rep.data["primal_objective"]["exact"], "3/2")
        self.assertEqual(rep.data["dual_bound"]["exact"], "3/2")
        self.assertEqual(rep.data["convexity"], "psd")

    def test_float_mode_cannot_prove_convexity_so_the_verdict_is_tolerance_level(self):
        code, rep = verify(QP, cert({"X2": 1}, {"R1": -1}), mode="float")
        self.assertEqual(code, 0)
        self.assertEqual(rep.detail, "PASS_OPTIMAL_TOL")
        self.assertFalse(rep.rigorous)
        self.assertEqual(rep.data["convexity"], "not_verified")

    def test_the_solvers_convexity_record_is_reported_but_the_exact_test_decides(self):
        c = cert({"X2": 1}, {"R1": -1})
        c["convexity"] = "tolerance"
        code, rep = verify(QP, c)
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_OPTIMAL", True))
        self.assertEqual(rep.data["solver_convexity"], "tolerance")
        self.assertEqual(rep.data["convexity"], "psd")

    def test_the_exact_psd_test_is_capped_and_says_so(self):
        code, rep = verify(QP, cert({"X2": 1}, {"R1": -1}), psd_cap=1)
        self.assertEqual(code, 0)
        self.assertEqual(rep.detail, "PASS_OPTIMAL_TOL")
        self.assertFalse(rep.rigorous)
        self.assertEqual(rep.data["convexity"], "not_verified")
        self.assertTrue(any("NOT rigorous" in line for line in rep.lines))

    def test_wrong_multiplier_or_point_is_rejected(self):
        self.assertEqual(verify(QP, cert({"X2": 1}, {"R1": -0.5}))[0], 1)
        self.assertEqual(verify(QP, cert({"X1": 0.5, "X2": 0.5}, {"R1": -1}, claimed=1.75))[0], 1)

    def test_the_claimed_objective_includes_the_quadratic_term(self):
        # Without the quadratic term the objective would be 3 - 2 = 1: a claim of 1 must fail.
        self.assertEqual(verify(QP, cert({"X2": 1}, {"R1": -1}, claimed=1.0))[0], 1)

    def test_an_indefinite_q_fails(self):
        text = QP.replace(" X2 X2 1\n", " X2 X1 2\n X2 X2 1\n")  # [[1,2],[2,1]]: eigenvalues 3 and -1
        c = cert({"X2": 1}, {"R1": -1}, text=text, q_nnz=3)
        code, rep = verify(text, c)
        self.assertEqual(code, 1)
        self.assertEqual(rep.data["convexity"], "not_psd")
        self.assertEqual(rep.detail, "FAIL")

    def test_a_singular_psd_q_with_a_zero_row_passes(self):
        # min 1/2 x1^2 - x1 with x2 absent from Q: Q = diag(1, 0) is PSD; x = (1, 0).
        text = """NAME S
ROWS
 N OBJ
 L R1
COLUMNS
 X1 OBJ -1 R1 1
 X2 OBJ 1 R1 1
RHS
 RHS R1 5
QUADOBJ
 X1 X1 1
ENDATA
"""
        # d = c + Qx - A^T y = (-1 + 1 - 0, 1 + 0 - 0) = (0, 1) at x = (1, 0), y = 0: x2 is at its lower bound 0 (d >= 0).
        c = make_cert(text, "optimal", {"x": {"X1": 1}, "y": {}, "claimed_objective": -0.5}, rows=1, cols=2, nnz=2)
        c["problem"]["quadratic"] = True
        c["problem"]["q_nnz"] = 1
        code, rep = verify(text, c)
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_OPTIMAL", True))
        self.assertEqual(rep.data["convexity"], "psd")

    def test_a_zero_pivot_with_a_nonzero_entry_in_its_row_is_not_psd(self):
        text = QP.replace(" X1 X1 1\n X2 X2 1\n", " X2 X1 1\n")  # [[0,1],[1,0]]
        c = cert({"X2": 1}, {"R1": -1}, text=text, q_nnz=1)
        code, rep = verify(text, c)
        self.assertEqual(rep.data["convexity"], "not_psd")
        self.assertEqual(code, 1)

    def test_maximization_negates_q_so_a_concave_objective_is_convex_after_the_sign_change(self):
        # max -x1^2/2 - x2^2/2 + x1 + 2 x2 - 3 (the negation of the model above); the model stores Q = -I and c = (1, 2).
        text = """NAME M
OBJSENSE
    MAX
ROWS
 N OBJ
 L R1
COLUMNS
 X1 OBJ 1 R1 1
 X2 OBJ 2 R1 1
RHS
 RHS OBJ 3 R1 1
QUADOBJ
 X1 X1 -1
 X2 X2 -1
ENDATA
"""
        # Minimization form: Q' = I, c' = (-1, -2), offset' = 3 (the RHS of the objective row is -offset, so offset = -3,
        # negated to +3); the optimum is the same point x = (0, 1), y' = -1, with model-sense objective -3/2.
        c = make_cert(text, "optimal", {"x": {"X2": 1}, "y": {"R1": -1}, "claimed_objective": -1.5}, rows=1, cols=2, nnz=2, sense="max")
        c["problem"]["quadratic"] = True
        c["problem"]["q_nnz"] = 2
        code, rep = verify(text, c)
        self.assertEqual((code, rep.detail), (0, "PASS_OPTIMAL"))
        self.assertEqual(rep.data["convexity"], "psd")

    def test_a_certificate_that_denies_the_quadratic_term_is_a_mismatch(self):
        c = cert({"X2": 1}, {"R1": -1})
        c["problem"]["quadratic"] = False
        self.assertEqual(verify(QP, c)[0], 1)


if __name__ == "__main__":
    unittest.main()
