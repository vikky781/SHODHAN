import unittest

from tests.helpers import TINY, TINY_PROB, make_cert, sha, verify


def optimal(x, y, claimed=9, text=TINY, **prob):
    p = dict(TINY_PROB)
    p.update(prob)
    return make_cert(text, "optimal", {"x": x, "y": y, "claimed_objective": claimed}, **p)


class OptimalTests(unittest.TestCase):
    def test_correct_certificate_passes_rigorously(self):
        code, rep = verify(TINY, optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.5}))
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_OPTIMAL", True))
        self.assertEqual(rep.data["primal_objective"]["exact"], "9")
        self.assertEqual(rep.data["dual_bound"]["exact"], "9")

    def test_float_mode_agrees_but_is_not_rigorous(self):
        code, rep = verify(TINY, optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.5}), mode="float")
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_OPTIMAL", False))

    def test_the_dual_bound_needs_no_sign_condition(self):
        # Any y gives a valid lower bound; y = 0 gives LB = 0 (all d >= 0 since c >= 0): a huge gap, so FAIL.
        code, rep = verify(TINY, optimal({"X": 3, "Y": 1}, {}))
        self.assertEqual(code, 1)
        # The bound is still reported and still valid: 0 <= 9.
        self.assertEqual(rep.data["dual_bound"]["exact"], "0")

    def test_wrong_y_is_rejected(self):
        code, rep = verify(TINY, optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.4}))
        self.assertEqual(code, 1)

    def test_a_dual_point_that_points_at_an_infinite_bound_proves_nothing(self):
        # y1 < 0 on a row with no upper bound: the bound is -infinity.
        code, rep = verify(TINY, optimal({"X": 3, "Y": 1}, {"R1": -1.0, "R2": 0.5}))
        self.assertEqual(code, 1)
        self.assertIsNone(rep.data["dual_bound"])

    def test_infeasible_primal_point_is_rejected(self):
        code, rep = verify(TINY, optimal({"X": 1, "Y": 1}, {"R1": 1.5, "R2": 0.5}, claimed=5))
        self.assertEqual(code, 1)

    def test_tiny_violation_within_tolerance_passes_but_is_not_rigorous(self):
        code, rep = verify(TINY, optimal({"X": 3, "Y": 1 - 1e-9}, {"R1": 1.5, "R2": 0.5}, claimed=9))
        self.assertEqual(code, 0)
        self.assertFalse(rep.rigorous)
        code, rep = verify(TINY, optimal({"X": 3, "Y": 1 - 1e-9}, {"R1": 1.5, "R2": 0.5}, claimed=9), primal_tol=1e-12)
        self.assertEqual(code, 1)

    def test_claimed_objective_must_match_the_exact_one(self):
        code, rep = verify(TINY, optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.5}, claimed=9.5))
        self.assertEqual(code, 1)

    def test_maximization_uses_the_minimization_form_multipliers(self):
        text = TINY.replace("NAME TINY", "NAME TINY\nOBJSENSE\n    MAX").replace(" G R1", " L R1").replace(" G R2", " L R2")
        # max 2x + 3y  s.t. x + y <= 4, x + 3y <= 6: optimum x = 3, y = 1 -> 9 (and the vertex (4, 0) gives 8).
        # Minimization form: min -2x - 3y; multipliers y' = (-1.5, -0.5) (rows at their upper bounds).
        cert = optimal({"X": 3, "Y": 1}, {"R1": -1.5, "R2": -0.5}, claimed=9, text=text, sense="max")
        code, rep = verify(text, cert)
        self.assertEqual((code, rep.detail), (0, "PASS_OPTIMAL"))
        self.assertEqual(rep.data["primal_objective"]["exact"], "9")
        # The same y with the wrong sign (a max-form multiplier) is not a valid bound.
        bad = optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.5}, claimed=9, text=text, sense="max")
        self.assertEqual(verify(text, bad)[0], 1)

    def test_file_hash_and_identity_are_checked(self):
        cert = optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.5})
        cert["problem"]["file_sha256"] = "0" * 64
        self.assertEqual(verify(TINY, cert)[0], 1)
        cert = optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.5}, rows=3)
        self.assertEqual(verify(TINY, cert)[0], 1)
        cert = optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.5}, sense="max")
        self.assertEqual(verify(TINY, cert)[0], 1)
        # Editing the model after the certificate was written changes the hash.
        cert = optimal({"X": 3, "Y": 1}, {"R1": 1.5, "R2": 0.5})
        self.assertEqual(verify(TINY.replace("RHS R1 4", "RHS R1 5"), cert)[0], 1)

    def test_unknown_names_and_malformed_bodies_fail_cleanly(self):
        self.assertEqual(verify(TINY, optimal({"NOPE": 3}, {"R1": 1.5}))[0], 1)
        self.assertEqual(verify(TINY, optimal({"X": 3, "Y": 1}, {"NOPE": 1.5}))[0], 1)
        cert = optimal({"X": 3, "Y": 1}, {})
        cert["x"] = [3, 1]
        self.assertEqual(verify(TINY, cert)[0], 1)

    def test_exact_arithmetic_sees_what_doubles_hide(self):
        # max x s.t. 10 x <= 1 (as min -x): x = 0.1 as a double times 10 is 1.00000000000000005..., slightly
        # above the exact decimal bound 1. The exact violation is reported (nonzero), within tolerance.
        text = "NAME E\nROWS\n N C\n L R\nCOLUMNS\n X C -1 R 10\nRHS\n RHS R 1\nENDATA\n"
        cert = make_cert(text, "optimal", {"x": {"X": 0.1}, "y": {"R": -0.1}, "claimed_objective": -0.1}, rows=1, cols=1, nnz=1)
        code, rep = verify(text, cert)
        self.assertEqual(code, 0)
        self.assertFalse(rep.rigorous)
        self.assertGreater(float(rep.data["max_primal_violation"]["decimal"]), 0.0)
        self.assertLess(float(rep.data["max_primal_violation"]["decimal"]), 1e-15)

    def test_status_other_is_inconclusive(self):
        cert = make_cert(TINY, "other", None, **TINY_PROB)
        self.assertEqual(verify(TINY, cert)[0], 2)


class DualNoiseTests(unittest.TestCase):
    """A multiplier that is zero in theory but 1e-14 in floating point meets an infinite bound."""

    MODEL = "NAME NOISE\nROWS\n N COST\n G R1\n E R2\nCOLUMNS\n X COST 1 R1 1\n F R1 1 R2 1\nRHS\n RHS R1 1\nBOUNDS\n FR BND F\nENDATA\n"
    PROB = dict(rows=2, cols=2, nnz=3)

    def cert(self, y2):
        return make_cert(self.MODEL, "optimal", {"x": {"X": 1, "F": 0}, "y": {"R1": 1, "R2": y2}}, **self.PROB)

    def test_exact_multipliers_are_rigorous(self):
        code, rep = verify(self.MODEL, self.cert(-1))
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_OPTIMAL", True))

    def test_noise_gives_tolerance_level_verdict(self):
        code, rep = verify(self.MODEL, self.cert(-1 + 1e-14))
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_OPTIMAL_TOL", False))

    def test_noise_is_not_accepted_when_strict(self):
        self.assertEqual(verify(self.MODEL, self.cert(-1 + 1e-14), dual_zero_tol=0)[0], 1)

    def claim_cert(self, y2, rigorous):
        cert = self.cert(y2)
        cert["dual_bound"] = {"rigorous": rigorous, "available": True}
        return cert

    def test_false_rigorous_claim_fails_in_exact_mode(self):
        code, rep = verify(self.MODEL, self.claim_cert(-1 + 1e-14, True))
        self.assertEqual(code, 1)
        self.assertTrue(any("claim is false" in line for line in rep.lines))

    def test_honest_non_rigorous_claim_passes(self):
        code, rep = verify(self.MODEL, self.claim_cert(-1 + 1e-14, False))
        self.assertEqual((code, rep.detail), (0, "PASS_OPTIMAL_TOL"))

    def test_float_mode_cannot_judge_a_rigorous_claim(self):
        code, rep = verify(self.MODEL, self.claim_cert(-1 + 1e-14, True), mode="float")
        self.assertEqual(code, 0)
        self.assertTrue(any("float mode cannot judge" in line for line in rep.lines))

    def test_large_wrong_sign_is_never_dropped(self):
        self.assertEqual(verify(self.MODEL, self.cert(-0.5))[0], 1)


class InfeasibleTests(unittest.TestCase):
    INF = "NAME INF\nROWS\n N C\n L R1\n G R2\nCOLUMNS\n X C 1 R1 1\n X R2 1\n Y C 2 R1 1\n Y R2 1\nRHS\n RHS R1 1 R2 3\nENDATA\n"

    def cert(self, y, text=None):
        return make_cert(text or self.INF, "infeasible", {"farkas": {"y": y}}, rows=2, cols=2, nnz=4)

    def test_valid_certificate(self):
        code, rep = verify(self.INF, self.cert({"R1": 1, "R2": -1}))
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_INFEASIBLE", True))
        self.assertEqual(verify(self.INF, self.cert({"R1": 2, "R2": -2}))[0], 0)  # scaling does not matter

    def test_wrong_signs_or_corrupted_multipliers_fail(self):
        self.assertEqual(verify(self.INF, self.cert({"R1": -1, "R2": 1}))[0], 0)  # the negated vector is valid too
        self.assertEqual(verify(self.INF, self.cert({"R1": 1, "R2": 1}))[0], 1)
        self.assertEqual(verify(self.INF, self.cert({"R1": 1, "R2": -2}))[0], 1)
        self.assertEqual(verify(self.INF, self.cert({}))[0], 1)

    def test_a_feasible_model_has_no_certificate(self):
        feasible = self.INF.replace("R1 1 R2 3", "R1 4 R2 3")
        for y in ({"R1": 1, "R2": -1}, {"R1": 0.3, "R2": -2}, {"R1": 1, "R2": 1}):
            self.assertEqual(verify(feasible, self.cert(y, feasible))[0], 1)

    def test_noise_on_an_infinite_bound_needs_the_tolerance(self):
        # Columns X (no upper bound) and F (free) get a coefficient of about 1e-15 in A^T y, which is zero in
        # theory; W carries a sizeable coefficient. Strictly the proof fails (infinite bounds); dropping
        # coefficients below 1e-12 of the largest recovers it, and the verdict says it is not rigorous.
        text = ("NAME F\nROWS\n N C\n L R1\n G R2\n E R3\nCOLUMNS\n X C 1 R1 1\n X R2 1\n F C 0 R1 1\n F R2 1\n"
                " W C 0 R3 1\nRHS\n RHS R1 1 R2 3\nBOUNDS\n FR BND F\n UP BND W 1\nENDATA\n")
        y = {"R1": 1, "R2": -1 + 1e-15, "R3": 1}
        cert = make_cert(text, "infeasible", {"farkas": {"y": y}}, rows=3, cols=3, nnz=5)
        code, rep = verify(text, cert, farkas_zero_tol=0)
        self.assertEqual(code, 1)
        code, rep = verify(text, cert)
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_INFEASIBLE_TOL", False))

    def test_float_mode_agrees(self):
        self.assertEqual(verify(self.INF, self.cert({"R1": 1, "R2": -1}), mode="float")[0], 0)


class UnboundedTests(unittest.TestCase):
    UNB = "NAME U\nROWS\n N C\n L R1\nCOLUMNS\n X C -1 R1 1\n Y C -1 R1 -1\nRHS\n RHS R1 2\nENDATA\n"

    def cert(self, point, ray, text=None):
        return make_cert(text or self.UNB, "unbounded", {"point": point, "ray": ray}, rows=1, cols=2, nnz=2)

    def test_valid(self):
        code, rep = verify(self.UNB, self.cert({"X": 1, "Y": 1}, {"X": 1, "Y": 1}))
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_UNBOUNDED", True))

    def test_each_corruption_is_caught(self):
        self.assertEqual(verify(self.UNB, self.cert({"X": 1, "Y": 1}, {"X": 1, "Y": 0}))[0], 1)    # leaves the row
        self.assertEqual(verify(self.UNB, self.cert({"X": 1, "Y": 1}, {"X": -1, "Y": -1}))[0], 1)  # leaves the columns
        self.assertEqual(verify(self.UNB, self.cert({"X": 5, "Y": 1}, {"X": 1, "Y": 1}))[0], 1)    # point infeasible
        self.assertEqual(verify(self.UNB, self.cert({"X": 1, "Y": 1}, {}))[0], 1)                  # zero ray
        worse = self.UNB.replace("C -1 R1 1", "C 1 R1 1").replace("C -1 R1 -1", "C 1 R1 -1")
        self.assertEqual(verify(worse, self.cert({"X": 1, "Y": 1}, {"X": 1, "Y": 1}, worse))[0], 1)  # objective gets worse

    def test_a_ray_alone_is_not_enough(self):
        cert = self.cert({"X": 1, "Y": 1}, {"X": 1, "Y": 1})
        del cert["point"]
        self.assertEqual(verify(self.UNB, cert)[0], 1)

    def test_a_ray_with_double_rounding_noise_passes_within_tolerance_only(self):
        text = "NAME U\nROWS\n N C\n E R1\nCOLUMNS\n X C -1 R1 0.1\n Y C 0 R1 -0.3\nRHS\n RHS R1 0\nENDATA\n"
        cert = make_cert(text, "unbounded", {"point": {}, "ray": {"X": 3, "Y": 1}}, rows=1, cols=2, nnz=2)
        code, rep = verify(text, cert)  # 0.1 * 3 - 0.3 is not exactly 0 in the exact decimal data? it is: 3/10 - 3/10 -> double noise
        self.assertEqual(code, 0)
        self.assertIn(rep.detail, ("PASS_UNBOUNDED", "PASS_UNBOUNDED_TOL"))

    def test_maximization_flips_the_improving_direction(self):
        text = self.UNB.replace("NAME U", "NAME U\nOBJSENSE\n    MAX").replace("C -1", "C 1")
        cert = make_cert(text, "unbounded", {"point": {"X": 1, "Y": 1}, "ray": {"X": 1, "Y": 1}}, rows=1, cols=2, nnz=2, sense="max")
        self.assertEqual(verify(text, cert)[0], 0)


class FeasibleTests(unittest.TestCase):
    MIP = "NAME M\nROWS\n N C\n L R1\nCOLUMNS\n MARKER 'MARKER' 'INTORG'\n X C -1 R1 1\n MARKER 'MARKER' 'INTEND'\n Y C -1 R1 1\nRHS\n RHS R1 4.5\nENDATA\n"

    def cert(self, x):
        return make_cert(self.MIP, "feasible", {"x": x}, rows=1, cols=2, nnz=2, n_integer=1)

    def test_feasible_and_integral(self):
        code, rep = verify(self.MIP, self.cert({"X": 3, "Y": 1.5}))
        self.assertEqual((code, rep.detail), (0, "PASS_FEASIBLE"))
        self.assertTrue(any("optimality not certified" in line for line in rep.lines))

    def test_fractional_or_infeasible_fails(self):
        self.assertEqual(verify(self.MIP, self.cert({"X": 2.5, "Y": 1}))[0], 1)
        self.assertEqual(verify(self.MIP, self.cert({"X": 4, "Y": 1}))[0], 1)


if __name__ == "__main__":
    unittest.main()
