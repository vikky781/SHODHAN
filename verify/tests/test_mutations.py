"""Mutation tests: corrupt valid certificates and check that KASAUTI rejects every corruption that is large
enough to matter.

The certificates come from a corpus directory written by shodhan_cert_corpus (environment variable
KASAUTI_CORPUS); without it the tests are skipped, visibly. Each mutation is classified by an ORACLE that is
independent of kasauti/checks.py: plain Fraction arithmetic on the parsed model, written here from the
definitions in docs/CERTIFICATES.md. A mutation is

  harmful  if it is clearly large: a bound violated by >= 1e-4 relative, a gap >= 1e-4 relative, a valid
           proof destroyed (Farkas intervals overlap, ray not in the recession cone, no improvement);
  benign   if the certificate is still clearly valid (violations <= 1e-8, gap <= 1e-8, proof intact);
  gray     otherwise: skipped and counted, because the verifier's tolerances (1e-6 / 1e-9) decide it.

The test requires 100% detection of harmful mutations, and reports how many benign ones were nevertheless
rejected.
"""

import collections
import copy
import glob
import json
import os
import random
import sys
import tempfile
import unittest
from fractions import Fraction as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kasauti import cli  # noqa: E402
from kasauti.mps import parse_mps, read_bytes  # noqa: E402
from tests.helpers import args  # noqa: E402

HARM = F(1, 10 ** 4)
BENIGN = F(1, 10 ** 8)
CORPUS = os.environ.get("KASAUTI_CORPUS")


# ---------------------------------------------------------------------------------------------------
class Oracle:
    """Independent exact definitions (no code shared with kasauti/checks.py)."""

    def __init__(self, model):
        self.m = model
        self.sgn = -1 if model.sense == "max" else 1
        self.rows = [[] for _ in range(model.n_rows)]
        for j, col in enumerate(model.col_entries):
            for i, a in col:
                self.rows[i].append((j, F(a)))
        self.c = [self.sgn * F(v) for v in model.col_cost]
        self.offset = self.sgn * F(model.offset)

    def activity(self, x):
        return [sum((a * x[j] for j, a in row), F(0)) for row in self.rows]

    def rel_violation(self, x):
        m, worst = self.m, F(0)
        act = self.activity(x)
        for i in range(m.n_rows):
            if m.row_lo[i] is not None and act[i] < F(m.row_lo[i]):
                worst = max(worst, (F(m.row_lo[i]) - act[i]) / (1 + abs(F(m.row_lo[i]))))
            if m.row_hi[i] is not None and act[i] > F(m.row_hi[i]):
                worst = max(worst, (act[i] - F(m.row_hi[i])) / (1 + abs(F(m.row_hi[i]))))
        for j in range(m.n_cols):
            if m.col_lo[j] is not None and x[j] < F(m.col_lo[j]):
                worst = max(worst, (F(m.col_lo[j]) - x[j]) / (1 + abs(F(m.col_lo[j]))))
            if m.col_hi[j] is not None and x[j] > F(m.col_hi[j]):
                worst = max(worst, (x[j] - F(m.col_hi[j])) / (1 + abs(F(m.col_hi[j]))))
        return worst

    def objective_min(self, x):
        return self.offset + sum((self.c[j] * x[j] for j in range(self.m.n_cols)), F(0))

    def strict_bound_min(self, y):
        """Strict weak-duality lower bound (minimization form) or None when a needed bound is infinite."""
        m = self.m
        lb = self.offset
        for i in range(m.n_rows):
            if y[i] > 0:
                if m.row_lo[i] is None:
                    return None
                lb += y[i] * F(m.row_lo[i])
            elif y[i] < 0:
                if m.row_hi[i] is None:
                    return None
                lb += y[i] * F(m.row_hi[i])
        for j, col in enumerate(m.col_entries):
            d = self.c[j] - sum((F(a) * y[i] for i, a in col), F(0))
            if d > 0:
                if m.col_lo[j] is None:
                    return None
                lb += d * F(m.col_lo[j])
            elif d < 0:
                if m.col_hi[j] is None:
                    return None
                lb += d * F(m.col_hi[j])
        return lb

    def farkas_valid(self, y):
        """Exact Farkas test: the interval of y^T (A x) over the columns misses the interval over the rows."""
        m = self.m
        r_lo, r_hi = F(0), F(0)
        for i in range(m.n_rows):
            lo = None if m.row_lo[i] is None else y[i] * F(m.row_lo[i])
            hi = None if m.row_hi[i] is None else y[i] * F(m.row_hi[i])
            if y[i] < 0:
                lo, hi = hi, lo
            if y[i] == 0:
                lo = hi = F(0)
            r_lo = None if (r_lo is None or lo is None) else r_lo + lo
            r_hi = None if (r_hi is None or hi is None) else r_hi + hi
        c_lo, c_hi = F(0), F(0)
        for j, col in enumerate(m.col_entries):
            g = sum((F(a) * y[i] for i, a in col), F(0))
            lo = None if m.col_lo[j] is None else g * F(m.col_lo[j])
            hi = None if m.col_hi[j] is None else g * F(m.col_hi[j])
            if g < 0:
                lo, hi = hi, lo
            if g == 0:
                lo = hi = F(0)
            c_lo = None if (c_lo is None or lo is None) else c_lo + lo
            c_hi = None if (c_hi is None or hi is None) else c_hi + hi
        # y^T A x lies in [c_lo, c_hi] over the columns and must equal y^T r, r in [r_lo, r_hi]:
        # infeasible iff the intervals are disjoint.
        if r_hi is not None and c_lo is not None and r_hi < c_lo:
            return True
        if r_lo is not None and c_hi is not None and c_hi < r_lo:
            return True
        return False

    def cone_violation(self, r):
        """Largest relative violation of the recession-cone conditions by the direction r."""
        m = self.m
        rmax = max([abs(v) for v in r] + [F(0)])
        if rmax == 0:
            return None
        worst = F(0)
        for j in range(m.n_cols):
            if r[j] > 0 and m.col_hi[j] is not None:
                worst = max(worst, r[j] / rmax)
            if r[j] < 0 and m.col_lo[j] is not None:
                worst = max(worst, -r[j] / rmax)
        for i, row in enumerate(self.rows):
            ar = sum((a * r[j] for j, a in row), F(0))
            mag = sum((abs(a * r[j]) for j, a in row), F(0))
            norm = mag if mag > 0 else rmax
            if m.row_lo[i] is not None and ar < 0:
                worst = max(worst, -ar / norm)
            if m.row_hi[i] is not None and ar > 0:
                worst = max(worst, ar / norm)
        return worst

    def rate_min(self, r):
        return sum((self.c[j] * r[j] for j in range(self.m.n_cols)), F(0))


# ---------------------------------------------------------------------------------------------------
class Result:
    def __init__(self):
        self.count = collections.Counter()

    def add(self, kind, label, detected):
        self.count[(kind, label)] += 1
        self.count[(kind, label, "detected" if detected else "accepted")] += 1


@unittest.skipUnless(CORPUS and os.path.isdir(CORPUS), "set KASAUTI_CORPUS to a directory written by shodhan_cert_corpus")
class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = random.Random(20260101)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.res = Result()
        cls.items = []
        for p in sorted(glob.glob(os.path.join(CORPUS, "*.cert.json"))):
            with open(p) as f:
                cert = json.load(f)
            if cert["status"] in ("optimal", "infeasible", "unbounded"):
                cls.items.append((p.replace(".cert.json", ".mps"), cert))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        res = cls.res.count
        harmful = sum(v for k, v in res.items() if len(k) == 2 and k[1] == "harmful")
        caught = sum(v for k, v in res.items() if len(k) == 3 and k[1] == "harmful" and k[2] == "detected")
        benign = sum(v for k, v in res.items() if len(k) == 2 and k[1] == "benign")
        benign_rej = sum(v for k, v in res.items() if len(k) == 3 and k[1] == "benign" and k[2] == "detected")
        gray = sum(v for k, v in res.items() if len(k) == 2 and k[1] == "gray")
        print("\n    mutation tests: harmful mutations detected %d of %d; benign mutations %d (rejected anyway: %d); gray zone skipped %d" %
              (caught, harmful, benign, benign_rej, gray))
        for kind in sorted({k[0] for k in res if len(k) == 2}):
            h = res[(kind, "harmful")]
            print("      %-22s harmful %4d detected %4d | benign %4d rejected %3d | gray %4d" % (
                kind, h, res[(kind, "harmful", "detected")], res[(kind, "benign")], res[(kind, "benign", "detected")], res[(kind, "gray")]))

    # -- helpers -------------------------------------------------------------------------------
    def verify_cert(self, mps, cert):
        path = os.path.join(self.tmp.name, "m.json")
        with open(path, "w") as f:
            json.dump(cert, f)
        code, _rep = cli.verify(mps, path, args(mode="exact"))
        return code

    def record(self, kind, label, mps, cert):
        if label == "gray":
            self.res.add(kind, "gray", False)
            return
        code = self.verify_cert(mps, cert)
        self.res.add(kind, label, code != 0)
        if label == "harmful":
            self.assertNotEqual(code, 0, "undetected harmful mutation (%s) of %s" % (kind, mps))

    @staticmethod
    def dense(model_names, sparse):
        return [F(sparse.get(n, 0)) for n in model_names]

    @staticmethod
    def sparse(names, vec):
        return {n: (float(v) if not isinstance(v, float) else v) for n, v in zip(names, vec) if v != 0}

    def classify_x(self, orc, x_new, obj_ref_min):
        viol = orc.rel_violation(x_new)
        dobj = orc.objective_min(x_new) - obj_ref_min
        scale = 1 + abs(obj_ref_min)
        if viol >= HARM or dobj >= HARM * scale:
            return "harmful"
        if viol <= BENIGN and abs(dobj) <= BENIGN * scale:
            return "benign"
        return "gray"

    # -- the tests -----------------------------------------------------------------------------
    def test_corruptions_are_detected(self):
        rng = self.rng
        optimal = [(m, c) for m, c in self.items if c["status"] == "optimal"]
        infeasible = [(m, c) for m, c in self.items if c["status"] == "infeasible"]
        unbounded = [(m, c) for m, c in self.items if c["status"] == "unbounded"]
        for mps, cert in rng.sample(optimal, min(len(optimal), 90)):
            self.mutate_optimal(mps, cert)
        for mps, cert in rng.sample(infeasible, min(len(infeasible), 40)):
            self.mutate_infeasible(mps, cert)
        for mps, cert in rng.sample(unbounded, min(len(unbounded), 40)):
            self.mutate_unbounded(mps, cert)
        for mps, cert in rng.sample(self.items, min(len(self.items), 40)):
            self.mutate_metadata(mps, cert)
        res = self.res.count
        harmful = sum(v for k, v in res.items() if len(k) == 2 and k[1] == "harmful")
        self.assertGreaterEqual(harmful, 500)

    def mutate_optimal(self, mps, cert):
        model = parse_mps(read_bytes(mps), exact=True)
        orc = Oracle(model)
        rng = self.rng
        x0 = self.dense(model.col_names, cert["x"])
        y0 = self.dense(model.row_names, cert["y"])
        ref = orc.objective_min(x0)

        def variants_x():
            j = rng.randrange(model.n_cols)
            for factor in (F(1, 100), F(1), F(100)):
                for sign in (1, -1):
                    x = list(x0)
                    x[j] = x[j] + sign * factor * (1 + abs(x[j]))
                    yield "x perturbed", x
            x = list(x0)
            x[j] = F(0) if x[j] != 0 else F(1)
            yield "x column zeroed/set", x
            if model.n_cols > 1:
                k = (j + 1 + rng.randrange(model.n_cols - 1)) % model.n_cols
                x = list(x0)
                x[j], x[k] = x[k], x[j]
                yield "x columns swapped", x
            yield "x scaled by 1.01", [v * F(101, 100) for v in x0]
            yield "x tiny noise", [v * (1 + F(rng.choice((-1, 1)), 10 ** 13)) for v in x0]

        for kind, x in variants_x():
            c2 = copy.deepcopy(cert)
            c2["x"] = self.sparse(model.col_names, [float(v) for v in x])
            xf = self.dense(model.col_names, c2["x"])  # what the verifier will actually read (the float values)
            self.record(kind, self.classify_x(orc, xf, ref), mps, c2)

        ymax = max([abs(v) for v in y0] + [F(0)])
        nonzero = [i for i in range(model.n_rows) if abs(y0[i]) >= ymax / 1000 and y0[i] != 0]

        def classify_y(y_new):
            lb = orc.strict_bound_min(y_new)
            scale = 1 + abs(ref)
            if lb is None:
                # no strict bound: harmful only if the offending value is large (the verifier may drop tiny ones)
                return "harmful" if self.large_offender(orc, y_new, ymax) else "gray"
            gap = ref - lb
            if gap >= HARM * scale or -gap >= HARM * scale:
                return "harmful"
            if abs(gap) <= BENIGN * scale:
                return "benign"
            return "gray"

        variants_y = []
        if nonzero:
            i = rng.choice(nonzero)
            y = list(y0)
            y[i] = -y[i]
            variants_y.append(("y sign flipped", y))
            y = list(y0)
            y[i] = F(0)
            variants_y.append(("y entry zeroed", y))
            y = list(y0)
            y[i] = y[i] * 2
            variants_y.append(("y entry doubled", y))
        if ymax > 0:
            variants_y.append(("y scaled by 0.5", [v / 2 for v in y0]))
            r = rng.randrange(model.n_rows)
            y = list(y0)
            y[r] += ymax / 10
            variants_y.append(("y entry shifted", y))
            variants_y.append(("y tiny noise", [v * (1 + F(rng.choice((-1, 1)), 10 ** 13)) for v in y0]))
        for kind, y in variants_y:
            c2 = copy.deepcopy(cert)
            c2["y"] = self.sparse(model.row_names, [float(v) for v in y])
            yf = self.dense(model.row_names, c2["y"])
            self.record(kind, classify_y(yf), mps, c2)

        c2 = copy.deepcopy(cert)
        c2["claimed_objective"] = cert["claimed_objective"] + (1 + abs(cert["claimed_objective"])) * 0.01
        self.record("claimed objective", "harmful", mps, c2)
        c2 = copy.deepcopy(cert)
        c2["claimed_objective"] = cert["claimed_objective"] * (1 + 1e-12)
        self.record("claimed objective", "benign", mps, c2)

    @staticmethod
    def large_offender(orc, y, ymax):
        """True if a value that needs an infinite bound is not tiny relative to its scale."""
        m = orc.m
        for i in range(m.n_rows):
            if y[i] > 0 and m.row_lo[i] is None and y[i] > ymax / 10 ** 6:
                return True
            if y[i] < 0 and m.row_hi[i] is None and -y[i] > ymax / 10 ** 6:
                return True
        for j, col in enumerate(m.col_entries):
            d = orc.c[j] - sum((F(a) * y[i] for i, a in col), F(0))
            scale = abs(orc.c[j]) + sum((abs(F(a) * y[i]) for i, a in col), F(0))
            if scale == 0:
                continue
            if (d > 0 and m.col_lo[j] is None or d < 0 and m.col_hi[j] is None) and abs(d) > scale / 10 ** 6:
                return True
        return False

    def mutate_infeasible(self, mps, cert):
        model = parse_mps(read_bytes(mps), exact=True)
        orc = Oracle(model)
        rng = self.rng
        y0 = self.dense(model.row_names, cert["farkas"]["y"])
        if not orc.farkas_valid(y0):
            # Valid only at tolerance level (tiny coefficients dropped): the strict oracle cannot classify
            # its mutations, so they are skipped (counted as gray).
            self.res.add("farkas (tolerance-level original)", "gray", False)
            return
        nz = [i for i in range(model.n_rows) if y0[i] != 0]
        variants = [("farkas negated", [-v for v in y0]), ("farkas scaled by 3", [v * 3 for v in y0])]
        if nz:
            i = rng.choice(nz)
            y = list(y0); y[i] = -y[i]; variants.append(("farkas entry sign flipped", y))
            y = list(y0); y[i] = F(0); variants.append(("farkas entry zeroed", y))
            ymax = max(abs(v) for v in y0)
            y = list(y0); y[rng.randrange(model.n_rows)] += ymax; variants.append(("farkas entry shifted", y))
            if len(nz) > 1:
                j = rng.choice([k for k in nz if k != i])
                y = list(y0); y[i], y[j] = y[j], y[i]; variants.append(("farkas entries swapped", y))
        variants.append(("farkas all zero", [F(0)] * model.n_rows))
        for kind, y in variants:
            c2 = copy.deepcopy(cert)
            c2["farkas"]["y"] = self.sparse(model.row_names, [float(v) for v in y])
            yf = self.dense(model.row_names, c2["farkas"]["y"])
            valid = orc.farkas_valid(yf)
            self.record(kind, "benign" if valid else "harmful", mps, c2)

    def mutate_unbounded(self, mps, cert):
        model = parse_mps(read_bytes(mps), exact=True)
        orc = Oracle(model)
        rng = self.rng
        r0 = self.dense(model.col_names, cert["ray"])
        p0 = self.dense(model.col_names, cert["point"])
        nz = [j for j in range(model.n_cols) if r0[j] != 0]

        def classify_ray(r):
            viol = orc.cone_violation(r)
            if viol is None:
                return "harmful"
            if orc.rate_min(r) >= 0:
                return "harmful"
            if viol >= HARM:
                return "harmful"
            if viol == 0:
                return "benign"
            return "gray"

        variants = [("ray negated", [-v for v in r0]), ("ray scaled by 7", [v * 7 for v in r0])]
        if nz:
            j = rng.choice(nz)
            r = list(r0); r[j] = -r[j]; variants.append(("ray component flipped", r))
            r = list(r0); r[j] = F(0); variants.append(("ray component zeroed", r))
            rmax = max(abs(v) for v in r0)
            r = list(r0); r[rng.randrange(model.n_cols)] += rmax; variants.append(("ray component shifted", r))
            if len(nz) > 1:
                k = rng.choice([q for q in nz if q != j])
                r = list(r0); r[j], r[k] = r[k], r[j]; variants.append(("ray components swapped", r))
        for kind, r in variants:
            c2 = copy.deepcopy(cert)
            c2["ray"] = self.sparse(model.col_names, [float(v) for v in r])
            rf = self.dense(model.col_names, c2["ray"])
            self.record(kind, classify_ray(rf), mps, c2)

        for factor in (F(1, 100), F(100)):
            j = rng.randrange(model.n_cols)
            p = list(p0)
            p[j] = p[j] + factor * (1 + abs(p[j]))
            c2 = copy.deepcopy(cert)
            c2["point"] = self.sparse(model.col_names, [float(v) for v in p])
            pf = self.dense(model.col_names, c2["point"])
            viol = orc.rel_violation(pf)
            self.record("point perturbed", "harmful" if viol >= HARM else ("benign" if viol <= BENIGN else "gray"), mps, c2)

    def mutate_metadata(self, mps, cert):
        c2 = copy.deepcopy(cert)
        h = c2["problem"]["file_sha256"]
        c2["problem"]["file_sha256"] = ("0" if h[0] != "0" else "1") + h[1:]
        self.record("sha256 edited", "harmful", mps, c2)
        c2 = copy.deepcopy(cert)
        c2["problem"]["rows"] = (c2["problem"]["rows"] or 0) + 1
        self.record("row count edited", "harmful", mps, c2)
        c2 = copy.deepcopy(cert)
        c2["status"] = {"optimal": "infeasible", "infeasible": "unbounded", "unbounded": "optimal"}[cert["status"]]
        self.record("status changed", "harmful", mps, c2)


if __name__ == "__main__":
    unittest.main()
