"""Mutation tests for QP certificates: corrupt valid certificates (and the model file) and check that KASAUTI rejects every
corruption that is large enough to matter.

The certificates come from the corpus written by shodhan_cert_corpus (environment variable KASAUTI_CORPUS, family "qp ...").
Each mutation is classified by an ORACLE written here from the definitions in docs/QP.md and docs/CERTIFICATES.md with plain
Fraction arithmetic (no code shared with kasauti/checks.py):

  harmful  a bound violated by >= 1e-4 relative, a gap >= 1e-4 relative, a lost proof, or a Q that is not positive semidefinite;
  benign   everything within 1e-8 and Q positive semidefinite;
  gray     otherwise (skipped and counted: the verifier's tolerances decide it).

The test requires 100% detection of harmful mutations.
"""

import collections
import copy
import glob
import json
import os
import random
import re
import sys
import tempfile
import unittest


def read_file(path):
    with open(path, "rb") as f:
        return f.read()

import hashlib
from fractions import Fraction as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kasauti import cli  # noqa: E402
from kasauti.mps import parse_mps  # noqa: E402
from tests.helpers import args  # noqa: E402

HARM = F(1, 10 ** 4)
BENIGN = F(1, 10 ** 8)
CORPUS = os.environ.get("KASAUTI_CORPUS")


class QpOracle:
    """Exact definitions: objective, violation, weak-duality bound with Q, positive semidefiniteness."""

    def __init__(self, model):
        self.m = model
        self.sgn = -1 if model.sense == "max" else 1
        self.c = [self.sgn * F(v) for v in model.col_cost]
        self.offset = self.sgn * F(model.offset)
        n = model.n_cols
        self.q = [dict() for _ in range(n)]  # full symmetric Q' = sgn Q
        for (i, j), v in model.quad.items():
            self.q[i][j] = self.sgn * F(v)
            if i != j:
                self.q[j][i] = self.sgn * F(v)
        self.rows = [[] for _ in range(model.n_rows)]
        for j, col in enumerate(model.col_entries):
            for i, a in col:
                self.rows[i].append((j, F(a)))

    def qx(self, x):
        return [sum((v * x[k] for k, v in self.q[j].items()), F(0)) for j in range(self.m.n_cols)]

    def objective_min(self, x):
        qx = self.qx(x)
        return self.offset + sum((self.c[j] * x[j] for j in range(self.m.n_cols)), F(0)) + sum((x[j] * qx[j] for j in range(self.m.n_cols)), F(0)) / 2

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

    def strict_bound_min(self, x, y):
        """LB(y; x~ = x) with the convex quadratic term, or None when a needed bound is infinite."""
        m = self.m
        qx = self.qx(x)
        lb = self.offset - sum((x[j] * qx[j] for j in range(m.n_cols)), F(0)) / 2
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
            d = self.c[j] + qx[j] - sum((F(a) * y[i] for i, a in col), F(0))
            if d > 0:
                if m.col_lo[j] is None:
                    return None
                lb += d * F(m.col_lo[j])
            elif d < 0:
                if m.col_hi[j] is None:
                    return None
                lb += d * F(m.col_hi[j])
        return lb

    def large_offender(self, x, y, ymax):
        m = self.m
        qx = self.qx(x)
        for i in range(m.n_rows):
            if y[i] > 0 and m.row_lo[i] is None and y[i] > ymax / 10 ** 6:
                return True
            if y[i] < 0 and m.row_hi[i] is None and -y[i] > ymax / 10 ** 6:
                return True
        for j, col in enumerate(m.col_entries):
            d = self.c[j] + qx[j] - sum((F(a) * y[i] for i, a in col), F(0))
            scale = abs(self.c[j]) + abs(qx[j]) + sum((abs(F(a) * y[i]) for i, a in col), F(0))
            if scale == 0:
                continue
            if (d > 0 and m.col_lo[j] is None or d < 0 and m.col_hi[j] is None) and abs(d) > scale / 10 ** 6:
                return True
        return False

    def psd(self):
        """Exact positive semidefiniteness by eliminating with every pivot >= 0 (zero pivot => zero row)."""
        n = self.m.n_cols
        a = [dict(r) for r in self.q]
        for k in range(n):
            piv = a[k].get(k, F(0))
            nb = {j: v for j, v in a[k].items() if j > k and v != 0}
            if piv < 0:
                return False
            if piv == 0:
                if nb:
                    return False
                continue
            for i, vi in nb.items():
                for j, vj in nb.items():
                    a[i][j] = a[i].get(j, F(0)) - vi * vj / piv
            for j in nb:
                a[j].pop(k, None)
            a[k] = {}
        return True


def dense(names, sparse):
    return [F(sparse.get(n, 0)) for n in names]


def sparse_of(names, vec):
    return {n: float(v) for n, v in zip(names, vec) if v != 0}


@unittest.skipUnless(CORPUS and os.path.isdir(CORPUS), "set KASAUTI_CORPUS to a directory written by shodhan_cert_corpus")
class QpMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = random.Random(20261005)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.count = collections.Counter()
        cls.items = []
        for p in sorted(glob.glob(os.path.join(CORPUS, "qp_*.cert.json"))):
            with open(p) as f:
                cert = json.load(f)
            if cert["status"] == "optimal" and cert.get("problem", {}).get("quadratic"):
                cls.items.append((p.replace(".cert.json", ".mps"), cert))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        c = cls.count
        kinds = sorted({k[0] for k in c if len(k) == 2})
        harmful = sum(c[(k, "harmful")] for k in kinds)
        caught = sum(c[(k, "harmful", "detected")] for k in kinds)
        benign = sum(c[(k, "benign")] for k in kinds)
        rejected = sum(c[(k, "benign", "detected")] for k in kinds)
        gray = sum(c[(k, "gray")] for k in kinds)
        print("\n    QP mutation tests: harmful mutations detected %d of %d; benign mutations %d (rejected anyway: %d); gray zone skipped %d" % (caught, harmful, benign, rejected, gray))
        for k in kinds:
            print("      %-26s harmful %4d detected %4d | benign %4d rejected %3d | gray %4d" % (
                k, c[(k, "harmful")], c[(k, "harmful", "detected")], c[(k, "benign")], c[(k, "benign", "detected")], c[(k, "gray")]))

    def run_verifier(self, mps_text_or_path, cert, from_text=False):
        mps_path = os.path.join(self.tmp.name, "m.mps")
        if from_text:
            with open(mps_path, "w", newline="") as f:
                f.write(mps_text_or_path)
        else:
            mps_path = mps_text_or_path
        cp = os.path.join(self.tmp.name, "m.json")
        with open(cp, "w") as f:
            json.dump(cert, f)
        code, _ = cli.verify(mps_path, cp, args(mode="exact"))
        return code

    def record(self, kind, label, code):
        self.count[(kind, label)] += 1
        if label == "gray":
            return
        self.count[(kind, label, "detected" if code != 0 else "accepted")] += 1
        if label == "harmful":
            self.assertNotEqual(code, 0, "undetected harmful mutation: " + kind)

    def classify(self, orc, x, y, ref):
        """Label of a (point, multipliers) pair for the model of `orc`."""
        if not orc.psd():
            return "harmful"
        viol = orc.rel_violation(x)
        scale = 1 + abs(ref)
        ymax = max([abs(v) for v in y] + [F(0)])
        lb = orc.strict_bound_min(x, y)
        if lb is None:
            return "harmful" if orc.large_offender(x, y, ymax) or viol >= HARM else "gray"
        gap = orc.objective_min(x) - lb
        if viol >= HARM or abs(gap) >= HARM * scale:
            return "harmful"
        if viol <= BENIGN and abs(gap) <= BENIGN * scale:
            return "benign"
        return "gray"

    def test_corruptions_are_detected(self):
        rng = self.rng
        self.assertGreaterEqual(len(self.items), 150)
        for mps, cert in rng.sample(self.items, min(len(self.items), 120)):
            model = parse_mps(read_file(mps), exact=True)
            orc = QpOracle(model)
            x0 = dense(model.col_names, cert["x"])
            y0 = dense(model.row_names, cert["y"])
            ref = orc.objective_min(x0)
            # --- x and y mutations on the original model ---
            j = rng.randrange(model.n_cols)
            xs = []
            for factor in (F(1, 100), F(1), F(100)):
                for sign in (1, -1):
                    x = list(x0)
                    x[j] = x[j] + sign * factor * (1 + abs(x[j]))
                    xs.append(("x perturbed", x))
            x = list(x0)
            x[j] = -x[j] if x[j] != 0 else F(1)
            xs.append(("x entry sign flipped", x))
            xs.append(("x scaled by 1.01", [v * F(101, 100) for v in x0]))
            xs.append(("x tiny noise", [v * (1 + F(rng.choice((-1, 1)), 10 ** 13)) for v in x0]))
            for kind, x in xs:
                c2 = copy.deepcopy(cert)
                c2["x"] = sparse_of(model.col_names, x)
                xf = dense(model.col_names, c2["x"])
                self.record(kind, self.classify(orc, xf, y0, ref), self.run_verifier(mps, c2))
            ymax = max([abs(v) for v in y0] + [F(0)])
            nz = [i for i in range(model.n_rows) if y0[i] != 0 and abs(y0[i]) >= ymax / 1000]
            ys = []
            if nz:
                i = rng.choice(nz)
                y = list(y0); y[i] = -y[i]; ys.append(("y sign flipped", y))
                y = list(y0); y[i] = F(0); ys.append(("y entry zeroed", y))
                y = list(y0); y[i] = y[i] * 2; ys.append(("y entry doubled", y))
            if ymax > 0:
                ys.append(("y scaled by 0.5", [v / 2 for v in y0]))
                y = list(y0); y[rng.randrange(model.n_rows)] += ymax / 10; ys.append(("y entry shifted", y))
                ys.append(("y tiny noise", [v * (1 + F(rng.choice((-1, 1)), 10 ** 13)) for v in y0]))
            for kind, y in ys:
                c2 = copy.deepcopy(cert)
                c2["y"] = sparse_of(model.row_names, y)
                yf = dense(model.row_names, c2["y"])
                self.record(kind, self.classify(orc, x0, yf, ref), self.run_verifier(mps, c2))
            c2 = copy.deepcopy(cert)
            c2["claimed_objective"] = cert["claimed_objective"] + (1 + abs(cert["claimed_objective"])) * 0.01
            self.record("claimed objective", "harmful", self.run_verifier(mps, c2))
            # --- mutations of the quadratic term in a copy of the model file (the hash in the certificate is updated, so the
            # corruption must be caught by the numbers, not by the hash) ---
            text = read_file(mps).decode()
            lines = text.split("\n")
            try:
                start = lines.index("QUADOBJ")
            except ValueError:
                continue
            end = next(k for k in range(start + 1, len(lines)) if lines[k].strip() == "ENDATA")
            if end == start + 1:
                continue
            for kind in ("Q entry scaled", "Q entry sign flipped", "Q diagonal made negative", "Q entry changed by 1e-12 relative"):
                k = rng.randrange(start + 1, end)
                parts = lines[k].split()
                if len(parts) != 3:
                    continue
                v = F(parts[2])
                if kind == "Q entry scaled":
                    nv = v * F(3, 2)
                elif kind == "Q entry sign flipped":
                    nv = -v
                elif kind == "Q diagonal made negative":
                    diag = [q for q in range(start + 1, end) if lines[q].split()[0] == lines[q].split()[1]]
                    if not diag:
                        continue
                    k = rng.choice(diag)
                    parts = lines[k].split()
                    nv = -abs(F(parts[2])) - 1
                else:
                    nv = v * (1 + F(1, 10 ** 12))
                mutated = list(lines)
                mutated[k] = "    %s  %s  %s" % (parts[0], parts[1], repr(float(nv)))
                mtext = "\n".join(mutated)
                c2 = copy.deepcopy(cert)
                c2["problem"]["file_sha256"] = hashlib.sha256(mtext.encode()).hexdigest()
                mmodel = parse_mps(mtext.encode(), exact=True)
                morc = QpOracle(mmodel)
                label = self.classify(morc, x0, y0, morc.objective_min(x0))
                self.record(kind, label, self.run_verifier(mtext, c2, from_text=True))
        c = self.count
        harmful = sum(v for k, v in c.items() if len(k) == 2 and k[1] == "harmful")
        self.assertGreaterEqual(harmful, 500)


if __name__ == "__main__":
    unittest.main()
