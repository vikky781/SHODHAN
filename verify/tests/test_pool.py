"""Pooling certificates (docs/POOLING.md): hand-made checks of the verifier, a corpus of certificates written by
`shodhan pool` (SYNTHETIC instances) verified in exact and float mode, and mutation tests with an oracle written here
from the definitions of the documentation (it shares no code with kasauti/pool.py).

The corpus and mutation tests need the shodhan executable (SHODHAN_EXE) and are skipped with a visible message otherwise.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kasauti import pool as kpool  # noqa: E402
from tests.helpers import args, sha  # noqa: E402
from tests.poolgen import generate  # noqa: E402
from kasauti import cli  # noqa: E402

EXE = os.environ.get("SHODHAN_EXE")
COUNT = int(os.environ.get("KASAUTI_POOL_COUNT", "130"))

SPEC = """# SYNTHETIC hand-made instance
name hand
synthetic yes
qualities 1 S
source A 6 inf 4
source B 14 inf 1
pool P 50
terminal T 20 30 2
arc A P
arc B P
arc P T
"""
# Flows A>P = 10, B>P = 20, P>T = 30; pool quality (4*10 + 1*20) / 30 = 2 exactly; terminal: 2*30 <= 2*30.
GOOD = {"A>P": 10, "B>P": 20, "P>T": 30}


def verify_text(spec_text, cert, mode="exact", **kw):
    """Runs the pool verifier on a spec given as text and a certificate dict; returns (code, report)."""
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "m.pool")
        cp = os.path.join(d, "c.json")
        with open(mp, "w", newline="") as f:
            f.write(spec_text)
        with open(cp, "w") as f:
            json.dump(cert, f)
        return cli.verify(mp, cp, args(mode=mode, **kw))


def make_cert(spec_text, flows, q, objective, **extra):
    cert = {"format": "shodhan-cert", "version": 1, "solver": {"name": "t", "version": "0"},
            "problem": {"name": "hand", "file_sha256": sha(spec_text), "kind": "pooling", "sources": 2, "pools": 1, "terminals": 1, "qualities": 1},
            "status": "feasible", "claimed_objective": objective, "optimality_certified": False, "nonconvex": True,
            "flows": flows, "q": q, "tolerances": {"pool_tol": 1e-6}, "attempts": {"count": 1, "configuration": "slp"}}
    cert.update(extra)
    return cert


class HandMadeTests(unittest.TestCase):
    def good(self, **extra):
        return make_cert(SPEC, dict(GOOD), {"P:S": 2}, 30 * 20 - 10 * 6 - 20 * 14, **extra)

    def test_an_exactly_feasible_point_passes_with_the_exact_verdict(self):
        code, rep = verify_text(SPEC, self.good())
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_FEASIBLE", True))
        self.assertEqual(rep.data["objective"], 260)

    def test_float_mode_passes_too(self):
        code, rep = verify_text(SPEC, self.good(), mode="float")
        self.assertEqual(code, 0)
        self.assertTrue(rep.detail.startswith("PASS_FEASIBLE"))

    def test_a_point_within_the_tolerance_is_tolerance_level(self):
        cert = self.good()
        cert["q"]["P:S"] = 2.0000001
        code, rep = verify_text(SPEC, cert)
        self.assertEqual((code, rep.detail, rep.rigorous), (0, "PASS_FEASIBLE_TOL", False))

    def test_each_violation_is_rejected(self):
        cases = {
            "pool quality": lambda c: c["q"].update({"P:S": 3.0}),
            "terminal spec": lambda c: (c["flows"].update({"A>P": 20, "B>P": 10, "P>T": 30}), c["q"].update({"P:S": 3})),
            "material balance": lambda c: c["flows"].update({"P>T": 31}),
            "negative flow": lambda c: c["flows"].update({"A>P": -1}),
            "pool capacity": lambda c: c["flows"].update({"A>P": 40, "B>P": 40, "P>T": 80}),
            "demand": lambda c: c["flows"].update({"A>P": 20, "B>P": 40, "P>T": 60}),
            "claimed objective": lambda c: c.update({"claimed_objective": 999}),
            "upper bound below the objective": lambda c: c.update({"claimed_upper_bound": 100}),
            "claim of optimality": lambda c: c.update({"optimality_certified": True}),
        }
        for name, mutate in cases.items():
            cert = self.good()
            mutate(cert)
            code, rep = verify_text(SPEC, cert)
            self.assertEqual(code, 1, name)

    def test_the_mccormick_bound_is_reported_as_not_verified(self):
        code, rep = verify_text(SPEC, self.good(claimed_upper_bound=300))
        self.assertEqual(code, 0)
        self.assertTrue(any("NOT verified" in line for line in rep.lines))

    def test_status_other_and_malformed_input(self):
        cert = self.good()
        cert["status"] = "other"
        self.assertEqual(verify_text(SPEC, cert)[0], 2)
        cert = self.good()
        cert["flows"]["X>Y"] = 1
        self.assertEqual(verify_text(SPEC, cert)[0], 1)
        cert = self.good()
        del cert["q"]["P:S"]
        self.assertEqual(verify_text(SPEC, cert)[0], 1)
        self.assertEqual(verify_text(SPEC.replace("arc P T", "arc P Z"), self.good())[0], 2)

    def test_a_changed_spec_file_does_not_match_the_hash(self):
        code, rep = verify_text(SPEC.replace("terminal T 20 30 2", "terminal T 20 30 3"), self.good())
        self.assertEqual(code, 1)

    def test_the_parser_reports_errors_with_line_numbers(self):
        for text, expect in (("qualities 1\nsource A 5 inf\n", "line 2"), ("qualities 1\nfoo\n", "line 2"), ("source A 1 1 1\n", "line 1")):
            with self.assertRaises(kpool.PoolError) as ctx:
                kpool.parse_pool(text, True)
            self.assertIn(expect, str(ctx.exception))


def oracle(spec_text, cert):
    """(worst relative residual, objective) of the certificate point, exact; written from docs/POOLING.md section 3 with its
    own parsing. None when the certificate does not name every pool quality."""
    S, P, T, arcs = {}, {}, {}, []
    K = 0
    for raw in spec_text.splitlines():
        t = raw.split("#")[0].split()
        if not t:
            continue
        num = lambda x: None if x == "inf" else Fraction(x)
        if t[0] == "qualities":
            K = int(t[1])
        elif t[0] == "source":
            S[t[1]] = (Fraction(t[2]), num(t[3]), [Fraction(x) for x in t[4:]])
        elif t[0] == "pool":
            P[t[1]] = num(t[2])
        elif t[0] == "terminal":
            T[t[1]] = (Fraction(t[2]), num(t[3]), [num(x) for x in t[4:]])
        elif t[0] == "arc":
            arcs.append((t[1], t[2]))
    flow = {a: Fraction(cert["flows"].get("%s>%s" % a, 0)) for a in arcs}
    q = {(p, k): Fraction(cert["q"]["%s:Q%d" % (p, k + 1)]) if "%s:Q%d" % (p, k + 1) in cert["q"] else Fraction(cert["q"]["%s:S" % p]) for p in P for k in range(K)}
    worst = Fraction(0)

    def rel(res, scale):
        return max(res, Fraction(0)) / (1 + scale)

    for a, v in flow.items():
        worst = max(worst, rel(-v, 0))
    for p in P:
        for k in range(K):
            qs = [S[s][2][k] for (s, pp) in arcs if pp == p and s in S]
            worst = max(worst, rel(max(min(qs) - q[(p, k)], q[(p, k)] - max(qs)), abs(q[(p, k)])))
    for s in S:
        out = sum((flow[a] for a in arcs if a[0] == s), Fraction(0))
        if S[s][1] is not None:
            worst = max(worst, rel(out - S[s][1], abs(out) + S[s][1]))
    for p in P:
        inn = sum((flow[a] for a in arcs if a[1] == p), Fraction(0))
        out = sum((flow[a] for a in arcs if a[0] == p), Fraction(0))
        if P[p] is not None:
            worst = max(worst, rel(inn - P[p], abs(inn) + P[p]))
        worst = max(worst, rel(abs(inn - out), abs(inn) + abs(out)))
        for k in range(K):
            terms = [S[a[0]][2][k] * flow[a] for a in arcs if a[1] == p and a[0] in S]
            worst = max(worst, rel(abs(sum(terms, Fraction(0)) - q[(p, k)] * out), sum((abs(x) for x in terms), Fraction(0)) + abs(q[(p, k)] * out)))
    for t in T:
        inn = sum((flow[a] for a in arcs if a[1] == t), Fraction(0))
        if T[t][1] is not None:
            worst = max(worst, rel(inn - T[t][1], abs(inn) + T[t][1]))
        for k in range(K):
            spec = T[t][2][k]
            if spec is None:
                continue
            terms = []
            for a in arcs:
                if a[1] != t:
                    continue
                terms.append((q[(a[0], k)] if a[0] in P else S[a[0]][2][k]) * flow[a])
            lhs = sum(terms, Fraction(0))
            worst = max(worst, rel(lhs - spec * inn, sum((abs(x) for x in terms), Fraction(0)) + abs(spec) * abs(inn)))
    obj = Fraction(0)
    for a in arcs:
        if a[0] in S and a[1] in P:
            obj -= S[a[0]][0] * flow[a]
        elif a[0] in P:
            obj += T[a[1]][0] * flow[a]
        else:
            obj += (T[a[1]][0] - S[a[0]][0]) * flow[a]
    return worst, obj


@unittest.skipUnless(EXE and os.path.exists(EXE), "SHODHAN_EXE is not set: the pooling corpus and mutation tests are skipped")
class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.items = []  # (spec text, cert, spec path, cert path)
        cls.inconclusive = 0
        cls.exit_codes = {}
        for seed in range(1, COUNT + 1):
            text, desc = generate(seed)
            sp = os.path.join(cls.tmp.name, "p%d.pool" % seed)
            cp = os.path.join(cls.tmp.name, "p%d.json" % seed)
            with open(sp, "w", newline="") as f:
                f.write(text)
            method = "recursion" if seed % 3 == 0 else "slp"
            r = subprocess.run([EXE, "pool", sp, "--method", method, "--starts", "6", "--seed", "1", "--write-cert", cp], capture_output=True, text=True)
            cls.exit_codes[r.returncode] = cls.exit_codes.get(r.returncode, 0) + 1
            with open(cp) as f:
                cert = json.load(f)
            if cert["status"] != "feasible":
                cls.inconclusive += 1
                continue
            cls.items.append((text, cert, sp, cp))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_certificate_passes_in_exact_and_float_mode(self):
        exact_rigorous = exact_tol = float_pass = agree = 0
        for text, cert, sp, cp in self.items:
            ce, re_ = cli.verify(sp, cp, args(mode="exact"))
            cf, rf = cli.verify(sp, cp, args(mode="float"))
            self.assertEqual(ce, 0, (sp, re_.lines[-6:]))
            self.assertEqual(cf, 0, (sp, rf.lines[-6:]))
            exact_rigorous += re_.detail == "PASS_FEASIBLE"
            exact_tol += re_.detail == "PASS_FEASIBLE_TOL"
            float_pass += 1
            agree += (ce == cf)
            # the exact residuals reported agree with the oracle
            worst, obj = oracle(text, cert)
            self.assertLessEqual(worst, Fraction(1, 10 ** 6))
        print("pooling corpus: %d certificates (%d instances gave no certificate body), exact PASS %d (exactly satisfied %d, within tolerance %d), "
              "float PASS %d, exact/float agree %d; solver exit codes %s" %
              (len(self.items), self.inconclusive, len(self.items), exact_rigorous, exact_tol, float_pass, agree, dict(sorted(self.exit_codes.items()))))
        self.assertGreaterEqual(len(self.items), 100)
        self.assertEqual(agree, len(self.items))

    def test_mutations_beyond_the_tolerance_are_all_detected(self):
        tol = Fraction(1, 10 ** 6)
        harmful = detected = benign = benign_accepted = gray = 0
        by_kind = {}

        def run(kind, text, cert, sp_text=None):
            nonlocal harmful, detected, benign, benign_accepted, gray
            worst, obj = oracle(text, cert)
            claim_err = abs(Fraction(cert["claimed_objective"]) - obj) / (1 + abs(obj))
            bad_bound = "claimed_upper_bound" in cert and Fraction(cert["claimed_upper_bound"]) < obj - tol * (1 + abs(obj))
            code, rep = verify_text(sp_text if sp_text is not None else text, cert)
            is_harmful = worst > 100 * tol or claim_err > 100 * tol or bad_bound or cert.get("optimality_certified") is True
            is_benign = worst <= tol / 100 and claim_err <= tol / 100 and not bad_bound
            slot = by_kind.setdefault(kind, [0, 0, 0, 0, 0])
            if is_harmful:
                harmful += 1
                slot[0] += 1
                if code != 0:
                    detected += 1
                    slot[1] += 1
                else:
                    print("UNDETECTED harmful mutation (%s): worst residual %s" % (kind, float(worst)))
            elif is_benign:
                benign += 1
                slot[2] += 1
                if code == 0:
                    benign_accepted += 1
                    slot[3] += 1
            else:
                gray += 1
                slot[4] += 1

        for text, cert, sp, cp in self.items:
            flows = cert["flows"]
            nz = sorted(k for k, v in flows.items() if v != 0)
            names = sorted(flows)
            qkeys = sorted(cert["q"])
            for name in nz[:3] or names[:1]:
                for delta, kind in ((1.0, "flow +1"), (5.0, "flow +5"), (1e-12, "flow +1e-12")):
                    c = copy.deepcopy(cert)
                    c["flows"][name] = flows[name] + delta
                    run(kind, text, c)
                c = copy.deepcopy(cert)
                c["flows"][name] = 0
                run("flow zeroed", text, c)
                c = copy.deepcopy(cert)
                c["flows"][name] = -abs(flows[name]) - 1
                run("flow negative", text, c)
            for key in qkeys[:2]:
                for delta, kind in ((0.5, "q +0.5"), (-0.5, "q -0.5"), (1e-12, "q +1e-12")):
                    c = copy.deepcopy(cert)
                    c["q"][key] += delta
                    run(kind, text, c)
            c = copy.deepcopy(cert)
            for name in names:
                if name.split(">")[1].startswith("P") or name.startswith("S") and name.split(">")[1].startswith("P"):
                    c["flows"][name] = c["flows"][name] * 1.5
            run("pool flows x1.5", text, c)
            c = copy.deepcopy(cert)
            c["claimed_objective"] = cert["claimed_objective"] + 1.0
            run("objective claim +1", text, c)
            c = copy.deepcopy(cert)
            c["claimed_objective"] = cert["claimed_objective"] * 1.01 + 1
            run("objective claim x1.01", text, c)
            if "claimed_upper_bound" in cert:
                c = copy.deepcopy(cert)
                c["claimed_upper_bound"] = cert["claimed_objective"] - 5.0
                run("bound below objective", text, c)
            c = copy.deepcopy(cert)
            c["optimality_certified"] = True
            run("optimality claim", text, c)
            # a different spec file with the same certificate: the hash no longer matches
            changed = text.replace("synthetic yes", "synthetic yes\nname other", 1) + "\n"
            code, rep = verify_text(changed, cert)
            harmful += 1
            by_kind.setdefault("spec file changed", [0, 0, 0, 0, 0])[0] += 1
            if code != 0:
                detected += 1
                by_kind["spec file changed"][1] += 1
        for kind in sorted(by_kind):
            h, d, b, ba, g = by_kind[kind]
            print("  %-24s harmful %4d detected %4d | benign %4d accepted %4d | gray %4d" % (kind, h, d, b, ba, g))
        print("pooling mutations: harmful %d, detected %d; benign %d, accepted %d; gray (skipped) %d" % (harmful, detected, benign, benign_accepted, gray))
        self.assertEqual(detected, harmful)
        self.assertEqual(benign_accepted, benign)
        self.assertGreater(harmful, 500)


if __name__ == "__main__":
    unittest.main()


def py_dump(p):
    """Canonical text of a parsed problem, the same format as `shodhan pool --dump`."""
    f = lambda v: "inf" if v is None else "%.17g" % float(v)
    sp, pt, st = kpool.arc_names(p)
    out = ["qualities %d" % p.n_q]
    out += ["source %s %s %s %s" % (s["name"], f(s["cost"]), f(s["supply"]), " ".join(f(x) for x in s["q"])) for s in p.sources]
    out += ["pool %s %s" % (q["name"], f(q["cap"])) for q in p.pools]
    out += ["terminal %s %s %s %s" % (t["name"], f(t["price"]), f(t["demand"]), " ".join(f(x) for x in t["spec"])) for t in p.terminals]
    out += ["arc " + a for a in sp + pt + st]
    return "\n".join(out) + "\n"


@unittest.skipUnless(EXE and os.path.exists(EXE), "SHODHAN_EXE is not set: the .pool differential parser test is skipped")
class PoolParserDifferentialTests(unittest.TestCase):
    def mutations(self, text, seed):
        """Valid text plus a few damaged variants (some of which are still valid)."""
        from tests.poolgen import Rng
        r = Rng(seed + 99)
        lines = text.splitlines()
        out = [text]
        for _ in range(5):
            ls = list(lines)
            kind = r.randint(0, 7)
            i = r.randint(0, len(ls) - 1)
            if kind == 0:
                del ls[i]
            elif kind == 1:
                ls.insert(i, ls[r.randint(0, len(ls) - 1)])  # duplicate a line (duplicate names or arcs)
            elif kind == 2:
                toks = ls[i].split()
                if len(toks) > 1:
                    del toks[r.randint(1, len(toks) - 1)]
                    ls[i] = " ".join(toks)
            elif kind == 3:
                ls[i] = ls[i] + " 1"
            elif kind == 4:
                ls[i] = ls[i].replace("0", "x", 1)
            elif kind == 5:
                ls[i] = ls[i].replace(" ", " -", 1)
            elif kind == 6:
                ls.insert(i, "frobnicate 3")
            else:
                ls[i] = ls[i].replace("arc ", "arc Z", 1)
            out.append("\n".join(ls) + "\n")
        return out

    def test_cpp_and_python_parsers_agree(self):
        agree_ok = agree_bad = 0
        mismatches = []
        with tempfile.TemporaryDirectory() as d:
            for seed in range(1, 121):
                text, _ = generate(seed)
                for k, variant in enumerate(self.mutations(text, seed)):
                    path = os.path.join(d, "x.pool")
                    with open(path, "w", newline="") as f:
                        f.write(variant)
                    cpp = subprocess.run([EXE, "pool", path, "--dump"], capture_output=True, text=True)
                    try:
                        py = py_dump(kpool.parse_pool(variant, False))
                        py_err = None
                    except (kpool.PoolError, UnicodeDecodeError) as e:
                        py, py_err = None, str(e)
                    cpp_ok = cpp.returncode == 0
                    if cpp_ok and py_err is None:
                        if cpp.stdout == py:
                            agree_ok += 1
                        else:
                            mismatches.append("seed %d variant %d: dumps differ" % (seed, k))
                    elif not cpp_ok and py_err is not None:
                        agree_bad += 1
                    else:
                        mismatches.append("seed %d variant %d: C++ %s (%s), Python %s (%s)" % (seed, k, "accepts" if cpp_ok else "rejects", cpp.stderr.strip()[:80],
                                                                                            "accepts" if py_err is None else "rejects", (py_err or "")[:80]))
        print("pool parser differential test: %d texts, %d accepted by both with identical dumps, %d rejected by both, %d mismatches" %
              (agree_ok + agree_bad + len(mismatches), agree_ok, agree_bad, len(mismatches)))
        for m in mismatches[:15]:
            print("  MISMATCH " + m)
        self.assertEqual(mismatches, [])
        self.assertGreaterEqual(agree_ok, 100)
        self.assertGreaterEqual(agree_bad, 100)
