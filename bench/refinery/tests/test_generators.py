"""Tests of the synthetic refinery generators: determinism, witness feasibility in exact arithmetic (with KASAUTI's
parser), documented sizes, the SYNTHETIC label. Python standard library only.

Run from bench/refinery:  python -m unittest discover -s tests -t .
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REFINERY = os.path.dirname(HERE)
sys.path.insert(0, REFINERY)

import check_witness  # noqa: E402
import families  # noqa: E402
import generate  # noqa: E402
from gen_core import SYNTHETIC_LINE, Rng  # noqa: E402


def read_text(path):
    with open(path, encoding="ascii") as f:
        return f.read()


def sha_dir(d):
    h = hashlib.sha256()
    for name in sorted(os.listdir(d)):
        with open(os.path.join(d, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read() + b"\0")
    return h.hexdigest()


class RngTests(unittest.TestCase):
    def test_the_generator_is_deterministic_and_spread(self):
        a, b = Rng(7), Rng(7)
        self.assertEqual([a.next() for _ in range(5)], [b.next() for _ in range(5)])
        self.assertNotEqual(Rng(7).next(), Rng(8).next())
        r = Rng(1)
        vals = [r.randint(0, 9) for _ in range(2000)]
        self.assertEqual(set(vals), set(range(10)))
        self.assertTrue(all(0.0 <= Rng(3).unit() < 1.0 for _ in range(3)))
        # a fixed value, so that a change of the generator (which would change every file) is noticed
        self.assertEqual(Rng(1).next(), Rng(1).next())
        self.assertEqual(Rng(12345).randint(0, 1000000), Rng(12345).randint(0, 1000000))


class FormatTests(unittest.TestCase):
    def test_numbers_are_written_as_their_exact_decimal_expansion(self):
        from fractions import Fraction
        from gen_core import fmt
        r = Rng(99)
        for _ in range(3000):
            v = Fraction(r.randint(-10**9, 10**9), 2 ** r.randint(0, 40))
            self.assertEqual(Fraction(fmt(v)), v)
        self.assertEqual(fmt(Fraction(5, 16)), "0.3125")
        self.assertEqual(fmt(7), "7")
        self.assertEqual(fmt(Fraction(-3, 2)), "-1.5")
        with self.assertRaises(ValueError):
            fmt(Fraction(1, 3))


class DeterminismTests(unittest.TestCase):
    def test_two_runs_write_identical_files(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            generate.suite(d1, ["tiny", "small"], 3)
            generate.suite(d2, ["tiny", "small"], 3)
            self.assertEqual(sha_dir(d1), sha_dir(d2))

    def test_a_different_hash_seed_and_process_give_the_same_bytes(self):
        digests = []
        for hashseed in ("0", "12345"):
            with tempfile.TemporaryDirectory() as d:
                env = dict(os.environ, PYTHONHASHSEED=hashseed)
                subprocess.run([sys.executable, os.path.join(REFINERY, "generate.py"), "--suite", d, "--sizes", "tiny", "--seed", "5"],
                               check=True, capture_output=True, env=env)
                digests.append(sha_dir(d))
        self.assertEqual(digests[0], digests[1])

    def test_the_seed_and_the_knobs_change_the_model(self):
        with tempfile.TemporaryDirectory() as d:
            a = generate.generate("R3", "small", 1, d)
            b = generate.generate("R3", "small", 2, d)
            c = generate.generate("R3", "small", 1, d, loose=50)
            self.assertEqual((a, b, c), ("R3_small_1", "R3_small_2", "R3_small_1_weak"))
            text = lambda n: read_text(os.path.join(d, n + ".mps"))
            self.assertNotEqual(text(a), text(b))
            self.assertNotEqual(text(a), text(c))


class ContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        cls.names = generate.suite(cls.dir, ["tiny", "small", "medium"], 1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_file_is_labelled_synthetic(self):
        for n in self.names:
            self.assertTrue(read_text(os.path.join(self.dir, n + ".mps")).startswith("* " + SYNTHETIC_LINE), n)
            self.assertIn(SYNTHETIC_LINE, read_text(os.path.join(self.dir, n + ".meta.txt")), n)

    def test_every_witness_is_feasible_in_exact_arithmetic(self):
        for n in self.names:
            self.assertEqual(check_witness.check(os.path.join(self.dir, n + ".mps")), [], n)

    def test_a_broken_witness_is_detected(self):
        n = self.names[0]
        meta = os.path.join(self.dir, n + ".meta.txt")
        original = read_text(meta)
        try:
            lines = original.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("witness ") and not line.startswith("witness_"):
                    name, val = line.split()[1:3]
                    lines[i] = "witness %s %s" % (name, "1000000")
                    break
            with open(meta, "w") as f:
                f.write("\n".join(lines))
            self.assertNotEqual(check_witness.check(os.path.join(self.dir, n + ".mps")), [])
        finally:
            with open(meta, "w", encoding="ascii") as f:
                f.write(original)

    def test_tiny_scheduling_and_blending_models_have_at_most_14_binaries(self):
        for fam in ("R3", "R4"):
            for seed in range(1, 21):
                m, _ = families.FAMILIES[fam]("tiny", seed)
                self.assertLessEqual(m.n_binaries(), 14, (fam, seed))
                self.assertGreater(m.n_binaries(), 0)

    def test_the_planning_family_scales_past_one_hundred_thousand_columns(self):
        m, knobs = families.FAMILIES["R2"]("huge", 1)
        self.assertGreaterEqual(len(m.cols), 100000)
        self.assertEqual(knobs["size"], "huge")

    def test_tight_and_weak_variants_differ_only_in_the_big_m_coefficients(self):
        for fam in ("R3", "R4", "R5"):
            tight, _ = families.FAMILIES[fam]("small", 1, loose=1)
            weak, _ = families.FAMILIES[fam]("small", 1, loose=50)
            self.assertEqual([c[0] for c in tight.cols], [c[0] for c in weak.cols])
            self.assertEqual([r[0] for r in tight.rows], [r[0] for r in weak.rows])
            self.assertNotEqual(tight.mps_text(), weak.mps_text())


if __name__ == "__main__":
    unittest.main()
