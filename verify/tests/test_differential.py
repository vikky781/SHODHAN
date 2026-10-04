"""Differential test of the two MPS parsers: `shodhan dump-model` versus KASAUTI's own parser.

Needs the shodhan executable (environment variable SHODHAN_EXE); skipped with a visible message otherwise.
Any disagreement is a bug in one of the parsers.
"""

import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kasauti.dump import compare_dumps, dump_model  # noqa: E402
from kasauti.mps import MpsError, parse_mps  # noqa: E402
from tests.mpsgen import generate, generate_qp  # noqa: E402

EXE = os.environ.get("SHODHAN_EXE")
COUNT = int(os.environ.get("KASAUTI_DIFF_COUNT", "300"))
QP_COUNT = int(os.environ.get("KASAUTI_DIFF_QP_COUNT", "400"))


@unittest.skipUnless(EXE and os.path.exists(EXE), "SHODHAN_EXE is not set: the differential parser test is skipped")
class DifferentialTests(unittest.TestCase):
    def test_generated_models_parse_identically(self):
        agree = both_reject = 0
        mismatches = []
        with tempfile.TemporaryDirectory() as d:
            for seed in range(COUNT):
                text, desc = generate(seed)
                path = os.path.join(d, "m%d.mps" % seed)
                with open(path, "w", newline="") as f:
                    f.write(text)
                cpp = subprocess.run([EXE, "dump-model", path], capture_output=True, text=True, encoding="utf-8", errors="surrogateescape")
                try:
                    py_model = parse_mps(text.encode())
                    py_text = dump_model(py_model)
                    py_err = None
                except MpsError as e:
                    py_text, py_err = None, str(e)
                if cpp.returncode != 0:
                    if py_err is not None:
                        both_reject += 1
                    else:
                        mismatches.append("seed %d (%s): C++ rejects (%s) but KASAUTI parses" % (seed, desc, cpp.stderr.strip()[:150]))
                    continue
                if py_err is not None:
                    mismatches.append("seed %d (%s): KASAUTI rejects (%s) but C++ parses" % (seed, desc, py_err[:150]))
                    continue
                diffs = compare_dumps(cpp.stdout, py_text, py_model.row_scale)
                if diffs:
                    mismatches.append("seed %d (%s): %s" % (seed, desc, "; ".join(diffs[:3])))
                else:
                    agree += 1
        print("differential parser test: %d models, %d identical, %d rejected by both, %d mismatches" % (COUNT, agree, both_reject, len(mismatches)))
        for m in mismatches[:20]:
            print("  MISMATCH " + m)
        self.assertEqual(mismatches, [])
        self.assertGreaterEqual(agree, 200)

    def test_generated_quadratic_models_parse_identically(self):
        agree = both_reject = 0
        valid_q = 0
        mismatches = []
        with tempfile.TemporaryDirectory() as d:
            for seed in range(QP_COUNT):
                text, desc = generate_qp(seed)
                path = os.path.join(d, "q%d.qps" % seed)
                with open(path, "w", newline="") as f:
                    f.write(text)
                cpp = subprocess.run([EXE, "dump-model", path], capture_output=True, text=True, encoding="utf-8", errors="surrogateescape")
                try:
                    py_model = parse_mps(text.encode())
                    py_text = dump_model(py_model)
                    py_err = None
                except MpsError as e:
                    py_text, py_err = None, str(e)
                if cpp.returncode != 0:
                    if py_err is not None:
                        both_reject += 1
                    else:
                        mismatches.append("seed %d (%s): C++ rejects (%s) but KASAUTI parses" % (seed, desc, cpp.stderr.strip()[:150]))
                    continue
                if py_err is not None:
                    mismatches.append("seed %d (%s): KASAUTI rejects (%s) but C++ parses" % (seed, desc, py_err[:150]))
                    continue
                diffs = compare_dumps(cpp.stdout, py_text, py_model.row_scale)
                if diffs:
                    mismatches.append("seed %d (%s): %s" % (seed, desc, "; ".join(diffs[:3])))
                else:
                    agree += 1
                    valid_q += 1 if py_model.n_quad else 0
        print("differential QP parser test: %d models, %d identical (%d with a quadratic term), %d rejected by both, %d mismatches"
              % (QP_COUNT, agree, valid_q, both_reject, len(mismatches)))
        for m in mismatches[:20]:
            print("  MISMATCH " + m)
        self.assertEqual(mismatches, [])
        self.assertGreaterEqual(valid_q, 200)
        self.assertGreater(both_reject, 0)



if __name__ == "__main__":
    unittest.main()
