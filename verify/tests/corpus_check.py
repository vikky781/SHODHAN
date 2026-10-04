"""Verifies every certificate of a corpus directory (written by shodhan_cert_corpus) with KASAUTI in BOTH
exact and float mode and prints the counts. Exit status 0 only if every certificate that is expected to pass
passes in exact mode, every certificate that certifies nothing (expect = inconclusive: an infeasibility proved
by branching, a MILP that is infeasible or unbounded) is reported INCONCLUSIVE and not as a pass, exact and
float verdicts agree, and the attempts field was recorded.

Usage: python verify/tests/corpus_check.py CORPUS_DIR
"""

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kasauti import cli  # noqa: E402
from tests.helpers import args as make_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    a = ap.parse_args()
    with open(os.path.join(a.corpus, "corpus.csv")) as f:
        manifest = list(csv.DictReader(f))
    per_family = collections.defaultdict(lambda: collections.Counter())
    attempts = collections.Counter()
    problems = []
    for row in manifest:
        name, family = row["name"], row["family"]
        expect = row.get("expect") or "pass"
        mps = os.path.join(a.corpus, name + ".mps")
        cert_path = os.path.join(a.corpus, name + ".cert.json")
        with open(cert_path) as f:
            cert = json.load(f)
        att = cert.get("attempts", {})
        if "count" not in att or "configuration" not in att:
            problems.append("%s: attempts field missing" % name)
        attempts["%s / %s attempt(s)" % (att.get("configuration"), att.get("count"))] += 1
        results = {}
        for mode in ("exact", "float"):
            code, rep = cli.verify(mps, cert_path, make_args(mode=mode))
            results[mode] = (code, rep.detail, rep.rigorous)
        ce, cf = results["exact"], results["float"]
        c = per_family[family]
        c["total"] += 1
        c[cert["status"]] += 1
        if expect == "pass":
            if ce[0] == 0:
                c["exact PASS"] += 1
                c["rigorous" if ce[2] else "tolerance-level"] += 1
            else:
                problems.append("%s: exact mode verdict %s (%s)" % (name, ce[1], family))
            if cf[0] == 0:
                c["float PASS"] += 1
        else:
            # Certifies nothing: must be INCONCLUSIVE (exit code 2) in both modes, never a pass.
            if ce[0] == 2 and cf[0] == 2:
                c["inconclusive (expected)"] += 1
            else:
                problems.append("%s: expected INCONCLUSIVE but exact mode gave %s and float mode %s (%s)" % (name, ce[1], cf[1], family))
        if ce[0] != cf[0]:
            problems.append("%s: exact (%s) and float (%s) verdicts disagree" % (name, ce[1], cf[1]))
        else:
            c["agree"] += 1
    print("KASAUTI corpus check: %d certificates" % len(manifest))
    for fam in sorted(per_family):
        c = per_family[fam]
        print("  %-24s total %3d | optimal %3d feasible %3d infeasible %3d unbounded %3d | exact PASS %3d (rigorous %3d, tolerance-checked %3d) | "
              "float PASS %3d | inconclusive as expected %3d | exact/float agree %3d"
              % (fam, c["total"], c["optimal"], c["feasible"], c["infeasible"], c["unbounded"], c["exact PASS"], c["rigorous"], c["tolerance-level"],
                 c["float PASS"], c["inconclusive (expected)"], c["agree"]))
    tot = sum(c["total"] for c in per_family.values())
    mips = sum(c["total"] for fam, c in per_family.items() if fam.startswith("mip "))
    print("  totals: %d certificates (%d MILP), exact PASS %d, float PASS %d, inconclusive as expected %d, verdicts agree %d, rigorous %d, tolerance-checked %d" % (
        tot, mips, sum(c["exact PASS"] for c in per_family.values()), sum(c["float PASS"] for c in per_family.values()),
        sum(c["inconclusive (expected)"] for c in per_family.values()), sum(c["agree"] for c in per_family.values()),
        sum(c["rigorous"] for c in per_family.values()), sum(c["tolerance-level"] for c in per_family.values())))
    print("  attempts (configuration / count):")
    for k, v in sorted(attempts.items(), key=lambda kv: -kv[1]):
        print("    %4d  %s" % (v, k))
    for p in problems:
        print("PROBLEM: " + p)
    return 1 if problems or tot < 300 else 0


if __name__ == "__main__":
    sys.exit(main())
