#!/usr/bin/env python
"""OPTIONAL external reference cross-check: solves the same MPS files with HiGHS (through the `highspy` package, if it is
installed in the current Python environment) and compares the objective values and statuses with a SHODHAN results CSV
(`run_suite.py` output). It is a correctness cross-check, NOT a speed comparison: no time is compared and no speed statement is made.

This is the only file in the repository that may import `highspy`. It is a reference script: never imported by the solver, the
generators or KASAUTI, never required, and listed explicitly in the allow-list of scripts/check_deps.py.

  python bench/refinery/crosscheck_highs.py RESULTS.csv MODELS_DIR [--rel-tol 1e-6] [--time-limit 120]

Without highspy it prints "skipped: highspy not installed" and exits 0.
"""

import argparse
import csv
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("models_dir")
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    ap.add_argument("--time-limit", type=float, default=120.0)
    args = ap.parse_args(argv)
    try:
        import highspy  # noqa: F401  (external reference solver, optional)
    except ImportError:
        print("skipped: highspy not installed")
        return 0
    import highspy
    mismatches = compared = skipped = 0
    with open(args.results, newline="", encoding="ascii") as f:
        for row in csv.DictReader(f):
            if row["status"] != "Optimal":
                skipped += 1
                continue
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("time_limit", args.time_limit)
            h.readModel(os.path.join(args.models_dir, row["name"] + ".mps"))
            h.run()
            status = h.modelStatusToString(h.getModelStatus())
            info = h.getInfo()
            ref = info.objective_function_value
            ours = float(row["objective"])
            compared += 1
            if status != "Optimal" or abs(ours - ref) > args.rel_tol * (1.0 + abs(ref)):
                mismatches += 1
                print("MISMATCH %-22s SHODHAN %s Optimal, HiGHS %s %.12g" % (row["name"], row["objective"], status, ref))
    print("HiGHS cross-check: %d compared, %d mismatches, %d not compared (SHODHAN status not Optimal)" % (compared, mismatches, skipped))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
