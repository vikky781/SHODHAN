#!/usr/bin/env python3
"""Compatibility wrapper: the LP set runner was generalized to LPs and MILPs in bench/run_set.py.

Usage is unchanged (python bench/run_lp_set.py DIR [options]); the default CSV name stays lp_results.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_set  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--csv" not in argv:
        argv = argv[:1] + ["--csv", "lp_results.csv"] + argv[1:] if argv and not argv[0].startswith("-") else ["--csv", "lp_results.csv"] + argv
    sys.exit(run_set.main(argv))
