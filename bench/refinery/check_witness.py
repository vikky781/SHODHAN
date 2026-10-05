#!/usr/bin/env python
"""Checks, in exact rational arithmetic, that the witness point of a generated model (the `witness` lines of its
`.meta.txt`) is feasible for the model as KASAUTI's parser reads it, and that the objective value recorded in the sidecar
equals the exact objective of the witness. Python standard library only; uses KASAUTI's MPS parser (verify/kasauti).

SYNTHETIC: structure follows textbook formulations; not plant or MRPL data.

  python bench/refinery/check_witness.py MODEL.mps [MODEL2.mps ...]      exit 0 when every witness is feasible
"""

import os
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "verify"))

from kasauti import mps  # noqa: E402


def read_meta(path):
    witness, objective = {}, None
    with open(path, encoding="ascii") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "witness":
                witness[parts[1]] = Fraction(parts[2])
            elif line.startswith("witness_objective:"):
                objective = Fraction(line.split(":", 1)[1].strip())
    return witness, objective


def check(mps_path):
    """Returns a list of problems (empty: the witness is feasible, integral and its objective matches the sidecar)."""
    model = mps.parse_mps(mps.read_bytes(mps_path), exact=True)
    witness, objective = read_meta(mps_path[:-4] + ".meta.txt")
    problems = []
    x = [witness.get(name, Fraction(0)) for name in model.col_names]
    for name in witness:
        if name not in model.col_index:
            problems.append("witness names an unknown column %s" % name)
    for j, v in enumerate(x):
        lo, hi = model.col_lo[j], model.col_hi[j]
        if lo is not None and v < lo or hi is not None and v > hi:
            problems.append("column %s = %s outside its bounds [%s, %s]" % (model.col_names[j], v, lo, hi))
        if model.col_integer[j] and v.denominator != 1:
            problems.append("integer column %s = %s is not integral" % (model.col_names[j], v))
    act = [Fraction(0)] * model.n_rows
    for j, ent in enumerate(model.col_entries):
        if x[j] != 0:
            for i, a in ent:
                act[i] += a * x[j]
    for i, a in enumerate(act):
        lo, hi = model.row_lo[i], model.row_hi[i]
        if lo is not None and a < lo or hi is not None and a > hi:
            problems.append("row %s: activity %s outside [%s, %s]" % (model.row_names[i], a, lo, hi))
    obj = model.offset + sum((model.col_cost[j] * x[j] for j in range(model.n_cols) if x[j] != 0), Fraction(0))
    if objective is not None and obj != objective:
        problems.append("objective of the witness is %s, the sidecar says %s" % (obj, objective))
    return problems


def main(argv):
    bad = 0
    for path in argv:
        problems = check(path)
        if problems:
            bad += 1
            print("%s: %d problem(s); first: %s" % (path, len(problems), problems[0]))
        else:
            print("%s: witness feasible (exact)" % path)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
