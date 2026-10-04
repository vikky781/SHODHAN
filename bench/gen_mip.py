#!/usr/bin/env python
"""Writes seeded MILP instances as MPS files (Python standard library only).

These are generated test instances, not benchmark instances: they say nothing about performance on real
problems. Families:

  knapsack N M SEED   multi-dimensional 0-1 knapsack: N items, M capacity rows (written as a minimization)
  setcover N M SEED   set cover: N sets, M elements
  setcover_sparse N M K SEED  set cover, every set covers K random elements
  facility F C SEED   capacitated facility location, F facilities, C customers (F*C + C + F rows)
  parity K SEED       K integer columns and one equation 2 sum(a_j x_j) = odd: the LP relaxation is feasible,
                      the MILP is infeasible (proved only by branching)

Usage: python bench/gen_mip.py FAMILY ARGS... OUT.mps
"""

import random
import sys


def write_mps(path, name, rows, cols, rhs, bounds, ranges=None):
    """rows: list of (name, type); cols: list of (name, is_int, {row: coef}); bounds: list of (type, col, value)."""
    out = ["NAME          %s" % name, "ROWS"]
    for rname, rtype in rows:
        out.append(" %s  %s" % (rtype, rname))
    out.append("COLUMNS")
    in_int = False
    marker = 0
    for cname, is_int, coefs in cols:
        if is_int and not in_int:
            out.append("    MARKER                 'MARKER'                 'INTORG'")
            in_int = True
        if not is_int and in_int:
            out.append("    MARKER                 'MARKER'                 'INTEND'")
            in_int = False
        for rname, v in coefs.items():
            out.append("    %-8s  %-8s  %.15g" % (cname, rname, v))
        marker += 1
    if in_int:
        out.append("    MARKER                 'MARKER'                 'INTEND'")
    out.append("RHS")
    for rname, v in rhs.items():
        out.append("    RHS       %-8s  %.15g" % (rname, v))
    if bounds:
        out.append("BOUNDS")
        for btype, cname, v in bounds:
            if v is None:
                out.append(" %s BND       %s" % (btype, cname))
            else:
                out.append(" %s BND       %-8s  %.15g" % (btype, cname, v))
    out.append("ENDATA")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def knapsack(n, m, seed, path):
    rng = random.Random(seed)
    rows = [("COST", "N")] + [("CAP%d" % i, "L") for i in range(m)]
    weights = [[rng.randint(10, 99) for _ in range(n)] for _ in range(m)]
    values = [sum(weights[i][j] for i in range(m)) // m + rng.randint(-10, 10) for j in range(n)]
    cols = []
    for j in range(n):
        coefs = {"COST": -float(values[j])}
        for i in range(m):
            coefs["CAP%d" % i] = float(weights[i][j])
        cols.append(("X%d" % j, True, coefs))
    rhs = {"CAP%d" % i: float(int(0.4 * sum(weights[i]))) for i in range(m)}
    write_mps(path, "KNAPSACK_%d_%d_%d" % (n, m, seed), rows, cols, rhs, [("BV", "X%d" % j, None) for j in range(n)])


def setcover(n, m, seed, path):
    rng = random.Random(seed)
    rows = [("COST", "N")] + [("E%d" % i, "G") for i in range(m)]
    covers = [[i for i in range(m) if rng.random() < 0.15] for _ in range(n)]
    for i in range(m):  # every element is covered by at least one set
        if not any(i in c for c in covers):
            covers[rng.randrange(n)].append(i)
    cols = []
    for j in range(n):
        coefs = {"COST": float(rng.randint(1, 20))}
        for i in covers[j]:
            coefs["E%d" % i] = 1.0
        cols.append(("S%d" % j, True, coefs))
    write_mps(path, "SETCOVER_%d_%d_%d" % (n, m, seed), rows, cols, {"E%d" % i: 1.0 for i in range(m)}, [("BV", "S%d" % j, None) for j in range(n)])


def parity(k, seed, path):
    rng = random.Random(seed)
    rows = [("COST", "N"), ("PAR", "E")]
    cols = []
    for j in range(k):
        cols.append(("X%d" % j, True, {"COST": float(rng.randint(1, 9)), "PAR": 2.0 * rng.randint(1, 3)}))
    rhs = {"PAR": 2.0 * rng.randint(2, 6) + 1.0}
    write_mps(path, "PARITY_%d_%d" % (k, seed), rows, cols, rhs, [("UP", "X%d" % j, 6.0) for j in range(k)])


def setcover_sparse(n, m, per_set, seed, path):
    """Set cover where every set covers `per_set` random elements (sparse: nnz = n * per_set)."""
    rng = random.Random(seed)
    rows = [("COST", "N")] + [("E%d" % i, "G") for i in range(m)]
    covers = [rng.sample(range(m), min(per_set, m)) for _ in range(n)]
    for i in range(m):
        if not any(i in c for c in covers):
            covers[rng.randrange(n)].append(i)
    cols = []
    for j in range(n):
        coefs = {"COST": float(rng.randint(1, 20))}
        for i in covers[j]:
            coefs["E%d" % i] = 1.0
        cols.append(("S%d" % j, True, coefs))
    write_mps(path, "SCSPARSE_%d_%d_%d" % (n, m, seed), rows, cols, {"E%d" % i: 1.0 for i in range(m)}, [("BV", "S%d" % j, None) for j in range(n)])


def facility(f, c, seed, path):
    """Capacitated facility location with F facilities and C customers: open[f] binary, x[f][c] in [0,1] continuous
    (share of customer c served by f), rows: demand (C), capacity (F) and the strong links x[f][c] <= open[f] (F*C)."""
    rng = random.Random(seed)
    dem = [rng.randint(5, 35) for _ in range(c)]
    total = sum(dem)
    cap = [int(total * rng.uniform(2.0, 3.5) / f) + 1 for _ in range(f)]
    rows = [("COST", "N")] + [("DEM%d" % k, "E") for k in range(c)] + [("CAP%d" % i, "L") for i in range(f)]
    rows += [("LNK%d_%d" % (i, k), "L") for i in range(f) for k in range(c)]
    cols = []
    for i in range(f):
        coefs = {"COST": float(rng.randint(300, 900))}
        coefs["CAP%d" % i] = -float(cap[i])
        for k in range(c):
            coefs["LNK%d_%d" % (i, k)] = -1.0
        cols.append(("Y%d" % i, True, coefs))
    for i in range(f):
        for k in range(c):
            coefs = {"COST": float(rng.randint(1, 40) * dem[k]) / 10.0, "DEM%d" % k: 1.0, "CAP%d" % i: float(dem[k]), "LNK%d_%d" % (i, k): 1.0}
            cols.append(("X%d_%d" % (i, k), False, coefs))
    bounds = [("BV", "Y%d" % i, None) for i in range(f)] + [("UP", "X%d_%d" % (i, k), 1.0) for i in range(f) for k in range(c)]
    write_mps(path, "FACILITY_%d_%d_%d" % (f, c, seed), rows, cols, {"DEM%d" % k: 1.0 for k in range(c)}, bounds)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    fam, out = argv[0], argv[-1]
    nums = [int(a) for a in argv[1:-1]]
    if fam == "knapsack" and len(nums) == 3:
        knapsack(nums[0], nums[1], nums[2], out)
    elif fam == "setcover" and len(nums) == 3:
        setcover(nums[0], nums[1], nums[2], out)
    elif fam == "setcover_sparse" and len(nums) == 4:
        setcover_sparse(nums[0], nums[1], nums[2], nums[3], out)
    elif fam == "facility" and len(nums) == 3:
        facility(nums[0], nums[1], nums[2], out)
    elif fam == "parity" and len(nums) == 2:
        parity(nums[0], nums[1], out)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
