#!/usr/bin/env python
"""Writes synthetic refinery / process case-study models (MPS + .meta.txt sidecar). Python standard library only.

SYNTHETIC: structure follows textbook formulations; not plant or MRPL data.

  python bench/refinery/generate.py --family R3 --size small --seed 1 [--loose 50] [--ties 0.3] [--spread 1] --out DIR
  python bench/refinery/generate.py --suite DIR [--sizes small,medium,large] [--seed 1]

The suite holds, for every size, R1 R2 R3 R4 R5 and the weak big-M variants of R3 R4 R5 (loose = 50). The files
are byte-identical for the same arguments on every platform (own random generator, exact dyadic data).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from families import FAMILIES, SIZES  # noqa: E402

WEAK_LOOSE = 50


def generate(family, size, seed, out, loose=1, ties=0.0, spread=0):
    model, knobs = FAMILIES[family](size, seed, loose=loose, ties=ties, spread=spread)
    model.write(out, model.name, knobs, seed)
    return model.name


def suite(out, sizes, seed):
    names = []
    for size in sizes:
        for family in ("R1", "R2", "R3", "R4", "R5"):
            names.append(generate(family, size, seed, out))
        for family in ("R3", "R4", "R5"):
            names.append(generate(family, size, seed, out, loose=WEAK_LOOSE))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES))
    ap.add_argument("--size", choices=SIZES + ("huge",), default="small")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--loose", type=int, default=1)
    ap.add_argument("--ties", type=float, default=0.0)
    ap.add_argument("--spread", type=int, default=0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--suite", metavar="DIR")
    ap.add_argument("--sizes", default="small,medium,large")
    args = ap.parse_args(argv)
    if args.suite:
        names = suite(args.suite, args.sizes.split(","), args.seed)
        print("wrote %d models to %s" % (len(names), args.suite))
        return 0
    if not args.family:
        ap.error("--family or --suite is required")
    print(generate(args.family, args.size, args.seed, args.out, args.loose, args.ties, args.spread))
    return 0


if __name__ == "__main__":
    sys.exit(main())
