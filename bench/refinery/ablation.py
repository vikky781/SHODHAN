#!/usr/bin/env python
"""Informational ablation on the SYNTHETIC scheduling (R3), blend/changeover (R4) and facility (R5) models:
{no structure tightening, with tightening} x {cuts off, cuts on}, with the weak big-M variants included. Python standard library only.

SYNTHETIC: structure follows textbook formulations; not plant or MRPL data.

  python bench/refinery/ablation.py --out DIR [--exe PATH] [--sizes small,medium] [--seed 1] [--time-limit 60] [--families R3,R4,R5]

For each model and configuration: the root bound (a run limited to the root node: the LP bound, or the bound after the cut loop when cuts
are on), the final status, objective, nodes and time of the full run. "Root gap closed" is measured against the plain configuration
(no tightening, no cuts) and the final objective of the model when some configuration proved it optimal. If the structure-based tightening
does nothing measurable, the summary says so. No claim beyond these models.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench"))

import generate  # noqa: E402
from run_set import parse_output  # noqa: E402

CONFIGS = [("plain", "off", "off"), ("tightening", "on", "off"), ("cuts", "off", "on"), ("tightening+cuts", "on", "on")]


def run(exe, mps, structure, cuts, limit, node_limit=None):
    cmd = [exe, "solve", mps, "--time-limit", str(limit), "--structure", structure, "--cuts", cuts]
    if node_limit is not None:
        cmd += ["--node-limit", str(node_limit)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit + 120)
        info = parse_output(p.stdout)
    except subprocess.TimeoutExpired:
        info = parse_output("")
        info["status"] = "Timeout"
    info["seconds"] = time.time() - t0
    return info


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "release", "shodhan"))
    ap.add_argument("--sizes", default="small,medium")
    ap.add_argument("--families", default="R3,R4,R5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args(argv)
    exe = os.path.abspath(args.exe)
    if not os.path.exists(exe) and os.path.exists(exe + ".exe"):
        exe += ".exe"
    models_dir = os.path.join(args.out, "models")
    names = []
    for size in args.sizes.split(","):
        for fam in args.families.split(","):
            names.append(generate.generate(fam, size, args.seed, models_dir))
            names.append(generate.generate(fam, size, args.seed, models_dir, loose=generate.WEAK_LOOSE))
    print("SYNTHETIC ablation: structure follows textbook formulations; not plant or MRPL data")
    rows = []
    summary = {}
    for name in names:
        mps = os.path.join(models_dir, name + ".mps")
        results = {}
        for label, structure, cuts in CONFIGS:
            root = run(exe, mps, structure, cuts, args.time_limit, node_limit=1)
            full = run(exe, mps, structure, cuts, args.time_limit)
            results[label] = (root, full)
        objective = None
        for label, (root, full) in results.items():
            if full["status"] == "Optimal":
                objective = num(full["objective"])
                break
        base = num(results["plain"][0]["bound"])
        for label, (root, full) in results.items():
            rb = num(root["bound"])
            closed = ""
            if objective is not None and base is not None and rb is not None and abs(objective - base) > 1e-9:
                closed = "%.1f" % (100.0 * (rb - base) / (objective - base))
            rows.append([name, label, root["bound"], closed, full["status"], full["objective"], full["nodes"], "%.3f" % full["seconds"]])
            summary.setdefault(label, []).append((closed, full["nodes"], full["seconds"], full["status"]))
            print("%-20s %-16s root bound %-16s gap closed %-7s %-9s obj %-14s nodes %-8s %7.2fs" %
                  (name, label, root["bound"] or "-", (closed + "%") if closed else "-", full["status"], full["objective"] or "-", full["nodes"] or "-", full["seconds"]))
            sys.stdout.flush()
    path = os.path.join(args.out, "ablation.csv")
    with open(path, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["name", "configuration", "root_bound", "root_gap_closed_percent", "status", "objective", "nodes", "seconds"])
        w.writerows(rows)
    print("\nsummary over %d models (root gap closed relative to the plain configuration, mean over the models where it is defined):" % len(names))
    means = {}
    for label, items in summary.items():
        vals = [float(c) for c, _, _, _ in items if c]
        nodes = [int(n) for _, n, _, st in items if n and st == "Optimal"]
        secs = [s for _, _, s, _ in items]
        means[label] = sum(vals) / len(vals) if vals else 0.0
        print("  %-16s mean root gap closed %6.2f%% (%d models), mean nodes %s over %d optimal runs, total time %.1f s" %
              (label, means[label], len(vals), ("%.1f" % (sum(nodes) / len(nodes))) if nodes else "-", len(nodes), sum(secs)))
    if abs(means.get("tightening", 0.0)) < 0.05 and abs(means.get("tightening+cuts", 0.0) - means.get("cuts", 0.0)) < 0.05:
        print("the structure-based tightening changed nothing measurable on these models")
    print("results in " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
