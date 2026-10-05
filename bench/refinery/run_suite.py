#!/usr/bin/env python
"""Generates the synthetic refinery suite, solves every model with SHODHAN, writes a certificate for each, verifies it with
KASAUTI and writes a CSV. Python standard library only.

SYNTHETIC: structure follows textbook formulations; not plant or MRPL data.

  python bench/refinery/run_suite.py --out DIR [--exe PATH] [--sizes small,medium,large] [--seed 1] [--time-limit 60]
                                      [--kasauti-timeout 600] [-- EXTRA SOLVE OPTIONS]

For every model: family, size, big-M variant, status, objective, best bound, gap, nodes, time, presolve and cut statistics, the
objective of the generator's witness point (a feasible point known by construction), and the KASAUTI verdict. KASAUTI runs in exact
mode when the model has few nonzeros and otherwise in float mode (with its explicit warning that the result is not an exact proof).

A model counts as WRONG when KASAUTI says FAIL, or when SHODHAN reports Optimal with an objective WORSE than the witness (the witness
is feasible, so the optimum cannot be worse), or when the solver crashes. Wrong answers are the first thing to diagnose.
Exit status: 0 when nothing is wrong or failed, 1 otherwise.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench"))

import generate  # noqa: E402
from run_set import LIMIT_STATUSES, parse_output  # noqa: E402

COLUMNS = ["name", "family", "size", "variant", "rows", "cols", "binaries", "status", "objective", "best_bound", "gap_percent", "nodes",
           "iterations", "seconds", "presolve_size", "cuts_added", "cuts_kept", "root_bound_no_cuts", "root_bound_cuts", "root_gap_closed_percent",
           "witness_objective", "kasauti", "kasauti_mode", "exit_code", "note"]


def read_meta(path):
    meta = {}
    with open(path, encoding="ascii") as f:
        for line in f:
            if ":" in line and not line.startswith("witness "):
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


def kasauti(mps_path, cert_path, timeout):
    """Returns (verdict, mode, note)."""
    cmd = [sys.executable, "-m", "kasauti", mps_path, cert_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(ROOT, "verify"), timeout=timeout)
    except subprocess.TimeoutExpired:
        return "KASAUTI_TIMEOUT", "", "kasauti exceeded %d s" % timeout
    verdicts = re.findall(r"VERDICT:\s+(\S+)", p.stdout)
    mode = "float" if "using float mode" in p.stdout or "mode float" in p.stdout else "exact"
    return (verdicts[-1] if verdicts else "NOVERDICT(%d)" % p.returncode), mode, ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "release", "shodhan"))
    ap.add_argument("--sizes", default="small,medium,large")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--kasauti-timeout", type=float, default=600.0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("extra", nargs="*", help="options passed to `shodhan solve` (after --)")
    args = ap.parse_args(argv)
    exe = os.path.abspath(args.exe)
    if not os.path.exists(exe) and os.path.exists(exe + ".exe"):
        exe += ".exe"
    if not os.path.exists(exe):
        print("executable not found: " + exe)
        return 1
    args.out = os.path.abspath(args.out)  # KASAUTI runs from verify/, so every path must be absolute
    models_dir = os.path.join(args.out, "models")
    certs_dir = os.path.join(args.out, "certs")
    os.makedirs(certs_dir, exist_ok=True)
    names = generate.suite(models_dir, args.sizes.split(","), args.seed)
    print("SYNTHETIC suite: structure follows textbook formulations; not plant or MRPL data")
    print("generated %d models in %s" % (len(names), models_dir))
    rows = []
    counts = {"solved": 0, "limit": 0, "failed": 0, "wrong": 0}
    for name in names:
        mps = os.path.join(models_dir, name + ".mps")
        meta = read_meta(os.path.join(models_dir, name + ".meta.txt"))
        cert = os.path.join(certs_dir, name + ".json")
        cmd = [exe, "solve", mps, "--time-limit", str(args.time_limit), "--write-cert", cert] + args.extra
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.time_limit + 120)
            info = parse_output(p.stdout)
            code = p.returncode
        except subprocess.TimeoutExpired:
            info = parse_output("")
            info["status"] = "Timeout"
            code = -1
        secs = time.time() - t0
        status = info["status"] or "NoStatus"
        note = []
        verdict, mode = "", ""
        if os.path.exists(cert):
            verdict, mode, knote = kasauti(mps, cert, args.kasauti_timeout)
            if knote:
                note.append(knote)
        wrong = False
        if verdict == "FAIL":
            wrong = True
            note.append("KASAUTI FAIL")
        if status == "Optimal":
            try:
                obj, wit = float(info["objective"]), float(meta["witness_objective"])
                if meta.get("sense", "min") == "min" and obj > wit + 1e-6 * (1 + abs(wit)):
                    wrong = True
                    note.append("Optimal objective worse than the witness (%s > %s)" % (obj, wit))
            except (ValueError, KeyError):
                pass
        if code in (-1,) or status in ("NoStatus",) and code not in (0, 4):
            note.append("no result / crash (exit %s)" % code)
            counts["failed"] += 1
        elif status == "Optimal":
            counts["solved"] += 1
        elif status in LIMIT_STATUSES:
            counts["limit"] += 1
        else:
            counts["failed"] += 1
            note.append("status " + status)
        if wrong:
            counts["wrong"] += 1
        parts = name.split("_")
        rows.append([name, meta.get("family", ""), parts[1], "weak" if name.endswith("_weak") else "tight", meta.get("rows", ""), meta.get("cols", ""),
                     meta.get("binaries", ""), status, info["objective"], info["bound"], info["gap"], info["nodes"], info["iterations"], "%.3f" % secs,
                     info["presolve"], info["cuts_added"], info["cuts_kept"], info["root_no_cuts"], info["root_cuts"], info["gap_closed"],
                     meta.get("witness_objective", ""), verdict, mode, code, "; ".join(note)])
        print("%-22s %-10s obj %-16s bound %-16s gap %-7s nodes %-8s %7.2fs  %-22s %s" % (name, status, info["objective"] or "-", info["bound"] or "-",
              (info["gap"] + "%") if info["gap"] else "-", info["nodes"] or "-", secs, verdict or "-", "; ".join(note)))
        sys.stdout.flush()
    path = args.csv or os.path.join(args.out, "suite.csv")
    with open(path, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    verdict_counts = {}
    for r in rows:
        verdict_counts[r[COLUMNS.index("kasauti")] or "-"] = verdict_counts.get(r[COLUMNS.index("kasauti")] or "-", 0) + 1
    print("\nsolved %d, stopped at a limit %d, failed %d, WRONG %d (of %d models); KASAUTI verdicts: %s; results in %s" %
          (counts["solved"], counts["limit"], counts["failed"], counts["wrong"], len(rows), dict(sorted(verdict_counts.items())), path))
    return 0 if counts["wrong"] == 0 and counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
