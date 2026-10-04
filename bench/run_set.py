#!/usr/bin/env python3
"""Runs `shodhan solve` on every .mps file of a directory (LPs and MILPs) and collects the results.

Standard library only. Usage:

  python bench/run_set.py DIR [--exe PATH] [--timeout SECONDS] [--csv OUT.csv]
                              [--reference REF.csv] [--rel-tol 1e-6] [-- EXTRA SOLVE OPTIONS]

For every instance the CSV gets: name, status, objective, best bound, gap (relative, percent), nodes,
iterations, seconds (wall clock of the process), the exit code and a note. LPs additionally report the
relative KKT residuals. A solve that exceeds the timeout is recorded with status Timeout. The reference
file (given by you) has two columns, name and optimal objective; a solved instance (status Optimal) whose
objective differs from the reference by more than rel-tol * (1 + |reference|) counts as a mismatch. No
reference values are built in. Instances that stop at a limit (NodeLimit, TimeLimit, Timeout) are counted
as "limit", not as failures; every other status that is not Optimal is a failure.
Exit status: 0 when nothing failed and nothing mismatched, 1 otherwise.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

LIMIT_STATUSES = ("NodeLimit", "TimeLimit", "Timeout", "IterationLimit")


def parse_output(text):
    info = {"status": "", "objective": "", "bound": "", "gap": "", "nodes": "", "iterations": "",
            "primal": "", "dual": "", "compl": "", "kkt_gap": ""}
    m = re.search(r"^Status:\s+(\S+)", text, re.M)
    if m:
        info["status"] = m.group(1)
    m = re.search(r"^Objective:\s+(\S+)", text, re.M)
    if m:
        info["objective"] = m.group(1)
    m = re.search(r"^Best bound:\s+(\S+)", text, re.M)
    if m:
        info["bound"] = m.group(1)
    m = re.search(r"^Gap:\s+\S+ absolute, (\S+)% relative", text, re.M)
    if m:
        info["gap"] = m.group(1)
    m = re.search(r"^Nodes:\s+(\d+) processed", text, re.M)
    if m:
        info["nodes"] = m.group(1)
    m = re.search(r"^Iterations:\s+(\d+)", text, re.M) or re.search(r"^LP iterations:\s+(\d+)", text, re.M)
    if m:
        info["iterations"] = m.group(1)
    for key, label in (("primal", "primal infeasibility"), ("dual", "dual infeasibility"),
                       ("compl", "complementarity"), ("kkt_gap", "duality gap")):
        m = re.search(r"^\s*" + label + r"\s+\S+ abs, (\S+) rel", text, re.M)
        if m:
            info[key] = m.group(1)
    return info


def read_reference(path):
    ref = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].strip().lower() in ("name", ""):
                continue
            try:
                ref[row[0].strip()] = float(row[1])
            except ValueError:
                pass
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--exe", default=os.path.join("build", "release", "shodhan"))
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per instance")
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--reference", help="CSV with columns name,objective")
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    ap.add_argument("extra", nargs="*", help="options passed to `shodhan solve` (put them after --)")
    args = ap.parse_args(argv)

    exe = os.path.abspath(args.exe)
    if not os.path.exists(exe) and os.path.exists(exe + ".exe"):
        exe += ".exe"
    if not os.path.exists(exe):
        print("executable not found: " + exe)
        return 1
    files = sorted(f for f in os.listdir(args.directory) if f.lower().endswith(".mps"))
    if not files:
        print("no .mps files in " + args.directory)
        return 1
    ref = read_reference(args.reference) if args.reference else {}
    solved = limit = failed = mismatch = missing_ref = 0
    rows = []
    for fname in files:
        name = fname[:-4]
        cmd = [exe, "solve", os.path.join(args.directory, fname)] + args.extra
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            info = parse_output(p.stdout)
            code = p.returncode
        except subprocess.TimeoutExpired:
            info = parse_output("")
            info["status"] = "Timeout"
            code = -1
        secs = time.time() - t0
        note = ""
        status = info["status"] or "NoStatus"
        if status == "Optimal":
            solved += 1
            if ref:
                if name not in ref:
                    missing_ref += 1
                    note = "no reference"
                else:
                    try:
                        obj = float(info["objective"])
                        if abs(obj - ref[name]) > args.rel_tol * (1.0 + abs(ref[name])):
                            mismatch += 1
                            note = "MISMATCH (reference %.12g)" % ref[name]
                    except ValueError:
                        mismatch += 1
                        note = "unreadable objective"
        elif status in LIMIT_STATUSES:
            limit += 1
        else:
            failed += 1
        rows.append([name, status, info["objective"], info["bound"], info["gap"], info["nodes"], info["iterations"],
                     "%.3f" % secs, info["primal"], info["dual"], info["compl"], info["kkt_gap"], code, note])
        print("%-22s %-14s obj %-16s bound %-16s gap %-8s nodes %-7s %8.2fs %s" %
              (name, status, info["objective"] or "-", info["bound"] or "-", (info["gap"] + "%") if info["gap"] else "-",
               info["nodes"] or "-", secs, note))
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "status", "objective", "best_bound", "gap_percent", "nodes", "iterations", "seconds",
                    "primal_rel", "dual_rel", "compl_rel", "kkt_gap_rel", "exit_code", "note"])
        w.writerows(rows)
    print("\nsolved %d, stopped at a limit %d, failed (any other status) %d, mismatches %d%s; results in %s" %
          (solved, limit, failed, mismatch, (", without reference %d" % missing_ref) if ref else "", args.csv))
    return 0 if failed == 0 and mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
