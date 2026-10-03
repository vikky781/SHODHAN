#!/usr/bin/env python3
"""End-to-end checks of the shodhan executable: exit codes and key output.

Usage: cli_check.py <path-to-shodhan> <models-dir>
"""
import os
import subprocess
import sys
import tempfile

failures = []


def run(exe, *args):
    p = subprocess.run([exe, *args], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def check(name, cond, detail=""):
    if cond:
        print("[ ok ] " + name)
    else:
        print("[FAIL] %s %s" % (name, detail))
        failures.append(name)


SOLVED_BY_PRESOLVE = (
    "NAME s\nROWS\n N COST\n G R1\nCOLUMNS\n X COST 1 R1 1\nRHS\n RHS R1 3\n"
    "BOUNDS\n FX BND X 3\nENDATA\n"
)
INFEASIBLE = (
    "NAME inf\nROWS\n N COST\n G R1\n L R2\nCOLUMNS\n X COST 1 R1 1\n X R2 1\n"
    "RHS\n RHS R1 5 R2 3\nENDATA\n"
)


def main():
    exe, models = sys.argv[1], sys.argv[2]
    lp = os.path.join(models, "tiny_lp.mps")
    mip = os.path.join(models, "tiny_mip.mps")
    pre = os.path.join(models, "tiny_presolve.mps")

    rc, out, _ = run(exe, "--version")
    check("version exits 0", rc == 0 and out.startswith("shodhan "), repr((rc, out)))

    rc, out, _ = run(exe, "--help")
    check("help exits 0", rc == 0 and "Usage" in out and "presolve" in out, repr((rc, out)))

    rc, _, err = run(exe)
    check("no arguments is a usage error (1)", rc == 1 and "Usage" in err, repr((rc, err)))

    rc, _, err = run(exe, "frobnicate")
    check("unknown command is a usage error (1)", rc == 1 and "unknown command" in err)

    rc, _, err = run(exe, "info")
    check("info without file is a usage error (1)", rc == 1)

    # ---- info ----
    rc, out, _ = run(exe, "info", lp)
    check("info on LP exits 0", rc == 0, repr(rc))
    check("info reports shape", "Rows:           4" in out and "Columns:        3" in out, out)
    check("info reports sense", "maximize" in out, out)
    check("info reports ranged row", "1 ranged" in out, out)
    check("info reports offset", "Objective offset: 10" in out, out)
    check("info prints the scaling-quality line", "Scaling:        coefficient ratio 4 -> 2" in out, out)

    rc, out, _ = run(exe, "info", mip)
    check("info on MIP exits 0", rc == 0, repr(rc))
    check("info reports column types", "1 continuous, 2 integer, 1 binary" in out, out)

    # ---- presolve ----
    rc, out, _ = run(exe, "presolve", pre)
    check("presolve exits 0", rc == 0, repr(rc))
    check("presolve reports status", "status:       Reduced" in out, out)
    check("presolve reports before/after sizes",
          "rows:         6 -> 3" in out and "columns:      7 -> 3" in out and "nonzeros:     12 -> 8" in out, out)
    check("presolve reports per-reduction counts",
          "empty rows                 1" in out and "doubleton equations        1" in out
          and "fixed columns              1" in out and "singleton rows             1" in out, out)
    check("presolve reports scaling ratio", "coefficient ratio:" in out and "after scaling" in out, out)
    check("presolve reports the time", " ms" in out and "time:" in out, out)

    rc, out, _ = run(exe, "presolve", mip)
    check("presolve on a MIP uses MIP-safe mode", rc == 0 and "MIP-safe" in out, out)
    rc, out, _ = run(exe, "presolve", lp, "--mip")
    check("presolve --mip forces MIP-safe mode", rc == 0 and "MIP-safe" in out, out)
    rc, out, _ = run(exe, "presolve", lp, "--no-dual-needed")
    check("presolve --no-dual-needed is acknowledged", rc == 0 and "not reconstructed" in out, out)

    with tempfile.TemporaryDirectory() as d:
        written = os.path.join(d, "presolved.mps")
        rc, out, _ = run(exe, "presolve", pre, "--write-presolved", written)
        check("presolve --write-presolved writes a file", rc == 0 and os.path.exists(written), out)
        rc2, out2, _ = run(exe, "info", written)
        check("the presolved file reads back with the reduced size",
              rc2 == 0 and "Rows:           3" in out2 and "Columns:        3" in out2, out2)

        bad = os.path.join(d, "bad.mps")
        with open(bad, "w") as f:
            f.write("NAME bad\nROWS\n N OBJ\nCOLUMNS\n X NOPE 1\nENDATA\n")
        rc, _, err = run(exe, "info", bad)
        check("malformed file is a read error (1)",
              rc == 1 and "bad.mps:5:" in err and "ReadError" in err, repr((rc, err)))
        rc, out, err = run(exe, "solve", bad)
        check("solve on malformed file exits 1, not 2", rc == 1 and "NotImplemented" not in out)
        rc, _, err = run(exe, "presolve", bad)
        check("presolve on malformed file is a read error (1)", rc == 1 and "bad.mps:5:" in err)

        solved = os.path.join(d, "solved.mps")
        with open(solved, "w") as f:
            f.write(SOLVED_BY_PRESOLVE)
        rc, out, _ = run(exe, "presolve", solved)
        check("presolve can solve a model completely", rc == 0 and "SolvedByPresolve" in out, out)
        rc, out, _ = run(exe, "solve", solved)
        check("solve reports a model solved by presolve honestly",
              rc == 0 and "Optimal (solved by presolve)" in out and "Objective: 3" in out
              and "KKT check (tolerance 1e-9): passed" in out, out)

        infeasible = os.path.join(d, "infeasible.mps")
        with open(infeasible, "w") as f:
            f.write(INFEASIBLE)
        rc, out, _ = run(exe, "presolve", infeasible)
        check("presolve reports an infeasible model", rc == 0 and "status:       Infeasible" in out, out)
        rc, out, _ = run(exe, "solve", infeasible)
        check("solve reports an infeasible model proven by presolve",
              rc == 0 and "Status: Infeasible (proven by presolve)" in out, out)

        rc, out, _ = run(exe, "presolve", infeasible, "--write-presolved", os.path.join(d, "none.mps"))
        check("nothing is written when presolve proves infeasibility",
              rc == 0 and "Nothing written" in out and not os.path.exists(os.path.join(d, "none.mps")), out)

    rc, _, err = run(exe, "presolve")
    check("presolve without file is a usage error (1)", rc == 1)
    rc, _, err = run(exe, "presolve", lp, "--bogus")
    check("presolve with an unknown option is a usage error (1)", rc == 1 and "unknown option" in err, err)
    rc, _, err = run(exe, "presolve", lp, "--write-presolved")
    check("--write-presolved without a file name is a usage error (1)", rc == 1, err)
    rc, _, err = run(exe, "presolve", os.path.join(models, "does_not_exist.mps"))
    check("presolve on a missing file is a read error (1)", rc == 1 and "error" in err, err)

    # ---- solve ----
    rc, out, _ = run(exe, "solve", lp)
    check("solve on a reducible model exits 2 (not implemented)", rc == 2, repr(rc))
    check("solve prints the summary, the presolve stats and the status",
          "Rows:" in out and "Presolve" in out and "Scaling" in out and "NotImplemented" in out, out)

    rc, _, err = run(exe, "info", os.path.join(models, "does_not_exist.mps"))
    check("missing file is a read error (1)", rc == 1 and "error" in err, repr((rc, err)))

    # ---- factor-bench ----
    rc, out, err = run(exe, "factor-bench", lp)
    check("factor-bench exits 0", rc == 0, repr((rc, err)))
    check("factor-bench says it is a diagnostic, not a benchmark", "not a benchmark" in out, out)
    check("factor-bench prints the factorization statistics",
          "m:                4" in out and "nnz(B):" in out and "nnz(L):" in out and "nnz(U):" in out
          and "fill ratio:" in out and "pivots:" in out and "factor time:" in out, out)
    check("factor-bench prints the solve counters and residuals",
          "50 random solves" in out and "hypersparse path:" in out and "dense path:" in out
          and "worst relative residual, ftran:" in out and "worst relative residual, btran:" in out, out)
    check("factor-bench runs the update sequence", "Forrest-Tomlin run" in out and "updates accepted:" in out, out)
    rc, out, _ = run(exe, "factor-bench", lp, "--threshold", "0.5", "--max-updates", "5")
    check("factor-bench accepts --threshold and --max-updates",
          rc == 0 and "u = 0.5" in out and "max updates 5" in out, out)
    rc, _, err = run(exe, "factor-bench")
    check("factor-bench without file is a usage error (1)", rc == 1 and "needs a file" in err, err)
    rc, _, err = run(exe, "factor-bench", lp, "--threshold", "7")
    check("factor-bench rejects a threshold above 1", rc == 1 and "--threshold" in err, err)
    rc, _, err = run(exe, "factor-bench", lp, "--max-updates", "-3")
    check("factor-bench rejects a negative update limit", rc == 1 and "--max-updates" in err, err)
    rc, _, err = run(exe, "factor-bench", lp, "--bogus")
    check("factor-bench rejects an unknown option", rc == 1 and "unknown option" in err, err)
    rc, _, err = run(exe, "factor-bench", os.path.join(models, "does_not_exist.mps"))
    check("factor-bench on a missing file is a read error (1)", rc == 1 and "error" in err, err)

    if failures:
        print("%d CLI check(s) failed" % len(failures))
        return 1
    print("all CLI checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
