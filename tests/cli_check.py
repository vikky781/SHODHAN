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
UNBOUNDED = (
    "NAME unb\nROWS\n N COST\n L R1\nCOLUMNS\n X COST -1 R1 1\n Y COST -1 R1 -1\n"
    "RHS\n RHS R1 2\nENDATA\n"
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
              rc == 0 and "Status:        Optimal" in out and "Objective:     3" in out and "solved by presolve" in out
              and "KKT check on the original model" in out and "passed" in out, out)

        infeasible = os.path.join(d, "infeasible.mps")
        with open(infeasible, "w") as f:
            f.write(INFEASIBLE)
        rc, out, _ = run(exe, "presolve", infeasible)
        check("presolve reports an infeasible model", rc == 0 and "status:       Infeasible" in out, out)
        rc, out, _ = run(exe, "solve", infeasible)
        check("solve reports an infeasible model with a verified certificate and exit code 3",
              rc == 3 and "Status:        Infeasible" in out and "Farkas multipliers verified" in out, repr((rc, out)))
        rc, out, _ = run(exe, "solve", infeasible, "--no-presolve")
        check("solve --no-presolve also proves infeasibility", rc == 3 and "Status:        Infeasible" in out and "Presolve:      off" in out, out)

        unbounded = os.path.join(d, "unbounded.mps")
        with open(unbounded, "w") as f:
            f.write(UNBOUNDED)
        rc, out, _ = run(exe, "solve", unbounded)
        check("solve reports an unbounded model with a verified ray and exit code 3",
              rc == 3 and "Status:        Unbounded" in out and "improving ray verified" in out, repr((rc, out)))
        rc, out, _ = run(exe, "solve", unbounded, "--no-presolve", "--no-scaling")
        check("solve --no-presolve --no-scaling also finds the unbounded ray",
              rc == 3 and "Status:        Unbounded" in out and "Scaling:       off" in out, out)

        sol = os.path.join(d, "tiny.sol")
        rc, out, _ = run(exe, "solve", lp, "--write-sol", sol)
        check("solve --write-sol writes the objective and the nonzero columns",
              rc == 0 and os.path.exists(sol), out)
        if os.path.exists(sol):
            with open(sol) as f:
                lines = f.read().splitlines()
            check("the solution file starts with the objective line", lines and lines[0].startswith("objective "), repr(lines[:3]))
            check("the solution file lists name value pairs",
                  all(len(l.split()) == 2 for l in lines[1:]) and len(lines) >= 2, repr(lines))
        rc, out, _ = run(exe, "solve", infeasible, "--write-sol", os.path.join(d, "none.sol"))
        check("no solution file is written for an infeasible model",
              rc == 3 and "Nothing written" in out and not os.path.exists(os.path.join(d, "none.sol")), out)

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
    check("solve on an LP exits 0 with status Optimal", rc == 0 and "Status:        Optimal" in out, repr((rc, out)))
    check("solve prints the summary, iterations, time and the KKT residuals",
          "Rows:" in out and "Iterations:" in out and "Time:" in out and "primal infeasibility" in out
          and "duality gap" in out and "passed" in out, out)
    check("solve reports the objective of the maximization model", "Objective:     94" in out, out)
    for flags in (["--no-presolve"], ["--no-scaling"], ["--no-perturb"], ["--no-presolve", "--no-scaling", "--no-perturb"],
                  ["--verbose"], ["--time-limit", "30", "--iter-limit", "1000"]):
        rc, out, _ = run(exe, "solve", lp, *flags)
        check("solve " + " ".join(flags) + " gives the same objective", rc == 0 and "Objective:     94" in out, repr((rc, out[-200:])))
    rc, out, _ = run(exe, "solve", lp, "--time-limit", "0", "--no-presolve")
    check("solve --time-limit 0 reports TimeLimit with exit code 4", rc == 4 and "Status:        TimeLimit" in out, repr((rc, out)))
    rc, out, _ = run(exe, "solve", lp, "--iter-limit", "0", "--no-presolve")
    check("solve --iter-limit 0 reports IterationLimit with exit code 4 or solves trivially",
          (rc == 4 and "IterationLimit" in out) or (rc == 0 and "Optimal" in out), repr((rc, out)))
    rc, out, _ = run(exe, "solve", mip)
    check("solve on a MILP solves it: Optimal, objective 0.2, a best bound, exit 0",
          rc == 0 and "Status:        Optimal" in out and "Objective:     0.2" in out and "Best bound:" in out and "Nodes:" in out, repr((rc, out[-300:])))
    # ---- quadratic programs ----
    qp = os.path.join(models, "tiny_qp.qps")
    rc, out, _ = run(exe, "solve", qp)
    check("solve on a convex QP uses the interior-point method: Optimal, objective 1.5, KKT passed, exit 0",
          rc == 0 and "Status:        Optimal" in out and "Objective:     1.5" in out and "interior point" in out and "KKT check" in out and "passed" in out, repr((rc, out[-400:])))
    rc, out, _ = run(exe, "solve", qp, "--method", "ipm", "--ipm-tol", "1e-10")
    check("solve --method ipm --ipm-tol 1e-10 on a QP gives objective 1.5", rc == 0 and "Objective:     1.5" in out, repr((rc, out[-300:])))
    rc, out, _ = run(exe, "solve", qp, "--method", "simplex")
    check("solve --method simplex on a QP is rejected with a clear message (NotImplemented, exit 2)",
          rc == 2 and "NotImplemented" in out and "cannot solve a quadratic program" in out, repr((rc, out[-300:])))
    rc, out, _ = run(exe, "solve", os.path.join(models, "tiny_nonconvex.qps"))
    check("solve on an indefinite QP reports NonConvex with exit 3", rc == 3 and "Status:        NonConvex" in out and "not positive semidefinite" in out, repr((rc, out[-300:])))
    rc, out, _ = run(exe, "solve", os.path.join(models, "tiny_miqp.qps"))
    check("solve on a QP with integer columns reports NotImplemented (MIQP planned) with exit 2", rc == 2 and "NotImplemented" in out and "MIQP" in out, repr((rc, out[-300:])))
    rc, out, _ = run(exe, "info", qp)
    check("info shows the quadratic term and its convexity", rc == 0 and "Quadratic term: 2 nonzeros" in out and "positive semidefinite" in out, repr((rc, out[-300:])))
    rc, out, _ = run(exe, "dump-model", qp)
    check("dump-model lists the quadratic entries", rc == 0 and "qnnz\t2" in out and "quad\tX1\tX1\t1" in out, repr((rc, out[-200:])))
    rc, _, err = run(exe, "solve", qp, "--method", "barrier")
    check("solve --method barrier is a usage error (1)", rc == 1, err)
    rc, _, err = run(exe, "solve", qp, "--ipm-tol", "0")
    check("solve --ipm-tol 0 is a usage error (1)", rc == 1, err)
    gen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "gen_mip.py")
    with tempfile.TemporaryDirectory() as md:
        knap = os.path.join(md, "knap.mps")
        par = os.path.join(md, "parity.mps")
        subprocess.run([sys.executable, gen, "knapsack", "30", "3", "11", knap], check=True)
        subprocess.run([sys.executable, gen, "parity", "5", "3", par], check=True)
        cert = os.path.join(md, "k.cert.json")
        rc, out, _ = run(exe, "solve", knap, "--seed", "1", "--write-cert", cert)
        check("solve on a knapsack MILP is optimal and writes a feasible certificate with the KASAUTI command",
              rc == 0 and "Status:        Optimal" in out and "Objective:     -687" in out and "Certificate:   feasible written to" in out and "python -m kasauti" in out
              and os.path.exists(cert), repr((rc, out[-400:])))
        with open(cert) as f:
            text = f.read()
        check("the MILP certificate says optimality is not certified", '"optimality_certified": false' in text.replace(":false", ": false") or '"optimality_certified":false' in text, text[:200])
        rc, out, _ = run(exe, "solve", knap, "--seed", "1", "--node-limit", "5")
        check("solve --node-limit 5 stops with NodeLimit, says whether an incumbent exists, exit 4",
              rc == 4 and "Status:        NodeLimit" in out and ("an incumbent exists" in out or "no incumbent was found" in out), repr((rc, out[-300:])))
        rc, out, _ = run(exe, "solve", knap, "--seed", "1", "--time-limit", "0")
        check("solve --time-limit 0 on a MILP exits 4 (time limit)", rc == 4 and "TimeLimit" in out, repr((rc, out[-300:])))
        pcert = os.path.join(md, "p.cert.json")
        rc, out, _ = run(exe, "solve", par, "--write-cert", pcert)
        check("solve on an infeasible MILP exits 3 and says no certificate exists",
              rc == 3 and "Status:        Infeasible" in out and "no certificate" in out, repr((rc, out[-300:])))
        rc, out, _ = run(exe, "solve", knap, "--branching", "mostfrac", "--node-select", "depth", "--heuristics", "off", "--mip-gap", "0", "--mip-abs-gap", "0")
        check("solve accepts --branching, --node-select, --heuristics and the gap options", rc == 0 and "Objective:     -687" in out, repr((rc, out[-300:])))
        rc, _, err = run(exe, "solve", knap, "--branching", "nope")
        check("solve with an unknown branching rule is a usage error (1)", rc == 1 and "unknown branching rule" in err, err)
        rc, _, err = run(exe, "solve", knap, "--node-select", "nope")
        check("solve with an unknown node selection is a usage error (1)", rc == 1 and "unknown node selection" in err, err)
        rc, _, err = run(exe, "solve", knap, "--heuristics", "maybe")
        check("solve --heuristics maybe is a usage error (1)", rc == 1, err)
        rc, out, _ = run(exe, "solve", knap, "--cuts", "on", "--cut-rounds", "3", "--presolve", "off", "--probing", "off")
        check("solve accepts --cuts, --cut-rounds, --presolve and --probing and prints the cut summary",
              rc == 0 and "Objective:     -687" in out and "Cuts:" in out, repr((rc, out[-300:])))
        rc, out, _ = run(exe, "solve", knap, "--cuts", "off")
        check("solve --cuts off runs no cut loop and finds the same optimum",
              rc == 0 and "Objective:     -687" in out and "Cuts:" not in out, repr((rc, out[-300:])))
        rc, _, err = run(exe, "solve", knap, "--cuts", "maybe")
        check("solve --cuts maybe is a usage error (1)", rc == 1, err)
        rc, _, err = run(exe, "solve", knap, "--cut-rounds", "-2")
        check("solve --cut-rounds -2 is a usage error (1)", rc == 1, err)
    # ---- gzip input: read with zlib when built with it, otherwise a clear error; the helper script always works ----
    gz = os.path.join(models, "tiny_lp.mps.gz")
    rc, out, err = run(exe, "info", gz)
    if rc == 0:
        check("a .mps.gz model is read (built with zlib)", "TINY_LP" in out and "Rows:           4" in out, out)
        rc2, out2, _ = run(exe, "solve", gz)
        check("a .mps.gz model is solved (built with zlib)", rc2 == 0 and "Objective:     94" in out2, out2)
    else:
        print("[note] this build has no zlib: the .gz read is not exercised, only the error message")
        check("without zlib a .gz model is a read error (1) that names the build option and the helper script",
              rc == 1 and "SHODHAN_ENABLE_ZLIB" in err and "gunzip_mps.py" in err, repr((rc, err)))
        check("SHODHAN_EXPECT_ZLIB=1 demands a zlib build", os.environ.get("SHODHAN_EXPECT_ZLIB") != "1",
              "this build was expected to have zlib")
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "gunzip_mps.py")
    with tempfile.TemporaryDirectory() as gd:
        expanded = os.path.join(gd, "expanded.mps")
        p = subprocess.run([sys.executable, helper, gz, expanded], capture_output=True, text=True)
        check("gunzip_mps.py expands a model that then solves", p.returncode == 0 and os.path.exists(expanded), p.stdout + p.stderr)
        rc3, out3, _ = run(exe, "solve", expanded)
        check("the expanded model has objective 94", rc3 == 0 and "Objective:     94" in out3, out3)
        bad = os.path.join(gd, "bad.mps.gz")
        with open(gz, "rb") as f:
            data = f.read()
        with open(bad, "wb") as f:
            f.write(data[: len(data) // 2])
        p = subprocess.run([sys.executable, helper, bad], capture_output=True, text=True)
        check("gunzip_mps.py reports a truncated file and leaves no partial output",
              p.returncode == 1 and not os.path.exists(os.path.join(gd, "bad.mps")) and not os.path.exists(os.path.join(gd, "bad.mps.part")), p.stdout + p.stderr)
    rc, _, err = run(exe, "solve")
    check("solve without file is a usage error (1)", rc == 1, err)
    rc, _, err = run(exe, "solve", lp, "--time-limit")
    check("solve --time-limit without a value is a usage error (1)", rc == 1, err)
    rc, _, err = run(exe, "solve", lp, "--time-limit", "abc")
    check("solve --time-limit abc is a usage error (1)", rc == 1, err)
    rc, _, err = run(exe, "solve", lp, "--iter-limit", "1.5")
    check("solve --iter-limit 1.5 is a usage error (1)", rc == 1, err)
    rc, _, err = run(exe, "solve", lp, "--bogus")
    check("solve with an unknown option is a usage error (1)", rc == 1 and "unknown option" in err, err)
    rc, _, err = run(exe, "solve", lp, "--write-sol")
    check("solve --write-sol without a path is a usage error (1)", rc == 1, err)

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
