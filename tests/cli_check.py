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


def main():
    exe, models = sys.argv[1], sys.argv[2]
    lp = os.path.join(models, "tiny_lp.mps")
    mip = os.path.join(models, "tiny_mip.mps")

    rc, out, _ = run(exe, "--version")
    check("version exits 0", rc == 0 and out.startswith("shodhan "), repr((rc, out)))

    rc, out, _ = run(exe, "--help")
    check("help exits 0", rc == 0 and "Usage" in out, repr((rc, out)))

    rc, _, err = run(exe)
    check("no arguments is a usage error (1)", rc == 1 and "Usage" in err, repr((rc, err)))

    rc, _, err = run(exe, "frobnicate")
    check("unknown command is a usage error (1)", rc == 1 and "unknown command" in err)

    rc, _, err = run(exe, "info")
    check("info without file is a usage error (1)", rc == 1)

    rc, out, _ = run(exe, "info", lp)
    check("info on LP exits 0", rc == 0, repr(rc))
    check("info reports shape", "Rows:           4" in out and "Columns:        3" in out, out)
    check("info reports sense", "maximize" in out, out)
    check("info reports ranged row", "1 ranged" in out, out)
    check("info reports offset", "Objective offset: 10" in out, out)

    rc, out, _ = run(exe, "info", mip)
    check("info on MIP exits 0", rc == 0, repr(rc))
    check("info reports column types", "1 continuous, 2 integer, 1 binary" in out, out)

    rc, out, _ = run(exe, "solve", lp)
    check("solve exits 2 (not implemented)", rc == 2, repr(rc))
    check("solve prints the summary and the status", "Rows:" in out and "NotImplemented" in out, out)

    rc, _, err = run(exe, "info", os.path.join(models, "does_not_exist.mps"))
    check("missing file is a read error (1)", rc == 1 and "error" in err, repr((rc, err)))

    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "bad.mps")
        with open(bad, "w") as f:
            f.write("NAME bad\nROWS\n N OBJ\nCOLUMNS\n X NOPE 1\nENDATA\n")
        rc, _, err = run(exe, "info", bad)
        check("malformed file is a read error (1)",
              rc == 1 and "bad.mps:5:" in err and "ReadError" in err, repr((rc, err)))
        rc, out, err = run(exe, "solve", bad)
        check("solve on malformed file exits 1, not 2", rc == 1 and "NotImplemented" not in out)

    if failures:
        print("%d CLI check(s) failed" % len(failures))
        return 1
    print("all CLI checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
