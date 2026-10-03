"""Shared helpers: write a model and a certificate to temporary files and run the verifier."""

import argparse
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kasauti import cli  # noqa: E402


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def args(**kw):
    ns = argparse.Namespace(mode="exact", primal_tol=1e-6, gap_tol=1e-6, ray_tol=1e-9, int_tol=1e-6, farkas_zero_tol=1e-12,
                            sol=None, report=None, model=None, certificate=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def make_cert(model_text, status, body=None, **problem):
    """A certificate dict for model_text with the correct hash; body is merged in."""
    cert = {
        "format": "shodhan-cert", "version": 1, "solver": {"name": "t", "version": "0"},
        "problem": {"name": "t", "file_sha256": sha(model_text), "rows": problem.get("rows"), "cols": problem.get("cols"),
                    "nnz": problem.get("nnz", 0), "sense": problem.get("sense", "min"), "n_integer": problem.get("n_integer", 0)},
        "status": status, "tolerances": {}, "attempts": {"count": 1, "configuration": "none"},
    }
    cert.update(body or {})
    return cert


def verify(model_text, cert, mode="exact", **kw):
    """Runs the verifier; returns (exit code, report)."""
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "m.mps")
        cp = os.path.join(d, "c.json")
        with open(mp, "w", newline="") as f:
            f.write(model_text)
        with open(cp, "w") as f:
            json.dump(cert, f)
        return cli.verify(mp, cp, args(mode=mode, **kw))


# min 2x + 3y  s.t.  x + y >= 4,  x + 3y >= 6,  x, y >= 0  -> x = 3, y = 1, objective 9.
TINY = """NAME TINY
ROWS
 N COST
 G R1
 G R2
COLUMNS
 X COST 2 R1 1
 X R2 1
 Y COST 3 R1 1
 Y R2 3
RHS
 RHS R1 4 R2 6
ENDATA
"""

TINY_PROB = dict(rows=2, cols=2, nnz=4)
