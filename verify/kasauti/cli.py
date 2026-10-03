"""Command line of KASAUTI: ``python -m kasauti MODEL.mps CERT.json [options]``.

Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (unsupported, malformed, status "other") or usage error.
"""

import argparse
import hashlib
import json
import sys

from . import __version__
from . import checks
from .mps import MpsError, Unsupported, parse_mps, read_bytes

EXACT_NNZ_LIMIT = 200000


def _build_parser():
    p = argparse.ArgumentParser(prog="kasauti", description="Independent verifier for SHODHAN LP certificates (standard library only).")
    p.add_argument("model", help="the model file (MPS, optionally .gz)")
    p.add_argument("certificate", nargs="?", help="the certificate (JSON)")
    p.add_argument("--sol", help="verify a .sol file (primal feasibility and objective only) instead of a certificate")
    p.add_argument("--mode", choices=("auto", "exact", "float"), default="auto",
                   help="exact (Fractions) or float (compensated sums); auto: exact when nnz <= %d, else float with a warning" % EXACT_NNZ_LIMIT)
    p.add_argument("--primal-tol", type=float, default=1e-6, help="relative bound violation accepted (default 1e-6)")
    p.add_argument("--gap-tol", type=float, default=1e-6, help="relative gap between the primal objective and the dual bound (default 1e-6)")
    p.add_argument("--ray-tol", type=float, default=1e-9, help="relative violation of the recession-cone conditions accepted (default 1e-9)")
    p.add_argument("--int-tol", type=float, default=1e-6, help="integrality tolerance for feasible certificates (default 1e-6)")
    p.add_argument("--farkas-zero-tol", type=float, default=1e-12,
                   help="relative size below which coefficients of A^T y are treated as zero if the strict check fails (0 = strict only; default 1e-12)")
    p.add_argument("--dual-zero-tol", type=float, default=1e-9,
                   help="relative size below which a reduced cost or multiplier that meets an infinite bound is treated as zero if the strict dual bound is -infinity (0 = strict only; default 1e-9, the solver accepts 1e-6)")
    p.add_argument("--report", help="write a JSON report to this file")
    p.add_argument("--version", action="version", version="kasauti " + __version__)
    return p


def _read_sol(path):
    values = {}
    claimed = None
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="surrogateescape")
    for no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if parts[0] == "objective" and len(parts) == 2 and claimed is None and no == 1:
            claimed = parts[1]
        elif len(parts) == 2:
            values[parts[0]] = parts[1]
        else:
            raise checks.CertError("%s:%d: cannot read the line" % (path, no))
    return values, claimed


def verify(model_path, cert_path, args, out=None):
    """Runs the verification and returns (exit code, Report). ``out`` receives the text."""
    emit = out or (lambda s: print(s))
    rep = checks.Report()
    opt = checks.Options(args.primal_tol, args.gap_tol, args.ray_tol, args.int_tol, args.farkas_zero_tol, getattr(args, "dual_zero_tol", 1e-9))

    try:
        data = read_bytes(model_path)
    except OSError as e:
        rep.say("cannot read the model file: %s" % e)
        rep.detail = "INCONCLUSIVE"
        return 2, rep
    sha = hashlib.sha256(data).hexdigest()

    cert = None
    values = claimed_sol = None
    try:
        if args.sol:
            values, claimed_sol = _read_sol(args.sol)
        else:
            if not cert_path:
                rep.say("usage: a certificate or --sol is required")
                rep.detail = "INCONCLUSIVE"
                return 2, rep
            with open(cert_path, "rb") as f:
                cert = json.loads(f.read().decode("utf-8", errors="surrogateescape"))
            if not isinstance(cert, dict) or cert.get("format") != "shodhan-cert":
                raise checks.CertError("not a shodhan-cert file")
            if cert.get("version") != 1:
                raise checks.CertError("unsupported certificate version %r" % (cert.get("version"),))
    except (OSError, ValueError, checks.CertError) as e:
        rep.say("cannot use the certificate: %s" % e)
        rep.detail = "INCONCLUSIVE"
        return 2, rep

    # Mode: decide from the size, which is known from a cheap exact-free float parse only for big files;
    # parse in float first when the file is large to count entries.
    mode = args.mode
    try:
        if mode == "auto":
            probe_exact = len(data) <= 4_000_000
            model = parse_mps(data, exact=probe_exact) if probe_exact else parse_mps(data, exact=False)
            if model.nnz <= EXACT_NNZ_LIMIT:
                mode = "exact"
                if not model.exact:
                    model = parse_mps(data, exact=True)
            else:
                mode = "float"
                if model.exact:
                    model = parse_mps(data, exact=False)
                rep.say("WARNING: %d nonzeros exceed %d: using float mode, the result is NOT an exact proof" % (model.nnz, EXACT_NNZ_LIMIT))
        else:
            model = parse_mps(data, exact=(mode == "exact"))
    except Unsupported as e:
        rep.say("unsupported: %s" % e)
        rep.detail = "INCONCLUSIVE"
        return 2, rep
    except MpsError as e:
        rep.say("cannot parse the model file: %s" % e)
        rep.detail = "INCONCLUSIVE"
        return 2, rep
    ar = checks.Arith(mode == "exact")
    rep.data.update({"mode": mode, "model": model_path, "model_sha256": sha})
    rep.say("model: %s, %d rows, %d columns, %d nonzeros, %d integer, sense %s; mode %s" %
            (model.name or "(unnamed)", model.n_rows, model.n_cols, model.nnz, model.n_integer, model.sense, mode))
    for w in model.warnings:
        rep.say("parser warning: %s" % w)

    ok = True
    try:
        if args.sol:
            ok = checks.check_solution_file(model, values, claimed_sol, ar, opt, rep)
        else:
            prob = cert.get("problem", {})
            sha_ok = prob.get("file_sha256") == sha
            rep.say("file integrity: sha256 %s %s the certificate (%s)" % (sha, "matches" if sha_ok else "DOES NOT MATCH", prob.get("file_sha256")))
            rep.check("file_sha256", sha_ok, model=sha, certificate=prob.get("file_sha256"))
            ident = []
            for key, have in (("rows", model.n_rows), ("cols", model.n_cols), ("sense", model.sense), ("n_integer", model.n_integer)):
                if prob.get(key) != have:
                    ident.append("%s: certificate says %r, the file has %r" % (key, prob.get(key), have))
            for line in ident:
                rep.say("problem mismatch: " + line)
            rep.check("problem_identity", not ident)
            if prob.get("nnz") != model.nnz:
                rep.say("note: nnz in the certificate (%r) differs from the parsed count (%d)" % (prob.get("nnz"), model.nnz))
            status = cert.get("status")
            rep.say("status claimed: %s" % status)
            rep.data["status_claimed"] = status
            ok = sha_ok and not ident
            if status == "optimal":
                ok = checks.check_optimal(model, cert, ar, opt, rep) and ok
            elif status == "infeasible":
                ok = checks.check_infeasible(model, cert, ar, opt, rep) and ok
            elif status == "unbounded":
                ok = checks.check_unbounded(model, cert, ar, opt, rep) and ok
            elif status == "feasible":
                ok = checks.check_feasible(model, cert, ar, opt, rep) and ok
            elif status == "other":
                rep.say("the certificate has status 'other': it certifies nothing")
                rep.detail = "INCONCLUSIVE"
                return 2, rep
            else:
                raise checks.CertError("unknown status %r" % (status,))
            if not (sha_ok and not ident) and rep.detail.startswith("PASS"):
                rep.detail = "FAIL"
    except checks.CertError as e:
        rep.say("malformed certificate: %s" % e)
        rep.detail = "FAIL"
        ok = False
    except (ValueError, TypeError) as e:
        rep.say("malformed certificate: %s" % e)
        rep.detail = "FAIL"
        ok = False

    if not rep.detail:
        rep.detail = "PASS" if ok else "FAIL"
    rep.verdict = checks.PASS if ok and rep.detail.startswith("PASS") else checks.FAIL
    rep.data["verdict"] = rep.detail
    rep.data["rigorous"] = rep.rigorous
    return (0 if rep.verdict == checks.PASS else 1), rep


def main(argv=None):
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code not in (0, None) else 0
    code, rep = verify(args.model, args.certificate, args, None)
    for line in rep.lines:
        print(line)
    label = rep.detail or ("PASS" if code == 0 else "FAIL")
    if code == 2:
        label = "INCONCLUSIVE"
    print("VERDICT: " + label + ("" if code != 0 else ("  (rigorous: exact arithmetic)" if rep.rigorous else "  (tolerance-checked, not a proof: see the lines above)")))
    if args.report:
        rep.data["verdict"] = label
        rep.data["lines"] = rep.lines
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep.data, f, indent=2, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
