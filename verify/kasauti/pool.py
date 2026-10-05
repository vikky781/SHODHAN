"""Verifier for pooling certificates (docs/POOLING.md, sections 2, 3 and 6).

Written from the documentation: its own parser of the `.pool` file and its own evaluation of the residuals of the
ORIGINAL nonlinear model, in exact rational arithmetic (default) or in floats. It checks feasibility within a stated
tolerance and recomputes the objective exactly. It does NOT check optimality, and it does NOT verify the McCormick bound
that the certificate claims: it reports "bound not verified".

Verdicts: PASS_FEASIBLE (every residual of the nonlinear model is exactly zero), PASS_FEASIBLE_TOL (every residual is
within the tolerance), FAIL, INCONCLUSIVE (status other, unsupported or malformed input).
"""

import math
from fractions import Fraction

from . import checks
from .mps import MpsError, parse_number

PASS = checks.PASS
FAIL = checks.FAIL


class PoolError(Exception):
    pass


class Problem:
    def __init__(self):
        self.name = ""
        self.synthetic = False
        self.n_q = 0
        self.q_names = []
        self.sources = []     # dicts: name, cost, supply (None = inf), q
        self.pools = []       # dicts: name, cap (None = inf)
        self.terminals = []   # dicts: name, price, demand (None = inf), spec (None = inf per quality)
        self.sp = []          # (source index, pool index)
        self.pt = []          # (pool index, terminal index)
        self.st = []          # (source index, terminal index)


def parse_pool(text, exact):
    """Parses the .pool format of docs/POOLING.md section 2. Raises PoolError with a line number."""
    p = Problem()
    names = {}

    def number(tok, line, allow_inf):
        v = parse_number(tok, exact)
        if isinstance(v, float) and math.isinf(v):
            if not allow_inf or v < 0:
                raise PoolError("line %d: infinity not allowed here" % line)
            return None
        return v

    have_q = False
    for line_no, raw in enumerate(text.splitlines(), 1):
        toks = []
        for t in raw.split():
            if t.startswith("#"):
                break
            toks.append(t)
        if not toks:
            continue
        kw = toks[0]
        try:
            if kw == "name":
                if len(toks) != 2:
                    raise PoolError("line %d: name takes one token" % line_no)
                p.name = toks[1]
            elif kw == "synthetic":
                if len(toks) != 2 or toks[1] not in ("yes", "no"):
                    raise PoolError("line %d: synthetic takes yes or no" % line_no)
                p.synthetic = toks[1] == "yes"
            elif kw == "qualities":
                if len(toks) < 2 or not toks[1].isdigit():
                    raise PoolError("line %d: qualities needs a count" % line_no)
                p.n_q = int(toks[1])
                if have_q or not 1 <= p.n_q <= 16:
                    raise PoolError("line %d: qualities is given once with a count from 1 to 16" % line_no)
                p.q_names = toks[2:] if len(toks) > 2 else ["Q%d" % (k + 1) for k in range(p.n_q)]
                if len(p.q_names) != p.n_q:
                    raise PoolError("line %d: wrong number of quality names" % line_no)
                have_q = True
            elif kw in ("source", "pool", "terminal"):
                if not have_q:
                    raise PoolError("line %d: qualities must come first" % line_no)
                name = toks[1]
                if name in names:
                    raise PoolError("line %d: duplicate name %s" % (line_no, name))
                if any(ch in name for ch in '>:"'):
                    raise PoolError("line %d: a name must not contain > : or a quote" % line_no)
                if kw == "source":
                    if len(toks) != 4 + p.n_q:
                        raise PoolError("line %d: a source needs a cost, a supply and %d qualities" % (line_no, p.n_q))
                    names[name] = ("s", len(p.sources))
                    p.sources.append({"name": name, "cost": number(toks[2], line_no, False), "supply": number(toks[3], line_no, True),
                                      "q": [number(t, line_no, False) for t in toks[4:]]})
                    if p.sources[-1]["supply"] is not None and p.sources[-1]["supply"] < 0:
                        raise PoolError("line %d: a supply must not be negative" % line_no)
                elif kw == "pool":
                    if len(toks) != 3:
                        raise PoolError("line %d: a pool needs a capacity" % line_no)
                    names[name] = ("p", len(p.pools))
                    p.pools.append({"name": name, "cap": number(toks[2], line_no, True)})
                    if p.pools[-1]["cap"] is not None and p.pools[-1]["cap"] < 0:
                        raise PoolError("line %d: a capacity must not be negative" % line_no)
                else:
                    if len(toks) != 4 + p.n_q:
                        raise PoolError("line %d: a terminal needs a price, a demand and %d specifications" % (line_no, p.n_q))
                    names[name] = ("t", len(p.terminals))
                    p.terminals.append({"name": name, "price": number(toks[2], line_no, False), "demand": number(toks[3], line_no, True),
                                        "spec": [number(t, line_no, True) for t in toks[4:]]})
                    if p.terminals[-1]["demand"] is not None and p.terminals[-1]["demand"] < 0:
                        raise PoolError("line %d: a demand must not be negative" % line_no)
            elif kw == "arc":
                if len(toks) != 3:
                    raise PoolError("line %d: an arc needs FROM and TO" % line_no)
                a, b = names[toks[1]], names[toks[2]]
                if a[0] == "s" and b[0] == "p":
                    lst = p.sp
                elif a[0] == "p" and b[0] == "t":
                    lst = p.pt
                elif a[0] == "s" and b[0] == "t":
                    lst = p.st
                else:
                    raise PoolError("line %d: an arc must go source->pool, pool->terminal or source->terminal" % line_no)
                if (a[1], b[1]) in lst:
                    raise PoolError("line %d: duplicate arc" % line_no)
                lst.append((a[1], b[1]))
            else:
                raise PoolError("line %d: unknown keyword %s" % (line_no, kw))
        except (IndexError, KeyError, ValueError, MpsError) as e:
            raise PoolError("line %d: %s" % (line_no, e))
    if not have_q or not p.sources or not p.terminals:
        raise PoolError("the file needs qualities, at least one source and at least one terminal")
    if not p.sp and not p.st:
        raise PoolError("no arc leaves a source")
    for i, pool in enumerate(p.pools):
        if not any(b == i for _, b in p.sp) or not any(a == i for a, _ in p.pt):
            raise PoolError("pool %s needs an incoming and an outgoing arc" % pool["name"])
    return p


def arc_names(p):
    """Names of the flow variables of a certificate: FROM>TO."""
    sp = ["%s>%s" % (p.sources[a]["name"], p.pools[b]["name"]) for a, b in p.sp]
    pt = ["%s>%s" % (p.pools[a]["name"], p.terminals[b]["name"]) for a, b in p.pt]
    st = ["%s>%s" % (p.sources[a]["name"], p.terminals[b]["name"]) for a, b in p.st]
    return sp, pt, st


def evaluate(p, ar, f_sp, f_pt, f_st, q):
    """The residuals of docs/POOLING.md section 3 (all relative) and the objective. q is indexed [pool][quality]."""
    zero = ar.zero
    one = ar.num(1)
    nS, nP, nT, K = len(p.sources), len(p.pools), len(p.terminals), p.n_q

    def rel(res, scale):
        return (res if res > 0 else zero) / (one + scale)

    res = {"bounds": zero, "supply": zero, "capacity": zero, "demand": zero, "material": zero, "quality_balance": zero, "terminal_quality": zero}
    for v in list(f_sp) + list(f_pt) + list(f_st):
        res["bounds"] = max(res["bounds"], rel(-v, zero))
    for pi in range(nP):
        for k in range(K):
            qs = [p.sources[s]["q"][k] for s, pool in p.sp if pool == pi]
            lo, hi = (min(qs), max(qs)) if qs else (None, None)
            if lo is None:
                qs = [s["q"][k] for s in p.sources]
                lo, hi = min(qs), max(qs)
            v = q[pi][k]
            res["bounds"] = max(res["bounds"], rel(max(lo - v, v - hi), ar.absval(v)))
    out_s = [zero] * nS
    in_p = [zero] * nP
    out_p = [zero] * nP
    in_t = [zero] * nT
    for (s, pi), v in zip(p.sp, f_sp):
        out_s[s] += v
        in_p[pi] += v
    for (pi, t), v in zip(p.pt, f_pt):
        out_p[pi] += v
        in_t[t] += v
    for (s, t), v in zip(p.st, f_st):
        out_s[s] += v
        in_t[t] += v
    for s in range(nS):
        lim = p.sources[s]["supply"]
        if lim is not None:
            res["supply"] = max(res["supply"], rel(out_s[s] - lim, ar.absval(out_s[s]) + lim))
    for pi in range(nP):
        lim = p.pools[pi]["cap"]
        if lim is not None:
            res["capacity"] = max(res["capacity"], rel(in_p[pi] - lim, ar.absval(in_p[pi]) + lim))
        res["material"] = max(res["material"], rel(ar.absval(in_p[pi] - out_p[pi]), ar.absval(in_p[pi]) + ar.absval(out_p[pi])))
        for k in range(K):
            terms = [p.sources[s]["q"][k] * v for (s, pool), v in zip(p.sp, f_sp) if pool == pi]
            lhs = ar.total(terms)
            scale = ar.total([ar.absval(t) for t in terms]) + ar.absval(q[pi][k] * out_p[pi])
            res["quality_balance"] = max(res["quality_balance"], rel(ar.absval(lhs - q[pi][k] * out_p[pi]), scale))
    for t in range(nT):
        lim = p.terminals[t]["demand"]
        if lim is not None:
            res["demand"] = max(res["demand"], rel(in_t[t] - lim, ar.absval(in_t[t]) + lim))
        for k in range(K):
            spec = p.terminals[t]["spec"][k]
            if spec is None:
                continue
            terms = [q[pi][k] * v for (pi, tt), v in zip(p.pt, f_pt) if tt == t]
            terms += [p.sources[s]["q"][k] * v for (s, tt), v in zip(p.st, f_st) if tt == t]
            lhs = ar.total(terms)
            scale = ar.total([ar.absval(x) for x in terms]) + ar.absval(spec) * ar.absval(in_t[t])
            res["terminal_quality"] = max(res["terminal_quality"], rel(lhs - spec * in_t[t], scale))
    objective = zero
    for (s, pi), v in zip(p.sp, f_sp):
        objective -= p.sources[s]["cost"] * v
    for (pi, t), v in zip(p.pt, f_pt):
        objective += p.terminals[t]["price"] * v
    for (s, t), v in zip(p.st, f_st):
        objective += (p.terminals[t]["price"] - p.sources[s]["cost"]) * v
    return res, objective


def verify_pool(model_path, data, sha, cert, args, rep):
    """Entry point used by cli.verify for a .pool file. Returns (exit code, report)."""
    mode = args.mode if args.mode in ("exact", "float") else "exact"
    exact = mode == "exact"
    ar = checks.Arith(exact)
    tol = ar.num(getattr(args, "pool_tol", 1e-6))
    gap_tol = ar.num(args.gap_tol)
    rep.data.update({"mode": mode, "model": model_path, "model_sha256": sha, "kind": "pooling"})
    try:
        text = data.decode("utf-8", errors="strict")
        p = parse_pool(text, exact)
    except (PoolError, UnicodeDecodeError) as e:
        rep.say("cannot parse the .pool file: %s" % e)
        rep.detail = "INCONCLUSIVE"
        return 2, rep
    rep.say("pooling problem %s: %d sources, %d pools, %d terminals, %d qualities, %d flow variables; mode %s%s" %
            (p.name or "(unnamed)", len(p.sources), len(p.pools), len(p.terminals), p.n_q, len(p.sp) + len(p.pt) + len(p.st), mode,
             "; SYNTHETIC" if p.synthetic else ""))
    prob = cert.get("problem", {})
    sha_ok = prob.get("file_sha256") == sha
    rep.say("file integrity: sha256 %s %s the certificate (%s)" % (sha, "matches" if sha_ok else "DOES NOT MATCH", prob.get("file_sha256")))
    rep.check("file_sha256", sha_ok, model=sha, certificate=prob.get("file_sha256"))
    ident = []
    if prob.get("kind") != "pooling":
        ident.append("kind: %r" % (prob.get("kind"),))
    for key, have in (("sources", len(p.sources)), ("pools", len(p.pools)), ("terminals", len(p.terminals)), ("qualities", p.n_q)):
        if prob.get(key) != have:
            ident.append("%s: certificate says %r, the file has %r" % (key, prob.get(key), have))
    for line in ident:
        rep.say("problem mismatch: " + line)
    rep.check("problem_identity", not ident)
    status = cert.get("status")
    rep.say("status claimed: %s" % status)
    if status == "other":
        rep.say("the certificate has status 'other': it certifies nothing")
        rep.detail = "INCONCLUSIVE"
        return 2, rep
    if status != "feasible":
        rep.say("a pooling certificate has status feasible or other, not %r" % (status,))
        rep.detail = "INCONCLUSIVE"
        return 2, rep
    ok = sha_ok and not ident
    try:
        flows = cert.get("flows")
        qs = cert.get("q")
        if not isinstance(flows, dict) or not isinstance(qs, dict):
            raise checks.CertError("flows and q must be objects")
        sp_n, pt_n, st_n = arc_names(p)
        known = set(sp_n + pt_n + st_n)
        for k in flows:
            if k not in known:
                raise checks.CertError("unknown flow %r" % k)

        def flow(name):
            v = flows.get(name, 0)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise checks.CertError("flow %s is not a number" % name)
            return ar.num(v)

        f_sp, f_pt, f_st = [flow(n) for n in sp_n], [flow(n) for n in pt_n], [flow(n) for n in st_n]
        q = []
        for pool in p.pools:
            row = []
            for k in range(p.n_q):
                key = "%s:%s" % (pool["name"], p.q_names[k])
                if key not in qs:
                    raise checks.CertError("missing pool quality %s" % key)
                if isinstance(qs[key], bool) or not isinstance(qs[key], (int, float)):
                    raise checks.CertError("pool quality %s is not a number" % key)
                row.append(ar.num(qs[key]))
            q.append(row)
        for key in qs:
            if not any(key == "%s:%s" % (pool["name"], p.q_names[k]) for pool in p.pools for k in range(p.n_q)):
                raise checks.CertError("unknown pool quality %r" % key)
    except (checks.CertError, ValueError) as e:
        rep.say("malformed certificate: %s" % e)
        rep.detail = "FAIL"
        return 1, rep
    res, objective = evaluate(p, ar, f_sp, f_pt, f_st, q)
    worst = max(res.values())
    for name, v in res.items():
        rep.say("residual %-16s %s" % (name, checks.fmt(v)))
        rep.check("residual_" + name, v <= tol, value=v, tolerance=tol)
    rep.say("worst relative residual of the nonlinear model: %s (tolerance %s)" % (checks.fmt(worst), checks.fmt(tol)))
    feasible = worst <= tol
    rep.say("objective (exact recomputation): %s" % checks.fmt(objective))
    claim_ok = True
    claimed = cert.get("claimed_objective")
    if claimed is not None:
        diff = ar.absval(ar.num(claimed) - objective)
        claim_ok = diff <= gap_tol * (1 + ar.absval(objective))
        rep.say("claimed objective %s versus exact %s: difference %s" % (checks.fmt(ar.num(claimed)), checks.fmt(objective), checks.fmt(diff)))
        rep.check("claimed_objective", claim_ok, claimed=ar.num(claimed), exact=objective, difference=diff)
    bound_ok = True
    ub = cert.get("claimed_upper_bound")
    if ub is not None:
        ubv = ar.num(ub)
        bound_ok = ubv >= objective - gap_tol * (1 + ar.absval(objective))
        rep.say("claimed upper bound %s: McCormick bound NOT verified (the verifier only checks that it is not below the objective of this feasible point: %s)" %
                (checks.fmt(ubv), "consistent" if bound_ok else "INCONSISTENT"))
        rep.check("upper_bound_consistent", bound_ok, bound=ubv, objective=objective)
    cert_ok = True
    if cert.get("optimality_certified") is True:
        cert_ok = False
        rep.say("the certificate claims optimality_certified: true; a pooling solution is local and nothing here can certify optimality")
    rep.check("no_optimality_claim", cert_ok)
    rep.say("the solution is a LOCAL one: optimality is not certified")
    ok = ok and feasible and claim_ok and bound_ok and cert_ok
    rep.data.update({"residuals": {k: v for k, v in res.items()}, "worst_residual": worst, "objective": objective})
    if not ok:
        rep.detail = "FAIL"
        rep.verdict = FAIL
        return 1, rep
    exactly_zero = worst == 0
    rep.detail = "PASS_FEASIBLE" if exactly_zero and exact else "PASS_FEASIBLE_TOL"
    rep.rigorous = exactly_zero and exact
    rep.verdict = PASS
    rep.data["verdict"] = rep.detail
    rep.data["rigorous"] = rep.rigorous
    return 0, rep
